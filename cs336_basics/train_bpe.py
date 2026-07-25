import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool
from collections import defaultdict


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def chunk_tokenize(chunk: str, special_tokens: list[str]) -> dict[str, int]:
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    split_chunks = re.split(pattern, chunk)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_token_count = {}
    for split_chunk in split_chunks:
        for m in re.finditer(PAT, split_chunk):
            pre_token_count[m.group()] = pre_token_count.get(m.group(), 0) + 1
    return pre_token_count


def bpe_algorithm(
    pre_token_count: dict,
    vocab: dict,
    vocab_size: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merge = []
    pair_counts = {}
    word_tokens = defaultdict(list)
    pair_to_words = defaultdict(set)
    word_id = 0
    word_freq = {}
    for word_id, (token, count) in enumerate(pre_token_count.items()):
        tokens = [bytes([b]) for b in token.encode("utf-8")]
        word_tokens[word_id] = tokens
        word_freq[word_id] = count
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            pair_to_words[pair].add(word_id)

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        max_pair = max(pair_counts, key=lambda k: (pair_counts[k], k))
        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        merge.append(max_pair)

        for word_id in pair_to_words[max_pair]:
            new_word_token = []
            for i in range(len(word_tokens[word_id]) - 1):
                pair = (word_tokens[word_id][i], word_tokens[word_id][i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) - word_freq[word_id]
            j = 0
            while j < len(word_tokens[word_id]):
                if (
                    word_tokens[word_id][j] == max_pair[0]
                    and j + 1 < len(word_tokens[word_id])
                    and word_tokens[word_id][j + 1] == max_pair[1]
                ):
                    new_word_token.append(max_pair[0] + max_pair[1])
                    j = j + 2
                else:
                    new_word_token.append(word_tokens[word_id][j])
                    j = j + 1
            word_tokens[word_id] = new_word_token
            for i in range(len(new_word_token) - 1):
                pair = (new_word_token[i], new_word_token[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + word_freq[word_id]
                pair_to_words[pair].add(word_id)
        del pair_counts[max_pair]
        del pair_to_words[max_pair]
    return vocab, merge


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for i, token in enumerate(special_tokens):
        vocab[256 + i] = token.encode("utf8")

    pre_token_count = {}
    ## Usage
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunks.append(chunk)

        results = {}
        args = [(chunk, special_tokens) for chunk in chunks]
        with Pool(processes=num_processes) as pool:
            results = pool.starmap(chunk_tokenize, args)

        for local_res in results:
            for token, count in local_res.items():
                pre_token_count[token] = pre_token_count.get(token, 0) + count
    # for token, count in pre_token_count.items():
    #    print(f"Token: {token}, Count: {count}")
    return bpe_algorithm(pre_token_count, vocab, vocab_size)
