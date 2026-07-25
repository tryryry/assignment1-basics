import ast
import re
from collections.abc import Iterable, Iterator
import regex as re

global_pattern = (
    r"'(?:[sdmt]|ll|ve|re)"
    r"| ?\p{L}+"
    r"| ?\p{N}+"
    r"| ?[^\s\p{L}\p{N}]+"
    r"|\s+(?!\S)"
    r"|\s+"
)


class tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        vocab = {}
        merges = []
        with open(vocab_filepath, "r", encoding="utf-8") as file:
            vocab = ast.literal_eval(file.read())
        with open(merges_filepath, "r", encoding="utf-8") as file:
            merges = ast.literal_eval(file.read())
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        bytes_to_int: dict[bytes, int] = {
            value: key for key, value in self.vocab.items()
        }
        if self.special_tokens:
            pattern = "|".join(
                re.escape(tok)
                for tok in sorted(self.special_tokens, key=len, reverse=True)
            )
            split_chunks = re.split(f"({pattern})", text)
        else:
            split_chunks = [text]
        res = []
        for chunk in split_chunks:
            whole_chunk_bytes = chunk.encode("utf-8")
            if whole_chunk_bytes in bytes_to_int:
                res.append(bytes_to_int[whole_chunk_bytes])
                continue
            pre_tokens = re.findall(global_pattern, chunk)
            for pre_token in pre_tokens:
                chunk_bytes_list = [
                    bytes([value]) for value in pre_token.encode("utf-8")
                ]
                need_merge = True
                while need_merge:
                    need_merge = False
                    for merge_bytes_tuple in self.merges:
                        for i, pair in enumerate(
                            zip(chunk_bytes_list, chunk_bytes_list[1:])
                        ):
                            if pair == merge_bytes_tuple:
                                chunk_bytes_list[i : i + 2] = [
                                    chunk_bytes_list[i] + chunk_bytes_list[i + 1]
                                ]
                                need_merge = True
                                break
                        if need_merge:
                            break

                for chunk_bytes in chunk_bytes_list:
                    if chunk_bytes in bytes_to_int:
                        res.append(bytes_to_int[chunk_bytes])
        return res

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:

        for str_ in iterable:
            yield from self.encode(str_)

    def decode(self, ids: list[int]) -> str:
        ids_bytes = b""
        for id in ids:
            if id in self.vocab:
                ids_bytes = ids_bytes + self.vocab[id]
            else:
                ids_bytes = ids_bytes + "\ufffd".encode("utf-8")
        return ids_bytes.decode("utf-8", errors="replace")
