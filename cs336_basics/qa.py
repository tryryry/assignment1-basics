from typing import Tuple

import torch
from torch import Tensor


# ====================================
# 题 8：GQA 的零拷贝 KV 扩展 + GQA attention
# 8a) expand_kv_for_gqa(k, v, num_q_heads):
#     k/v: [B, H_kv, S, D]，num_q_heads 是 H_kv 的整数倍，G = H_q // H_kv
#     返回 (k_exp, v_exp)，形状 [B, H_kv, G, S, D]
#     硬性要求：必须与输入共享 storage（用 expand，不许 repeat / repeat_interleave /
#              contiguous / reshape 到 4 维——那一步一定会拷贝）。
#     思考：为什么 expand 出来的维度再 reshape 合并就必须拷贝？
# 8b) gqa_attention(q, k, v, causal):
#     q: [B, H_q, S, D]，k/v: [B, H_kv, S, D]
#     要求用 8a 的 5 维 expand 结果 + 把 q 也 view 成 [B, H_kv, G, S, D] 来算，
#     全程不物化 [B, H_q, S, D] 的 KV（这就是 SGLang/vLLM 里真实的做法）
#     分组约定：q head i 对应 kv head i // G（同组 q head 连续）
#     返回 [B, H_q, S, D]
# ============================================================


# 8a
def expand_kv_for_gqa(k: Tensor, v: Tensor, num_q_heads: int) -> Tuple[Tensor, Tensor]:
    G = num_q_heads // k.shape(1)
    k = k.unsqueeze(2)  # [B, H_kv, S, D] -> [B, H_kv, 1,  S, D]
    k = k.expand(-1, -1, G, -1, -1)  # [B, H_kv, 1,  S, D] -> [B, H_kv, G,  S, D]
    v = v.unsqueeze(2)
    v = v.expand(-1, -1, G, -1, -1)
    return k, v


# 8b
# G = 4
# Q 头 0~3   → KV 头 0
# Q 头 4~7   → KV 头 1
def gqa_attention(q: Tensor, k: Tensor, v: Tensor, causal: bool) -> Tensor:
    B, H_q, S, D = q.shape
    H_kv = k.shape(1)
    G = H_q // H_kv
    q = q.view(B, H_kv, G, S, D)  # [B, H_q, S, D] -> [B, H_kv, G, S, D]

    k, v = expand_kv_for_gqa(k, v, H_q)

    # att
    score = torch.matmul(q, k.transpose(-2, -1)) / (D**0.5)
    if causal:
        mask = torch.ones(S, S).triu(diagonal=1)
        score = score.masked_fill(mask, float("-inf"))
    att = torch.softmax(score, dim=-1)
    out = torch.matmul(att, v)

    return out.view(B, H_q, S, D)


# ============================================================
# 题 8c：张量并行的 Attention 模块（一个完整 attention block）
# 类 TPAttention —— 复刻 SGLang/vLLM 里一层 attention 的 TP 前向：
#   forward(hidden_states) 输入 [B, S, hidden]，输出 [B, S, hidden]。
#   内部持有「本 rank 的分片权重」：
#     - q_proj / k_proj / v_proj 是 column parallel：按 head 切 out_features；
#       q head 均分；kv head 够分时均分，不够分（num_kv_heads < tp_size）时被复制。
#     - o_proj 是 row parallel：按 in_features(=H_q*D) 切列，本 rank 只持有
#       和自己 q head 对应的那几列。
#   计算流程：qkv 投影 → reshape 成 head → gqa_attention（本地）→ o_proj 部分和
#            → all_reduce(SUM)。
#   同时覆盖 MHA（num_kv_heads == num_q_heads）与 GQA（num_kv_heads < num_q_heads）。
#   为便于判分，用 from_full() 从「完整权重」按 rank 切片构造；
#   各 rank forward 后做 all_reduce（或单进程下把各 rank 输出相加）应 == 单卡整权重前向。
# ============================================================
class TPAttention(torch.nn.Module):
    def __init__(self, num_q_heads, num_kv_heads, head_dim, tp_size, rank, causal):
        super().__init__()

        self.num_q_heads = num_q_heads
        self.head_dim = head_dim
        self.tp_size = tp_size
        self.rank = rank
        self.causal = causal
        self.local_q_heads = num_q_heads // tp_size

        hidden_size = num_q_heads * head_dim
        self.local_q_size = self.local_q_heads * head_dim
        self.q_proj = torch.nn.Parameter(torch.empty(self.local_q_size, hidden_size))
        self.k_proj = torch.nn.Parameter(torch.empty(self.local_q_size, hidden_size))
        self.v_proj = torch.nn.Parameter(torch.empty(self.local_q_size, hidden_size))
        self.o_proj = torch.nn.Parameter(torch.empty(hidden_size, self.local_q_size))

    # q_weight [H_q, D, h] o_weight [h, H_q, D]
    @classmethod
    def from_full(
        cls,
        q_weight,
        k_weight,
        v_weight,
        o_weight,
        num_q_heads,
        num_kv_heads,
        head_dim,
        tp_size,
        rank,
        causal,
    ):
        module = cls(num_q_heads, num_kv_heads, head_dim, tp_size, rank, causal)
        q_start = rank * module.local_q_heads
        q_end = q_start + module.local_q_heads
        q_weight_split = q_weight[q_start:q_end]
        k_weight_split = k_weight[q_start:q_end]
        v_weight_split = v_weight[q_start:q_end]

        o_weight_split = o_weight[:, q_start:q_end, D]

        return module

    def forward(self, hidden_states: Tensor):
        B, S, hidden_size = hidden_states.shape
        Q = torch.matmul(hidden_states, self.q_proj.transpose(-1, -2))
        K = torch.matmul(hidden_states, self.k_proj.transpose(-1, -2))
        V = torch.matmul(hidden_states, self.v_proj.transpose(-1, -2))
        Q = Q.reshape(B, S, self.local_q_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, S, self.local_kv_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, S, self.local_kv_heads, self.head_dim).transpose(1, 2)

        out = gqa_attention(Q, K, V, self.causal)
        return torch.matmul(
            out.transpose(1, 2).reshape(B, S, self.local_q_size),
            self.o_proj.transpose(-1, -2),
        )


# ============================================================
# 题 13：padded <-> ragged 互转（cu_seqlens 是 FlashAttention 的输入约定）
#   pad_to_ragged(padded, seq_lens):
#       padded: [B, S_max, D]，seq_lens: [B] int64 2 3
#       返回 (flat, cu_seqlens)
#       flat: [sum(seq_lens), D]，按 batch 顺序把有效 token 拼起来
#       cu_seqlens: [B+1] int32，前缀和，cu_seqlens[0] == 0
#   ragged_to_pad(flat, cu_seqlens, pad_value=0.0):
#       逆操作，返回 [B, S_max, D]，S_max = max(seq 长度)，padding 位填 pad_value
# 要求：都不允许 python 循环（提示：arange 广播出 mask，再布尔索引 / index_put）
# ============================================================
def pad_to_ragged(padded, seq_lens):
    B, S_max, D = padded.shape
    mask = torch.arange(S_max)[None, :] < seq_lens[:, None]

    # mask = torch.zeros(B, S_max, dtype=torch.bool)
    # for b in range(B):
    #     for s in range(S_max):
    #         mask[b, s] = s < seq_lens[b]
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)

    return padded[mask], cu_seqlens


def ragged_to_pad(flat: Tensor, cu_seqlens: Tensor, pad_value=0.0):
    B = cu_seqlens.shape[0] - 1
    D = flat.shape[1]

    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    S_max = torch.max(seq_lens)
    mask = torch.arange(S_max)[None, :] < seq_lens[:, None]

    out = torch.full((B, S_max, D), pad_value)
    out[mask] = flat
    return out


# ============================================================
# 题 14：变长（块对角）causal attention
#   q_flat/k_flat/v_flat: [T, H, D]，T = sum(seq_lens)，多条序列首尾拼接
#   cu_seqlens: [B+1]
#   语义：每个 token 只能注意到"同一条序列内且不晚于自己"的 token
#   返回 [T, H, D]
# 要求：不允许在 B 上写 python 循环；构造 [T, T] 的块对角 causal mask 一次算完。
# 提示：先用 cu_seqlens 生成每个 token 的 seq_id 和序列内 position，
#      mask = (seq_id_i == seq_id_j) & (pos_j <= pos_i)
# ============================================================
# flatten


def varlen_attention(
    q_flat: Tensor, k_flat: Tensor, v_flat: Tensor, cu_seqlens: Tensor
) -> Tensor:
    T, H, D = q_flat.shape
    B = cu_seqlens.shape[0] - 1
    score = torch.matmul(
        q_flat.transpose(0, 1), k_flat.transpose(0, 1).transpose(-1, -2)
    ) / (D**0.5)
    # mask
    seq_len = cu_seqlens[1:] - cu_seqlens[:-1]
    # [0 0  1 1 B-1]
    seq_id = torch.arange(B).repeat_interleave(seq_len)  # [T]
    # [0 1 T-1] - [ 0 ]
    seq_pos = torch.arange(T) - cu_seqlens[seq_id]  # [T]
    # for i in range(T):
    #     for j in range(T):
    #         mask[i][j] = (seq_id[i] == seq_id[j]) and (seq_pos[i] >= seq_pos[j])
    same_seq = seq_id[:, None] == seq_id[None, :]  # [T, T]
    causal_seq = seq_pos[:, None] >= seq_pos[None, :]  # [T, T]
    mask = same_seq & causal_seq
    score = score.masked_fill(~mask, float("-inf"))

    att = torch.softmax(score, dim=-1)
    out = torch.matmul(att, v_flat.transpose(0, 1))
    return out.transpose(0, 1)


# ============================================================
# 题 15：连续批处理 prefill 的元信息
#   prefix_lens: [B] int64，命中前缀缓存的长度（这部分 KV 已存在，不再计算）
#   extend_lens: [B] int64，本次需要前向的 token 数
#   返回 (positions, extend_start_loc)
#     positions: [sum(extend_lens)] int64，
#                第 b 条序列贡献 prefix_lens[b] .. prefix_lens[b]+extend_lens[b]-1
#     extend_start_loc: [B] int64，每条序列在拼接后张量里的起始下标（即 extend_lens 的
#                       exclusive 前缀和）
# 要求：无 python 循环。提示 repeat_interleave + arange 减去 offset。
# ============================================================
# prefix_lens: [2 0] extend_lens: [3 2]


def build_position_ids(
    prefix_lens: Tensor, extend_lens: Tensor
) -> Tuple[Tensor, Tensor]:
    B = prefix_lens.shape[0]
    T = torch.sum(extend_lens)
    extend_start_loc = (
        torch.cumsum(extend_lens, dim=0) - extend_lens
    )  # [3 5] - [3 2] = [0 3]

    offset_id = torch.arange(B).repeat_interleave(
        extend_lens
    )  # [sum(extend_lens)] [0 0 0 1 1]
    offset = torch.arange(T) - extend_lens[offset_id]  # [0 1 2 0 1]
    pos = (
        prefix_lens.repeat_interleave(extend_lens) + offset
    )  # [2 2 2 0 0] + [0 1 2 0 1] = [2 3 4 0 1]

    return pos, extend_start_loc
