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
    k = k.unsqueeze(2) # [B, H_kv, S, D] -> [B, H_kv, 1,  S, D]
    k = k.expand(-1, -1, G, -1, -1) # [B, H_kv, 1,  S, D] -> [B, H_kv, G,  S, D] 
    v = v.unsqueeze(2)
    v = v.expand(-1, -1, G, -1, -1) 
    return k, v

# 8b
# G = 4
# Q 头 0~3   → KV 头 0
# Q 头 4~7   → KV 头 1
def gqa_attention(q:Tensor, k: Tensor, v: Tensor, causal: bool) -> Tensor:
    B, H_q, S, D = q.shape
    H_kv = k.shape(1)
    G = H_q // H_kv
    q = q.view(B, H_kv, G, S, D) # [B, H_q, S, D] -> [B, H_kv, G, S, D] 
    
    k, v = expand_kv_for_gqa(k, v, H_q)
    
    # att
    score = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
    if causal:
        mask = torch.ones(S, S).triu(diagonal=1)
        score = score.masked_fill(mask, float("-inf"))
    att= torch.softmax(score, dim=-1)
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
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tp_size= tp_size
        self.rank = rank
        self.causal = causal
        self.local_q_heads = num_q_heads // tp_size
        if num_kv_heads > tp_size:
            self.local_kv_heads = num_kv_heads // tp_size
        else:
            self.local_kv_heads = 1

        hidden_size = num_q_heads * head_dim
        self.local_q_size = self.local_q_heads * head_dim
        self.local_kv_size = self.local_kv_heads * head_dim
        self.q_proj = torch.nn.Parameter(torch.empty(self.local_q_size, hidden_size))
        self.k_proj = torch.nn.Parameter(torch.empty(self.local_kv_size, hidden_size))
        self.v_proj = torch.nn.Parameter(torch.empty(self.local_kv_size, hidden_size))
        self.o_proj = torch.nn.Parameter(torch.empty(hidden_size, self.local_q_size))

    def forward(self, hidden_states: Tensor):
        B, S, hidden_size = hidden_states.shape
        Q = torch.matmul(hidden_states, self.q_proj.transpose(-1, -2))
        K = torch.matmul(hidden_states, self.k_proj.transpose(-1, -2))
        V = torch.matmul(hidden_states, self.v_proj.transpose(-1, -2))
        Q = Q.reshape(B, S, self.local_q_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, S, self.local_kv_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, S, self.local_kv_heads, self.head_dim).transpose(1, 2)

        out = gqa_attention(Q, K, V, self.causal)
        return torch.matmul(out.transpose(1, 2).reshape(B, S, self.local_q_size), self.o_proj.transpose(-1, -2))