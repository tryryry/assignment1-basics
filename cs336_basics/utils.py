import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum


class Linear(torch.nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.trunc_normal_(self.w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w.T


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embedding = torch.nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        torch.nn.init.trunc_normal_(self.embedding)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.g = torch.nn.Parameter(torch.empty(d_model))
        torch.nn.init.trunc_normal_(self.g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        # 避免溢出
        x = x.to(torch.float32)
        x = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) * self.g
        return x.to(in_dtype)


class Swiglu(torch.nn.Module):
    def __init__(self, d_modle: int, d_ff: int):
        super().__init__()
        self.d_modle = d_modle
        self.d_ff = d_ff

    def forward(
        self,
        in_features: Float[Tensor, " ... d_model"],
        w1_weight: Float[Tensor, " d_ff d_model"],
        w2_weight: Float[Tensor, " d_model d_ff"],
        w3_weight: Float[Tensor, " d_ff d_model"],
    ):
        # W1 @ x
        linear1 = Linear(self.d_modle, self.d_ff)
        linear1.load_state_dict({"w": w1_weight})
        x1 = linear1.forward(in_features)
        # SiLu(x)
        x1 = x1 * torch.sigmoid(x1)
        # W3 @ x
        linear3 = Linear(self.d_modle, self.d_ff)
        linear3.load_state_dict({"w": w3_weight})
        x3 = linear3.forward(in_features)
        x = x1 * x3
        # W2 @ x
        linear2 = Linear(self.d_ff, self.d_modle)
        linear2.load_state_dict({"w": w2_weight})
        return linear2(x)


def RotaryPositionalEmbedding(
    theta: float,
    d_k: int,
    max_seq_len: int,
    x: torch.Tensor,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    freq = theta ** (-torch.arange(0, d_k, 2) / d_k)
    angles = token_positions[..., None] * freq
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos
    return torch.stack(
        (x_rot_even, x_rot_odd),
        dim=-1,
    ).flatten(-2)


def Softmax(in_features: Float[Tensor, " ..."], dim: int):
    max = torch.amax(in_features, dim=dim, keepdim=True)
    shifted_features = in_features - max
    exp_features = torch.exp(shifted_features)
    return exp_features / torch.sum(exp_features, dim=dim, keepdim=True)


def Scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    scores = scores / (Q.shape[-1] ** (0.5))
    scores = scores.masked_fill(~mask, float("-inf"))
    attention = Softmax(scores, -1)
    return einsum(attention, V, "... queries keys, ... keys d_v -> ... queries d_v")
