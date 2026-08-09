import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum
from einops import rearrange


class Linear(torch.nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.w = torch.nn.Parameter(
            torch.empty(out_features, in_features, **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w.T


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embedding = torch.nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, **factory_kwargs)
        )
        torch.nn.init.trunc_normal_(self.embedding)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]


class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {"device": device, "dtype": dtype}
        self.g = torch.nn.Parameter(torch.empty(d_model, **factory_kwargs))
        torch.nn.init.trunc_normal_(self.g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        # 避免溢出
        x = x.to(torch.float32)
        x = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) * self.g
        return x.to(in_dtype)


class Swiglu(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)

    def forward(
        self,
        x: Float[Tensor, " ... d_model"],
    ):
        z = self.w1(x)
        return self.w2(torch.sigmoid(z) * z * self.w3(x))


class Rope(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        freq = self.theta ** (
            -torch.arange(0, self.d_k, 2, device=device) / self.d_k
        )  # (d_k/2, )
        positions = torch.arange(max_seq_len, device=device)  # (max_seq_len, )
        angles = positions[..., None] * freq  # (max_seq_len, d_k/2)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)


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


class MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        d_k=None,
        d_v=None,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.d_k = d_k if d_k is not None else d_model 
        self.d_v = d_v if d_v is not None else d_model 

        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(d_in, self.d_k)
        self.k_proj = Linear(d_in, self.d_k)
        self.v_proj = Linear(d_in, self.d_v)
        self.o_proj = Linear(self.d_v, d_model)
        if max_seq_len is not None:
            self.rope = Rope(
                theta=theta, d_k=self.d_k // num_heads, max_seq_len=max_seq_len
            )
        else:
            self.rope = None

    def forward(
        self,
        x: Float[Tensor, " ... sequence_length d_in"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = rearrange(Q, "... seq (h d) -> ... h seq d", h=self.num_heads)
        K = rearrange(K, "... seq (h d) -> ... h seq d", h=self.num_heads)
        V = rearrange(V, "... seq (h d) -> ... h seq d", h=self.num_heads)
        if token_positions is not None and self.rope is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        seq = x.shape[-2]
        mask = torch.tril(
            torch.ones(seq, seq, dtype=torch.bool, device=x.device), diagonal=0
        )
        O_Att = Scaled_dot_product_attention(Q, K, V, mask)
        O_Att = rearrange(O_Att, "... h seq d -> ... seq (h d)")
        return self.o_proj(O_Att)
    
class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        
        self.attn = MultiHeadAttention(d_in=d_model, d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, theta=theta)
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        self.ffn= Swiglu(d_model=d_model, d_ff=d_ff)

    def forward(self, x: Float[Tensor, " batch sequence_length d_model"]):
        token_positions = torch.arange(x.shape[-2],device=x.device,)
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))
    
class TransformerLM(torch.nn.Module):
    def __init__()