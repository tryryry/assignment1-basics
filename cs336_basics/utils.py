import torch


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
        x = x.to(torch.float32)
        x = x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps) * self.g
        return x.to(in_dtype)

    def rms(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = torch.mean(x * x, dim=-1, keepdim=True)
        return torch.sqrt(x_mean + self.eps)
