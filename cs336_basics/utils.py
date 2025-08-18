import torch


class Linear(torch.nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.w = torch.nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.trunc_normal_(self.w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w.T
