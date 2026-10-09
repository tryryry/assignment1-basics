from typing import Tuple

import torch
from torch import Tensor


#  b s h
#
def quantize_per_token_int8(x: Tensor) -> Tuple[Tensor, Tensor]:
    x = x.float()
    x_max = x.abs().amax(dim=-1, keepdim=True)  # b s 1
    scale = x_max / 127  # b s 1
    scale = torch.where(scale == 0, 1.0, scale)
    q = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def dequantize_per_token_int8(q: Tensor, scale: Tensor) -> Tensor:
    return q * scale


# todo int8 代码 vllm

x = torch.tensor(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
)
print(x.shape)  # [2,3]
print(x.ndim)  # 2
print(x.numel())  #
# dim=0 2
# dim=1 3
# dim=-1 3

# 整数索引去掉对应维度，切片保留对应维度。
print(x[0].shape)  # [3] tensor([1, 2, 3])
print(x[0:1].shape)  # [1, 3] tensor([[1, 2, 3]])
print(x[:, 1].shape)  # [2] tensor([2, 5])
print(x[:, 1:2].shape)  # [2, 1] tensor([[2], [5]])
print(x[1, 2].shape)  # [] tensor(6)
