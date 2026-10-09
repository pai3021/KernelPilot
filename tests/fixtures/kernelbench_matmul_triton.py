import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(a, b, c, n: tl.constexpr, block: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offsets_m = pid_m * block + tl.arange(0, block)
    offsets_n = pid_n * block + tl.arange(0, block)
    offsets_k = tl.arange(0, block)
    acc = tl.zeros((block, block), tl.float32)
    for k in range(0, n, block):
        a_ptrs = a + offsets_m[:, None] * n + (k + offsets_k[None, :])
        b_ptrs = b + (k + offsets_k[:, None]) * n + offsets_n[None, :]
        acc += tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), input_precision="ieee")
    c_ptrs = c + offsets_m[:, None] * n + offsets_n[None, :]
    tl.store(c_ptrs, acc)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        n = a.shape[0]
        out = torch.empty((n, n), device=a.device, dtype=a.dtype)
        grid = (triton.cdiv(n, 32), triton.cdiv(n, 32))
        matmul_kernel[grid](a, b, out, n=n, block=32, num_warps=4)
        return out
