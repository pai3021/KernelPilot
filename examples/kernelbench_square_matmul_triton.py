import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a, b, c,
    N: tl.constexpr,
    STRIDE_AM: tl.constexpr, STRIDE_AK: tl.constexpr,
    STRIDE_BK: tl.constexpr, STRIDE_BN: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Nearby programs share B tiles while progressing through a group of A rows.
    pid = tl.program_id(0)
    num_m = tl.cdiv(N, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    programs_per_group = GROUP_M * num_n
    group = pid // programs_per_group
    first_m = group * GROUP_M
    group_size = tl.minimum(num_m - first_m, GROUP_M)
    pid_in_group = pid % programs_per_group
    pid_m = first_m + pid_in_group % group_size
    pid_n = pid_in_group // group_size

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ks = tl.arange(0, BLOCK_K)
    a_ptrs = a + rows[:, None] * STRIDE_AM + ks[None, :] * STRIDE_AK
    b_ptrs = b + ks[:, None] * STRIDE_BK + cols[None, :] * STRIDE_BN
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Keep every K contribution. tf32x3 uses three tensor-core products to
    # approximate FP32 multiplication without ordinary TF32's input truncation.
    for k_block in range(tl.cdiv(N, BLOCK_K)):
        valid_k = k_block * BLOCK_K + ks < N
        a_tile = tl.load(a_ptrs, mask=(rows[:, None] < N) & valid_k[None, :], other=0.0)
        b_tile = tl.load(b_ptrs, mask=valid_k[:, None] & (cols[None, :] < N), other=0.0)
        acc = tl.dot(a_tile, b_tile, acc, input_precision="tf32x3")
        a_ptrs += BLOCK_K * STRIDE_AK
        b_ptrs += BLOCK_K * STRIDE_BK

    c_ptrs = c + rows[:, None] * N + cols[None, :]
    tl.store(c_ptrs, acc, mask=(rows[:, None] < N) & (cols[None, :] < N))


@torch.no_grad()
def run(a, b):
    n = a.shape[0]
    out = torch.empty((n, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(n, 128) * triton.cdiv(n, 128),)
    matmul_kernel[grid](
        a, b, out,
        N=n,
        STRIDE_AM=a.stride(0), STRIDE_AK=a.stride(1),
        STRIDE_BK=b.stride(0), STRIDE_BN=b.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_M=8,
        num_warps=8, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return run(a, b)
