import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _scan_add(lhs, rhs):
    return lhs + rhs


@triton.jit
def _reverse_cumsum_kernel(
    x_ptr,
    y_ptr,
    x_stride_row,
    y_stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(axis=0)
    offsets = tl.arange(0, BLOCK)
    reverse_offsets = BLOCK - 1 - offsets

    values = tl.load(x_ptr + row * x_stride_row + reverse_offsets)
    scanned = tl.associative_scan(values, axis=0, combine_fn=_scan_add)
    tl.store(y_ptr + row * y_stride_row + reverse_offsets, scanned)

class Model(nn.Module):
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cumsum(x.flip(self.dim), dim=self.dim).flip(self.dim)

batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]


class ModelNew(Model):
    def forward(self, x):
        y = torch.empty_like(x)
        _reverse_cumsum_kernel[(x.shape[0],)](
            x,
            y,
            x.stride(0),
            y.stride(0),
            BLOCK=x.shape[1],
            num_warps=32,
            num_stages=1,
        )
        return y
