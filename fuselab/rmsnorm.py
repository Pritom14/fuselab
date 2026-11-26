"""Fused RMSNorm in Triton (inference, forward only).

y = x / sqrt(mean(x^2) + eps) * w, computed per row in fp32 and written back in x's dtype.

The point of fusing: the unfused PyTorch version makes several passes over x in GPU
memory (square, mean, rsqrt, multiply, multiply). This kernel reads each row once,
does all the maths in registers, and writes the result once.
"""

import torch
import triton
import triton.language as tl

MAX_COLS = 16384  # one row per program; wider rows would spill registers


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    x_row_stride,
    out_row_stride,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)  # one program per row (token)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # one trip in
    x = tl.load(x_ptr + row * x_row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # all the maths stays on-chip
    mean_sq = tl.sum(x * x, axis=0) / n_cols
    y = x * tl.rsqrt(mean_sq + eps) * w

    # one trip out (stored in out's dtype)
    tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6, num_warps: int | None = None) -> torch.Tensor:
    """RMSNorm over the last dimension of ``x`` with weight ``w``."""
    if x.shape[-1] != w.shape[0]:
        raise ValueError(f"last dim of x ({x.shape[-1]}) != len(w) ({w.shape[0]})")
    n_cols = x.shape[-1]
    if n_cols > MAX_COLS:
        raise ValueError(f"hidden size {n_cols} > {MAX_COLS} is not supported by this kernel")

    x2 = x.reshape(-1, n_cols)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    w = w.contiguous()
    out = torch.empty_like(x2)

    block = triton.next_power_of_2(n_cols)
    if num_warps is None:
        num_warps = min(max(block // 256, 1), 8)

    _rmsnorm_kernel[(x2.shape[0],)](
        x2, w, out, x2.stride(0), out.stride(0), n_cols, eps, BLOCK=block, num_warps=num_warps
    )
    return out.reshape(x.shape)


def rmsnorm_reference(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Plain PyTorch, fp32 maths, same formula. Used for correctness and as the 'eager' baseline."""
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    return y.to(x.dtype)
