"""Fused residual add + RMSNorm + per-token INT8 quantisation (inference, forward only).

What a serving engine runs before every INT8 matmul, in one pass:

    residual = x + residual                      (written back in place, in the input dtype)
    y        = residual / sqrt(mean(residual^2) + eps) * w        (fp32)
    scale    = max(|y|) / 127                    (one fp32 per row / token)
    q        = round(y / scale)                  (int8, round half away from zero)

Unfused PyTorch makes about eight passes over the row in GPU memory (add, store, square, mean,
rsqrt, multiply, multiply, abs-max, divide, round, cast). This kernel reads x and residual once and
writes residual, q and scale once.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_COLS = 16384
QMAX = 127.0
MIN_AMAX = 1e-8  # a row of zeros gets a tiny scale instead of dividing by zero


@triton.jit
def _add_rmsnorm_quant_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    q_ptr,
    scale_ptr,
    x_stride,
    res_stride,
    q_stride,
    n_cols,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    # one trip in
    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0)
    res = tl.load(res_ptr + row * res_stride + cols, mask=mask, other=0.0)
    r = x + res  # in the input dtype, like the unfused add
    tl.store(res_ptr + row * res_stride + cols, r, mask=mask)  # residual updated in place

    rf = r.to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = rf * tl.rsqrt(tl.sum(rf * rf, axis=0) / n_cols + eps) * w

    amax = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-8)
    v = y * (127.0 / amax)
    qf = tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))  # round half away from zero
    qf = tl.minimum(tl.maximum(qf, -127.0), 127.0)

    # one trip out
    tl.store(q_ptr + row * q_stride + cols, qf.to(tl.int8), mask=mask)
    tl.store(scale_ptr + row, amax / 127.0)


def add_rmsnorm_quant(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float = 1e-6, num_warps: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (q int8 [..., N], scale fp32 [..., 1], residual). ``residual`` is updated in place."""
    if x.shape != residual.shape or x.dtype != residual.dtype:
        raise ValueError("x and residual must have the same shape and dtype")
    n = x.shape[-1]
    if w.shape[0] != n:
        raise ValueError(f"last dim of x ({n}) != len(w) ({w.shape[0]})")
    if n > MAX_COLS:
        raise ValueError(f"hidden size {n} > {MAX_COLS} is not supported by this kernel")
    if not residual.is_contiguous():
        raise ValueError("residual must be contiguous (it is written in place)")

    x2 = x.reshape(-1, n)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    r2 = residual.view(-1, n)
    w = w.contiguous()
    q = torch.empty(x2.shape, dtype=torch.int8, device=x.device)
    scale = torch.empty((x2.shape[0], 1), dtype=torch.float32, device=x.device)

    block = triton.next_power_of_2(n)
    if num_warps is None:
        num_warps = min(max(block // 256, 1), 8)
    _add_rmsnorm_quant_kernel[(x2.shape[0],)](
        x2, r2, w, q, scale, x2.stride(0), r2.stride(0), q.stride(0), n, eps, BLOCK=block, num_warps=num_warps
    )
    return q.reshape(x.shape), scale.reshape(*x.shape[:-1], 1), residual


def add_rmsnorm_quant_reference(
    x: torch.Tensor, residual: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Plain PyTorch, same formula and rounding. Does NOT modify ``residual``; returns the new one."""
    r = x + residual
    rf = r.float()
    y = rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    amax = y.abs().amax(-1, keepdim=True).clamp_min(MIN_AMAX)
    v = y * (QMAX / amax)
    q = torch.where(v >= 0, torch.floor(v + 0.5), torch.ceil(v - 0.5)).clamp(-QMAX, QMAX).to(torch.int8)
    return q, amax / QMAX, r
