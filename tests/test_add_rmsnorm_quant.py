"""Correctness of fused residual add + RMSNorm + INT8 quantisation against the PyTorch reference.

Runs on a CUDA GPU, or on CPU with TRITON_INTERPRET=1 (slow, small shapes only).
"""

import os

import pytest
import torch

from fuselab.add_rmsnorm_quant import add_rmsnorm_quant, add_rmsnorm_quant_reference

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu" and not INTERPRET:
    pytest.skip("needs a CUDA GPU or TRITON_INTERPRET=1", allow_module_level=True)

DTYPES = [torch.float32, torch.float16]
if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    DTYPES.append(torch.bfloat16)

ROWS = [1, 7, 64] if INTERPRET else [1, 7, 128, 4096]
COLS = [128, 1000, 2560] if INTERPRET else [128, 1000, 2560, 4096, 8192]


def _check(x, res, w):
    want_q, want_scale, want_res = add_rmsnorm_quant_reference(x, res, w)
    res_in = res.clone()
    q, scale, out_res = add_rmsnorm_quant(x, res_in, w)
    assert out_res is res_in, "residual is updated in place and returned"
    # the residual add happens in the input dtype, so it matches exactly
    torch.testing.assert_close(res_in, want_res, atol=0, rtol=0)
    torch.testing.assert_close(scale, want_scale, atol=1e-6, rtol=1e-4)
    assert q.dtype == torch.int8 and q.shape == x.shape and int(q.abs().max()) <= 127
    # fp32 rounding order can move a value sitting exactly on .5 by one step; nothing more
    diff = (q.int() - want_q.int()).abs()
    assert int(diff.max()) <= 1 and float((diff > 0).float().mean()) < 1e-3
    # and the int8 values are a faithful encoding of the normalised row
    y = (want_res.float() * torch.rsqrt(want_res.float().pow(2).mean(-1, keepdim=True) + 1e-6)) * w.float()
    assert float(((q.float() * scale) - y).abs().max()) <= float(scale.max()) * 0.5 + 1e-5


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("cols", COLS)
def test_matches_reference(dtype, rows, cols):
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device=DEVICE, dtype=dtype)
    res = torch.randn(rows, cols, device=DEVICE, dtype=dtype)
    w = torch.randn(cols, device=DEVICE, dtype=dtype)
    _check(x, res, w)


def test_zero_row_does_not_divide_by_zero():
    x = torch.zeros(3, 256, device=DEVICE)
    res = torch.zeros(3, 256, device=DEVICE)
    q, scale, _ = add_rmsnorm_quant(x, res, torch.ones(256, device=DEVICE))
    assert int(q.abs().max()) == 0 and torch.isfinite(scale).all()


def test_3d_input():
    torch.manual_seed(1)
    x = torch.randn(2, 5, 512, device=DEVICE)
    res = torch.randn(2, 5, 512, device=DEVICE)
    w = torch.randn(512, device=DEVICE)
    q, scale, _ = add_rmsnorm_quant(x, res.clone(), w)
    want_q, want_scale, _ = add_rmsnorm_quant_reference(x, res, w)
    assert q.shape == (2, 5, 512) and scale.shape == (2, 5, 1)
    assert int((q.int() - want_q.int()).abs().max()) <= 1


def test_rejects_bad_inputs():
    x = torch.randn(2, 64, device=DEVICE)
    with pytest.raises(ValueError):
        add_rmsnorm_quant(x, torch.randn(2, 32, device=DEVICE), torch.ones(64, device=DEVICE))
    with pytest.raises(ValueError):
        add_rmsnorm_quant(x, torch.randn(2, 64, device=DEVICE), torch.ones(32, device=DEVICE))
