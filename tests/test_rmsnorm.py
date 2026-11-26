"""Correctness of the fused RMSNorm against the fp32 PyTorch reference.

Runs on a CUDA GPU, or on CPU with TRITON_INTERPRET=1 (slow, small shapes only).
"""

import os

import pytest
import torch

from fuselab import rmsnorm, rmsnorm_reference

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu" and not INTERPRET:
    pytest.skip("needs a CUDA GPU or TRITON_INTERPRET=1", allow_module_level=True)

DTYPES = [torch.float32, torch.float16]
if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
    DTYPES.append(torch.bfloat16)

TOL = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}

ROWS = [1, 7, 64] if INTERPRET else [1, 7, 128, 4096]
COLS = [128, 1000, 2560] if INTERPRET else [128, 1000, 2560, 4096, 8192]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("cols", COLS)
def test_matches_reference(dtype, rows, cols):
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device=DEVICE, dtype=dtype)
    w = torch.randn(cols, device=DEVICE, dtype=dtype)
    got = rmsnorm(x, w)
    want = rmsnorm_reference(x, w)
    assert got.dtype == x.dtype and got.shape == x.shape
    torch.testing.assert_close(got, want, atol=TOL[dtype], rtol=TOL[dtype])


def test_3d_input_and_non_contiguous_rows():
    torch.manual_seed(0)
    big = torch.randn(4, 6, 512, device=DEVICE)
    x = big[:, ::2, :]  # rows not contiguous in memory
    w = torch.randn(512, device=DEVICE)
    torch.testing.assert_close(rmsnorm(x, w), rmsnorm_reference(x, w), atol=1e-5, rtol=1e-5)


def test_rejects_wrong_weight_length():
    x = torch.randn(2, 64, device=DEVICE)
    with pytest.raises(ValueError):
        rmsnorm(x, torch.ones(32, device=DEVICE))
