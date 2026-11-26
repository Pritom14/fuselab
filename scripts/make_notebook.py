"""Build the self-contained Kaggle notebook from the repo's source files (so they never drift)."""

import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FILES = [
    "fuselab/__init__.py",
    "fuselab/rmsnorm.py",
    "fuselab/add_rmsnorm_quant.py",
    "tests/test_rmsnorm.py",
    "tests/test_add_rmsnorm_quant.py",
    "bench/bench_rmsnorm.py",
    "bench/bench_add_rmsnorm_quant.py",
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


cells = [
    md(
        """
# fuselab: fused LLM inference kernels in Triton (step 1 RMSNorm, step 2 add + RMSNorm + INT8)

RMSNorm runs dozens of times for every token an LLM generates. Plain PyTorch does it in
several passes over GPU memory; this kernel reads each row once, does all the maths on-chip,
and writes once. This notebook:

1. checks the kernel matches PyTorch (fp32 reference) across shapes and dtypes,
2. benchmarks it against eager PyTorch, `F.rms_norm`, `torch.compile` and Liger-Kernel,
3. plots effective memory bandwidth against the GPU's peak.

**To run on Kaggle:** Settings → Accelerator → **GPU T4 x2** (or P100), Internet **on**, then *Run All*.
"""
    ),
    code(
        """
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
import torch, triton
print("torch", torch.__version__, "| triton", triton.__version__, "| cuda", torch.version.cuda)
"""
    ),
    code("!pip install -q --no-deps liger-kernel"),
    code("!mkdir -p fuselab tests bench results"),
]
for rel in FILES:
    src = open(os.path.join(ROOT, rel)).read()
    cells.append(code(f"%%writefile {rel}\n{src}"))
cells += [
    md("## 1. Correctness"),
    code("!python -m pytest -q tests"),
    md("## 2. Benchmark"),
    code("!python bench/bench_rmsnorm.py --dtype float16 --out results"),
    code("!python bench/bench_add_rmsnorm_quant.py --dtype float16 --out results"),
    md("## 3. Results"),
    code(
        """
import glob
import pandas as pd
from IPython.display import Image, display

for p in sorted(glob.glob("results/*.png")):
    display(Image(p))
for p in sorted(glob.glob("results/*.csv")):
    print(p)
    display(pd.read_csv(p).pivot_table(index=["hidden", "rows"], columns="impl", values="us"))
"""
    ),
]

for i, c in enumerate(cells):
    c["id"] = f"cell-{i:02d}"  # nbformat 4.5 wants cell ids
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
out = os.path.join(ROOT, "notebooks", "fuselab_kaggle.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", os.path.relpath(out, ROOT))
