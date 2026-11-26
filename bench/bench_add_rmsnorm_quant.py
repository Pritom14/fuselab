"""Benchmark fused residual add + RMSNorm + INT8 quantisation.

Baselines:
  eager          plain PyTorch (about eight separate GPU kernels)
  torch.compile  PyTorch's compiler, which fuses it into its own Triton kernel(s)
  norm-only      residual add in PyTorch -> fuselab's RMSNorm kernel -> quantise in PyTorch
                 (what you get by fusing just the norm)
  fuselab        everything in one kernel

Metric: microseconds per call, and effective bandwidth = bytes that must move / time.
Bytes: read x and residual, write residual, read w, write int8 q and one fp32 scale per row.

Usage: python bench/bench_add_rmsnorm_quant.py [--dtype float16|bfloat16|float32] [--out results]
"""

import argparse
import csv
import os
import sys

import torch
import triton
import triton.testing

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fuselab import add_rmsnorm_quant, add_rmsnorm_quant_reference, rmsnorm  # noqa: E402

PEAK_GBPS = {"T4": 320, "P100": 732, "L4": 300, "A10": 600, "A100-SXM4-40GB": 1555, "A100-SXM4-80GB": 2039,
             "H100 80GB HBM3": 3350, "H100 PCIe": 2000, "RTX 4090": 1008, "RTX 5090": 1792}
HIDDEN = [2560, 4096]
ROWS = [1, 16, 64, 256, 1024, 4096, 16384]
IMPLS = ["eager", "torch.compile", "norm-only", "fuselab"]


def peak_for(name):
    return next((v for k, v in PEAK_GBPS.items() if k in name), None)


def norm_only(x, res, w, eps):
    r = x + res
    y = rmsnorm(r, w, eps).float()
    amax = y.abs().amax(-1, keepdim=True).clamp_min(1e-8)
    v = y * (127.0 / amax)
    q = torch.where(v >= 0, torch.floor(v + 0.5), torch.ceil(v - 0.5)).clamp(-127, 127).to(torch.int8)
    return q, amax / 127.0, r


def build_impls(eps):
    for name in ("recompile_limit", "cache_size_limit"):  # each new shape recompiles; don't fall back to eager
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, 64)
    torch._dynamo.reset()
    compiled = torch.compile(add_rmsnorm_quant_reference, dynamic=False)
    return {
        "eager": lambda x, r, w: add_rmsnorm_quant_reference(x, r, w, eps),
        "torch.compile": lambda x, r, w: compiled(x, r, w, eps),
        "norm-only": lambda x, r, w: norm_only(x, r, w, eps),
        "fuselab": lambda x, r, w: add_rmsnorm_quant(x, r, w, eps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU")
    dtype, gpu, eps = getattr(torch, a.dtype), torch.cuda.get_device_name(), 1e-6
    peak = peak_for(gpu)
    print(f"GPU: {gpu} (peak {peak or '?'} GB/s) | torch {torch.__version__} | triton {triton.__version__} | {a.dtype}")

    out = []
    for n in HIDDEN:
        impls = build_impls(eps)
        for m in ROWS:
            x = torch.randn(m, n, device="cuda", dtype=dtype)
            res = torch.randn(m, n, device="cuda", dtype=dtype)
            w = torch.randn(n, device="cuda", dtype=dtype)
            want_q, _, _ = add_rmsnorm_quant_reference(x, res, w, eps)
            # timing input: x = 0 keeps the in-place residual from drifting over thousands of calls
            # (same memory traffic, no overflow)
            x0 = torch.zeros_like(x)
            e = x.element_size()
            moved = 3 * m * n * e + n * e + m * n + 4 * m
            for name, fn in impls.items():
                try:
                    q = fn(x, res.clone(), w)[0]
                    assert int((q.int() - want_q.int()).abs().max()) <= 1, "int8 output differs from reference"
                    ms = triton.testing.do_bench(lambda: fn(x0, res, w), warmup=25, rep=100, return_mode="median")
                except Exception as err:
                    print(f"  {name:13s} n={n:5d} m={m:6d}  FAILED: {type(err).__name__}: {str(err)[:120]}")
                    continue
                gbps = moved / (ms * 1e-3) / 1e9
                out.append({"gpu": gpu, "dtype": a.dtype, "hidden": n, "rows": m, "impl": name,
                            "us": round(ms * 1e3, 2), "gbps": round(gbps, 1),
                            "pct_peak": round(100 * gbps / peak, 1) if peak else ""})
                print(f"  {name:13s} n={n:5d} m={m:6d}  {ms*1e3:8.2f} us  {gbps:7.1f} GB/s")

    os.makedirs(a.out, exist_ok=True)
    tag = f"{gpu.replace(' ', '_').replace('/', '_')}_{a.dtype}"
    path = os.path.join(a.out, f"add_rmsnorm_quant_{tag}.csv")
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print(f"saved {path}")
    for n in HIDDEN:
        for m in (1, ROWS[-1]):
            sub = {r["impl"]: r for r in out if r["hidden"] == n and r["rows"] == m}
            base = sub.get("eager")
            for name, r in sub.items():
                sp = f"{base['us'] / r['us']:.2f}x vs eager" if base else ""
                print(f"  hidden={n:5d} rows={m:6d} {name:13s} {r['us']:8.2f} us {r['gbps']:7.1f} GB/s "
                      f"{r['pct_peak']}% of peak  {sp}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(HIDDEN), figsize=(6 * len(HIDDEN), 4.5), sharey=True)
        for ax, n in zip(axes, HIDDEN):
            for name in IMPLS:
                pts = [(r["rows"], r["gbps"]) for r in out if r["hidden"] == n and r["impl"] == name]
                if pts:
                    ax.plot(*zip(*pts), marker="o", label=name, linewidth=2.5 if name == "fuselab" else 1.2)
            if peak:
                ax.axhline(peak, color="black", linestyle="--", linewidth=1)
                ax.text(ROWS[0], peak * 1.02, f"peak {peak} GB/s", fontsize=8)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("tokens in batch (rows)")
            ax.set_title(f"add + RMSNorm + INT8, hidden={n}, {a.dtype}")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("effective bandwidth (GB/s)")
        axes[-1].legend()
        fig.suptitle(gpu)
        fig.tight_layout()
        png = os.path.join(a.out, f"add_rmsnorm_quant_{tag}.png")
        fig.savefig(png, dpi=130)
        print(f"saved {png}")
    except ImportError:
        print("matplotlib not installed; skipped the plot")


if __name__ == "__main__":
    main()
