"""Benchmark the fused Triton RMSNorm against honest baselines.

Baselines:
  eager          plain PyTorch, fp32 maths (several separate GPU kernels)
  F.rms_norm     PyTorch's built-in op
  torch.compile  PyTorch's compiler, which writes its own fused Triton kernel
  liger          Liger-Kernel's hand-written Triton RMSNorm
  fuselab        this repo

Metric: effective memory bandwidth (GB/s) = bytes that must move / time. RMSNorm is
memory-bound, so the ceiling is the GPU's peak memory bandwidth.

Usage: python bench/bench_rmsnorm.py [--dtype float16|bfloat16|float32] [--out results]
"""

import argparse
import csv
import os
import sys

import torch
import triton
import triton.testing

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fuselab import rmsnorm, rmsnorm_reference  # noqa: E402

# Peak DRAM bandwidth in GB/s (vendor spec sheets)
PEAK_GBPS = {
    "T4": 320,
    "P100": 732,
    "L4": 300,
    "A10": 600,
    "A100-SXM4-40GB": 1555,
    "A100-SXM4-80GB": 2039,
    "A100 80GB PCIe": 1935,
    "H100 80GB HBM3": 3350,
    "H100 PCIe": 2000,
    "RTX 4090": 1008,
    "RTX 5090": 1792,
}

HIDDEN = [2560, 4096]  # Qwen3-4B, Llama-3-8B
ROWS = [1, 16, 64, 256, 1024, 4096, 16384]  # tokens in the batch


def peak_for(name: str) -> float | None:
    for key, v in PEAK_GBPS.items():
        if key in name:
            return v
    return None


def build_impls(eps: float):
    impls = {"eager": lambda x, w: rmsnorm_reference(x, w, eps)}
    if hasattr(torch.nn.functional, "rms_norm"):
        impls["F.rms_norm"] = lambda x, w: torch.nn.functional.rms_norm(x, (x.shape[-1],), w, eps)
    # One compiled function per hidden size (build_impls is called per size), and a high recompile
    # limit: every new batch shape recompiles, and past the default limit (8) torch.compile silently
    # falls back to eager, which made it look as slow as eager in the first run.
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, 64)
    torch._dynamo.reset()
    compiled = torch.compile(rmsnorm_reference, dynamic=False)
    impls["torch.compile"] = lambda x, w: compiled(x, w, eps)
    try:
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction

        # "gemma" casting = all maths in fp32 with offset 0, i.e. the same formula as ours
        impls["liger"] = lambda x, w: LigerRMSNormFunction.apply(x, w, eps, 0.0, "gemma")
    except Exception as e:  # liger missing or API changed
        print(f"skipping liger: {type(e).__name__}: {e}")
    impls["fuselab"] = lambda x, w: rmsnorm(x, w, eps)
    return impls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        sys.exit("needs a CUDA GPU")

    dtype = getattr(torch, args.dtype)
    gpu = torch.cuda.get_device_name()
    peak = peak_for(gpu)
    eps = 1e-6
    impls = build_impls(eps)  # rebuilt per hidden size below
    print(f"GPU: {gpu} (peak {peak or '?'} GB/s) | torch {torch.__version__} | triton {triton.__version__} | {args.dtype}")

    rows_out = []
    for n in HIDDEN:
        impls = build_impls(eps)
        for m in ROWS:
            x = torch.randn(m, n, device="cuda", dtype=dtype)
            w = torch.randn(n, device="cuda", dtype=dtype)
            want = rmsnorm_reference(x, w, eps)
            moved = (2 * x.numel() + w.numel()) * x.element_size()  # read x + w, write y
            for name, fn in impls.items():
                try:
                    got = fn(x, w)
                    torch.testing.assert_close(got.float(), want.float(), atol=2e-2, rtol=2e-2)
                    ms = triton.testing.do_bench(lambda: fn(x, w), warmup=25, rep=100, return_mode="median")
                except Exception as e:
                    print(f"  {name:13s} n={n:5d} m={m:6d}  FAILED: {type(e).__name__}: {str(e)[:120]}")
                    continue
                gbps = moved / (ms * 1e-3) / 1e9
                rows_out.append(
                    {
                        "gpu": gpu,
                        "dtype": args.dtype,
                        "hidden": n,
                        "rows": m,
                        "impl": name,
                        "us": round(ms * 1e3, 2),
                        "gbps": round(gbps, 1),
                        "pct_peak": round(100 * gbps / peak, 1) if peak else "",
                    }
                )
                print(f"  {name:13s} n={n:5d} m={m:6d}  {ms*1e3:8.2f} us  {gbps:7.1f} GB/s")

    os.makedirs(args.out, exist_ok=True)
    tag = f"{gpu.replace(' ', '_').replace('/', '_')}_{args.dtype}"
    csv_path = os.path.join(args.out, f"rmsnorm_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        wr.writeheader()
        wr.writerows(rows_out)
    print(f"saved {csv_path}")

    # summary at the largest batch: speed-up over eager and % of peak
    print("\nlargest batch (rows=%d):" % ROWS[-1])
    for n in HIDDEN:
        sub = {r["impl"]: r for r in rows_out if r["hidden"] == n and r["rows"] == ROWS[-1]}
        base = sub.get("eager")
        for name, r in sub.items():
            speed = f"{base['us'] / r['us']:.2f}x vs eager" if base else ""
            print(f"  hidden={n:5d} {name:13s} {r['gbps']:7.1f} GB/s  {r['pct_peak']}% of peak  {speed}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(HIDDEN), figsize=(6 * len(HIDDEN), 4.5), sharey=True)
        for ax, n in zip(axes, HIDDEN):
            for name in impls:
                pts = [(r["rows"], r["gbps"]) for r in rows_out if r["hidden"] == n and r["impl"] == name]
                if pts:
                    ax.plot(*zip(*pts), marker="o", label=name, linewidth=2.5 if name == "fuselab" else 1.2)
            if peak:
                ax.axhline(peak, color="black", linestyle="--", linewidth=1)
                ax.text(ROWS[0], peak * 1.02, f"peak {peak} GB/s", fontsize=8)
            ax.set_xscale("log", base=2)
            ax.set_xlabel("tokens in batch (rows)")
            ax.set_title(f"RMSNorm, hidden={n}, {args.dtype}")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("effective bandwidth (GB/s)")
        axes[-1].legend()
        fig.suptitle(gpu)
        fig.tight_layout()
        png = os.path.join(args.out, f"rmsnorm_{tag}.png")
        fig.savefig(png, dpi=130)
        print(f"saved {png}")
    except ImportError:
        print("matplotlib not installed; skipped the plot")


if __name__ == "__main__":
    main()
