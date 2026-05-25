"""
run_all.py
──────────────────────────────────────────────
Runs the full FFT Convolution & Denoising Engine:
  1. Main 2-D and 3-D pipeline demos
  2. Compression sweep
  3. Denoising strategy comparison
  4. Benchmark (image-size and kernel-size scaling)

All output images saved to outputs/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fft_engine import (
    run_pipeline_2d,
    run_pipeline_3d,
    plot_2d_results,
    plot_3d_results,
    plot_compression_sweep,
)
from denoising import run_denoising_comparison
from benchmark import benchmark_image_sizes, benchmark_kernel_sizes, plot_benchmarks

print("\n" + "█" * 60)
print("  High-Dimensional FFT Fast Convolution & Denoising Engine")
print("█" * 60)

# ── 1. 2-D pipeline ───────────────────────────────────────
res2d = run_pipeline_2d(H=256, W=256, kernel_size=15, sigma=3.0, keep_fraction=0.25)
plot_2d_results(res2d)
plot_compression_sweep(res2d["image"], res2d["kernel"])

# ── 2. 3-D pipeline ───────────────────────────────────────
res3d = run_pipeline_3d(nF=8, H=64, W=64, keep_fraction=0.15)
plot_3d_results(res3d)

# ── 3. Denoising strategies ───────────────────────────────
run_denoising_comparison(H=256, W=256, noise_std=0.20)

# ── 4. Benchmarks ─────────────────────────────────────────
sz_data = benchmark_image_sizes(sizes=[64, 128, 256, 512])
ks_data = benchmark_kernel_sizes(image_size=256, kernel_sizes=[5, 11, 15, 21, 31])
plot_benchmarks(sz_data, ks_data)

print("\n" + "█" * 60)
print("  ✓  All outputs saved to outputs/")
print("█" * 60)
