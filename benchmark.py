"""
benchmark.py
────────────────────────────────────────────────────────────
Systematically measures wall-clock time of:
  • Spatial convolution (F.conv2d)
  • FFT-based convolution + compression

across a range of image sizes and kernel sizes, then plots
the scaling curve.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from fft_engine import (
    make_synthetic_image,
    make_gaussian_kernel,
    fft_of_tensor,
    fft_of_kernel,
    frequency_filter,
    ifft_to_spatial,
    ceil_log2,
)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

REPEATS = 5      # average over N runs to reduce noise


def time_spatial(image: torch.Tensor, kernel: torch.Tensor) -> float:
    H, W   = image.shape
    kH, kW = kernel.shape
    x = image.unsqueeze(0).unsqueeze(0)
    k = kernel.unsqueeze(0).unsqueeze(0)
    pad = (kH // 2, kW // 2)

    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        _ = F.conv2d(x, k, padding=pad)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3     # ms


def time_fft(image: torch.Tensor, kernel: torch.Tensor,
             keep_fraction: float = 1.0) -> float:
    H, W   = image.shape
    kH, kW = kernel.shape
    pad_h  = int(2 ** np.ceil(np.log2(H + kH - 1)))
    pad_w  = int(2 ** np.ceil(np.log2(W + kW - 1)))
    pad_shape = (pad_h, pad_w)

    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        Xd = fft_of_tensor(image, pad_shape)
        Xk = fft_of_kernel(kernel, pad_shape)
        Y, _  = frequency_filter(Xd, Xk, keep_fraction)
        out   = ifft_to_spatial(Y, (H, W))
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e3     # ms


def benchmark_image_sizes(
    sizes=(64, 128, 256, 512, 1024),
    kernel_size: int = 15,
    sigma: float = 3.0,
):
    print("\n" + "=" * 55)
    print("  Benchmark: Image Size Scaling")
    print("=" * 55)
    print(f"{'Size':>8}  {'Spatial (ms)':>14}  {'FFT (ms)':>10}  {'Speedup':>9}")
    print("-" * 55)

    spatial_ms, fft_ms, speedups, sz_list = [], [], [], []

    for s in sizes:
        img    = make_synthetic_image(s, s)
        kernel = make_gaussian_kernel(kernel_size, sigma)

        t_sp  = time_spatial(img, kernel)
        t_fft = time_fft(img, kernel)
        sp    = t_sp / t_fft

        spatial_ms.append(t_sp)
        fft_ms.append(t_fft)
        speedups.append(sp)
        sz_list.append(s)

        print(f"{s:>8}  {t_sp:>14.3f}  {t_fft:>10.3f}  {sp:>9.2f}×")

    return sz_list, spatial_ms, fft_ms, speedups


def benchmark_kernel_sizes(
    image_size: int = 256,
    kernel_sizes=(5, 11, 15, 21, 31, 51, 71),
    sigma: float = 3.0,
):
    print("\n" + "=" * 55)
    print(f"  Benchmark: Kernel Size Scaling  (image {image_size}×{image_size})")
    print("=" * 55)
    print(f"{'KSize':>7}  {'Spatial (ms)':>14}  {'FFT (ms)':>10}  {'Speedup':>9}")
    print("-" * 55)

    img = make_synthetic_image(image_size, image_size)
    spatial_ms, fft_ms, speedups, ksz_list = [], [], [], []

    for ks in kernel_sizes:
        if ks > min(image_size, image_size):
            continue
        kernel  = make_gaussian_kernel(ks, sigma)
        t_sp    = time_spatial(img, kernel)
        t_fft   = time_fft(img, kernel)
        sp      = t_sp / t_fft

        spatial_ms.append(t_sp)
        fft_ms.append(t_fft)
        speedups.append(sp)
        ksz_list.append(ks)

        print(f"{ks:>7}  {t_sp:>14.3f}  {t_fft:>10.3f}  {sp:>9.2f}×")

    return ksz_list, spatial_ms, fft_ms, speedups


def plot_benchmarks(
    sz_data: tuple,
    ks_data: tuple,
    save_path: str = "outputs/benchmark.png",
):
    sz_list, sp_sz, fft_sz, speedup_sz = sz_data
    ks_list, sp_ks, fft_ks, speedup_ks = ks_data

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#0f0f12")
    accent = "#00d4ff"
    warm   = "#ff6b35"

    # ── panel 1: image size scaling ───────────────────────────
    ax = axes[0]
    ax.set_facecolor("#1a1a22")
    ax.plot(sz_list, sp_sz,  "o-", color=warm,   lw=2, label="Spatial (F.conv2d)")
    ax.plot(sz_list, fft_sz, "s-", color=accent,  lw=2, label="FFT pipeline")
    ax.set_xlabel("Image dimension (px)", color="#cccccc")
    ax.set_ylabel("Time (ms)", color="#cccccc")
    ax.set_title("Time vs Image Size", color="white", pad=10)
    ax.legend(framealpha=0.3, labelcolor="white")
    ax.set_yscale("log")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    # ── panel 2: kernel size scaling ──────────────────────────
    ax = axes[1]
    ax.set_facecolor("#1a1a22")
    ax.plot(ks_list, sp_ks,  "o-", color=warm,   lw=2, label="Spatial")
    ax.plot(ks_list, fft_ks, "s-", color=accent,  lw=2, label="FFT")
    ax.set_xlabel("Kernel size (px)", color="#cccccc")
    ax.set_ylabel("Time (ms)", color="#cccccc")
    ax.set_title("Time vs Kernel Size  (256×256 image)", color="white", pad=10)
    ax.legend(framealpha=0.3, labelcolor="white")
    ax.set_yscale("log")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    # ── panel 3: speedup ──────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor("#1a1a22")
    ax.bar(
        [str(s) for s in sz_list], speedup_sz,
        color=accent, alpha=0.8, width=0.5, label="vs image size"
    )
    ax.axhline(1, color="#555566", lw=1, ls="--")
    ax.set_xlabel("Image size", color="#cccccc")
    ax.set_ylabel("Speedup (×)", color="#cccccc")
    ax.set_title("FFT Speedup over Spatial Conv", color="white", pad=10)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")

    plt.suptitle(
        "FFT Fast Convolution – Benchmark Results",
        color="white", fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Saved → {save_path}")


if __name__ == "__main__":
    sz_data = benchmark_image_sizes(sizes=[64, 128, 256, 512])
    ks_data = benchmark_kernel_sizes(image_size=256, kernel_sizes=[5, 11, 15, 21, 31])
    plot_benchmarks(sz_data, ks_data)
    print("\n✓ Benchmark complete.")
