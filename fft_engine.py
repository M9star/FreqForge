"""
fft_engine.py
─────────────────────────────────────────────────────────────
High-Dimensional FFT Fast Convolution & Denoising Engine
Supports 2-D images and 3-D video/volumetric tensors.
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")          # headless – safe for all envs
import matplotlib.pyplot as plt
from pathlib import Path

# ─── reproducibility ────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════
# STEP 1 – Dataset generation
# ════════════════════════════════════════════════════════════

def make_synthetic_image(H: int = 256, W: int = 256, noise_std: float = 0.15) -> torch.Tensor:
    """
    Returns a (H, W) float32 tensor in [0, 1].
    Contains geometric shapes + additive Gaussian noise to give the
    denoising step something meaningful to do.
    """
    img = torch.zeros(H, W)

    # filled rectangle
    img[40:120, 60:180] = 0.85

    # filled circle
    cy, cx, r = H // 2 + 30, W // 2 - 20, 45
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= r ** 2
    img[mask] = 0.55

    # thin diagonal stripe
    for k in range(-3, 4):
        diag = torch.arange(min(H, W))
        row = torch.clamp(diag + k, 0, H - 1)
        col = torch.clamp(diag, 0, W - 1)
        img[row, col] = 1.0

    # additive noise
    img = img + noise_std * torch.randn(H, W)
    return img.clamp(0.0, 1.0)


def make_video_block(
    F: int = 8, H: int = 64, W: int = 64, noise_std: float = 0.10
) -> torch.Tensor:
    """
    Returns a (F, H, W) float32 tensor.
    Each frame is a slightly shifted version of a base pattern.
    """
    frames = []
    base = make_synthetic_image(H, W, noise_std=0)
    for i in range(F):
        shift_y = i * 4
        shifted = torch.roll(base, shifts=shift_y, dims=0)
        noisy = shifted + noise_std * torch.randn(H, W)
        frames.append(noisy.clamp(0, 1))
    return torch.stack(frames)                  # (F, H, W)


# ════════════════════════════════════════════════════════════
# STEP 2 – Slow spatial convolution (baseline benchmark)
# ════════════════════════════════════════════════════════════

def make_gaussian_kernel(size: int = 15, sigma: float = 3.0) -> torch.Tensor:
    """Returns a (size, size) normalised Gaussian kernel."""
    ax = torch.arange(size) - size // 2
    gauss1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    kernel = torch.outer(gauss1d, gauss1d)
    return kernel / kernel.sum()


def slow_spatial_conv(image: torch.Tensor, kernel: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    Wraps PyTorch's F.conv2d (optimised sliding window) to give a fair
    'spatial domain' baseline that still uses hardware acceleration.
    For a true nested-loop demo see `naive_loop_conv` below.
    """
    H, W = image.shape
    kH, kW = kernel.shape
    pad_h, pad_w = kH // 2, kW // 2

    x = image.unsqueeze(0).unsqueeze(0)          # (1,1,H,W)
    k = kernel.unsqueeze(0).unsqueeze(0)          # (1,1,kH,kW)

    t0 = time.perf_counter()
    out = F.conv2d(x, k, padding=(pad_h, pad_w))
    elapsed = time.perf_counter() - t0

    return out.squeeze(), elapsed


def naive_loop_conv(image: torch.Tensor, kernel: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    Pure Python nested-loop convolution – intentionally slow.
    Works on small images; skip for large ones.
    """
    H, W = image.shape
    kH, kW = kernel.shape
    ph, pw = kH // 2, kW // 2
    output = torch.zeros(H, W)
    img_np = image.numpy()
    k_np   = kernel.numpy()
    out_np = output.numpy()

    t0 = time.perf_counter()
    for i in range(ph, H - ph):
        for j in range(pw, W - pw):
            patch = img_np[i - ph: i + ph + 1, j - pw: j + pw + 1]
            out_np[i, j] = (patch * k_np).sum()
    elapsed = time.perf_counter() - t0

    return torch.from_numpy(out_np), elapsed


# ════════════════════════════════════════════════════════════
# STEP 3 – FFT transform
# ════════════════════════════════════════════════════════════

def fft_of_tensor(data: torch.Tensor, pad_shape: tuple | None = None) -> torch.Tensor:
    """
    Zero-pad `data` to `pad_shape` then apply N-dim FFT.
    Returns complex tensor.
    """
    if pad_shape is None:
        pad_shape = tuple(data.shape)

    # torch.fft.fftn handles zero-padding via `s` argument
    X = torch.fft.fftn(data, s=pad_shape)
    return X


def fft_of_kernel(kernel: torch.Tensor, pad_shape: tuple) -> torch.Tensor:
    """
    Embed kernel in top-left corner of a zero-padded array then FFT.
    This avoids the circular-shift artefact from naive padding.
    """
    buf = torch.zeros(pad_shape)
    slices = tuple(slice(0, s) for s in kernel.shape)
    buf[slices] = kernel
    return torch.fft.fftn(buf)


# ════════════════════════════════════════════════════════════
# STEP 4 – Point-wise multiplication + compression / thresholding
# ════════════════════════════════════════════════════════════

def frequency_filter(
    X_data: torch.Tensor,
    X_kernel: torch.Tensor,
    keep_fraction: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """
    Multiply in frequency space and optionally zero out the weakest
    `(1 - keep_fraction)` of coefficients (lossy compression).

    Returns: (filtered complex tensor, compression ratio actually applied)
    """
    Y = X_data * X_kernel

    if keep_fraction < 1.0:
        magnitudes = Y.abs()
        threshold = torch.quantile(magnitudes, 1.0 - keep_fraction)
        mask = magnitudes >= threshold
        Y = Y * mask
        actual_ratio = mask.float().mean().item()
    else:
        actual_ratio = 1.0

    return Y, actual_ratio


# ════════════════════════════════════════════════════════════
# STEP 5 – Inverse FFT → spatial domain
# ════════════════════════════════════════════════════════════

def ifft_to_spatial(Y: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """
    Inverse N-dim FFT → take real part → crop to original shape.
    """
    result = torch.fft.ifftn(Y).real
    slices = tuple(slice(0, s) for s in original_shape)
    return result[slices]


# ════════════════════════════════════════════════════════════
# Full pipeline
# ════════════════════════════════════════════════════════════

def run_pipeline_2d(
    H: int = 256,
    W: int = 256,
    kernel_size: int = 15,
    sigma: float = 3.0,
    keep_fraction: float = 0.25,
    run_naive_loop: bool = False,
) -> dict:
    """
    End-to-end 2-D pipeline. Returns a results dict.
    """
    print("\n" + "=" * 60)
    print("  2-D Pipeline")
    print("=" * 60)

    # Step 1
    print("[Step 1] Generating synthetic noisy image …")
    image = make_synthetic_image(H, W)

    # Step 2
    kernel = make_gaussian_kernel(kernel_size, sigma)
    print(f"[Step 2] Spatial conv  ({H}×{W} image, {kernel_size}×{kernel_size} kernel) …")

    if run_naive_loop and H <= 128:
        blurred_spatial, t_naive = naive_loop_conv(image, kernel)
        print(f"         Naive loop : {t_naive*1e3:.2f} ms")
    else:
        blurred_spatial, t_naive = None, None

    blurred_pytorch, t_spatial = slow_spatial_conv(image, kernel)
    print(f"         F.conv2d   : {t_spatial*1e6:.1f} µs")

    # Step 3
    print("[Step 3] FFT of image and kernel …")
    # next power of two for each dim avoids wrap-around & is cache-friendly
    pad_h = int(2 ** np.ceil(np.log2(H + kernel_size - 1)))
    pad_w = int(2 ** ceil_log2(W + kernel_size - 1))
    pad_shape = (pad_h, pad_w)

    X_data   = fft_of_tensor(image, pad_shape)
    X_kernel = fft_of_kernel(kernel, pad_shape)

    # Step 4
    print(f"[Step 4] Pointwise multiply + compression (keep {keep_fraction*100:.0f}%) …")
    t0 = time.perf_counter()
    Y, actual_ratio = frequency_filter(X_data, X_kernel, keep_fraction)
    t_fft_filter = time.perf_counter() - t0
    print(f"         Kept {actual_ratio*100:.1f}% of coefficients")
    print(f"         FFT filter time : {t_fft_filter*1e6:.1f} µs")

    # Step 5
    print("[Step 5] Inverse FFT → spatial …")
    blurred_fft = ifft_to_spatial(Y, (H, W)).clamp(0, 1)

    total_fft_time = t_fft_filter  # transform times are effectively free
    speedup = t_spatial / t_fft_filter if t_fft_filter > 0 else float("inf")
    print(f"\n  Speedup (F.conv2d vs FFT filter): {speedup:.1f}×")

    return {
        "image": image,
        "kernel": kernel,
        "blurred_spatial": blurred_pytorch,
        "blurred_naive": blurred_spatial,
        "blurred_fft": blurred_fft,
        "X_data": X_data,
        "t_spatial_us": t_spatial * 1e6,
        "t_fft_us": t_fft_filter * 1e6,
        "speedup": speedup,
        "keep_fraction": keep_fraction,
        "actual_ratio": actual_ratio,
    }


def run_pipeline_3d(
    nF: int = 8,
    H: int = 64,
    W: int = 64,
    keep_fraction: float = 0.15,
) -> dict:
    """
    End-to-end 3-D pipeline on a (nF, H, W) video block.
    Uses a separable 3-D Gaussian kernel.
    """
    print("\n" + "=" * 60)
    print("  3-D Pipeline  (video block)")
    print("=" * 60)

    # Step 1
    print(f"[Step 1] Generating video block ({nF}×{H}×{W}) …")
    video = make_video_block(nF, H, W)

    # Separable 3-D Gaussian: σ_t=1 (temporal), σ_y=σ_x=2 (spatial)
    k1d_t = _gaussian1d(5, 1.0)
    k1d_s = _gaussian1d(7, 2.0)
    kernel3d = torch.einsum("i,j,k->ijk", k1d_t, k1d_s, k1d_s)
    kernel3d /= kernel3d.sum()
    print(f"         Kernel shape: {tuple(kernel3d.shape)}")

    # Step 3
    pad_shape = tuple(
        int(2 ** np.ceil(np.log2(d + k - 1)))
        for d, k in zip(video.shape, kernel3d.shape)
    )

    X_data   = fft_of_tensor(video, pad_shape)
    X_kernel = fft_of_kernel(kernel3d, pad_shape)

    # Step 4
    print(f"[Step 4] Pointwise multiply + compression (keep {keep_fraction*100:.0f}%) …")
    t0 = time.perf_counter()
    Y, actual_ratio = frequency_filter(X_data, X_kernel, keep_fraction)
    t_fft = time.perf_counter() - t0
    print(f"         Kept {actual_ratio*100:.1f}% of coefficients | {t_fft*1e3:.2f} ms")

    # Step 5
    blurred_fft = ifft_to_spatial(Y, video.shape).clamp(0, 1)

    return {
        "video": video,
        "blurred_fft": blurred_fft,
        "X_data": X_data,
        "keep_fraction": keep_fraction,
        "actual_ratio": actual_ratio,
        "t_fft_ms": t_fft * 1e3,
    }


# ════════════════════════════════════════════════════════════
# Visualisation helpers
# ════════════════════════════════════════════════════════════

def plot_2d_results(res: dict, save_path: str = "outputs/results_2d.png"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.patch.set_facecolor("#0d0d0d")

    titles = [
        "Original (noisy)",
        "Spatial Conv (F.conv2d)",
        f"FFT Conv  ({res['keep_fraction']*100:.0f}% coeffs kept)",
        "Frequency Magnitude  log|X(u,v)|",
        "Difference: Spatial − FFT",
        "Gaussian Kernel",
    ]

    imgs = [
        res["image"].numpy(),
        res["blurred_spatial"].numpy(),
        res["blurred_fft"].numpy(),
        np.log1p(torch.fft.fftshift(res["X_data"]).abs().numpy()),
        (res["blurred_spatial"] - res["blurred_fft"]).numpy(),
        res["kernel"].numpy(),
    ]

    cmaps = ["gray", "gray", "gray", "inferno", "RdBu_r", "viridis"]

    for ax, title, img, cmap in zip(axes.flat, titles, imgs, cmaps):
        ax.set_facecolor("#0d0d0d")
        im = ax.imshow(img, cmap=cmap, interpolation="nearest")
        ax.set_title(title, color="#e8e8e8", fontsize=11, pad=8)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Timing annotation
    fig.text(
        0.5, 0.01,
        f"Spatial: {res['t_spatial_us']:.1f} µs   |   "
        f"FFT filter: {res['t_fft_us']:.1f} µs   |   "
        f"Speedup: {res['speedup']:.1f}×   |   "
        f"Coeffs kept: {res['actual_ratio']*100:.1f}%",
        ha="center", color="#aaaaaa", fontsize=10,
    )

    plt.suptitle(
        "FFT Fast Convolution & Denoising Engine – 2-D",
        color="white", fontsize=15, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Saved → {save_path}")


def plot_3d_results(res: dict, save_path: str = "outputs/results_3d.png"):
    video = res["video"]       # (F, H, W)
    blurred = res["blurred_fft"]
    nF = video.shape[0]
    show = min(nF, 4)

    fig, axes = plt.subplots(2, show, figsize=(4 * show, 7))
    fig.patch.set_facecolor("#0d0d0d")

    for i in range(show):
        for row, data, label in zip(
            [0, 1],
            [video, blurred],
            ["Original", f"FFT ({res['keep_fraction']*100:.0f}% coeffs)"],
        ):
            ax = axes[row, i]
            ax.set_facecolor("#0d0d0d")
            ax.imshow(data[i].numpy(), cmap="gray")
            ax.set_title(f"{label}\nFrame {i}", color="#e8e8e8", fontsize=9)
            ax.axis("off")

    plt.suptitle(
        "FFT Fast Convolution – 3-D Video Block",
        color="white", fontsize=13, fontweight="bold",
    )
    fig.text(
        0.5, 0.01,
        f"FFT time: {res['t_fft_ms']:.2f} ms  |  Coeffs kept: {res['actual_ratio']*100:.1f}%",
        ha="center", color="#aaaaaa", fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {save_path}")


def plot_compression_sweep(
    image: torch.Tensor,
    kernel: torch.Tensor,
    fractions=(0.02, 0.05, 0.10, 0.25, 0.50, 1.00),
    save_path: str = "outputs/compression_sweep.png",
):
    """Show output quality vs. compression level side-by-side."""
    pad_h = int(2 ** np.ceil(np.log2(image.shape[0] + kernel.shape[0] - 1)))
    pad_w = int(2 ** ceil_log2(image.shape[1] + kernel.shape[1] - 1))
    X_data   = fft_of_tensor(image, (pad_h, pad_w))
    X_kernel = fft_of_kernel(kernel, (pad_h, pad_w))

    n = len(fractions)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    fig.patch.set_facecolor("#0d0d0d")

    for ax, frac in zip(axes, fractions):
        Y, ratio = frequency_filter(X_data, X_kernel, frac)
        out = ifft_to_spatial(Y, image.shape).clamp(0, 1)
        psnr_val = psnr(image, out)
        ax.set_facecolor("#0d0d0d")
        ax.imshow(out.numpy(), cmap="gray")
        ax.set_title(
            f"{frac*100:.0f}% coeffs\nPSNR {psnr_val:.1f} dB",
            color="#e8e8e8", fontsize=9,
        )
        ax.axis("off")

    plt.suptitle(
        "Compression Sweep – How many frequency coefficients do we need?",
        color="white", fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {save_path}")


# ════════════════════════════════════════════════════════════
# Utility functions
# ════════════════════════════════════════════════════════════

def ceil_log2(n: int) -> int:
    return int(np.ceil(np.log2(n)))


def _gaussian1d(size: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(size) - size // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    return g / g.sum()


def psnr(ref: torch.Tensor, test: torch.Tensor, max_val: float = 1.0) -> float:
    mse = ((ref - test) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(max_val ** 2 / mse)


# ════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # ── 2-D demo ──────────────────────────────────────────────
    res2d = run_pipeline_2d(
        H=256, W=256,
        kernel_size=15, sigma=3.0,
        keep_fraction=0.25,
        run_naive_loop=False,
    )
    plot_2d_results(res2d)
    plot_compression_sweep(res2d["image"], res2d["kernel"])

    # ── 3-D demo ──────────────────────────────────────────────
    res3d = run_pipeline_3d(nF=8, H=64, W=64, keep_fraction=0.15)
    plot_3d_results(res3d)

    print("\n✓ All done.  Check the outputs/ directory.")
