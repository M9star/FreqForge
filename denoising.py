"""
denoising.py
────────────────────────────────────────────────────────────
Frequency-domain denoising strategies:

  1. Ideal low-pass filter  (hard cutoff in frequency space)
  2. Gaussian low-pass      (smooth rolloff)
  3. Wiener-inspired filter (SNR-adaptive)
  4. Threshold / hard shrinkage (sparse compression)

Includes PSNR / SNR metrics and a summary plot.
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from fft_engine import (
    make_synthetic_image,
    make_gaussian_kernel,
    fft_of_tensor,
    fft_of_kernel,
    ifft_to_spatial,
    psnr,
)

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


# ════════════════════════════════════════════════════════════
# Filter masks
# ════════════════════════════════════════════════════════════

def _freq_grid(H: int, W: int) -> torch.Tensor:
    """
    Returns a (H, W) tensor of normalised frequency radii in [0, 1].
    0 = DC, 1 = Nyquist corner.
    """
    fy = torch.fft.fftfreq(H)   # range [-0.5, 0.5)
    fx = torch.fft.fftfreq(W)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")
    return torch.sqrt(FY ** 2 + FX ** 2) / (np.sqrt(2) / 2)   # normalise to [0,1]


def ideal_lowpass(H: int, W: int, cutoff: float = 0.3) -> torch.Tensor:
    """Hard binary mask: pass frequencies with radius ≤ cutoff."""
    return (_freq_grid(H, W) <= cutoff).float()


def gaussian_lowpass(H: int, W: int, sigma: float = 0.2) -> torch.Tensor:
    """Smooth Gaussian rolloff mask."""
    r = _freq_grid(H, W)
    return torch.exp(-(r ** 2) / (2 * sigma ** 2))


def wiener_filter(
    X_noisy: torch.Tensor,
    signal_var: float = 0.5,
    noise_var: float = 0.05,
) -> torch.Tensor:
    """
    Simplified Wiener filter mask in the frequency domain:
        H(u,v) = S_signal / (S_signal + S_noise)
    Approximated using |X|^2 as a proxy for power spectrum.
    """
    power = X_noisy.abs() ** 2
    # smooth the power estimate with a small Gaussian
    H, W = X_noisy.shape
    smooth_k = make_gaussian_kernel(5, 1.5)
    pad = (5 // 2, 5 // 2)
    power_smooth = torch.nn.functional.conv2d(
        power.real.unsqueeze(0).unsqueeze(0),
        smooth_k.unsqueeze(0).unsqueeze(0),
        padding=pad,
    ).squeeze()
    mask = power_smooth / (power_smooth + noise_var)
    return mask.float()


# ════════════════════════════════════════════════════════════
# Apply a mask in frequency space
# ════════════════════════════════════════════════════════════

def apply_freq_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    FFT → multiply by mask → IFFT → clamp to [0,1].
    mask must match image shape.
    """
    X = torch.fft.fftn(image)
    Y = X * mask
    out = torch.fft.ifftn(Y).real
    return out.clamp(0, 1)


# ════════════════════════════════════════════════════════════
# Hard shrinkage (frequency thresholding)
# ════════════════════════════════════════════════════════════

def hard_shrinkage(image: torch.Tensor, keep_fraction: float = 0.10) -> torch.Tensor:
    """Zero out the weakest (1-keep_fraction) frequency coefficients."""
    X   = torch.fft.fftn(image)
    mag = X.abs()
    thr = torch.quantile(mag, 1.0 - keep_fraction)
    Y   = X * (mag >= thr)
    return torch.fft.ifftn(Y).real.clamp(0, 1)


# ════════════════════════════════════════════════════════════
# Metric helpers
# ════════════════════════════════════════════════════════════

def snr(reference: torch.Tensor, noisy: torch.Tensor) -> float:
    signal_power = (reference ** 2).mean().item()
    noise_power  = ((reference - noisy) ** 2).mean().item()
    if noise_power == 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


# ════════════════════════════════════════════════════════════
# Full denoising comparison
# ════════════════════════════════════════════════════════════

def run_denoising_comparison(
    H: int = 256,
    W: int = 256,
    noise_std: float = 0.20,
):
    print("\n" + "=" * 60)
    print("  Denoising Comparison")
    print("=" * 60)

    # clean reference + noisy input
    clean  = make_synthetic_image(H, W, noise_std=0.0)
    noisy  = (clean + noise_std * torch.randn(H, W)).clamp(0, 1)

    X_noisy = torch.fft.fftn(noisy)

    methods = {}

    # Ideal low-pass
    m = ideal_lowpass(H, W, cutoff=0.25)
    methods["Ideal LP\n(cutoff 0.25)"] = apply_freq_mask(noisy, m)

    # Gaussian low-pass
    m = gaussian_lowpass(H, W, sigma=0.15)
    methods["Gaussian LP\n(σ=0.15)"] = apply_freq_mask(noisy, m)

    # Wiener
    m = wiener_filter(X_noisy, noise_var=noise_std ** 2)
    methods["Wiener\nfilter"] = apply_freq_mask(noisy, m)

    # Hard shrinkage 5 %
    methods["Hard shrink\n5% coeffs"] = hard_shrinkage(noisy, keep_fraction=0.05)

    # Hard shrinkage 15 %
    methods["Hard shrink\n15% coeffs"] = hard_shrinkage(noisy, keep_fraction=0.15)

    # Print metrics
    print(f"\n{'Method':<22}  {'PSNR (dB)':>10}  {'SNR (dB)':>10}")
    print("-" * 46)
    baseline_psnr = psnr(clean, noisy)
    print(f"{'Noisy input':<22}  {baseline_psnr:>10.2f}  {snr(clean, noisy):>10.2f}")
    for name, out in methods.items():
        label = name.replace("\n", " ")
        print(f"{label:<22}  {psnr(clean, out):>10.2f}  {snr(clean, out):>10.2f}")

    _plot_denoising(clean, noisy, methods)
    return clean, noisy, methods


def _plot_denoising(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    methods: dict,
    save_path: str = "outputs/denoising_comparison.png",
):
    n_methods = len(methods)
    ncols = n_methods + 2        # clean + noisy + methods
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 4))
    fig.patch.set_facecolor("#0d0d0d")

    panels = {"Clean\nreference": clean, "Noisy\ninput": noisy}
    panels.update(methods)

    for ax, (title, img) in zip(axes, panels.items()):
        ax.set_facecolor("#0d0d0d")
        ax.imshow(img.numpy(), cmap="gray", vmin=0, vmax=1)
        p = psnr(clean, img)
        ax.set_title(
            f"{title}\nPSNR {p:.1f} dB" if not title.startswith("Clean") else title,
            color="#e8e8e8", fontsize=9,
        )
        ax.axis("off")

    plt.suptitle(
        "Frequency-Domain Denoising Strategies",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Saved → {save_path}")


# ════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_denoising_comparison(H=256, W=256, noise_std=0.20)
    print("\n✓ Denoising comparison done.")
