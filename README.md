# High-Dimensional FFT Fast Convolution & Denoising Engine

A from-scratch Python/PyTorch pipeline for understanding, benchmarking,
and experimenting with frequency-domain signal processing on 2-D images
and 3-D video blocks.

---

## Project Structure

```
fft_engine/
├── fft_engine.py          ← Core engine (Steps 1-5, full pipeline)
├── denoising.py           ← Frequency-domain denoising strategies
├── benchmark.py           ← Scaling benchmarks (size & kernel)
├── run_all.py             ← Single entry point to run everything
├── requirements.txt
├── outputs/               ← All generated plots land here
│   ├── results_2d.png
│   ├── results_3d.png
│   ├── compression_sweep.png
│   ├── denoising_comparison.png
│   └── benchmark.png
└── README.md
```

---

## Quick Start

```bash
pip install -r requirements.txt
python run_all.py
```

Or run individual modules:

```bash
python fft_engine.py   # 2-D + 3-D pipeline + compression sweep
python denoising.py    # Denoising strategy comparison
python benchmark.py    # Timing benchmarks
```

---

## The 5-Step Build Plan (what the code implements)

### Step 1 — Generate / Load Dataset

`make_synthetic_image(H, W)` in `fft_engine.py` returns a `(H, W)` float32
tensor in [0, 1] containing geometric shapes (rectangle, circle, diagonal
stripe) plus additive Gaussian noise.

`make_video_block(F, H, W)` stacks `F` slightly-shifted noisy frames into a
`(F, H, W)` tensor for the 3-D pipeline.

### Step 2 — Spatial Convolution (Baseline)

Two implementations:
- `slow_spatial_conv` — uses `torch.nn.functional.conv2d` (hardware-optimised
  sliding window). This is the "spatial baseline" for timing comparisons.
- `naive_loop_conv` — pure Python nested loops; intentionally slow, used only
  on small images to demonstrate the O(N² × K²) cost.

### Step 3 — FFT Transform

`fft_of_tensor(data, pad_shape)` applies `torch.fft.fftn` with zero-padding
to the next power-of-two size. This prevents circular wrap-around artifacts.

`fft_of_kernel(kernel, pad_shape)` embeds the kernel in a zero-padded buffer
(top-left corner) before transforming, which avoids a phase-shift artifact.

### Step 4 — Pointwise Filtering + Compression

`frequency_filter(X_data, X_kernel, keep_fraction)`:
1. **Convolution** — multiply the two complex tensors element-wise.
   This is equivalent to convolution in the spatial domain (convolution theorem).
2. **Compression** — optionally zero out all coefficients whose magnitude
   falls below the `(1 - keep_fraction)` quantile.
   Setting `keep_fraction=0.05` means only 5 % of frequencies survive.

### Step 5 — Inverse FFT

`ifft_to_spatial(Y, original_shape)` applies `torch.fft.ifftn`,
takes the real part, then crops back to the original spatial dimensions.

---

## Denoising Strategies (`denoising.py`)

| Method | How it works |
|---|---|
| **Ideal low-pass** | Hard binary mask: pass all frequencies below a cutoff radius |
| **Gaussian low-pass** | Smooth exponential rolloff — fewer ringing artifacts |
| **Wiener filter** | Adaptive mask: `S_signal / (S_signal + S_noise)` per coefficient |
| **Hard shrinkage** | Zero out the weakest X% of frequency coefficients |

All methods are compared with PSNR and SNR metrics against a clean reference.

---

## Benchmarks (`benchmark.py`)

Two sweeps:

1. **Image size scaling** — measures Spatial vs FFT time as image grows from
   64×64 to 512×512 (or 1024×1024). FFT complexity is O(N² log N) vs
   O(N² × K²) for spatial — the gap widens dramatically with larger kernels.

2. **Kernel size scaling** — for a fixed 256×256 image, increases kernel from
   5×5 to 71×71. Spatial cost grows as K², FFT cost stays nearly flat.

---

## Key Concepts Illustrated

### Convolution Theorem
Convolution in the spatial domain = pointwise multiplication in the
frequency domain:

```
h[x,y] * g[x,y]  ←FFT→  H(u,v) · G(u,v)
```

This lets us replace an O(N²K²) sliding-window sum with an O(N² log N) FFT.

### Zero-Padding
When computing linear (non-circular) convolution via FFT, we must zero-pad
the input to at least `N + K − 1` samples in each dimension, then crop.
Without padding, the FFT computes *circular* convolution and you see
wrap-around artifacts at the image edges.

Padding to the next power of two (`2^⌈log₂(N+K-1)⌉`) is a well-known trick
to maximise FFT cache efficiency (Cooley-Tukey radix-2 algorithm).

### Frequency Sparsity
Natural images and video are approximately sparse in the frequency domain:
most energy lives in low-frequency coefficients. This is why JPEG and MP4
compression work — you can null out the majority of frequency coefficients
and still preserve perceptual quality.

The `keep_fraction` parameter in `frequency_filter` lets you explore this
trade-off interactively. The compression sweep plot shows how PSNR degrades
as you keep fewer and fewer coefficients.

---

## Extending the Project

- **Load a real image**: replace `make_synthetic_image` with
  `torchvision.io.read_image` or `PIL.Image.open`.
- **3-D medical volumes**: replace the video block with a CT/MRI scan loaded
  from NIfTI format (`nibabel`) — `torch.fft.fftn` handles arbitrary dims.
- **Edge detection**: swap the Gaussian kernel for a Laplacian or
  Sobel kernel and observe the frequency-domain representation.
- **Adaptive thresholding**: try soft shrinkage (`sign(X) * max(|X| - λ, 0)`)
  instead of hard thresholding — this is wavelet denoising applied in the
  Fourier basis.
- **GPU acceleration**: move tensors to CUDA (`tensor.to("cuda")`) before
  calling `torch.fft.fftn` for a substantial speedup on large volumes.

---

## Output Files

| File | What it shows |
|---|---|
| `outputs/results_2d.png` | Original / spatial blur / FFT blur / frequency magnitude / diff / kernel |
| `outputs/results_3d.png` | Side-by-side frames before and after 3-D FFT blur |
| `outputs/compression_sweep.png` | Quality vs. fraction of coefficients kept |
| `outputs/denoising_comparison.png` | Five denoising methods with PSNR scores |
| `outputs/benchmark.png` | Timing curves for spatial vs FFT across sizes |
