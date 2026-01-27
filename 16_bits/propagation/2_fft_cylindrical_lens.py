# -*- coding: utf-8 -*-
"""
Optical Simulation: SLM Phase Modulation & Cylindrical Lens Propagation
Output: Input Pattern vs. Simulated Focal Plane Intensity (Phase Mode)

Physics:
- The SLM modulates the PHASE of the light (0-2pi), not the amplitude.
- A vertical cylindrical lens performs a 1D Fourier Transform along the horizontal axis.
- The 'Background' (black pixels) transmits light with Phase 0, resulting in a strong
  Zero-Order peak at the center of the focal plane.

Author: Fabrizio Coccetti
Date: 2026-01-23
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
from pathlib import Path

# --- CONFIGURATION ---
# Filename of the BMP to load
# FILENAME = "SLM_CERN_event1221_480x480_r8_c8_w108.bmp"
FILENAME = "08_bits/propagation/img/SLM_CERN_event1221_480x480_r8_c8_w108_grating.bmp"

# Output (saved next to the input image by default)
SAVE_CCD_IMAGES = True            # save simulated CCD intensity images
SAVE_FIGURE = True                # save the matplotlib figure (input + CCD)
SAVE_CCD_LOG = True               # save log1p(intensity) (recommended for visibility)
SHOW_FIGURE = True                # keep current interactive behavior

# Physical Parameters
# 'w108' in filename implies Gray Level 108 corresponds to a Phase Shift of PI (3.14 rad)
PI_GRAY_LEVEL = 108.0 


def _normalize_to_uint16(arr: np.ndarray) -> np.ndarray:
    """Normalize array to 0..65535 for PNG saving (robust to outliers)."""
    a = np.asarray(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = a - a.min()
    mx = a.max()
    if mx <= 0:
        return np.zeros(a.shape, dtype=np.uint16)
    a = a / mx
    return np.round(a * 65535.0).astype(np.uint16)

def _cmap_to_uint8_rgb(arr: np.ndarray, cmap: str = "inferno") -> np.ndarray:
    """Map array to an RGB uint8 image using a matplotlib colormap."""
    a = np.asarray(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = a - a.min()
    mx = a.max()
    if mx <= 0:
        a = np.zeros(a.shape, dtype=np.float64)
    else:
        a = a / mx  # 0..1

    rgba = mpl.colormaps.get_cmap(cmap)(a)  # (..., 4) float in 0..1
    rgb = np.round(rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb

def main():
    # 1. Resolve input path (FILENAME may already include folders)
    filename = Path(FILENAME)
    candidates: list[Path]
    if filename.is_absolute():
        candidates = [filename]
    else:
        # Prefer relative to current working directory, then relative to repo root
        # (so "08_bits/propagation/img/..." works no matter where you run from).
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            Path.cwd() / filename,
            repo_root / filename,
        ]

    img_path: Path | None = next((p for p in candidates if p.exists()), None)
    if img_path is None:
        print(f"ERROR: File not found: {FILENAME}")
        print("Tried:")
        for p in candidates:
            print(f" - {p}")
        return

    img_path = img_path.resolve()
    print(f"Loading Image: {img_path}")

    # 2. Load and Prepare Image
    try:
        img = Image.open(str(img_path)).convert('L') # Convert to 8-bit Grayscale
        data = np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"Error reading image: {e}")
        return

    print(f"Image Statistics - Max: {data.max()}, Min: {data.min()}")

    # 3. Physics Simulation (Phase Modulation)
    # Convert Gray Levels to Phase Radians
    # Formula: Phase = (GrayLevel / 108) * Pi
    phase_map = (data / PI_GRAY_LEVEL) * np.pi
    
    # Create the Optical Field (Complex Number)
    # Amplitude is 1.0 everywhere (light passes through), only Phase varies.
    input_field = 1.0 * np.exp(1j * phase_map)

    # 4. Cylindrical Lens Propagation (1D FFT)
    # The lens focuses along the X-axis (rows), so we apply FFT on axis=1.
    # We use fftshift to move the Zero-Order (DC component) to the center.
    fft_field = np.fft.fftshift(np.fft.fft(input_field, axis=1), axes=1)
    
    # Calculate Intensity (Amplitude Squared)
    intensity = np.abs(fft_field)**2
    
    # Use Logarithmic Scale for visualization
    # (Real CCDs have limited dynamic range, but Log helps us see the faint diffraction orders
    # that are otherwise hidden by the massive central peak).
    intensity_log = np.log1p(intensity)

    # 4b. Save "final image" (simulated CCD) and/or figure
    if SAVE_CCD_IMAGES or SAVE_FIGURE:
        out_dir = img_path.parent
        stem = img_path.stem

        if SAVE_CCD_IMAGES:
            # Save as 16-bit grayscale PNGs (preserves dynamic range better than 8-bit)
            out_linear = out_dir / f"{stem}_CCD_intensity_linear.png"
            Image.fromarray(_normalize_to_uint16(intensity), mode="I;16").save(str(out_linear))
            print(f"Saved: {out_linear}")

            # Also save a colored (inferno) version for easier visual inspection
            out_linear_cmap = out_dir / f"{stem}_CCD_intensity_linear_inferno.png"
            Image.fromarray(_cmap_to_uint8_rgb(intensity, cmap="inferno"), mode="RGB").save(str(out_linear_cmap))
            print(f"Saved: {out_linear_cmap}")

            if SAVE_CCD_LOG:
                out_log = out_dir / f"{stem}_CCD_intensity_log.png"
                Image.fromarray(_normalize_to_uint16(intensity_log), mode="I;16").save(str(out_log))
                print(f"Saved: {out_log}")

                out_log_cmap = out_dir / f"{stem}_CCD_intensity_log_inferno.png"
                Image.fromarray(_cmap_to_uint8_rgb(intensity_log, cmap="inferno"), mode="RGB").save(str(out_log_cmap))
                print(f"Saved: {out_log_cmap}")

    # 5. Plotting (2 Subplots)
    fig = plt.figure(figsize=(12, 6))

    # Plot 1: Input Image (The Phase Pattern)
    plt.subplot(1, 2, 1)
    plt.title("SLM Input Pattern")
    plt.imshow(data, cmap='gray', aspect='auto')
    plt.colorbar(label="Gray Level (0-255)")
    plt.xlabel("SLM x-axis (px)")
    plt.ylabel("SLM y-axis (px)")

    # Plot 2: Simulated CCD Response
    plt.subplot(1, 2, 2)
    plt.title("CCD Intensity (Linear Scale)\n(Cylindrical Lens Focus)")
    plt.imshow(intensity, cmap='inferno', aspect='auto')
    plt.colorbar(label="Intensity (a.u.)")
    plt.xlabel("Focal Position (Spatial Frequency k_x)")
    plt.ylabel("Vertical Position (px)")

    plt.tight_layout()
    if SAVE_FIGURE:
        out_fig = img_path.parent / f"{img_path.stem}_FFT_cylindrical_lens.png"
        fig.savefig(str(out_fig), dpi=200, bbox_inches="tight")
        print(f"Saved: {out_fig}")

    if SHOW_FIGURE:
        plt.show()
        print("Simulation Complete. Displaying figure.")
    else:
        plt.close(fig)
        print("Simulation Complete. Figure saving only (SHOW_FIGURE=False).")

if __name__ == "__main__":
    main()
