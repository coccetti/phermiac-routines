# -*- coding: utf-8 -*-
"""
Optical Simulation: SLM Phase Modulation & Cylindrical Lens Propagation
Output: publication-style plots (original vs grating vs CCD intensity)

What’s new vs 2_fft_cylindrical_lens.py:
- Produces a single 3-panel "final" PNG:
  (1) original (no grating) in pure black/white (0/255),
  (2) input with grating (as-is),
  (3) CCD intensity on linear scale.
- Produces an additional 2-panel PNG:
  original (no grating) BW + CCD intensity with LOG scale (colored).
- Adds an extra "overview" plot (2x2) useful for quick inspection.

Physics:
- The SLM modulates the PHASE of the light (0-2pi), not the amplitude.
- A vertical cylindrical lens performs a 1D Fourier Transform along the horizontal axis.

Author: Fabrizio Coccetti
Date: 2026-02-03
"""

from __future__ import annotations

import os
from pathlib import Path

# Make matplotlib cache writable (avoids slow temp cache + fontconfig warnings on some systems)
_MPLCONFIGDIR = Path(__file__).resolve().parent / ".mplconfig"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
# Fontconfig may also need a writable cache dir on some systems
os.environ.setdefault("XDG_CACHE_HOME", str(_MPLCONFIGDIR))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
import csv


# --- CONFIGURATION ---
# Filename of the BMP to load (this is typically the *_grating.bmp output of 0_apply_grating_separation.py)
FILENAME = "16_bits/propagation/img/SLM_16x16bits_CERN-01_event3_CERN-02_sec3_960x960_r16_c16_w108_grating.bmp"

# Output (saved next to the input image by default)
SAVE_CCD_IMAGES = True            # save simulated CCD intensity images (linear + optional log1p) as PNG
SAVE_CCD_LOG = True               # save log1p(intensity) PNGs (recommended for visibility)
SAVE_PLOTS = True                 # save the requested matplotlib plots
SHOW_FIGURE = False               # show interactive windows (default off to avoid hanging terminals)

# --- CCD "vertical lines" readout (macropixel-row peaks) ---
# Each macropixel row in the SLM corresponds to a horizontal band on the CCD (same y pixels).
MACROPIXEL_HEIGHT = 60
SAVE_MACROROW_PEAKS_CSV = True

# If True, ignore the central (zero-order) region when searching peaks, so you pick the steered vertical line.
EXCLUDE_CENTER_ZERO_ORDER = True
CENTER_EXCLUSION_HALF_WIDTH_PX = 35  # exclude columns [cx-hw, cx+hw]

# Physical Parameters
# 'w108' in filename implies Gray Level 108 corresponds to a Phase Shift of PI (3.14 rad)
PI_GRAY_LEVEL = 108.0

# Plot styling
# NOTE: The CCD intensity has a huge dynamic range; for "linear scale" plots we keep a linear mapping
# but CLIP vmax to a high percentile so the vertical lines are visible (still linear, just saturated).
LINEAR_CMAP = "inferno"
LOG_CMAP = "viridis"
LINEAR_VMAX_PERCENTILE = 99.9  # 99.5..99.95 are reasonable; lower = more contrast, more clipping


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


def _resolve_path(filename: str) -> Path | None:
    """Resolve input path. Tries CWD and repo root."""
    p = Path(filename)
    if p.is_absolute():
        return p if p.exists() else None

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [Path.cwd() / p, repo_root / p]
    return next((c.resolve() for c in candidates if c.exists()), None)


def _load_grayscale_u8(path: Path) -> np.ndarray:
    img = Image.open(str(path)).convert("L")
    return np.array(img, dtype=np.uint8)


def _find_original_no_grating(img_path: Path) -> Path | None:
    """
    Best-effort: if input is '*_grating.bmp', try to load the matching non-grating BMP
    next to it by stripping '_grating' from the stem.
    """
    stem = img_path.stem
    if "_grating" not in stem:
        return None

    candidate = img_path.with_name(stem.replace("_grating", "") + img_path.suffix)
    if candidate.exists():
        return candidate.resolve()
    return None


def _as_bw_0_255(arr_u8: np.ndarray) -> np.ndarray:
    """Force to pure 0/255 for display."""
    a = np.asarray(arr_u8, dtype=np.uint8)
    return np.where(a > 0, 255, 0).astype(np.uint8)


def _lognorm_for_intensity(intensity: np.ndarray) -> LogNorm:
    """Choose a reasonable LogNorm even with zeros present."""
    i = np.asarray(intensity, dtype=np.float64)
    vmax = float(np.nanmax(i)) if np.isfinite(i).any() else 1.0
    if vmax <= 0:
        return LogNorm(vmin=1e-12, vmax=1.0)

    positive = i[i > 0]
    if positive.size == 0:
        vmin = 1e-12
    else:
        # Avoid vmin being too tiny; use percentile for robustness, but not below vmax*1e-6.
        vmin = float(np.percentile(positive, 1))
        vmin = max(vmin, vmax * 1e-6)
    return LogNorm(vmin=vmin, vmax=vmax)


def _macrorow_peaks(
    intensity: np.ndarray,
    macro_h: int,
    *,
    exclude_center: bool,
    center_half_width: int,
) -> list[dict[str, int | float]]:
    """
    For each macropixel row (horizontal band of height macro_h), find the brightest CCD spot (linear intensity).

    Returns a list of dicts with:
      macro_row, y_start, y_end, peak_x, peak_y, peak_intensity
    """
    I = np.asarray(intensity, dtype=np.float64)
    h, w = I.shape
    cx = w // 2

    results: list[dict[str, int | float]] = []
    num_rows = int(np.ceil(h / macro_h)) if macro_h > 0 else 0
    for r in range(num_rows):
        y0 = r * macro_h
        y1 = min((r + 1) * macro_h, h)
        if y0 >= h:
            break

        band = I[y0:y1, :].copy()
        if exclude_center and center_half_width > 0:
            x0 = max(0, cx - center_half_width)
            x1 = min(w, cx + center_half_width + 1)
            band[:, x0:x1] = -np.inf

        flat_idx = int(np.nanargmax(band))
        by, bx = np.unravel_index(flat_idx, band.shape)
        peak_val = float(band[by, bx])
        results.append(
            {
                "macro_row": r,
                "y_start": int(y0),
                "y_end": int(y1),
                "peak_x": int(bx),
                "peak_y": int(y0 + by),
                "peak_intensity": peak_val,
            }
        )

    return results


def main() -> None:
    img_path = _resolve_path(FILENAME)
    if img_path is None:
        print(f"ERROR: File not found: {FILENAME}")
        return

    print(f"Loading (with grating): {img_path}")
    data_grating = _load_grayscale_u8(img_path).astype(np.float32)
    print(f"Input statistics - Max: {data_grating.max()}, Min: {data_grating.min()}")

    orig_path = _find_original_no_grating(img_path)
    if orig_path is not None:
        print(f"Loading (no grating):  {orig_path}")
        data_orig_u8 = _load_grayscale_u8(orig_path)
    else:
        # Fallback: we cannot reliably "undo" the modulo grating; use the current input.
        print("WARNING: Could not find a matching non-grating BMP (stem without '_grating').")
        print("         Using the grating BMP also as the 'original' for plotting.")
        data_orig_u8 = np.clip(data_grating, 0, 255).astype(np.uint8)

    data_orig_bw = _as_bw_0_255(data_orig_u8)

    # --- Physics Simulation (Phase Modulation) ---
    phase_map = (data_grating / PI_GRAY_LEVEL) * np.pi
    input_field = 1.0 * np.exp(1j * phase_map)

    # Cylindrical Lens Propagation (1D FFT along x-axis, axis=1)
    fft_field = np.fft.fftshift(np.fft.fft(input_field, axis=1), axes=1)
    intensity = np.abs(fft_field) ** 2
    intensity_log1p = np.log1p(intensity)

    # Robust vmax for linear visualization (keeps scale linear, but avoids a single huge peak
    # washing out the whole colormap).
    finite_I = intensity[np.isfinite(intensity)]
    if finite_I.size:
        linear_vmax = float(np.percentile(finite_I, LINEAR_VMAX_PERCENTILE))
    else:
        linear_vmax = 1.0
    if not np.isfinite(linear_vmax) or linear_vmax <= 0:
        linear_vmax = float(np.nanmax(intensity)) if np.isfinite(intensity).any() else 1.0
    linear_vmin = 0.0

    out_dir = img_path.parent
    stem = img_path.stem

    # --- Readout: one intensity value per macropixel row (vertical-line spot) ---
    peaks = _macrorow_peaks(
        intensity,
        MACROPIXEL_HEIGHT,
        exclude_center=EXCLUDE_CENTER_ZERO_ORDER,
        center_half_width=CENTER_EXCLUSION_HALF_WIDTH_PX,
    )
    if SAVE_MACROROW_PEAKS_CSV:
        out_csv = out_dir / f"{stem}_CCD_macrorow_peaks_linear.csv"
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["macro_row", "y_start", "y_end", "peak_x", "peak_y", "peak_intensity"],
            )
            w.writeheader()
            w.writerows(peaks)
        print(f"Saved: {out_csv}")
    if peaks:
        vals = np.array([p["peak_intensity"] for p in peaks], dtype=float)
        print(
            f"Macrorow peaks (linear): N={len(peaks)}, "
            f"min={vals.min():.3g}, median={np.median(vals):.3g}, max={vals.max():.3g}"
        )

    # --- Save CCD images (optional) ---
    if SAVE_CCD_IMAGES:
        out_linear = out_dir / f"{stem}_CCD_intensity_linear.png"
        Image.fromarray(_normalize_to_uint16(intensity)).save(str(out_linear))
        print(f"Saved: {out_linear}")

        out_linear_cmap = out_dir / f"{stem}_CCD_intensity_linear_{LINEAR_CMAP}.png"
        Image.fromarray(_cmap_to_uint8_rgb(intensity, cmap=LINEAR_CMAP)).save(str(out_linear_cmap))
        print(f"Saved: {out_linear_cmap}")

        if SAVE_CCD_LOG:
            out_log = out_dir / f"{stem}_CCD_intensity_log1p.png"
            Image.fromarray(_normalize_to_uint16(intensity_log1p)).save(str(out_log))
            print(f"Saved: {out_log}")

            out_log_cmap = out_dir / f"{stem}_CCD_intensity_log1p_{LOG_CMAP}.png"
            Image.fromarray(_cmap_to_uint8_rgb(intensity_log1p, cmap=LOG_CMAP)).save(str(out_log_cmap))
            print(f"Saved: {out_log_cmap}")

    # --- Requested plots ---
    figs: list[plt.Figure] = []

    if SAVE_PLOTS or SHOW_FIGURE:
        # 1) Final 3-panel plot (original BW, input grating, CCD intensity linear)
        fig1, ax = plt.subplots(1, 3, figsize=(18, 6))
        figs.append(fig1)

        im0 = ax[0].imshow(data_orig_bw, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax[0].set_title("Original (no grating)\nBW (0/255)")
        ax[0].set_xlabel("x (px)")
        ax[0].set_ylabel("y (px)")
        c0 = fig1.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        c0.set_label("Gray level")

        im1 = ax[1].imshow(np.clip(data_grating, 0, 255), cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax[1].set_title("SLM input (with grating)\n(as loaded)")
        ax[1].set_xlabel("x (px)")
        ax[1].set_ylabel("y (px)")
        c1 = fig1.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        c1.set_label("Gray level")

        im2 = ax[2].imshow(
            intensity,
            cmap=LINEAR_CMAP,
            vmin=linear_vmin,
            vmax=linear_vmax,
            aspect="auto",
            interpolation="nearest",
            resample=False,
        )
        ax[2].set_title(
            "CCD intensity (linear scale)\n(cylindrical lens focus)\n"
            "(per-macropixel-row peak values annotated at right)"
        )
        ax[2].set_xlabel("k_x (FFT-shifted index)")
        ax[2].set_ylabel("y (px)")
        c2 = fig1.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        c2.set_label(f"Intensity (a.u.) [clipped at p{LINEAR_VMAX_PERCENTILE:g}]")
        # Put numerical peak intensity values (linear scale) for each macropixel row.
        # NOTE: cyan dots are shown only on the LOG plot (below), not on linear plots.
        if peaks:
            peak_vals = np.array([float(p["peak_intensity"]) for p in peaks], dtype=float)
            vmax = float(np.max(peak_vals)) if peak_vals.size else 0.0
            if vmax > 0:
                common_exp = int(np.floor(np.log10(vmax)))
                scale = 10.0**common_exp
            else:
                common_exp = 0
                scale = 1.0

            # Write the shared exponent once (so all row labels share it).
            # Example: "×10^5" meaning each label is mantissa in that scale.
            ax[2].text(
                0.02,
                0.98,
                rf"$\times 10^{{{common_exp}}}$",
                transform=ax[2].transAxes,
                ha="left",
                va="top",
                fontsize=11,
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.45),
            )

            x_text = intensity.shape[1] - 5
            for p in peaks:
                y_mid = 0.5 * (float(p["y_start"]) + float(p["y_end"]))
                mantissa = float(p["peak_intensity"]) / scale if scale != 0 else 0.0
                ax[2].text(
                    x_text,
                    y_mid,
                    f"{mantissa:6.2f}",
                    color="white",
                    fontsize=8,
                    ha="right",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="black", ec="none", alpha=0.45),
                )

        fig1.tight_layout()
        if SAVE_PLOTS:
            out_fig1 = out_dir / f"{stem}_FFT_cylindrical_lens_FINAL.png"
            fig1.savefig(str(out_fig1), dpi=220, bbox_inches="tight")
            print(f"Saved: {out_fig1}")

        # 2) Original BW + CCD intensity log plot (colored)
        fig2, ax2 = plt.subplots(1, 2, figsize=(13, 6))
        figs.append(fig2)

        im20 = ax2[0].imshow(data_orig_bw, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax2[0].set_title("Original (no grating)\nBW (0/255)")
        ax2[0].set_xlabel("x (px)")
        ax2[0].set_ylabel("y (px)")
        c20 = fig2.colorbar(im20, ax=ax2[0], fraction=0.046, pad=0.04)
        c20.set_label("Gray level")

        im21 = ax2[1].imshow(
            intensity_log1p,
            cmap=LOG_CMAP,
            aspect="auto",
            interpolation="nearest",
            resample=False,
        )
        ax2[1].set_title("CCD intensity (log1p)\n(colored)")
        ax2[1].set_xlabel("k_x (FFT-shifted index)")
        ax2[1].set_ylabel("y (px)")
        c21 = fig2.colorbar(im21, ax=ax2[1], fraction=0.046, pad=0.04)
        c21.set_label("log1p(Intensity)")
        if peaks:
            ax2[1].scatter(
                [p["peak_x"] for p in peaks],
                [p["peak_y"] for p in peaks],
                s=10,
                c="cyan",
                marker="o",
                linewidths=0,
                alpha=0.9,
            )

        fig2.tight_layout()
        if SAVE_PLOTS:
            out_fig2 = out_dir / f"{stem}_FFT_cylindrical_lens_LOG.png"
            fig2.savefig(str(out_fig2), dpi=220, bbox_inches="tight")
            print(f"Saved: {out_fig2}")

        # 3) Extra "brilliant" overview plot (2x2) for fast inspection
        fig3, ax3 = plt.subplots(2, 2, figsize=(16, 12))
        figs.append(fig3)

        im30 = ax3[0, 0].imshow(data_orig_bw, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax3[0, 0].set_title("Original (no grating) BW")
        ax3[0, 0].set_xlabel("x (px)")
        ax3[0, 0].set_ylabel("y (px)")
        fig3.colorbar(im30, ax=ax3[0, 0], fraction=0.046, pad=0.04).set_label("Gray level")

        im31 = ax3[0, 1].imshow(np.clip(data_grating, 0, 255), cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax3[0, 1].set_title("Input (with grating)")
        ax3[0, 1].set_xlabel("x (px)")
        ax3[0, 1].set_ylabel("y (px)")
        fig3.colorbar(im31, ax=ax3[0, 1], fraction=0.046, pad=0.04).set_label("Gray level")

        im32 = ax3[1, 0].imshow(
            intensity,
            cmap=LINEAR_CMAP,
            vmin=linear_vmin,
            vmax=linear_vmax,
            aspect="auto",
            interpolation="nearest",
            resample=False,
        )
        ax3[1, 0].set_title("CCD intensity (linear)")
        ax3[1, 0].set_xlabel("k_x (FFT-shifted index)")
        ax3[1, 0].set_ylabel("y (px)")
        fig3.colorbar(im32, ax=ax3[1, 0], fraction=0.046, pad=0.04).set_label("Intensity (a.u.)")

        # Use log1p(intensity) for the overview plot to match the standalone saved image
        # and avoid visual striping artifacts that can appear with LogNorm+resampling.
        im33 = ax3[1, 1].imshow(
            intensity_log1p,
            cmap=LOG_CMAP,
            aspect="auto",
            interpolation="nearest",
            resample=False,
        )
        ax3[1, 1].set_title("CCD intensity (log1p)")
        ax3[1, 1].set_xlabel("k_x (FFT-shifted index)")
        ax3[1, 1].set_ylabel("y (px)")
        fig3.colorbar(im33, ax=ax3[1, 1], fraction=0.046, pad=0.04).set_label("log1p(Intensity)")

        fig3.tight_layout()
        if SAVE_PLOTS:
            out_fig3 = out_dir / f"{stem}_FFT_cylindrical_lens_OVERVIEW.png"
            fig3.savefig(str(out_fig3), dpi=220, bbox_inches="tight")
            print(f"Saved: {out_fig3}")

    if SHOW_FIGURE and figs:
        plt.show()
        print("Simulation Complete. Displaying figures.")
    else:
        for f in figs:
            plt.close(f)
        print("Simulation Complete. Figures saved only (SHOW_FIGURE=False).")


if __name__ == "__main__":
    main()

