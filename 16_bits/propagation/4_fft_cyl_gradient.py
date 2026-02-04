# -*- coding: utf-8 -*-
"""
Optical Simulation: SLM Phase Modulation & Cylindrical Lens Propagation
Variant: add macropixel-column amplitude gradient (1,2,3,...)

Compared to 3_fft_cylindrical_lens.py:
- Same phase-only encoding from the input BMP (gray -> phase).
- Adds an amplitude term that is CONSTANT within each macropixel:
    leftmost macropixel column amplitude = 1,
    next column amplitude = 2,
    ...
  repeated identically for each row.

Author: Fabrizio Coccetti
Date: 2026-02-03
"""

from __future__ import annotations

import os
import re
import csv
from pathlib import Path

# Make matplotlib cache writable (avoids slow temp cache + fontconfig warnings on some systems)
_MPLCONFIGDIR = Path(__file__).resolve().parent / ".mplconfig"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPLCONFIGDIR))

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.patheffects as pe
from PIL import Image


# --- CONFIGURATION ---
# Filename of the BMP to load (typically the *_grating.bmp output of 0_apply_grating_separation.py)
FILENAME = "16_bits/propagation/img/SLM_16x16bits_CERN-01_event3_CERN-02_sec3_960x960_r16_c16_w108_grating.bmp"

# Output (saved next to the input image by default)
SAVE_CCD_IMAGES = True            # save simulated CCD intensity images (linear + optional log1p) as PNG
SAVE_CCD_LOG = True               # save log1p(intensity) PNGs (recommended for visibility)
SAVE_PLOTS = True                 # save the requested matplotlib plots
SHOW_FIGURE = False               # show interactive windows (default off to avoid hanging terminals)

# --- CCD "vertical lines" readout (macropixel-row peaks) ---
MACROPIXEL_HEIGHT = 60
SAVE_MACROROW_PEAKS_CSV = True
EXCLUDE_CENTER_ZERO_ORDER = True
CENTER_EXCLUSION_HALF_WIDTH_PX = 35  # exclude columns [cx-hw, cx+hw]

# Amplitude term
# - "fixed": A(x,y) = 1 everywhere
# - "gradient": A is constant in each macropixel column, decreasing left->right (16..1 for c16)
# AMPLITUDE_MODE = "gradient"  # "fixed" | "gradient"
AMPLITUDE_MODE = "fixed"  # "fixed" | "gradient"
DEFAULT_MACRO_COLS = 16  # used if we can't parse cXX from filename

# Physical Parameters
# 'w108' in filename implies Gray Level 108 corresponds to a Phase Shift of PI (3.14 rad)
PI_GRAY_LEVEL = 108.0

# Plot styling
LINEAR_CMAP = "inferno"
LOG_CMAP = "viridis"

# Linear CCD color normalization (consistent comparisons across events)
LINEAR_VMAX_MODE = "theoretical_fraction"  # "theoretical_fraction" | "fixed" | "peaks_percentile"
LINEAR_VMAX_FRACTION_OF_THEORETICAL_MAX = 0.50
LINEAR_VMAX_FIXED = None
LINEAR_VMAX_PEAKS_PERCENTILE = 99.0
LINEAR_VMAX_PEAKS_MARGIN = 1.10
# Ensure we don't clip the true peak in the *plots* (so the brightest pixel gets the top color)
LINEAR_ENSURE_NO_CLIP_IN_PLOTS = True
LINEAR_DATA_MAX_MARGIN = 1.00  # multiply data max by this (>=1). Use 1.02 for tiny headroom.

# Display convention for legends/colorbars:
# express grayscale patterns in phase units [0; π] with mapping 0 -> 0 and 255 -> π.
DISPLAY_PHASE_MAX = np.pi
DISPLAY_PHASE_SCALE = DISPLAY_PHASE_MAX / 255.0


def _normalize_to_uint16(arr: np.ndarray) -> np.ndarray:
    """Normalize array to 0..65535 for PNG saving."""
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
        a = a / mx
    rgba = mpl.colormaps.get_cmap(cmap)(a)  # (..., 4) float
    return np.round(rgba[..., :3] * 255.0).astype(np.uint8)


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
    stem = img_path.stem
    if "_grating" not in stem:
        return None
    candidate = img_path.with_name(stem.replace("_grating", "") + img_path.suffix)
    return candidate.resolve() if candidate.exists() else None


def _as_bw_0_255(arr_u8: np.ndarray) -> np.ndarray:
    a = np.asarray(arr_u8, dtype=np.uint8)
    return np.where(a > 0, 255, 0).astype(np.uint8)


def _phase_colorbar(cbar: mpl.colorbar.Colorbar) -> None:
    cbar.set_label("Phase (rad)  [0 → 0, 255 → π]")
    cbar.set_ticks([0.0, np.pi / 2.0, np.pi])
    cbar.set_ticklabels(["0", "π/2", "π"])


def _lognorm_for_intensity(intensity: np.ndarray) -> LogNorm:
    i = np.asarray(intensity, dtype=np.float64)
    vmax = float(np.nanmax(i)) if np.isfinite(i).any() else 1.0
    if vmax <= 0:
        return LogNorm(vmin=1e-12, vmax=1.0)
    positive = i[i > 0]
    if positive.size == 0:
        vmin = 1e-12
    else:
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
        results.append(
            {
                "macro_row": r,
                "y_start": int(y0),
                "y_end": int(y1),
                "peak_x": int(bx),
                "peak_y": int(y0 + by),
                "peak_intensity": float(band[by, bx]),
            }
        )
    return results


def _parse_macro_cols_from_filename(path: Path) -> int | None:
    """
    Parse 'cXX' from filenames like ..._r16_c16_...
    Returns None if not present.
    """
    m = re.search(r"_c(\d+)(?:_|$)", path.stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _amplitude_gradient_map(height: int, width: int, macro_cols: int) -> np.ndarray:
    """
    Build amplitude map constant in each macropixel column.
    Column 0 amplitude macro_cols, column 1 amplitude (macro_cols-1), ... rightmost column amplitude 1.
    Repeated for all rows (no y dependence).
    """
    macro_cols = max(1, int(macro_cols))
    macro_w = max(1, int(round(width / macro_cols)))

    amps = np.arange(macro_cols, 0, -1, dtype=np.float32)
    row = np.repeat(amps, macro_w)
    if row.size < width:
        row = np.pad(row, (0, width - row.size), mode="edge")
    elif row.size > width:
        row = row[:width]

    return np.tile(row[None, :], (height, 1))


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
        print("WARNING: Could not find a matching non-grating BMP; using the grating BMP as 'original'.")
        data_orig_u8 = np.clip(data_grating, 0, 255).astype(np.uint8)

    data_orig_bw = _as_bw_0_255(data_orig_u8)

    # --- Physics Simulation (Phase Modulation) ---
    phase_map = (data_grating / PI_GRAY_LEVEL) * np.pi

    # --- Amplitude gradient term (constant per macropixel column) ---
    h, w = phase_map.shape
    if AMPLITUDE_MODE.lower() == "gradient":
        macro_cols = _parse_macro_cols_from_filename(img_path) or DEFAULT_MACRO_COLS
        amp_map = _amplitude_gradient_map(h, w, macro_cols)
        amp_note = f"Amplitude gradient \"{macro_cols} to 1\""
    else:
        amp_map = np.ones((h, w), dtype=np.float32)
        amp_note = "uniform amplitude"

    # Create the Optical Field (Complex Number): amplitude * exp(i*phase)
    input_field = amp_map * np.exp(1j * phase_map)

    # Cylindrical Lens Propagation (1D FFT along x-axis, axis=1)
    fft_field = np.fft.fftshift(np.fft.fft(input_field, axis=1), axes=1)
    intensity = np.abs(fft_field) ** 2
    intensity_log1p = np.log1p(intensity)

    out_dir = img_path.parent
    stem = img_path.stem

    # --- Readout: one intensity value per macropixel row ---
    peaks = _macrorow_peaks(
        intensity,
        MACROPIXEL_HEIGHT,
        exclude_center=EXCLUDE_CENTER_ZERO_ORDER,
        center_half_width=CENTER_EXCLUSION_HALF_WIDTH_PX,
    )
    if SAVE_MACROROW_PEAKS_CSV:
        out_csv = out_dir / f"{stem}_CCD_macrorow_peaks_linear.csv"
        with out_csv.open("w", newline="") as f:
            wcsv = csv.DictWriter(
                f,
                fieldnames=["macro_row", "y_start", "y_end", "peak_x", "peak_y", "peak_intensity"],
            )
            wcsv.writeheader()
            wcsv.writerows(peaks)
        print(f"Saved: {out_csv}")
    if peaks:
        vals = np.array([float(p["peak_intensity"]) for p in peaks], dtype=float)
        print(
            f"Macrorow peaks (linear): N={len(peaks)}, "
            f"min={vals.min():.3g}, median={np.median(vals):.3g}, max={vals.max():.3g}"
        )

    # --- Linear CCD colormap scaling ---
    linear_vmin = 0.0
    # Theoretical maximum for an unnormalized FFT row is (sum |amplitude|)^2 (if phases align).
    amp_row = amp_map[0, :].astype(np.float64)
    theoretical_max = float(np.sum(np.abs(amp_row)) ** 2) if amp_row.size else 1.0

    if LINEAR_VMAX_MODE == "theoretical_fraction":
        linear_vmax = float(LINEAR_VMAX_FRACTION_OF_THEORETICAL_MAX) * theoretical_max
        linear_vmax_note = rf"vmax={LINEAR_VMAX_FRACTION_OF_THEORETICAL_MAX:g}·(Σ|A|)²"
    elif LINEAR_VMAX_MODE == "fixed":
        if LINEAR_VMAX_FIXED is None:
            linear_vmax = float(LINEAR_VMAX_FRACTION_OF_THEORETICAL_MAX) * theoretical_max
            linear_vmax_note = rf"vmax={LINEAR_VMAX_FRACTION_OF_THEORETICAL_MAX:g}·(Σ|A|)² [fallback]"
        else:
            linear_vmax = float(LINEAR_VMAX_FIXED)
            linear_vmax_note = f"vmax=fixed={linear_vmax:.3g}"
    elif LINEAR_VMAX_MODE == "peaks_percentile":
        if peaks:
            peak_vals = np.array([float(p["peak_intensity"]) for p in peaks], dtype=float)
            pctl = float(np.percentile(peak_vals, LINEAR_VMAX_PEAKS_PERCENTILE))
            linear_vmax = max(1.0, pctl * float(LINEAR_VMAX_PEAKS_MARGIN))
            linear_vmax_note = (
                f"vmax=p{LINEAR_VMAX_PEAKS_PERCENTILE:g}(peaks)×{LINEAR_VMAX_PEAKS_MARGIN:g}={linear_vmax:.3g}"
            )
        else:
            linear_vmax = float(LINEAR_VMAX_FRACTION_OF_THEORETICAL_MAX) * theoretical_max
            linear_vmax_note = rf"vmax={LINEAR_VMAX_FRACTION_OF_THEORETICAL_MAX:g}·(Σ|A|)² [no peaks]"
    else:
        linear_vmax = float(LINEAR_VMAX_FRACTION_OF_THEORETICAL_MAX) * theoretical_max
        linear_vmax_note = rf"vmax={LINEAR_VMAX_FRACTION_OF_THEORETICAL_MAX:g}·(Σ|A|)² [unknown mode]"

    if not np.isfinite(linear_vmax) or linear_vmax <= 0:
        linear_vmax = float(np.nanmax(intensity)) if np.isfinite(intensity).any() else 1.0
        linear_vmax_note = f"vmax=data_max={linear_vmax:.3g} [fallback]"

    # For correct visualization: ensure the plot scale includes the true peak (avoid clipping).
    # This prevents the brightest pixel from saturating at the same color as other near-max pixels.
    finite_I = intensity[np.isfinite(intensity)]
    data_max = float(np.max(finite_I)) if finite_I.size else 1.0
    if LINEAR_ENSURE_NO_CLIP_IN_PLOTS and np.isfinite(data_max) and data_max > 0:
        data_max *= float(LINEAR_DATA_MAX_MARGIN)
        if data_max > linear_vmax:
            linear_vmax = data_max
            linear_vmax_note = f"{linear_vmax_note} → max(data)={linear_vmax:.3g}"

    print(f"Amplitude term: {amp_note}")
    print(f"Linear CCD colormap scaling: vmin={linear_vmin:g}, vmax={linear_vmax:.3g} ({linear_vmax_note})")

    # Use one shared Normalize object for ALL linear CCD subplots (FINAL + OVERVIEW),
    # so the colormap mapping is guaranteed identical.
    linear_norm = mpl.colors.Normalize(vmin=linear_vmin, vmax=linear_vmax, clip=True)

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

    # --- Plots (same layout as v3) ---
    figs: list[plt.Figure] = []

    if SAVE_PLOTS or SHOW_FIGURE:
        # FINAL: original BW, grating input, CCD linear
        fig1, ax = plt.subplots(1, 3, figsize=(18, 6))
        figs.append(fig1)

        im0 = ax[0].imshow(
            data_orig_bw.astype(np.float32) * DISPLAY_PHASE_SCALE,
            cmap="gray",
            vmin=0.0,
            vmax=DISPLAY_PHASE_MAX,
            aspect="auto",
        )
        ax[0].set_title(r"Original Phase" "\n" r"[Black: $\phi = 0$; White: $\phi = \pi$]")
        ax[0].set_xlabel("x (px)")
        ax[0].set_ylabel("y (px)")
        c0 = fig1.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)
        _phase_colorbar(c0)

        im1 = ax[1].imshow(
            np.clip(data_grating, 0, 255) * DISPLAY_PHASE_SCALE,
            cmap="gray",
            vmin=0.0,
            vmax=DISPLAY_PHASE_MAX,
            aspect="auto",
        )
        ax[1].set_title("SLM input (with grating)\n(as loaded)")
        ax[1].set_xlabel("x (px)")
        ax[1].set_ylabel("y (px)")
        c1 = fig1.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)
        _phase_colorbar(c1)

        im2 = ax[2].imshow(
            intensity,
            cmap=LINEAR_CMAP,
            norm=linear_norm,
            aspect="auto",
            interpolation="nearest",
            resample=False,
        )
        ax[2].set_title(
            "CCD intensity (linear scale)\nCylindrical lens focus\n"
            f"{amp_note}\nPeak intensity values written at right"
        )
        ax[2].set_xlabel("k_x (FFT-shifted index)")
        ax[2].set_ylabel("y (px)")
        c2 = fig1.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)
        c2.set_label(f"Intensity (a.u.) [{linear_vmax_note}]")

        # Per-row peak numbers (shared exponent)
        if peaks:
            peak_vals = np.array([float(p["peak_intensity"]) for p in peaks], dtype=float)
            vmaxp = float(np.max(peak_vals)) if peak_vals.size else 0.0
            if vmaxp > 0:
                common_exp = int(np.floor(np.log10(vmaxp)))
                scale = 10.0**common_exp
            else:
                common_exp = 0
                scale = 1.0

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
                peak_rgba = mpl.colormaps.get_cmap(LINEAR_CMAP)(linear_norm(float(p["peak_intensity"])))
                ax[2].text(
                    x_text,
                    y_mid,
                    f"{mantissa:6.2f}",
                    color=peak_rgba,
                    fontsize=8,
                    ha="right",
                    va="center",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                )

        fig1.tight_layout()
        if SAVE_PLOTS:
            out_fig1 = out_dir / f"{stem}_FFT_cylindrical_lens_FINAL.png"
            fig1.savefig(str(out_fig1), dpi=220, bbox_inches="tight")
            print(f"Saved: {out_fig1}")

        # LOG: original BW + CCD log1p colored, with cyan dots
        fig2, ax2 = plt.subplots(1, 2, figsize=(13, 6))
        figs.append(fig2)

        im20 = ax2[0].imshow(
            data_orig_bw.astype(np.float32) * DISPLAY_PHASE_SCALE,
            cmap="gray",
            vmin=0.0,
            vmax=DISPLAY_PHASE_MAX,
            aspect="auto",
        )
        ax2[0].set_title(
            r"Original Phase" "\n" r"[Black: $\phi = 0$; White: $\phi = \pi$]"
        )
        ax2[0].set_xlabel("x (px)")
        ax2[0].set_ylabel("y (px)")
        c20 = fig2.colorbar(im20, ax=ax2[0], fraction=0.046, pad=0.04)
        _phase_colorbar(c20)

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

        # OVERVIEW: 2x2
        fig3, ax3 = plt.subplots(2, 2, figsize=(16, 12))
        figs.append(fig3)

        im30 = ax3[0, 0].imshow(
            data_orig_bw.astype(np.float32) * DISPLAY_PHASE_SCALE,
            cmap="gray",
            vmin=0.0,
            vmax=DISPLAY_PHASE_MAX,
            aspect="auto",
        )
        ax3[0, 0].set_title(r"Original Phase" "\n" r"[Black: $\phi = 0$; White: $\phi = \pi$]")
        ax3[0, 0].set_xlabel("x (px)")
        ax3[0, 0].set_ylabel("y (px)")
        _phase_colorbar(fig3.colorbar(im30, ax=ax3[0, 0], fraction=0.046, pad=0.04))

        im31 = ax3[0, 1].imshow(
            np.clip(data_grating, 0, 255) * DISPLAY_PHASE_SCALE,
            cmap="gray",
            vmin=0.0,
            vmax=DISPLAY_PHASE_MAX,
            aspect="auto",
        )
        ax3[0, 1].set_title("Input (with grating)")
        ax3[0, 1].set_xlabel("x (px)")
        ax3[0, 1].set_ylabel("y (px)")
        _phase_colorbar(fig3.colorbar(im31, ax=ax3[0, 1], fraction=0.046, pad=0.04))

        im32 = ax3[1, 0].imshow(
            intensity,
            cmap=LINEAR_CMAP,
            norm=linear_norm,
            aspect="auto",
            interpolation="nearest",
            resample=False,
        )
        ax3[1, 0].set_title("CCD intensity (linear)")
        ax3[1, 0].set_xlabel("k_x (FFT-shifted index)")
        ax3[1, 0].set_ylabel("y (px)")
        fig3.colorbar(im32, ax=ax3[1, 0], fraction=0.046, pad=0.04).set_label(
            f"Intensity (a.u.) [{linear_vmax_note}]"
        )

        # Same peak intensity values at right as in *_FINAL.png
        if peaks:
            peak_vals = np.array([float(p["peak_intensity"]) for p in peaks], dtype=float)
            vmaxp = float(np.max(peak_vals)) if peak_vals.size else 0.0
            if vmaxp > 0:
                common_exp = int(np.floor(np.log10(vmaxp)))
                scale = 10.0**common_exp
            else:
                common_exp = 0
                scale = 1.0

            ax3[1, 0].text(
                0.02,
                0.98,
                rf"$\times 10^{{{common_exp}}}$",
                transform=ax3[1, 0].transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color="white",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.45),
            )

            x_text = intensity.shape[1] - 5
            for p in peaks:
                y_mid = 0.5 * (float(p["y_start"]) + float(p["y_end"]))
                mantissa = float(p["peak_intensity"]) / scale if scale != 0 else 0.0
                peak_rgba = mpl.colormaps.get_cmap(LINEAR_CMAP)(linear_norm(float(p["peak_intensity"])))
                ax3[1, 0].text(
                    x_text,
                    y_mid,
                    f"{mantissa:6.2f}",
                    color=peak_rgba,
                    fontsize=7,
                    ha="right",
                    va="center",
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                )

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
            # Slightly higher DPI to preserve narrow bright peaks in the overview layout
            fig3.savefig(str(out_fig3), dpi=300, bbox_inches="tight")
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

