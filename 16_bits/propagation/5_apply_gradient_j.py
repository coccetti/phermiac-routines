# -*- coding: utf-8 -*-
"""
Algorithm: Apply Macropixel-Column Gradient (Amplitude-like weighting)

Description:
    Loads an SLM pattern (8-bit grayscale) and applies ONLY the macropixel-column
    gradient used in 4_fft_cyl_gradient.py (no grating).

    For each macropixel column:
      leftmost column factor = MACRO_COLS
      next column factor     = MACRO_COLS-1
      ...
      rightmost column factor = 1

    The factor is constant inside each macropixel column and identical for all rows.

IMPORTANT (representation):
    A real "amplitude" term cannot be encoded directly in a phase-only SLM bitmap.
    Here we apply the gradient to the bitmap gray values via a *normalized multiplicative*
    scaling (factor / max_factor) so the result remains within [0,255].

Output:
    Writes a new BMP with suffix "_gradient.bmp" in the same img folder.

Author: Fabrizio Coccetti
Date: 2026-02-03
"""

from __future__ import annotations

import os
import re
import numpy as np
from PIL import Image


# --- CONFIGURATION ---
BASE_DIR = "16_bits/propagation"
IMG_SUBDIR = "img"

INPUT_FILENAME = "SLM_16x16bits_CERN-01_event3_CERN-02_sec3_960x960_r16_c16_w108.bmp"

# If present, macro columns are parsed from filename token "_cXX_"
DEFAULT_MACRO_COLS = 16

# How to apply the gradient to the grayscale image:
# - "multiply_normalized": output = input * (factor / max_factor)
APPLY_MODE = "multiply_normalized"


def _parse_macro_cols_from_filename(filename: str) -> int | None:
    m = re.search(r"_c(\d+)(?:_|$)", os.path.splitext(filename)[0])
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _gradient_row(width: int, macro_cols: int) -> np.ndarray:
    """
    Returns a 1D float array of length 'width' with values decreasing left->right:
    [macro_cols .. 1] constant per macropixel column.
    """
    macro_cols = max(1, int(macro_cols))
    macro_w = max(1, int(round(width / macro_cols)))

    amps = np.arange(macro_cols, 0, -1, dtype=np.float32)  # left=macro_cols, right=1
    row = np.repeat(amps, macro_w)
    if row.size < width:
        row = np.pad(row, (0, width - row.size), mode="edge")
    elif row.size > width:
        row = row[:width]
    return row.astype(np.float32)


def main() -> None:
    print("--- Apply Macropixel-Column Gradient ---")

    input_path = os.path.join(BASE_DIR, IMG_SUBDIR, INPUT_FILENAME)
    if not os.path.exists(input_path):
        # allow running from same folder
        if os.path.exists(INPUT_FILENAME):
            input_path = INPUT_FILENAME
        else:
            raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading: {input_path}")
    img = Image.open(input_path).convert("L")
    data = np.array(img, dtype=np.float32)
    h, w = data.shape

    macro_cols = _parse_macro_cols_from_filename(INPUT_FILENAME) or DEFAULT_MACRO_COLS
    print(f"Image size: {w}x{h}, macro_cols: {macro_cols}, apply_mode: {APPLY_MODE}")

    g = _gradient_row(w, macro_cols)  # (w,)
    g2 = np.tile(g[None, :], (h, 1))  # (h,w)

    if APPLY_MODE == "multiply_normalized":
        g_norm = g2 / float(np.max(g2))  # 1.0 on left, 1/macro_cols on right
        out = data * g_norm
    else:
        raise ValueError(f"Unknown APPLY_MODE: {APPLY_MODE}")

    out_u8 = np.clip(np.round(out), 0, 255).astype(np.uint8)

    # Output filename
    name, ext = os.path.splitext(INPUT_FILENAME)
    output_filename = f"{name}_gradient{ext}"
    output_dir = os.path.join(BASE_DIR, IMG_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    Image.fromarray(out_u8).save(output_path)
    print(f"SUCCESS: Saved gradient image to:\n{output_path}")


if __name__ == "__main__":
    main()

