#!/usr/bin/env python3
"""
Algorithm 9: read bits and make SLM BMP image(s)

- Defines a macropixel of size 60x60 pixels.
- Reads a CSV containing 0/1 digits (typically one bit-string per row).
- The output BMP size is computed automatically:
  - number of macropixel columns = number of digits per line
  - number of macropixel rows    = number of lines
  - output pixels = (cols * 60) x (rows * 60)
- Saves a BMP image with characteristics consistent with `16_bits/slm/slm_clean.png`
  and with the same grayscale scaling as
  `16_bits/slm2/plots/slm_chessboard_720x480_r24_c16_w108.bmp`
  (8-bit grayscale: black=0, white=108). Values are fixed (no template read).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image


# =========================
# Input and Output paths
# =========================
# All paths are resolved relative to this script's folder (`xx_bits/`).
BASE_DIR = Path(__file__).resolve().parent

# Input: CSV with 8 digits per row (either "01010101" or "0,1,0,1,0,1,0,1")
# BITS_CSV = BASE_DIR / "data" / "SLM_16bits_CERN-01_event1221_CERN-02_sec1221_nbit16.dat"
# BITS_CSV = BASE_DIR / "data" / "SLM_16bits_CERN-01_event1221.dat"
BITS_CSV = BASE_DIR / "data" / "SLM_16x16bits_CERN-01_event3.dat"
# BITS_CSV = BASE_DIR / "data" / "SLM_16x16bits_CERN-01_event3_CERN-02_sec3.dat"
# BITS_CSV = BASE_DIR / "data" / "SLM_16x16bits_CERN-02_sec3_nbit16.dat"
# BITS_CSV = BASE_DIR / "data" / "SLM_16x16bits_CERN-02_sec3.dat"

# Output BMP
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_FILENAME = OUTPUT_DIR / "SLM_16x16bits_CERN-01_event3.bmp"

# Geometry
MACROPIXEL_SIZE = 60  # px

# Match `..._w108.bmp` style (8-bit grayscale)
IMAGE_MODE = "L"
BLACK_VALUE = 0
WHITE_VALUE = 108


def _parse_digits_from_csv_row(raw_row: List[str]) -> Optional[List[str]]:
    if not raw_row:
        return None
    # Accept either:
    # - single column like "01010101"
    # - N columns like "0,1,0,1,..."
    if len(raw_row) == 1:
        s = raw_row[0].strip().replace(" ", "")
        if not s:
            return None
        return list(s)
    digits = [c.strip() for c in raw_row if c.strip() != ""]
    return digits if digits else None


def _read_bits_matrix(csv_path: Path) -> Tuple[List[List[int]], int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Bits CSV not found: {csv_path}")

    rows: List[List[int]] = []
    expected_cols: Optional[int] = None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for raw_row in reader:
            digits = _parse_digits_from_csv_row(raw_row)
            if digits is None:
                continue

            if expected_cols is None:
                expected_cols = len(digits)
                if expected_cols <= 0:
                    raise ValueError(f"Could not infer number of digits per line from row: {raw_row!r}")
            elif len(digits) != expected_cols:
                raise ValueError(
                    f"Inconsistent digits-per-line: expected {expected_cols}, got {len(digits)} in row: {raw_row!r}"
                )

            if any(d not in ("0", "1") for d in digits):
                raise ValueError(f"Non-binary digit found in row: {raw_row!r}")

            rows.append([1 if d == "1" else 0 for d in digits])

    if not rows:
        raise ValueError(f"No data rows found in: {csv_path}")

    if expected_cols is None:
        raise ValueError(f"Could not infer digits-per-line from: {csv_path}")

    return rows, expected_cols


def _make_image_from_bits(rows: List[List[int]], cols: int) -> Image.Image:
    if cols <= 0:
        raise ValueError(f"Invalid cols: {cols}")
    if not rows:
        raise ValueError("No rows provided")

    out_w = cols * MACROPIXEL_SIZE
    out_h = len(rows) * MACROPIXEL_SIZE
    img = Image.new(IMAGE_MODE, (out_w, out_h), color=BLACK_VALUE)

    macro_black = Image.new(IMAGE_MODE, (MACROPIXEL_SIZE, MACROPIXEL_SIZE), color=BLACK_VALUE)
    macro_white = Image.new(IMAGE_MODE, (MACROPIXEL_SIZE, MACROPIXEL_SIZE), color=WHITE_VALUE)

    for r, row in enumerate(rows):
        if len(row) != cols:
            raise ValueError(f"Row length mismatch at row {r}: expected {cols}, got {len(row)}")
        for c, bit in enumerate(row):
            tile = macro_white if bit == 1 else macro_black
            img.paste(tile, (c * MACROPIXEL_SIZE, r * MACROPIXEL_SIZE))

    return img


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a BMP made of macropixels from a CSV of 0/1 digits."
    )
    parser.add_argument("--csv", default=str(BITS_CSV), help="Input CSV containing 8 digits per row.")
    parser.add_argument("--out", default=str(OUTPUT_FILENAME), help="Output BMP filename.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    rows, cols = _read_bits_matrix(csv_path)
    img = _make_image_from_bits(rows, cols)

    # Append computed info to output filename, e.g.:
    # "slm_bits_480x480_r8_c8_w108.bmp"
    w, h = img.size
    r = len(rows)
    c = cols
    if out_path.suffix.lower() != ".bmp":
        # Keep it simple: always write BMP files
        out_path = out_path.with_suffix(".bmp")
    out_path = out_path.with_name(f"{out_path.stem}_{w}x{h}_r{r}_c{c}_w{WHITE_VALUE}{out_path.suffix}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "BMP")

    print(f"Read {len(rows)} rows x {cols} cols from {csv_path}")
    print(f"Wrote BMP {w}x{h} to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

