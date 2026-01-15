#!/usr/bin/env python3
"""
Algorithm 8: make bits table

Reads:
 - ./08_bits/data/CERN-01_event1221_nbit8.csv  (uses ONLY the first non-empty line as mask)
 - ./08_bits/data/CERN-02_sec1221_nbit8.csv    (processes every non-empty line)

For each line of the second file, performs a "binary add without carry" with the
mask, digit-by-digit (equivalent to XOR for 0/1 digits), and writes the result
to a one-column CSV file.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


# =========================
# Input file paths
# =========================
# These are the default paths used when you run the script without CLI arguments.
# They are resolved relative to THIS script's folder (`08_bits/`).
BASE_DIR = Path(__file__).resolve().parent
EVENT_CSV = BASE_DIR / "data" / "CERN-01_event1221_nbit8.csv"
SECONDS_CSV = BASE_DIR / "data" / "CERN-02_sec1221_nbit8.csv"
OUT_CSV = BASE_DIR / "data" / "SLM_bits_CERN-01_event1221_CERN-02_sec1221_nbit8.csv"


def _read_bit_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    lines: List[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s:
            continue
        lines.append(s)
    return lines


def _parse_bits(bit_string: str) -> List[int]:
    if any(c not in "01" for c in bit_string):
        raise ValueError(f"Non-binary digit found in: {bit_string!r}")
    return [1 if c == "1" else 0 for c in bit_string]


def _xor_bits(a: List[int], b: List[int]) -> List[int]:
    if len(a) != len(b):
        raise ValueError(f"Bit-length mismatch: {len(a)} != {len(b)}")
    return [(x + y) & 1 for x, y in zip(a, b)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Digit-wise binary add-without-carry (XOR) between an event mask and each line of a seconds file."
    )
    parser.add_argument(
        "--event",
        default=str(EVENT_CSV),
        help="Event CSV path (first non-empty line is used as mask).",
    )
    parser.add_argument(
        "--seconds",
        default=str(SECONDS_CSV),
        help="Seconds CSV path (each non-empty line is processed).",
    )
    parser.add_argument(
        "--out",
        default=str(OUT_CSV),
        help="Output CSV path (one column).",
    )
    args = parser.parse_args()

    event_path = Path(args.event)
    seconds_path = Path(args.seconds)
    out_path = Path(args.out)

    event_lines = _read_bit_lines(event_path)
    if not event_lines:
        raise ValueError(f"No non-empty lines found in event file: {event_path}")

    mask_bits = _parse_bits(event_lines[0])

    seconds_lines = _read_bit_lines(seconds_path)
    if not seconds_lines:
        raise ValueError(f"No non-empty lines found in seconds file: {seconds_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for line in seconds_lines:
            bits = _parse_bits(line)
            combined = _xor_bits(mask_bits, bits)
            writer.writerow(["".join(str(d) for d in combined)])

    print(f"Wrote {len(seconds_lines)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

