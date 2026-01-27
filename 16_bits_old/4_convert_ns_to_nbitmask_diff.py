#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read nanoseconds from a CERN-0*.csv file and convert each value to an n-bit binary string
after applying a mask to the _highest_ n bits.

nanoseconds: List of nanoseconds from the CERN-0*.csv file
n_bits: Number of bits to keep after applying the mask
shift_value: Number of bits to shift to the right (to apply the mask to the highest n bits)
mask: Mask to apply to the highest n bits
nanoseconds_nbit: List of n-bit binary strings
"""

import csv
import numpy as np

def read_nanoseconds(file_path):
    nanoseconds = []
    with open(file_path, mode='r') as file:
        for line in file:
            line = line.strip()
            if line:  # Skip empty lines
                # Convert the floating point time difference to nanoseconds
                time_diff_seconds = float(line)
                nanoseconds.append(int(time_diff_seconds * 1e9))  # Convert to nanoseconds
    return np.array(nanoseconds)

def convert_to_nbit_binary_string(number, n):
    masked_number = (number >> shift_value) & mask  # Apply mask to highest n bits (after shifting)
    return format(masked_number, f'0{n}b')

def save_to_csv(data, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Nanoseconds_nbit"])
        for item in data:
            writer.writerow([item])

if __name__ == "__main__":
    file_path = 'data/time_differences_ABS_C01-event_C02-1sec.csv'
    nanoseconds = read_nanoseconds(file_path)
    n_bits = 16  # number of bits to keep [THIS IS THE NUMBER YOU WANNA CHANGE!!]
    shift_value = 30 - n_bits  # 30 is the number of bits for nanoseconds
    mask = (1 << n_bits) - 1
    print(f"Number of bits to keep: {n_bits}")
    print(f"Mask: {mask}")
    print(f"Shift Value: {shift_value}")
    nanoseconds_nbit = [convert_to_nbit_binary_string(ns, n_bits) for ns in nanoseconds]
    print(f"np.array of nanoseconds converted into {n_bits} bits:\n{nanoseconds_nbit}")

    # Generate output filename by adding "_bits" to the input filename
    from pathlib import Path
    input_path = Path(file_path)
    output_file_path = input_path.parent / f"{input_path.stem}_bits{input_path.suffix}"
    save_to_csv(nanoseconds_nbit, output_file_path)
    print(f"Converted nanoseconds saved to {output_file_path}")
