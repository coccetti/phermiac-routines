#!/usr/bin/env python3
"""
SLM Algorithm Implementation - Full Frame Correction Version
Author: Fabrizio Coccetti (PhD Level Implementation)
Date: September, 2025 (Updated Dec 2025)

DESCRIPTION:
This script generates a phase mask for a Hamamatsu LCOS-SLM.
1. Loads the Wavefront Correction file (CAL_...) in its NATIVE resolution.
2. Generates the Signal Matrix (720x480) from CERN binary data.
3. Centers the Signal Matrix within the full SLM frame defined by the Correction file.
4. Adds a Blazed Phase Grating (Sawtooth) to the entire frame for 0-order separation.
5. Computes the final phase modulo 2pi and maps it to 8-bit gray levels.

KEY CONSTRAINTS:
- The Correction File determines the final output resolution (e.g., 1272x1024).
- The Correction File is NEVER resized.
- The Signal is strictly added to the central region.
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from PIL import Image

# --- Configuration Parameters ---
# Signal Dimensions (The active area of interest)
SIGNAL_ROWS = 720           # Height of the signal pattern
SIGNAL_COLS = 480           # Width of the signal pattern

# Macropixel Grid Configuration
NUMBERS_TO_COMPARE = 24     # Number of macropixel rows
INPUT_RESOLUTION_BITS = 16  # Number of macropixel columns

# SLM Calibration & Hardware
PI_WHITE_VALUE = 108        # Gray value corresponding to a pi phase shift
GRATING_PERIOD = 16         # Period of the blazed grating in pixels
DEFAULT_SLM_W = 1272        # Fallback width if calibration file missing
DEFAULT_SLM_H = 1024        # Fallback height if calibration file missing

# Input Files
TELESCOPE_01_FILE = "CERN-01_event1_nbit16.csv"
TELESCOPE_02_FILE = "CERN-02_sec1_nbit16.csv"
CORRECTION_FILE = "CAL_LSH0905569_1064nm.bmp"

def load_wavefront_correction_native(filename):
    """
    Loads the Hamamatsu correction BMP in its native resolution.
    
    Args:
        filename (str): Path to the BMP file.
        
    Returns:
        tuple: (correction_phase_matrix, width, height)
        
    Note:
        Returns a zero matrix and default dimensions if file is missing.
    """
    if not os.path.exists(filename):
        print(f"WARNING: Correction file '{filename}' not found.")
        print(f"Using default empty correction (Size: {DEFAULT_SLM_W}x{DEFAULT_SLM_H})")
        return np.zeros((DEFAULT_SLM_H, DEFAULT_SLM_W), dtype=float), DEFAULT_SLM_W, DEFAULT_SLM_H
        
    print(f"Loading wavefront correction from: {filename}")
    try:
        with Image.open(filename) as img:
            width, height = img.size
            print(f"Native Correction Resolution: {width}x{height}")
            
            # Convert to numpy array (Grayscale)
            bmp_data = np.array(img.convert('L'), dtype=float)
            
            # Normalize 0-255 to 0-2pi 
            # (Standard Hamamatsu LUT: 0->0, 255->2pi approx, dependent on wavelength)
            # We assume the BMP is a full dynamic range phase map.
            correction_phase = (bmp_data / 255.0) * (2 * np.pi)
            
            return correction_phase, width, height
            
    except Exception as e:
        print(f"ERROR loading correction file: {e}")
        return np.zeros((DEFAULT_SLM_H, DEFAULT_SLM_W), dtype=float), DEFAULT_SLM_W, DEFAULT_SLM_H

def create_signal_matrix(rows, cols):
    """
    Create the signal pixel matrix structure (initialized to 0).
    """
    return np.zeros((rows, cols), dtype=float)

def assign_macropixel_value(matrix, row_idx, col_idx, value, macro_h, macro_w):
    """Assign a value to a specific macropixel region."""
    r_start = row_idx * macro_h
    r_end = r_start + macro_h
    c_start = col_idx * macro_w
    c_end = c_start + macro_w
    matrix[r_start:r_end, c_start:c_end] = value

def read_binary_data(filename):
    """Read binary string from CSV."""
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            return next(reader)[0]
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None

def generate_blazed_grating(rows, cols, period_pixels=16):
    """
    Generates a linear phase ramp (sawtooth) over the full frame.
    """
    _, x_indices = np.indices((rows, cols))
    # Phase = 2pi * (x / Period)
    return (2 * np.pi * x_indices / period_pixels)

def pad_signal_to_center(signal_matrix, full_rows, full_cols):
    """
    Places the signal matrix into the center of a zero-filled full-frame matrix.
    """
    sig_rows, sig_cols = signal_matrix.shape
    
    if sig_rows > full_rows or sig_cols > full_cols:
        raise ValueError(f"Signal size ({sig_rows}x{sig_cols}) exceeds SLM size ({full_rows}x{full_cols})")
    
    # Calculate centering offsets
    row_offset = (full_rows - sig_rows) // 2
    col_offset = (full_cols - sig_cols) // 2
    
    full_frame_signal = np.zeros((full_rows, full_cols), dtype=float)
    full_frame_signal[row_offset:row_offset+sig_rows, col_offset:col_offset+sig_cols] = signal_matrix
    
    print(f"Signal centered at offset: (y={row_offset}, x={col_offset})")
    return full_frame_signal

def process_slm_algorithm():
    print("\n--- SLM Algorithm: Full Frame Correction Mode ---")
    
    # --- Step 1: Load Wavefront Correction (Defines the Master Resolution) ---
    correction_phase, slm_width, slm_height = load_wavefront_correction_native(CORRECTION_FILE)
    slm_rows = slm_height
    slm_cols = slm_width
    
    # --- Step 2: Generate the Signal Matrix (The "Inner" Image) ---
    print(f"\nGenerating Signal Matrix ({SIGNAL_ROWS}x{SIGNAL_COLS})...")
    signal_matrix = create_signal_matrix(SIGNAL_ROWS, SIGNAL_COLS)
    
    # Calculate macropixel sizes for the signal area
    mp_rows = SIGNAL_ROWS // NUMBERS_TO_COMPARE
    mp_cols = SIGNAL_COLS // INPUT_RESOLUTION_BITS
    print(f"Macropixel size: {mp_rows}x{mp_cols} pixels")
    
    # Process CERN-01
    cern01_data = read_binary_data(TELESCOPE_01_FILE)
    if cern01_data:
        for col, bit in enumerate(cern01_data):
            val = np.pi if bit == '1' else 0
            for row in range(NUMBERS_TO_COMPARE):
                assign_macropixel_value(signal_matrix, row, col, val, mp_rows, mp_cols)
                
    # Process CERN-02
    try:
        with open(TELESCOPE_02_FILE, 'r') as f:
            reader = csv.reader(f)
            cern02_rows = [r[0] for i, r in enumerate(reader) if i < NUMBERS_TO_COMPARE]
            
        for r_idx, bits in enumerate(cern02_rows):
            for c_idx, bit in enumerate(bits):
                add_val = np.pi if bit == '1' else 0
                # Apply XOR logic in phase domain (0+pi=pi, pi+pi=2pi~0)
                current = signal_matrix[r_idx*mp_rows, c_idx*mp_cols]
                signal_matrix[r_idx*mp_rows : (r_idx+1)*mp_rows, 
                              c_idx*mp_cols : (c_idx+1)*mp_cols] = (current + add_val) % (2*np.pi)
    except FileNotFoundError:
        print("CERN-02 file not found, skipping.")

    # --- Step 3: Center the Signal on the Master Frame ---
    print("\nCentering signal on full frame...")
    padded_signal_phase = pad_signal_to_center(signal_matrix, slm_rows, slm_cols)
    
    # --- Step 4: Generate Blazed Grating (Full Frame) ---
    print(f"Generating Phase Grating (Period: {GRATING_PERIOD}px)...")
    grating_phase = generate_blazed_grating(slm_rows, slm_cols, GRATING_PERIOD)
    
    # --- Step 5: Compute Total Phase ---
    # Formula: Final = (Correction + Grating + Signal) % 2pi
    # The signal is 0 in the periphery, so there it's just Correction + Grating.
    print("Combining Correction + Grating + Signal...")
    total_phase = (correction_phase + grating_phase + padded_signal_phase) % (2 * np.pi)
    
    # --- Step 6: Convert to 8-bit Bitmap ---
    # Determine the gray level scaling factor
    # PI_WHITE_VALUE is the gray level for pi (108). So 2pi is 216.
    TWO_PI_GRAY = 2 * PI_WHITE_VALUE
    if TWO_PI_GRAY > 255:
        print(f"Warning: 2pi gray level ({TWO_PI_GRAY}) clipped to 255.")
        TWO_PI_GRAY = 255
        
    print(f"Mapping Phase 0-2pi to Gray 0-{TWO_PI_GRAY}")
    bmp_float = (total_phase / (2 * np.pi)) * TWO_PI_GRAY
    bmp_uint8 = bmp_float.astype(np.uint8)
    
    # --- Step 7: Save Outputs ---
    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        
    output_filename = os.path.join(plots_folder, f'slm_final_centered_g{GRATING_PERIOD}.bmp')
    Image.fromarray(bmp_uint8, mode='L').save(output_filename)
    
    print("\n" + "="*50)
    print(f"SUCCESS. Output saved to: {output_filename}")
    print(f"Final Image Dimensions: {slm_width} x {slm_height}")
    print(f"Signal Region: {SIGNAL_ROWS} x {SIGNAL_COLS} (Centered)")
    print(f"Wavefront Correction: Applied full frame from {CORRECTION_FILE}")
    print("="*50)

    # Optional Debug: Save the centered signal mask to verify position
    debug_mask = (padded_signal_phase > 0).astype(np.uint8) * 255
    Image.fromarray(debug_mask, mode='L').save(os.path.join(plots_folder, 'debug_signal_placement.png'))

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    process_slm_algorithm()
    