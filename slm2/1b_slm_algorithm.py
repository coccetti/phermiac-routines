#!/usr/bin/env python3
"""
SLM Algorithm Implementation - ROI Grating Mode
Author: Fabrizio Coccetti (PhD Level Implementation)
Date: September, 2025 (Updated Dec 2025)

DESCRIPTION:
This script generates a phase mask for a Hamamatsu LCOS-SLM with Region-of-Interest (ROI) control.
1. Loads the Wavefront Correction file (CAL_...) as the Master Frame (e.g., 1272x1024).
2. Generates the Signal Matrix (720x480) from CERN binary data.
3. Generates a Blazed Phase Grating ONLY for the 720x480 ROI.
4. Combines [Correction + Signal + Grating] strictly within the central ROI.
5. Leaves the outer regions as [Correction only] (no grating, no signal).

This ensures diffraction separation happens only for the active signal area.
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
    Returns the phase matrix (0-2pi) and dimensions.
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
            # (Standard Hamamatsu LUT: 0->0, 255->2pi approx)
            correction_phase = (bmp_data / 255.0) * (2 * np.pi)
            
            return correction_phase, width, height
            
    except Exception as e:
        print(f"ERROR loading correction file: {e}")
        return np.zeros((DEFAULT_SLM_H, DEFAULT_SLM_W), dtype=float), DEFAULT_SLM_W, DEFAULT_SLM_H

def create_signal_matrix(rows, cols):
    """Create the signal pixel matrix structure (initialized to 0)."""
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
    Generates a linear phase ramp (sawtooth) for a specific ROI size.
    """
    _, x_indices = np.indices((rows, cols))
    # Phase = 2pi * (x / Period)
    return (2 * np.pi * x_indices / period_pixels)

def process_slm_algorithm():
    print("\n--- SLM Algorithm: Inner-ROI Grating Mode ---")
    
    # --- Step 1: Load Wavefront Correction (Base Image) ---
    # This defines the full frame size (e.g., 1272x1024)
    full_frame_phase, slm_width, slm_height = load_wavefront_correction_native(CORRECTION_FILE)
    
    # --- Step 2: Generate the Signal Matrix (ROI Only) ---
    print(f"\nGenerating Signal Matrix ({SIGNAL_ROWS}x{SIGNAL_COLS})...")
    signal_matrix = create_signal_matrix(SIGNAL_ROWS, SIGNAL_COLS)
    
    # Calculate macropixel sizes
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
                current = signal_matrix[r_idx*mp_rows, c_idx*mp_cols]
                # Phase addition logic
                signal_matrix[r_idx*mp_rows : (r_idx+1)*mp_rows, 
                              c_idx*mp_cols : (c_idx+1)*mp_cols] = (current + add_val) % (2*np.pi)
    except FileNotFoundError:
        print("CERN-02 file not found, skipping.")

    # --- Step 3: Generate Grating (ROI Only) ---
    print(f"Generating Phase Grating for Inner Image ({SIGNAL_ROWS}x{SIGNAL_COLS}, Period={GRATING_PERIOD}px)...")
    # We generate the grating ONLY for the size of the signal
    roi_grating = generate_blazed_grating(SIGNAL_ROWS, SIGNAL_COLS, GRATING_PERIOD)

    # --- Step 4: Embed ROI into Full Frame ---
    print("\nEmbedding ROI (Signal + Grating) into Full Frame Center...")
    
    # Calculate centering offsets
    if SIGNAL_ROWS > slm_height or SIGNAL_COLS > slm_width:
        raise ValueError(f"Signal size ({SIGNAL_ROWS}x{SIGNAL_COLS}) larger than SLM ({slm_height}x{slm_width})")

    r_off = (slm_height - SIGNAL_ROWS) // 2
    c_off = (slm_width - SIGNAL_COLS) // 2
    print(f"ROI Offset: Top={r_off}, Left={c_off}")

    # Extract the corresponding central part of the correction
    # We must add to the EXISTING correction in that region
    correction_roi = full_frame_phase[r_off : r_off + SIGNAL_ROWS, 
                                      c_off : c_off + SIGNAL_COLS]
    
    # Combine components for the ROI:
    # ROI = (Correction_Slice + Signal + Grating) % 2pi
    combined_roi_phase = (correction_roi + signal_matrix + roi_grating) % (2 * np.pi)
    
    # Update the full frame
    # Outside the ROI, full_frame_phase remains just the Correction (as loaded in Step 1)
    full_frame_phase[r_off : r_off + SIGNAL_ROWS, 
                     c_off : c_off + SIGNAL_COLS] = combined_roi_phase

    # --- Step 5: Convert to 8-bit Bitmap ---
    # Map [0, 2pi] -> [0, 2*PI_WHITE_VALUE]
    TWO_PI_GRAY = 2 * PI_WHITE_VALUE
    if TWO_PI_GRAY > 255:
        print(f"Warning: 2pi gray level ({TWO_PI_GRAY}) clipped to 255.")
        TWO_PI_GRAY = 255
        
    print(f"Mapping Phase 0-2pi to Gray 0-{TWO_PI_GRAY}")
    
    # We apply modulo 2pi again to the whole frame just to be safe (though logic ensures it)
    final_phase = full_frame_phase % (2 * np.pi)
    
    bmp_float = (final_phase / (2 * np.pi)) * TWO_PI_GRAY
    bmp_uint8 = bmp_float.astype(np.uint8)
    
    # --- Step 6: Save Outputs ---
    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        
    output_filename = os.path.join(plots_folder, f'slm_final_roi_g{GRATING_PERIOD}.bmp')
    Image.fromarray(bmp_uint8, mode='L').save(output_filename)
    
    print("\n" + "="*50)
    print(f"SUCCESS. Output saved to: {output_filename}")
    print(f"Total Resolution: {slm_width} x {slm_height}")
    print(f"Inner ROI: {SIGNAL_ROWS} x {SIGNAL_COLS} (contains Signal + Grating)")
    print(f"Outer Area: Contains only Wavefront Correction")
    print("="*50)

    # Optional: Save debug images to verify ROI placement
    # Debug 1: Mask showing where the grating is applied
    debug_mask = np.zeros((slm_height, slm_width), dtype=np.uint8)
    debug_mask[r_off:r_off+SIGNAL_ROWS, c_off:c_off+SIGNAL_COLS] = 255
    Image.fromarray(debug_mask, mode='L').save(os.path.join(plots_folder, 'debug_roi_mask.png'))

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    process_slm_algorithm()
    