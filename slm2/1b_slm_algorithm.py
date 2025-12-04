#!/usr/bin/env python3
"""
SLM Algorithm Implementation - Alternating Row Grating Shift
Author: Fabrizio Coccetti
Date: September, 2025 (Updated Dec 2025)

DESCRIPTION:
This script generates a phase mask for a Hamamatsu LCOS-SLM with Region-of-Interest (ROI) control.
1. Loads the Wavefront Correction file (CAL_...) as the Master Frame.
2. Generates the Signal Matrix (720x480) from CERN binary data.
3. Generates a SPECIAL Blazed Phase Grating for the ROI:
   - Base Period: 16 pixels (Global shift).
   - Alternating Shift: 
     * Odd Macropixel Rows (1, 3, ...): Grating shifted by +4 pixels.
     * Even Macropixel Rows (2, 4, ...): Grating shifted by -4 pixels.
4. Combines [Correction + Signal + Grating] strictly within the central ROI.
5. Leaves the outer regions as [Correction only].
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
GRATING_PERIOD = 16         # Base period of the blazed grating in pixels
DEFAULT_SLM_W = 1272        # Fallback width
DEFAULT_SLM_H = 1024        # Fallback height

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
            bmp_data = np.array(img.convert('L'), dtype=float)
            # Normalize 0-255 to 0-2pi 
            correction_phase = (bmp_data / 255.0) * (2 * np.pi)
            return correction_phase, width, height
    except Exception as e:
        print(f"ERROR loading correction file: {e}")
        return np.zeros((DEFAULT_SLM_H, DEFAULT_SLM_W), dtype=float), DEFAULT_SLM_W, DEFAULT_SLM_H

def create_signal_matrix(rows, cols):
    return np.zeros((rows, cols), dtype=float)

def assign_macropixel_value(matrix, row_idx, col_idx, value, macro_h, macro_w):
    r_start = row_idx * macro_h
    r_end = r_start + macro_h
    c_start = col_idx * macro_w
    c_end = c_start + macro_w
    matrix[r_start:r_end, c_start:c_end] = value

def read_binary_data(filename):
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            return next(reader)[0]
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None

def generate_alternating_grating(rows, cols, period, num_macro_rows):
    """
    Generates a blazed grating with alternating shifts for each macropixel row.
    - Rows 0, 2, 4... (1st, 3rd...): Shift +4px
    - Rows 1, 3, 5... (2nd, 4th...): Shift -4px
    """
    grating_matrix = np.zeros((rows, cols), dtype=float)
    x_indices = np.arange(cols)
    
    # Calculate height of each macropixel row
    mp_height = rows // num_macro_rows
    
    print(f"Generating Alternating Grating:")
    print(f" - Period: {period}px")
    print(f" - Macropixel Row Height: {mp_height}px")
    print(f" - Odd Rows (1,3..): Shift +4px")
    print(f" - Even Rows (2,4..): Shift -4px")

    for i in range(num_macro_rows):
        r_start = i * mp_height
        r_end = r_start + mp_height
        
        # Apply shifts based on row index (0-based)
        # User request: "prima riga" (idx 0) -> +4, "seconda riga" (idx 1) -> -4
        if i % 2 == 0:
            shift = 4.0
        else:
            shift = -4.0
            
        # Formula: Phase = 2pi * (x + shift) / Period
        # This shifts the grating phase spatially
        row_phase = (2 * np.pi * (x_indices + shift) / period)
        
        # Assign to the slice
        grating_matrix[r_start:r_end, :] = row_phase
        
    return grating_matrix

def process_slm_algorithm():
    print("\n--- SLM Algorithm: Alternating ROI Grating Mode ---")
    
    # --- Step 1: Load Wavefront Correction (Base Image) ---
    full_frame_phase, slm_width, slm_height = load_wavefront_correction_native(CORRECTION_FILE)
    
    # --- Step 2: Generate the Signal Matrix (ROI Only) ---
    print(f"\nGenerating Signal Matrix ({SIGNAL_ROWS}x{SIGNAL_COLS})...")
    signal_matrix = create_signal_matrix(SIGNAL_ROWS, SIGNAL_COLS)
    
    mp_rows = SIGNAL_ROWS // NUMBERS_TO_COMPARE
    mp_cols = SIGNAL_COLS // INPUT_RESOLUTION_BITS
    
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
                signal_matrix[r_idx*mp_rows : (r_idx+1)*mp_rows, 
                              c_idx*mp_cols : (c_idx+1)*mp_cols] = (current + add_val) % (2*np.pi)
    except FileNotFoundError:
        pass

    # --- Step 3: Generate Alternating Grating (ROI Only) ---
    roi_grating = generate_alternating_grating(SIGNAL_ROWS, SIGNAL_COLS, 
                                             GRATING_PERIOD, NUMBERS_TO_COMPARE)

    # --- Step 4: Embed ROI into Full Frame ---
    print("\nEmbedding ROI into Full Frame Center...")
    
    if SIGNAL_ROWS > slm_height or SIGNAL_COLS > slm_width:
        raise ValueError("Signal size larger than SLM size")

    r_off = (slm_height - SIGNAL_ROWS) // 2
    c_off = (slm_width - SIGNAL_COLS) // 2
    
    # Extract correction ROI
    correction_roi = full_frame_phase[r_off : r_off + SIGNAL_ROWS, 
                                      c_off : c_off + SIGNAL_COLS]
    
    # Combine: ROI = (Correction + Signal + Grating) % 2pi
    combined_roi_phase = (correction_roi + signal_matrix + roi_grating) % (2 * np.pi)
    
    # Update Full Frame
    full_frame_phase[r_off : r_off + SIGNAL_ROWS, 
                     c_off : c_off + SIGNAL_COLS] = combined_roi_phase

    # --- Step 5: Convert to 8-bit Bitmap ---
    TWO_PI_GRAY = 2 * PI_WHITE_VALUE
    if TWO_PI_GRAY > 255:
        print(f"Warning: 2pi gray level ({TWO_PI_GRAY}) clipped to 255.")
        TWO_PI_GRAY = 255
        
    print(f"Mapping Phase 0-2pi to Gray 0-{TWO_PI_GRAY}")
    
    final_phase = full_frame_phase % (2 * np.pi)
    bmp_uint8 = ((final_phase / (2 * np.pi)) * TWO_PI_GRAY).astype(np.uint8)
    
    # --- Step 6: Save Outputs ---
    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        
    output_filename = os.path.join(plots_folder, f'slm_final_shifted_rows.bmp')
    Image.fromarray(bmp_uint8, mode='L').save(output_filename)
    
    print(f"SUCCESS. Output saved to: {output_filename}")

    # Debug: Save visual check of the grating transitions
    # Show a vertical slice to see the phase jumps between rows
    plt.figure(figsize=(10, 6))
    plt.plot(final_phase[r_off:r_off+2*mp_rows, c_off+10], label='Phase Profile (Column 10)')
    plt.title('Vertical Phase Profile (Crossing Macropixel Rows)')
    plt.xlabel('Pixel Row')
    plt.ylabel('Phase (rad)')
    plt.grid(True)
    plt.savefig(os.path.join(plots_folder, 'debug_phase_profile.png'))
    plt.close()

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    process_slm_algorithm()
    