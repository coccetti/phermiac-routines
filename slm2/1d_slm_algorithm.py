#!/usr/bin/env python3
"""
SLM Algorithm Implementation - Enhanced with Horizontal and Alternating Gratings
Author: Fabrizio Coccetti
Date: December 2025

DESCRIPTION:
This script generates phase masks for a Hamamatsu LCOS-SLM with Region-of-Interest (ROI) control.
1. Loads the Wavefront Correction file (CAL_...) as the Master Frame.
2. Generates the Signal Matrix (720x480) from CERN binary data (KEPT AS IS - CORRECT).
3. Saves the ROI (Signal Matrix) as a BMP file.
4. Adds a horizontal grating for the ROI to shift the Signal Matrix by a user-defined amount.
5. Generates an Alternating Grating for the ROI with 2 shifts (one for odd rows, one for even rows).
6. Saves the full image (Correction Image + (ROI with grating)) in BMP format.
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
DEFAULT_SLM_W = 1272        # Fallback width
DEFAULT_SLM_H = 1024        # Fallback height

# Grating Parameters
WAVELENGTH = 1064e-9        # Wavelength in meters (1064 nm)
DX = 12.5e-6                # Pixel pitch in meters (12.5 micrometers)

# Horizontal Grating Parameters (for shifting the Signal Matrix)
# The grating period determines the deflection angle
# Smaller period = larger deflection angle
HORIZONTAL_GRATING_PERIOD = 16.0  # Grating period in pixels (user can adjust this)
# Alternative: Use frequency-based approach
# HORIZONTAL_GRATING_FREQUENCY = -0.2  # cycles/mm (negative for left deflection)

# Alternating Grating Parameters (2 shifts for odd/even rows)
# These are spatial shifts in pixels applied to the grating pattern
ALTERNATING_SHIFT_ODD = 4.0      # Spatial shift in pixels for odd rows (1, 3, 5, ...)
ALTERNATING_SHIFT_EVEN = -4.0    # Spatial shift in pixels for even rows (2, 4, 6, ...)
ALTERNATING_GRATING_PERIOD = 16.0  # Base grating period in pixels

# Input Files
# TELESCOPE_01_FILE = "CERN-01_event1_nbit16.csv"
# TELESCOPE_02_FILE = "CERN-02_sec1_nbit16.csv"
TELESCOPE_01_FILE = "CERN-01_test0.csv"
TELESCOPE_02_FILE = "CERN-02_test1.csv"
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
    """Creates an empty signal matrix."""
    return np.zeros((rows, cols), dtype=float)

def assign_macropixel_value(matrix, row_idx, col_idx, value, macro_h, macro_w):
    """Assigns a value to a macropixel region in the matrix."""
    r_start = row_idx * macro_h
    r_end = r_start + macro_h
    c_start = col_idx * macro_w
    c_end = c_start + macro_w
    matrix[r_start:r_end, c_start:c_end] = value

def read_binary_data(filename):
    """Reads binary data from a CSV file."""
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            return next(reader)[0]
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None

def generate_horizontal_grating(rows, cols, period):
    """
    Generates a horizontal blazed grating that creates a phase gradient along x-direction.
    This grating deflects light horizontally when used with a cylindrical lens.
    
    The phase pattern follows: phase = 2π * x / period
    This creates a sawtooth pattern that repeats every 'period' pixels.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        period: Grating period in pixels (determines deflection angle)
                Smaller period = larger deflection angle
    
    Returns:
        Grating matrix with horizontal phase gradient (0-2π range)
    """
    grating_matrix = np.zeros((rows, cols), dtype=float)
    x_indices = np.arange(cols)
    
    # Create a blazed grating: phase = 2π * x / period
    # This creates a linear phase ramp that repeats every 'period' pixels
    phase_gradient = (2 * np.pi * x_indices / period) % (2 * np.pi)
    
    # Apply the same gradient to all rows
    for row in range(rows):
        grating_matrix[row, :] = phase_gradient
    
    # Calculate deflection angle for information
    # sin(theta) = m * lambda / (period * dx)
    # For first order (m=1):
    deflection_angle = np.arcsin(WAVELENGTH / (period * DX)) * 180 / np.pi
    
    print(f"Generated Horizontal Grating:")
    print(f" - Period: {period:.2f} pixels")
    print(f" - Period in physical units: {period * DX * 1e6:.2f} μm")
    print(f" - Theoretical deflection angle (1st order): {deflection_angle:.4f} degrees")
    
    return grating_matrix

def generate_alternating_grating(rows, cols, num_macro_rows, period, shift_odd, shift_even):
    """
    Generates an Alternating Grating for the ROI with 2 spatial shifts:
    - One spatial shift for odd rows (1, 3, 5, ...)
    - Another spatial shift for even rows (2, 4, 6, ...)
    
    This is a multigrating with 2 values that alternates between macropixel rows.
    Each row has a proper blazed grating pattern, but shifted spatially.
    
    Formula: phase = 2π * (x + shift) / period
    This creates a blazed grating that is spatially shifted by 'shift' pixels.
    
    Args:
        rows: Number of rows
        cols: Number of columns
        num_macro_rows: Number of macropixel rows
        period: Base grating period in pixels
        shift_odd: Spatial shift in pixels for odd macropixel rows (1, 3, 5, ...)
        shift_even: Spatial shift in pixels for even macropixel rows (2, 4, 6, ...)
    
    Returns:
        Alternating grating matrix with proper phase gradients
    """
    grating_matrix = np.zeros((rows, cols), dtype=float)
    x_indices = np.arange(cols)
    
    # Calculate height of each macropixel row
    mp_height = rows // num_macro_rows
    
    print(f"Generating Alternating Grating (Multigrating with 2 spatial shifts):")
    print(f" - Base period: {period:.2f} pixels")
    print(f" - Odd rows spatial shift: {shift_odd:.2f} pixels")
    print(f" - Even rows spatial shift: {shift_even:.2f} pixels")
    print(f" - Macropixel Row Height: {mp_height}px")
    print(f" - Total macropixel rows: {num_macro_rows}")

    for i in range(num_macro_rows):
        r_start = i * mp_height
        r_end = r_start + mp_height
        
        # Determine if this is an odd or even row (1-indexed: 1st row is odd, 2nd is even, etc.)
        # Row index i: 0=1st (odd), 1=2nd (even), 2=3rd (odd), ...
        if i % 2 == 0:  # Odd row (1-indexed: 1, 3, 5, ...)
            spatial_shift = shift_odd
        else:  # Even row (1-indexed: 2, 4, 6, ...)
            spatial_shift = shift_even
        
        # Create blazed grating with spatial shift: phase = 2π * (x + shift) / period
        # This shifts the grating pattern spatially while maintaining the phase gradient
        row_phase = (2 * np.pi * (x_indices + spatial_shift) / period) % (2 * np.pi)
        
        # Apply to all pixels in this macropixel row
        grating_matrix[r_start:r_end, :] = row_phase
        
    return grating_matrix

def save_phase_matrix_as_bmp(phase_matrix, filename, pi_white_value=108):
    """
    Converts a phase matrix (0-2pi) to an 8-bit BMP file.
    
    Args:
        phase_matrix: Phase matrix in radians (0-2pi)
        filename: Output filename
        pi_white_value: Gray value corresponding to pi phase shift
    """
    TWO_PI_GRAY = 2 * pi_white_value
    if TWO_PI_GRAY > 255:
        print(f"Warning: 2pi gray level ({TWO_PI_GRAY}) clipped to 255.")
        TWO_PI_GRAY = 255
    
    # Normalize phase to 0-2pi and convert to 8-bit
    normalized_phase = phase_matrix % (2 * np.pi)
    bmp_uint8 = ((normalized_phase / (2 * np.pi)) * TWO_PI_GRAY).astype(np.uint8)
    
    # Save as BMP
    Image.fromarray(bmp_uint8, mode='L').save(filename)
    print(f"Saved BMP: {filename}")

def process_slm_algorithm():
    print("\n--- SLM Algorithm: Enhanced with Horizontal and Alternating Gratings ---")
    
    # Create output directory
    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    
    # --- Step 1: Load Wavefront Correction (Base Image) ---
    full_frame_phase, slm_width, slm_height = load_wavefront_correction_native(CORRECTION_FILE)
    
    # --- Step 2: Generate the Signal Matrix (ROI Only) - KEPT AS IS (CORRECT) ---
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
    
    print("Signal Matrix generated successfully.")
    
    # --- Step 3: Save ROI (Signal Matrix) as BMP ---
    roi_filename = os.path.join(plots_folder, 'roi_signal_matrix.bmp')
    save_phase_matrix_as_bmp(signal_matrix, roi_filename, PI_WHITE_VALUE)
    print(f"ROI Signal Matrix saved to: {roi_filename}")
    
    # --- Step 4: Generate Horizontal Grating for ROI ---
    print(f"\n--- Generating Horizontal Grating ---")
    horizontal_grating = generate_horizontal_grating(SIGNAL_ROWS, SIGNAL_COLS, 
                                                      HORIZONTAL_GRATING_PERIOD)
    
    # Combine Signal Matrix with Horizontal Grating
    signal_with_horizontal = (signal_matrix + horizontal_grating) % (2 * np.pi)
    
    # --- Step 5: Embed ROI with Horizontal Grating into Full Frame ---
    print("\nEmbedding ROI with Horizontal Grating into Full Frame Center...")
    
    if SIGNAL_ROWS > slm_height or SIGNAL_COLS > slm_width:
        raise ValueError("Signal size larger than SLM size")

    r_off = (slm_height - SIGNAL_ROWS) // 2
    c_off = (slm_width - SIGNAL_COLS) // 2
    
    # Extract correction ROI
    correction_roi = full_frame_phase[r_off : r_off + SIGNAL_ROWS, 
                                      c_off : c_off + SIGNAL_COLS]
    
    # Combine: ROI = (Correction + Signal + Horizontal Grating) % 2pi
    combined_roi_phase = (correction_roi + signal_with_horizontal) % (2 * np.pi)
    
    # Create full frame with horizontal grating
    full_frame_with_horizontal = full_frame_phase.copy()
    full_frame_with_horizontal[r_off : r_off + SIGNAL_ROWS, 
                               c_off : c_off + SIGNAL_COLS] = combined_roi_phase
    
    # Save full image with horizontal grating
    horizontal_output = os.path.join(plots_folder, 'slm_full_with_horizontal_grating.bmp')
    save_phase_matrix_as_bmp(full_frame_with_horizontal, horizontal_output, PI_WHITE_VALUE)
    print(f"Full image with Horizontal Grating saved to: {horizontal_output}")
    
    # --- Step 6: Generate Alternating Grating for ROI ---
    print(f"\n--- Generating Alternating Grating (Multigrating with 2 values) ---")
    alternating_grating = generate_alternating_grating(SIGNAL_ROWS, SIGNAL_COLS, 
                                                       NUMBERS_TO_COMPARE,
                                                       ALTERNATING_GRATING_PERIOD,
                                                       ALTERNATING_SHIFT_ODD, 
                                                       ALTERNATING_SHIFT_EVEN)
    
    # Combine Signal Matrix with Alternating Grating
    signal_with_alternating = (signal_matrix + alternating_grating) % (2 * np.pi)
    
    # --- Step 7: Embed ROI with Alternating Grating into Full Frame ---
    print("\nEmbedding ROI with Alternating Grating into Full Frame Center...")
    
    # Combine: ROI = (Correction + Signal + Alternating Grating) % 2pi
    combined_roi_alternating = (correction_roi + signal_with_alternating) % (2 * np.pi)
    
    # Create full frame with alternating grating
    full_frame_with_alternating = full_frame_phase.copy()
    full_frame_with_alternating[r_off : r_off + SIGNAL_ROWS, 
                                c_off : c_off + SIGNAL_COLS] = combined_roi_alternating
    
    # Save full image with alternating grating
    alternating_output = os.path.join(plots_folder, 'slm_full_with_alternating_grating.bmp')
    save_phase_matrix_as_bmp(full_frame_with_alternating, alternating_output, PI_WHITE_VALUE)
    print(f"Full image with Alternating Grating saved to: {alternating_output}")
    
    # --- Step 8: Optional - Save alternating grating alone for visualization ---
    alternating_grating_filename = os.path.join(plots_folder, 'alternating_grating_only.bmp')
    save_phase_matrix_as_bmp(alternating_grating, alternating_grating_filename, PI_WHITE_VALUE)
    
    print("\n=== PROCESSING COMPLETE ===")
    print(f"Output files saved in '{plots_folder}' directory:")
    print(f"  1. roi_signal_matrix.bmp - ROI Signal Matrix only")
    print(f"  2. slm_full_with_horizontal_grating.bmp - Full image with horizontal grating")
    print(f"  3. slm_full_with_alternating_grating.bmp - Full image with alternating grating")
    print(f"  4. alternating_grating_only.bmp - Alternating grating visualization")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    process_slm_algorithm()
