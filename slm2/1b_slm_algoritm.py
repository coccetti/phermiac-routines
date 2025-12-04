#!/usr/bin/env python3
"""
SLM (Spatial Light Modulator) Algorithm Implementation - DEFINITIVE VERSION
Includes:
- Macropixel signal generation (CERN-01/02)
- Blazed Grating (Phase Grating) for 0-order separation
- Wavefront Correction (Flatness correction) loading

This script implements the algorithm described in algorithm.txt:
- Creates a matrix divided into macropixels
- Reads binary data from CSV files and assigns phase values
- Superimposes a linear phase grating and wavefront correction
- Generates 8-bit BMP files for Hamamatsu SLM hardware

CONFIGURATION:
- Modify PI_WHITE_VALUE (default: 108) to change the gray value for PI.
  NOTE: For grating, the max value will be 2 * PI_WHITE_VALUE (up to 255).
- Modify GRATING_PERIOD (default: 16) to change the diffraction angle.

September, 2025 (Updated Dec 2025)
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from PIL import Image

# --- Configuration Parameters ---
NUMBERS_TO_COMPARE = 24     # Number of macropixel rows (height)
INPUT_RESOLUTION_BITS = 16  # Number of macropixel columns (width)
TOTAL_ROWS = 720            # Total number of pixel rows (Active area or full)
TOTAL_COLS = 480            # Total number of pixel columns
PI_WHITE_VALUE = 108        # Gray value for phase shift of π (0-255)
                            # 2π will be calculated as 2 * PI_WHITE_VALUE
GRATING_PERIOD = 16         # Period of the blazed grating in pixels
TELESCOPE_01_FILE = "CERN-01_event1_nbit16.csv"  # CSV file for CERN-01 data
TELESCOPE_02_FILE = "CERN-02_sec1_nbit16.csv"    # CSV file for CERN-02 data
CORRECTION_FILE = "CAL_LSH0905569_1064nm.bmp"    # Hamamatsu wavefront correction file

def create_slm_matrix():
    """
    Create the SLM pixel matrix and macropixel structure.
    Returns: (pixel_matrix, macropixel_rows, macropixel_cols)
    """
    input_resolution_bits = INPUT_RESOLUTION_BITS
    numbers_to_compare = NUMBERS_TO_COMPARE
    total_rows = TOTAL_ROWS
    total_cols = TOTAL_COLS
    
    # Create pixel matrix initialized to 0
    pixel_matrix = np.zeros((total_rows, total_cols), dtype=float)
    
    # Calculate macropixel dimensions
    macropixel_rows = total_rows // numbers_to_compare
    macropixel_cols = total_cols // input_resolution_bits
    
    print(f"Matrix size: {total_rows} x {total_cols}")
    print(f"Macropixel size: {macropixel_rows} x {macropixel_cols}")
    print(f"Number of macropixels: {numbers_to_compare} rows x {input_resolution_bits} columns")
    
    return pixel_matrix, macropixel_rows, macropixel_cols

def assign_macropixel_value(matrix, macropixel_row, macropixel_col, value, macropixel_rows, macropixel_cols):
    """Assign a value to all pixels in a specific macropixel."""
    start_row = macropixel_row * macropixel_rows
    end_row = start_row + macropixel_rows
    start_col = macropixel_col * macropixel_cols
    end_col = start_col + macropixel_cols
    
    matrix[start_row:end_row, start_col:end_col] = value

def read_binary_data(filename):
    """Read binary data from a CSV file."""
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            data = next(reader)[0]
            return data
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None

def generate_blazed_grating(rows, cols, period_pixels=16, direction='horizontal'):
    """
    Generates a linear phase ramp (sawtooth) 0 to 2pi.
    """
    y_indices, x_indices = np.indices((rows, cols))
    
    if direction == 'horizontal':
        # Phase = 2pi * (x / Period)
        grating_phase = (2 * np.pi * x_indices / period_pixels)
    else:
        # Phase = 2pi * (y / Period)
        grating_phase = (2 * np.pi * y_indices / period_pixels)
        
    return grating_phase

def load_wavefront_correction(filename, rows, cols):
    """
    Loads the Hamamatsu correction BMP and converts it to phase (0-2pi).
    Assumes the input BMP covers 0-2pi over 0-255 (standard Hamamatsu).
    """
    if not os.path.exists(filename):
        print(f"Warning: Correction file {filename} not found. Proceeding without correction.")
        return np.zeros((rows, cols), dtype=float)
        
    print(f"Loading wavefront correction from {filename}...")
    try:
        with Image.open(filename) as img:
            # Resize if necessary to match the current ROI
            if img.size != (cols, rows):
                print(f"Resizing correction map from {img.size} to ({cols}, {rows})")
                img = img.resize((cols, rows), Image.Resampling.BILINEAR)
            
            # Convert to numpy array and normalize
            # Standard Hamamatsu BMPs: 0-255 maps to 0-2pi linearly
            bmp_data = np.array(img.convert('L'), dtype=float)
            correction_phase = (bmp_data / 255.0) * (2 * np.pi)
            return correction_phase
    except Exception as e:
        print(f"Error loading correction file: {e}. Using zero correction.")
        return np.zeros((rows, cols), dtype=float)

def save_image_with_grid(matrix, macropixel_rows, macropixel_cols, numbers_to_compare, 
                        input_resolution_bits, filename, title, show=True):
    """Save and optionally display a visualization image."""
    # Visualization only shows the binary signal structure (ignoring grating for clarity)
    binary_image = (matrix > 1.0).astype(int) # Threshold at ~1.0 rad
    
    plt.figure(figsize=(12, 8))
    plt.imshow(binary_image, cmap='gray', aspect='equal')
    plt.title(title)
    
    # Add grid lines
    for i in range(numbers_to_compare + 1):
        plt.axhline(y=i * macropixel_rows, color='red', linewidth=0.5, alpha=0.5)
    for i in range(input_resolution_bits + 1):
        plt.axvline(x=i * macropixel_cols, color='red', linewidth=0.5, alpha=0.5)
    
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def process_slm_algorithm():
    print("Starting SLM Algorithm Implementation (Definitive)")
    print("=" * 50)
    print(f"Matrix size: {TOTAL_ROWS} x {TOTAL_COLS} pixels")
    
    # Use global configuration parameters
    input_resolution_bits = INPUT_RESOLUTION_BITS
    numbers_to_compare = NUMBERS_TO_COMPARE
    
    # --- Step 1: Create Signal Matrix (Binary 0 or pi) ---
    pixel_matrix, macropixel_rows, macropixel_cols = create_slm_matrix()
    
    # Read CERN-01
    cern01_data = read_binary_data(TELESCOPE_01_FILE)
    if cern01_data:
        print(f"Processing CERN-01 data ({len(cern01_data)} bits)...")
        for col_idx, digit in enumerate(cern01_data):
            value = np.pi if digit == '1' else 0
            for row_idx in range(numbers_to_compare):
                assign_macropixel_value(pixel_matrix, row_idx, col_idx, value, 
                                      macropixel_rows, macropixel_cols)

    # Read CERN-02
    try:
        with open(TELESCOPE_02_FILE, 'r') as file:
            reader = csv.reader(file)
            cern02_data = [row[0] for i, row in enumerate(reader) if i < numbers_to_compare]
        
        print(f"Processing CERN-02 data ({len(cern02_data)} rows)...")
        for row_idx, binary_string in enumerate(cern02_data):
            for col_idx, digit in enumerate(binary_string):
                add_val = np.pi if digit == '1' else 0
                # Logical XOR equivalent for phase: (a + b) % 2pi
                current_val = pixel_matrix[row_idx * macropixel_rows, col_idx * macropixel_cols]
                new_val = (current_val + add_val) % (2 * np.pi)
                assign_macropixel_value(pixel_matrix, row_idx, col_idx, new_val, 
                                      macropixel_rows, macropixel_cols)
    except FileNotFoundError:
        print(f"Warning: {TELESCOPE_02_FILE} not found. Skipping.")

    # Create plots folder
    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    # Save visualization of the SIGNAL only (before grating)
    vis_filename = os.path.join(plots_folder, 'slm_signal_pattern.png')
    save_image_with_grid(pixel_matrix, macropixel_rows, macropixel_cols, 
                        numbers_to_compare, input_resolution_bits,
                        vis_filename, 'Raw Signal Pattern (No Grating)', show=True)

    # --- Step 2: Add Grating and Correction (The Hamamatsu Way) ---
    print("\nApplying Phase Grating and Wavefront Correction...")
    
    # 2a. Generate Linear Grating
    grating_phase = generate_blazed_grating(TOTAL_ROWS, TOTAL_COLS, 
                                          period_pixels=GRATING_PERIOD, 
                                          direction='horizontal')
    
    # 2b. Load Wavefront Correction
    correction_phase = load_wavefront_correction(CORRECTION_FILE, TOTAL_ROWS, TOTAL_COLS)
    
    # 2c. Combine: Signal + Grating + Correction (Modulo 2pi)
    # The result represents the phase wrapped between 0 and 2pi
    total_phase_matrix = (pixel_matrix + grating_phase + correction_phase) % (2 * np.pi)
    
    # --- Step 3: Convert to 8-bit Gray Levels for Hardware ---
    print("\nConverting to 8-bit BMP for SLM Hardware...")
    
    # Calculate the gray level corresponding to 2pi
    # PI_WHITE_VALUE is the gray level for pi. So 2pi is double that.
    TWO_PI_GRAY = 2 * PI_WHITE_VALUE
    
    if TWO_PI_GRAY > 255:
        print(f"WARNING: 2*PI gray value ({TWO_PI_GRAY}) > 255. Clipping to 255.")
        TWO_PI_GRAY = 255
        
    print(f"Mapping: 0 rad -> 0 gray, 2π rad -> {TWO_PI_GRAY} gray")
    
    # Map [0, 2pi) -> [0, TWO_PI_GRAY]
    bmp_data_float = (total_phase_matrix / (2 * np.pi)) * TWO_PI_GRAY
    bmp_data_uint8 = bmp_data_float.astype(np.uint8)
    
    # --- Step 4: Save Final BMP ---
    final_bmp_name = os.path.join(plots_folder, f'slm_final_g{GRATING_PERIOD}_w{PI_WHITE_VALUE}.bmp')
    Image.fromarray(bmp_data_uint8, mode='L').save(final_bmp_name)
    
    print(f"SUCCESS: Final SLM file saved as: {final_bmp_name}")
    print(f"Includes: Signal + Grating (T={GRATING_PERIOD}px) + Correction ({CORRECTION_FILE})")

    # Optional: Save a visual check of the phase (zoomed in to see grating)
    # Just show top-left 200x200 pixels
    plt.figure(figsize=(10, 10))
    plt.imshow(total_phase_matrix[:200, :200], cmap='gray')
    plt.title(f"Detail: Top-Left 200x200 pixels\n(Visible Grating Period: {GRATING_PERIOD}px)")
    plt.colorbar(label='Phase (rad)')
    plt.savefig(os.path.join(plots_folder, 'slm_phase_detail_zoom.png'))
    plt.close()

    return total_phase_matrix

if __name__ == "__main__":
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    process_slm_algorithm()
    