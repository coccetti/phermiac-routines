# -*- coding: utf-8 -*-
"""
Algorithm 10: Apply Dual-Grating Separation
Description:
    Loads an SLM pattern and applies two different phase gratings:
    - Even Macropixel Rows -> Grating A (e.g., Steer Left)
    - Odd Macropixel Rows  -> Grating B (e.g., Steer Right)
    
    This separates the signals spatially on the CCD plane (after cylindrical lens),
    creating two interrupted vertical lines for easier readout.

Physics:
    Phase is additive. Final_Phase = (Signal_Phase + Grating_Phase) % 2pi.
    We use modulo 217 (where 217 ~= 2pi) to maintain phase linearity.

Author: Fabrizio Coccetti
Date: 2026-01-23
"""

import os
import numpy as np
from PIL import Image

# --- CONFIGURATION ---

# 1. File Paths
BASE_DIR = "16_bits/propagation"  # Adjust to your working folder
IMG_SUBDIR = "img"
# INPUT_FILENAME = "SLM_16x16bits_CERN-01_event3_CERN-02_sec3_960x960_r16_c16_w108.bmp"
INPUT_FILENAME = "SLM_16x16bits_CERN-01_event3_CERN-02_sec3_960x960_r16_c16_w108_gradient.bmp"
INPUT_PATH = os.path.join(BASE_DIR, IMG_SUBDIR, INPUT_FILENAME)
# Add suffix "_GratingSeparated" before file extension
name, ext = os.path.splitext(INPUT_FILENAME)
OUTPUT_FILENAME = f"{name}_grating{ext}"

# 2. Geometry
MACROPIXEL_HEIGHT = 60  # Height of one row of macropixels in pixels
SLM_WIDTH = 480         # Should match input image width

# 3. Phase Calibration
# 217 corresponds to 2*Pi (Full Wave). 
# The grating must wrap around this value.
MAX_PHASE_VAL = 217 

# 4. Grating Parameters (Beam Steering)
# Period in pixels. Smaller period = Larger deflection angle.
# Positive period = Slope up (Steer one way)
# Negative period = Slope down (Steer opposite way)
PERIOD_EVEN = 20  # Grating for Even rows (e.g., +X shift)
PERIOD_ODD  = -20 # Grating for Odd rows  (e.g., -X shift)

def load_image(path):
    if not os.path.exists(path):
        # Try looking in current dir if base dir fails
        if os.path.exists(INPUT_FILENAME):
            path = INPUT_FILENAME
        else:
            raise FileNotFoundError(f"Input file not found: {path}")
    
    print(f"Loading: {path}")
    return Image.open(path).convert('L')

def generate_grating_row(width, period, max_val):
    """
    Generates a 1D sawtooth phase grating (ramp).
    Formula: Phase(x) = (x * max_val / period) % max_val
    """
    x = np.arange(width)
    if period == 0:
        return np.zeros(width, dtype=np.float32)
    
    # Calculate linear ramp
    # We use float for precision then modulus
    ramp = (x / period) * max_val
    
    # Wrap phase (Sawtooth)
    grating_line = np.mod(ramp, max_val)
    return grating_line

def main():
    print("--- SLM Dual-Grating Application ---")
    
    # 1. Load Input
    # Prefer the configured path (BASE_DIR/img/...) but allow running from the same folder
    input_path = INPUT_PATH if os.path.exists(INPUT_PATH) else INPUT_FILENAME
    try:
        img = load_image(input_path)
        data = np.array(img, dtype=np.float32)
    except Exception as e:
        print(f"Error: {e}")
        return

    height, width = data.shape
    print(f"Image Size: {width}x{height}")
    print(f"Macropixel Height: {MACROPIXEL_HEIGHT}")

    # 2. Generate 1D Gratings
    print(f"Generating Gratings (Period Even: {PERIOD_EVEN}, Odd: {PERIOD_ODD})...")
    grating_line_even = generate_grating_row(width, PERIOD_EVEN, MAX_PHASE_VAL)
    grating_line_odd  = generate_grating_row(width, PERIOD_ODD,  MAX_PHASE_VAL)

    # 3. Apply Gratings Row by Row
    # Create output array
    final_data = np.zeros_like(data)

    num_macro_rows = int(np.ceil(height / MACROPIXEL_HEIGHT))
    print(f"Processing {num_macro_rows} macropixel rows...")

    for i in range(num_macro_rows):
        # Define the slice for the current macropixel row
        y_start = i * MACROPIXEL_HEIGHT
        y_end = min((i + 1) * MACROPIXEL_HEIGHT, height)
        if y_start >= height:
            break
        
        # Select Input Signal for this band
        signal_band = data[y_start:y_end, :]
        
        # Select Grating based on index (Even vs Odd)
        if i % 2 == 0:
            # Even Row (0, 2, 4...)
            current_grating = grating_line_even
            type_str = "EVEN (+Shift)"
        else:
            # Odd Row (1, 3, 5...)
            current_grating = grating_line_odd
            type_str = "ODD  (-Shift)"
            
        # Broadcast 1D grating to 2D band shape
        # (Grating is constant along Y, varies along X)
        grating_band = np.tile(current_grating, (y_end - y_start, 1))
        
        # --- PHYSICS CORE ---
        # Total Phase = (Signal + Grating) % 2pi
        # We sum the Gray Levels and modulo by 217
        combined_band = np.mod(signal_band + grating_band, MAX_PHASE_VAL)
        
        # Store in final image
        final_data[y_start:y_end, :] = combined_band
        
        # Progress info (print every macropixel row)
        print(f" Macropixel row {i}/{num_macro_rows - 1}: {type_str} applied to y={y_start}:{y_end}")

    # 4. Save Output
    # Save into the img subfolder (same place as the input BMP)
    output_dir = os.path.join(BASE_DIR, IMG_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, OUTPUT_FILENAME)
    
    final_img = Image.fromarray(np.clip(final_data, 0, 255).astype(np.uint8))
    final_img.save(output_path)
    
    print(f"\nSUCCESS: Saved Grating-Separated Image to:\n{output_path}")

if __name__ == "__main__":
    main()
