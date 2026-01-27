# -*- coding: utf-8 -*-
"""
Hamamatsu LCOS-SLM Pattern Generator (Model LSH0905569)
Target Wavelength: 1064 nm | Linearity Limit (2pi): 217

UPDATED PATHS: Uses '08_bits/slm/mylib' for source files.

LOGIC RECAP:
1. Signal 'w108' implies Gray Level 108 = Pi phase.
2. System Limit is 217 = 2Pi phase.
3. Calibration (CAL) is standard 0-255. It must be rescaled to 0-217.
4. Final Phase = (Signal + Rescaled_Cal) % 217.

Author: Fabrizio Coccetti
Date: 2026-01-18
"""

import os
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
# Base directory relative to where the script is run
BASE_PATH = "08_bits/slm"
# Updated to 'mylib' as per user repository structure
LIB_PATH = os.path.join(BASE_PATH, "mylib") 
RES_PATH = os.path.join(BASE_PATH, "results")

# SLM Physical Resolution
SLM_WIDTH = 1272
SLM_HEIGHT = 1024

# SYSTEM CRITICAL VALUES
# The value that corresponds to a 2pi phase shift on your device at 1064nm
MAX_PHASE_VALUE = 217 

# Files
CAL_FILENAME = "CAL_LSH0905569_1064nm.bmp"
SIG_FILENAME = "SLM_test_checkerboard_nbit8_480x480_r8_c8_w108.bmp"
OUTPUT_FILENAME = "SLM_Pattern_Ready_217_Limit.bmp"

def load_slm_bitmap(filepath):
    """Loads a BMP file as a numpy array."""
    if not os.path.exists(filepath):
        # Print absolute path to help debugging
        abs_path = os.path.abspath(filepath)
        raise FileNotFoundError(f"Critical File Missing: {filepath}\n(Checked absolute: {abs_path})")
    with Image.open(filepath) as img:
        return np.array(img.convert('L'), dtype=np.float32)

def center_signal_on_canvas(signal_arr, canvas_shape):
    """Centers the signal on a zero-filled canvas."""
    canvas_h, canvas_w = canvas_shape
    sig_h, sig_w = signal_arr.shape
    
    y_start = max(0, (canvas_h - sig_h) // 2)
    x_start = max(0, (canvas_w - sig_w) // 2)
    
    # Create empty canvas
    canvas = np.zeros(canvas_shape, dtype=np.float32)
    
    # Place signal (handling cropping if necessary)
    out_y_end = min(canvas_h, y_start + sig_h)
    out_x_end = min(canvas_w, x_start + sig_w)
    
    # Slice of signal to use
    sig_slice_h = out_y_end - y_start
    sig_slice_w = out_x_end - x_start
    
    canvas[y_start:out_y_end, x_start:out_x_end] = \
        signal_arr[0:sig_slice_h, 0:sig_slice_w]
        
    return canvas

def main():
    print(f"--- Hamamatsu SLM Generation (Limit: {MAX_PHASE_VALUE}) ---")
    print(f"Working Directory: {os.getcwd()}")
    
    # Ensure output directory exists
    os.makedirs(RES_PATH, exist_ok=True)
    
    # 1. Load Calibration (Range 0-255)
    cal_path = os.path.join(LIB_PATH, CAL_FILENAME)
    print(f"Loading Calibration: {cal_path}")
    try:
        cal_data_raw = load_slm_bitmap(cal_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # Check Dimensions
    real_h, real_w = cal_data_raw.shape
    if (real_h, real_w) != (SLM_HEIGHT, SLM_WIDTH):
        print(f"WARNING: Calibration size {cal_data_raw.shape} != Expected SLM size ({SLM_HEIGHT}x{SLM_WIDTH}).")
        slm_h, slm_w = real_h, real_w
    else:
        slm_h, slm_w = SLM_HEIGHT, SLM_WIDTH

    # 2. RESCALE CALIBRATION
    # Compressing the 0-255 calibration map into the 0-217 range.
    print(f"Rescaling Calibration from [0-255] to [0-{MAX_PHASE_VALUE}]...")
    cal_scale_factor = MAX_PHASE_VALUE / 255.0
    cal_data_rescaled = cal_data_raw * cal_scale_factor

    # 3. Load Signal (Range 0-108 -> fits in 0-217)
    sig_path = os.path.join(LIB_PATH, SIG_FILENAME)
    print(f"Loading Signal: {sig_path}")
    try:
        sig_data = load_slm_bitmap(sig_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    # 4. Center Signal
    print("Centering Signal...")
    centered_signal = center_signal_on_canvas(sig_data, (slm_h, slm_w))
    
    # 5. Combine and Wrap (Modulo 217)
    print("Applying correction and wrapping phase...")
    total_phase = centered_signal + cal_data_rescaled
    
    # Modulo operation
    final_data = np.mod(total_phase, MAX_PHASE_VALUE)
    
    # Convert to 8-bit Integer for BMP
    final_img_array = final_data.astype(np.uint8)
    
    # 6. Save Result
    out_path = os.path.join(RES_PATH, OUTPUT_FILENAME)
    Image.fromarray(final_img_array).save(out_path)
    print(f"SUCCESS: Corrected Image saved to: {out_path}")
    
    # 7. Generate Dark Frame (Rescaled Calibration only)
    dark_data = np.mod(cal_data_rescaled, MAX_PHASE_VALUE).astype(np.uint8)
    dark_path = os.path.join(RES_PATH, "SLM_Dark_Frame_217.bmp")
    Image.fromarray(dark_data).save(dark_path)
    print(f"SUCCESS: Dark Frame saved to: {dark_path}")

if __name__ == "__main__":
    main()
