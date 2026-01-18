# -*- coding: utf-8 -*-
"""
Hamamatsu LCOS-SLM Pattern Generator (Model LSH0905569) - LINEAR REGION CORRECTION
Target Wavelength: 1064 nm
Phase Depth Setting: 0 to 2pi mapped to 0-217 Gray Levels.

SCIENTIFIC LOGIC:
1. System Linearity Limit: The user specifies 217 as the maximum linear phase response (2pi).
2. Calibration Mismatch Correction: The factory CAL file uses the full 0-255 range. 
   To combine them physically, we rescale the CAL file to the 0-217 range 
   so that the 'Phase per Gray Level' matches the Signal.
3. Modulo Arithmetic: The final phase addition is wrapped modulo 217 to ensure 
   we stay within the liquid crystal's linear response region (0-2pi).

Author: Gemini Fabrizio Coccetti
Date: 2026-01-18
"""

import os
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
BASE_PATH = "08_bits/slm"
LIB_PATH = os.path.join(BASE_PATH, "mylib")
RES_PATH = os.path.join(BASE_PATH, "results")

# SLM Physical Limits
SLM_WIDTH = 1272
SLM_HEIGHT = 1024

# SYSTEM CRITICAL VALUES
# The value that corresponds to a 2pi phase shift (or max linear range)
MAX_PHASE_VALUE = 217 
# Note: In modulo arithmetic, the range is [0, MAX_PHASE_VALUE-1].
# A value of 217 is physically equivalent to 0 (2pi == 0).

# Files
CAL_FILENAME = "CAL_LSH0905569_1064nm.bmp"
SIG_FILENAME = "SLM_test_checkerboard_nbit8_480x480_r8_c8_w108.bmp"
OUTPUT_FILENAME = "SLM_Pattern_Ready_217_Limit.bmp"

def load_slm_bitmap(filepath):
    """Loads a BMP file as a numpy array."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Critical File Missing: {filepath}")
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
    os.makedirs(RES_PATH, exist_ok=True)
    
    # 1. Load Calibration (Range 0-255)
    cal_path = os.path.join(LIB_PATH, CAL_FILENAME)
    print(f"Loading Calibration: {CAL_FILENAME}")
    try:
        cal_data_raw = load_slm_bitmap(cal_path)
    except FileNotFoundError as e:
        print(e)
        return

    # Check Dimensions
    real_h, real_w = cal_data_raw.shape
    if (real_h, real_w) != (SLM_HEIGHT, SLM_WIDTH):
        print(f"WARNING: Calibration size {cal_data_raw.shape} != SLM size.")
        # We adopt calibration size to be safe
        slm_h, slm_w = real_h, real_w
    else:
        slm_h, slm_w = SLM_HEIGHT, SLM_WIDTH

    # 2. RESCALE CALIBRATION (The Fix)
    # We compress the 0-255 calibration map into the 0-217 range.
    # This ensures that "Maximum Backplane Curvature" maps to "Maximum Correctable Value".
    print(f"Rescaling Calibration from [0-255] to [0-{MAX_PHASE_VALUE}]...")
    cal_scale_factor = MAX_PHASE_VALUE / 255.0
    cal_data_rescaled = cal_data_raw * cal_scale_factor

    # 3. Load Signal (Already Range 0-217)
    sig_path = os.path.join(LIB_PATH, SIG_FILENAME)
    print(f"Loading Signal: {SIG_FILENAME}")
    try:
        sig_data = load_slm_bitmap(sig_path)
    except FileNotFoundError:
        print("Signal not found, generating dummy.")
        sig_data = np.zeros((480, 480))

    # 4. Center Signal
    centered_signal = center_signal_on_canvas(sig_data, (slm_h, slm_w))
    
    # 5. Combine and Wrap (Modulo 217)
    # Total Phase = (Signal + Calibration) % 2pi
    # Here 2pi is represented by MAX_PHASE_VALUE (217)
    print("Applying correction and wrapping phase...")
    
    total_phase = centered_signal + cal_data_rescaled
    
    # Modulo operation with floating point, then cast to integer
    final_data = np.mod(total_phase, MAX_PHASE_VALUE)
    
    # Convert to 8-bit Integer
    final_img_array = final_data.astype(np.uint8)
    
    # Sanity Check
    print(f"Output Statistics: Min={final_img_array.min()}, Max={final_img_array.max()}")
    if final_img_array.max() > MAX_PHASE_VALUE:
        print("WARNING: Values exceed specified limit!")

    # 6. Save Result
    out_path = os.path.join(RES_PATH, OUTPUT_FILENAME)
    Image.fromarray(final_img_array).save(out_path)
    print(f"Saved Final Image: {out_path}")
    
    # 7. Generate Dark Frame (Zero Signal + Rescaled Calibration)
    # This is the 'Black' reference for your camera
    dark_data = np.mod(cal_data_rescaled, MAX_PHASE_VALUE).astype(np.uint8)
    dark_path = os.path.join(RES_PATH, "SLM_Dark_Frame_217.bmp")
    Image.fromarray(dark_data).save(dark_path)
    print(f"Saved Dark Frame: {dark_path}")

if __name__ == "__main__":
    main()
