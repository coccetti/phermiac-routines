# -*- coding: utf-8 -*-
"""
Hamamatsu LCOS-SLM Display Controller
Loads the generated Phase Pattern and displays it on the SLM device via Image_Control.dll.

Requirements:
- 'mylib/Image_Control.dll' must be present.
- Hamamatsu Drivers installed.
- SLM connected as Extended Monitor (usually Monitor 2).

Author: Fabrizio Coccetti
Date: 2026-01-18
"""

import os
import sys
import ctypes
import time
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(BASE_PATH, "mylib")
RESULTS_DIR = os.path.join(BASE_PATH, "results")

# DLL Path
DLL_PATH = os.path.join(LIB_DIR, "Image_Control.dll")

# Image to Display
IMAGE_TO_SHOW = "SLM_Pattern_Ready_217_Limit.bmp"
# IMAGE_TO_SHOW = "SLM_Dark_Frame_217.bmp" # Uncomment to show Dark Frame

# SLM Monitor ID (Usually 2 for secondary display)
SLM_MONITOR_ID = 2 
SLM_WIDTH = 1272
SLM_HEIGHT = 1024

def load_hamamatsu_dll():
    """Loads the Hamamatsu Image_Control DLL."""
    if not os.path.exists(DLL_PATH):
        print(f"CRITICAL ERROR: DLL not found at {DLL_PATH}")
        sys.exit(1)
    
    try:
        # Load the DLL using ctypes
        # We use LoadLibrary to handle the specific path
        slm_lib = ctypes.cdll.LoadLibrary(DLL_PATH)
        print(f"Successfully loaded: {DLL_PATH}")
        return slm_lib
    except Exception as e:
        print(f"Error loading DLL: {e}")
        print("Ensure you are running 64-bit Python if the DLL is 64-bit (or 32-bit/32-bit).")
        sys.exit(1)

def main():
    print("--- Hamamatsu SLM Display Control ---")
    
    # 1. Load DLL
    slm_lib = load_hamamatsu_dll()
    
    # 2. Initialize SLM SDK
    # Defines return types for safety
    slm_lib.Instance.restype = ctypes.c_long
    slm_lib.Window_Open.restype = ctypes.c_long
    slm_lib.Window_Show.restype = ctypes.c_long
    slm_lib.Window_Close.restype = ctypes.c_long
    slm_lib.Delete.restype = ctypes.c_long
    slm_lib.Window_Array_to_Display.argtypes = [ctypes.c_long, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_long, ctypes.c_long]
    slm_lib.Window_Array_to_Display.restype = ctypes.c_long

    # Create Instance
    ret = slm_lib.Instance()
    if ret != 0:
        print(f"Error initializing Instance. Code: {ret}")
        return

    print("SLM Instance Initialized.")

    # 3. Open Window on SLM Monitor
    print(f"Opening Window on Monitor ID: {SLM_MONITOR_ID}...")
    # Note: Window_Open takes the Monitor ID. 
    # If it fails, check Windows Display Settings to confirm SLM is Monitor 2.
    ret = slm_lib.Window_Open(SLM_MONITOR_ID)
    if ret != 0:
        print(f"Error Opening Window on Monitor {SLM_MONITOR_ID}. Code: {ret}")
        print("Tip: Check if SLM is connected and extended as a desktop monitor.")
        slm_lib.Delete()
        return

    # Show the Window (makes it visible/fullscreen)
    slm_lib.Window_Show(SLM_MONITOR_ID)
    
    # 4. Load the BMP Image
    img_path = os.path.join(RESULTS_DIR, IMAGE_TO_SHOW)
    print(f"Loading Image: {img_path}")
    
    if not os.path.exists(img_path):
        print("Image file not found!")
        slm_lib.Window_Close(SLM_MONITOR_ID)
        slm_lib.Delete()
        return

    try:
        img = Image.open(img_path).convert('L') # Ensure 8-bit Grayscale
        img_width, img_height = img.size
        
        if (img_width, img_height) != (SLM_WIDTH, SLM_HEIGHT):
            print(f"Warning: Image size ({img_width}x{img_height}) does not match SLM ({SLM_WIDTH}x{SLM_HEIGHT}).")
        
        # Convert Image to C-compatible Byte Array
        img_data = np.array(img, dtype=np.uint8)
        
        # Create pointer to data
        data_ptr = img_data.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        
        # 5. Send to SLM
        print("Sending data to SLM...")
        # Window_Array_to_Display(WindowID, DataPointer, Width, Height)
        # Note: Depending on the DLL version, WindowID might be the MonitorID or an internal handle.
        # Usually for this library, it matches the ID passed to Window_Open.
        ret = slm_lib.Window_Array_to_Display(SLM_MONITOR_ID, data_ptr, img_width, img_height)
        
        if ret == 0:
            print(">>> IMAGE SUCCESSFULLY DISPLAYED ON SLM <<<")
        else:
            print(f"Error sending image data. Code: {ret}")

        # 6. Keep Alive
        print("\nPress Ctrl+C to stop and close the SLM window.")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nClosing...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        # 7. Cleanup
        print("Releasing SLM resources...")
        slm_lib.Window_Close(SLM_MONITOR_ID)
        slm_lib.Delete()
        print("Done.")

if __name__ == "__main__":
    main()

