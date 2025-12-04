# -*- coding: utf-8 -*-
"""
This script loads "phermiac_pattern.bmp" and displays it centered on the SLM
by calculating and using the xShift and yShift parameters.

The image is displayed at its original size. The script will always
re-calculate the centering based on the current image dimensions.

Based on LCOS-SLM_python_sample_01.py.
16 Oct 2025
"""

import os
from PIL import Image
import numpy as np
from ctypes import *

def showOn2ndDisplay(monitorNo, windowNo, img_width, xShift, img_height, yShift, array):
    """
    Displays the given array on the LCOS-SLM using Image_Control.dll.
    The window is positioned using xShift and yShift.

    Args:
        monitorNo (int): The monitor number for the SLM.
        windowNo (int): The window number to use for display.
        img_width (int): The width of the image to display.
        xShift (int): The horizontal offset for the display window.
        img_height (int): The height of the image to display.
        yShift (int): The vertical offset for the display window.
        array (c_uint8 array): The image data to be displayed.
    """
    try:
        # Load the DLL library for SLM control
        Lcoslib = windll.LoadLibrary("slm2/Image_Control.dll")

        # Configure the SLM display window settings, including the offset
        Window_Settings = Lcoslib.Window_Settings
        Window_Settings.argtypes = [c_int, c_int, c_int, c_int]
        Window_Settings.restype = c_int
        print(f"Setting window with shift: (x={xShift}, y={yShift})")
        Window_Settings(monitorNo, windowNo, xShift, yShift)

        # Send the image array to the positioned SLM display window
        Window_Array_to_Display = Lcoslib.Window_Array_to_Display
        Window_Array_to_Display.argtypes = [c_void_p, c_int, c_int, c_int, c_int]
        Window_Array_to_Display.restype = c_int
        Window_Array_to_Display(array, img_width, img_height, windowNo, img_width * img_height)

        # Wait for user input before closing the window
        input("Press ENTER to close the SLM window...")

        # Terminate the SLM display window
        Window_Term = Lcoslib.Window_Term
        Window_Term.argtypes = [c_int]
        Window_Term.restype = c_int
        Window_Term(windowNo)

        print("SLM window closed successfully.")
        return 0
    except FileNotFoundError:
        print("\n--- ERROR ---")
        print("'Image_Control.dll' not found.")
        print("Please ensure the DLL file is in the same directory as this script.")
        print("---------------")
        return -1
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return -1

def write_shifted_image_to_slm(image_path):
    """
    Loads a BMP image, calculates the centering shifts, and displays it.

    Args:
        image_path (str): The file path of the image to display.
    """
    # --- SLM Configuration ---
    slm_width = 1272
    slm_height = 1024

    # SLM monitor and window settings
    monitor_number = 2
    window_number = 0

    # --- Image Loading and Processing ---
    if not os.path.exists(image_path):
        print(f"Error: The image file was not found at '{image_path}'")
        return

    print(f"Loading image from: {image_path}")
    try:
        with Image.open(image_path) as img:
            # Get the dimensions of the input image
            image_width, image_height = img.size
            print(f"Input image dimensions: {image_width}x{image_height}")

            # --- Centering Calculation using Shifts ---
            x_shift = (slm_width - image_width) // 2
            y_shift = (slm_height - image_height) // 2
            
            print(f"Calculated shifts for centering: (xShift={x_shift}, yShift={y_shift})")

            # --- Array Preparation for SLM ---
            # The array now only needs to be the size of the image itself
            array_size = image_width * image_height
            FARRAY = c_uint8 * array_size
            farray = FARRAY(0) 

            # Convert image to grayscale and get pixel data
            pixels = np.array(img.convert("L"))
            flat_pixels = pixels.flatten()

            # Copy the pixel data directly into the ctypes array
            memmove(farray, flat_pixels.ctypes.data, array_size)

    except Exception as e:
        print(f"An error occurred while processing the image: {e}")
        return

    print("Image data prepared. Sending to SLM with calculated shifts...")
    # Call the display function with the image size and calculated shifts
    showOn2ndDisplay(monitor_number, window_number, image_width, x_shift, image_height, y_shift, farray)


if __name__ == "__main__":
    # Define the path to the BMP image file
    image_file_to_display = "./slm2/plots/slm_final_roi_g16.bmp"
    
    # Execute the main function
    write_shifted_image_to_slm(image_file_to_display)
