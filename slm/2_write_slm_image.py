# -*- coding: utf-8 -*-
"""
This script loads a specified image file and displays it on an LCOS-SLM.
It is based on the LCOS-SLM_python_sample_01.py example.

pip install Pillow
Place the Image_Control.dll file in the slm directory
the script, it will load slm/slm_clean.png

"""

import os
from PIL import Image
import numpy as np
from ctypes import *

def showOn2ndDisplay(monitorNo, windowNo, x, xShift, y, yShift, array):
    """
    Displays the given array on the LCOS-SLM.
    This function assumes that 'Image_Control.dll' is available in the script's
    directory or in the system's PATH.

    Args:
        monitorNo (int): The monitor number for the SLM.
        windowNo (int): The window number to use for display.
        x (int): The width of the display area in pixels.
        xShift (int): The horizontal shift of the display area.
        y (int): The height of the display area in pixels.
        yShift (int): The vertical shift of the display area.
        array (c_uint8 array): The image data to be displayed.
    """
    try:
        # Load the DLL library for SLM control
        Lcoslib = windll.LoadLibrary("Image_Control.dll")

        # Configure the SLM display window settings
        Window_Settings = Lcoslib.Window_Settings
        Window_Settings.argtypes = [c_int, c_int, c_int, c_int]
        Window_Settings.restype = c_int
        Window_Settings(monitorNo, windowNo, xShift, yShift)

        # Send the image array to the SLM display
        Window_Array_to_Display = Lcoslib.Window_Array_to_Display
        Window_Array_to_Display.argtypes = [c_void_p, c_int, c_int, c_int, c_int]
        Window_Array_to_Display.restype = c_int
        Window_Array_to_Display(array, x, y, windowNo, x * y)

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

def write_image_to_slm(image_path):
    """
    Loads a PNG image, converts it to the required format, and displays it on the SLM.

    Args:
        image_path (str): The file path of the image to display.
    """
    # --- LCOS-SLM Configuration ---
    # Set pixel dimensions based on the output of slm_algorithm.py
    x_resolution = 1280
    y_resolution = 1024

    # SLM monitor and window settings
    monitor_number = 2  # Assumes SLM is configured as the second monitor
    window_number = 0
    x_shift = 0
    y_shift = 0

    # --- Image Loading and Processing ---
    # Verify that the image file exists
    if not os.path.exists(image_path):
        print(f"Error: The image file was not found at '{image_path}'")
        return

    print(f"Loading image from: {image_path}")
    try:
        # Open the image using the Pillow library
        with Image.open(image_path) as img:
            # Convert the image to 8-bit grayscale ('L' mode)
            im_gray = img.convert("L")
            image_width, image_height = img.size

            print(f"Image dimensions: {image_width}x{image_height}")

            # Check if image dimensions match the SLM resolution
            if image_width != x_resolution or image_height != y_resolution:
                print(f"Warning: Image size ({image_width}x{image_height}) does not match SLM resolution ({x_resolution}x{y_resolution}).")

            # --- Array Preparation for SLM ---
            array_size = x_resolution * y_resolution
            FARRAY = c_uint8 * array_size
            farray = FARRAY(0) # Initialize array with zeros

            # Convert the grayscale image to a NumPy array and flatten it
            pixels = np.array(im_gray)
            flat_pixels = pixels.flatten()

            # Convert pixel values to SLM phase values
            # Black (0) → phase shift 0 → SLM value 0
            # White (255) → phase shift π → SLM value 108
            # 2π corresponds to SLM value 217
            slm_values = np.zeros_like(flat_pixels, dtype=np.uint8)
            for i in range(len(flat_pixels)):
                pixel_value = flat_pixels[i]
                if pixel_value == 0:  # Black pixel
                    slm_values[i] = 0  # Phase shift 0
                elif pixel_value == 255:  # White pixel
                    slm_values[i] = 108  # Phase shift π
                else:  # Grayscale values - linear interpolation
                    # Map 0-255 to 0-108 (0 to π)
                    slm_values[i] = int((pixel_value / 255.0) * 108)

            # Copy the SLM values into the ctypes array
            for i in range(min(len(slm_values), array_size)):
                farray[i] = slm_values[i]

    except Exception as e:
        print(f"An error occurred while processing the image: {e}")
        return

    print("Image data prepared. Sending to SLM...")
    # Call the function to display the image on the SLM
    showOn2ndDisplay(monitor_number, window_number, x_resolution, x_shift, y_resolution, y_shift, farray)


if __name__ == "__main__":
    # Define the path to the image file to be displayed on the SLM
    # This assumes the script is run from the parent directory of 'slm/'
    image_file_to_display = "slm/slm_clean.png"
    
    # Execute the main function
    write_image_to_slm(image_file_to_display)
