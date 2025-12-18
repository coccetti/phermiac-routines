#!/usr/bin/env python3
"""
SLM (Spatial Light Modulator) Algorithm Implementation - Chessboard Pattern

This script creates a chessboard pattern for SLM testing:
- Creates a matrix divided into evenly-sized macropixels
- Generates an alternating black and white chessboard pattern
- Assigns phase values (0 for black, π for white) to macropixels
- Outputs two 8-bit BMP files:
  1. Adjusted size with uniform macropixel sizes
  2. Original size (1024x1272) with unused pixels set to 0 (black)

CONFIGURATION:
- Modify INPUT_RESOLUTION_BITS (default: 16) to change number of macropixel columns
- Modify NUMBERS_TO_COMPARE (default: 24) to change number of macropixel rows
- Modify WHITE_VALUE (default: 108) to change the gray value for white pixels in BMP
- These parameters control the macropixel grid size and resulting image dimensions

September, 2025
"""

import numpy as np
import os
from PIL import Image

# Configuration parameters - modify these to change the SLM pattern
NUMBERS_TO_COMPARE = 24     # 24 Number of macropixel rows (height)
INPUT_RESOLUTION_BITS = 16  # 16 Number of macropixel columns (width)
PI_WHITE_VALUE = 108           # Value for white pixels: 108, 217

def create_slm_matrix():
    """
    Create the SLM pixel matrix and macropixel structure.
    
    Returns:
        tuple: (pixel_matrix, macropixel_rows, macropixel_cols)
    """
    # Use global configuration parameters
    input_resolution_bits = INPUT_RESOLUTION_BITS
    numbers_to_compare = NUMBERS_TO_COMPARE
    total_rows = 720 # 1024 rows
    total_cols = 480 # 1272 columns
    
    # Create pixel matrix initialized to 0
    pixel_matrix = np.zeros((total_rows, total_cols), dtype=float)
    
    # Calculate macropixel dimensions to ensure even division
    # 1024 // 24 = 42 with remainder 16, so we use 42 pixels per macropixel
    # 1272 // 16 = 79 with remainder 8, so we use 79 pixels per macropixel
    macropixel_rows = total_rows // numbers_to_compare  # 1024 // 24 = 42
    macropixel_cols = total_cols // input_resolution_bits  # 1272 // 16 = 79
    
    # Adjust total dimensions to ensure perfect fit
    adjusted_total_rows = macropixel_rows * numbers_to_compare  # 42 * 24 = 1008
    adjusted_total_cols = macropixel_cols * input_resolution_bits  # 79 * 16 = 1264
    
    print(f"Original matrix size: {total_rows} x {total_cols}")
    print(f"Adjusted matrix size: {adjusted_total_rows} x {adjusted_total_cols}")
    print(f"Macropixel size: {macropixel_rows} x {macropixel_cols}")
    print(f"Number of macropixels: {numbers_to_compare} rows x {input_resolution_bits} columns")
    print(f"Unused pixels: {total_rows - adjusted_total_rows} rows, {total_cols - adjusted_total_cols} columns")
    
    return pixel_matrix, macropixel_rows, macropixel_cols, adjusted_total_rows, adjusted_total_cols, total_rows, total_cols

def assign_macropixel_value(matrix, macropixel_row, macropixel_col, value, macropixel_rows, macropixel_cols):
    """
    Assign a value to all pixels in a specific macropixel.
    
    Args:
        matrix: The pixel matrix
        macropixel_row: Row index of the macropixel (0 to numbers_to_compare-1)
        macropixel_col: Column index of the macropixel (0 to input_resolution_bits-1)
        value: Value to assign (0 or pi)
        macropixel_rows: Height of each macropixel in pixels
        macropixel_cols: Width of each macropixel in pixels
    """
    # Calculate pixel coordinates for this macropixel
    start_row = macropixel_row * macropixel_rows
    end_row = start_row + macropixel_rows
    start_col = macropixel_col * macropixel_cols
    end_col = start_col + macropixel_cols
    
    # Assign value to all pixels in the macropixel
    matrix[start_row:end_row, start_col:end_col] = value



def process_slm_algorithm():
    """
    Main function to process the SLM algorithm with chessboard pattern.
    """
    print("Starting SLM Algorithm Implementation - Chessboard Pattern")
    print("=" * 60)
    print(f"Configuration: {INPUT_RESOLUTION_BITS} columns x {NUMBERS_TO_COMPARE} rows macropixels")
    print(f"White value: {PI_WHITE_VALUE} (0-255 range)")
    
    # Use global configuration parameters
    input_resolution_bits = INPUT_RESOLUTION_BITS
    numbers_to_compare = NUMBERS_TO_COMPARE
    
    # Step 1: Create the pixel matrix and macropixel structure
    pixel_matrix, macropixel_rows, macropixel_cols, adjusted_rows, adjusted_cols, total_rows, total_cols = create_slm_matrix()
    
    # Step 2: Create chessboard pattern
    print("\nCreating chessboard pattern...")
    for row_idx in range(numbers_to_compare):
        for col_idx in range(input_resolution_bits):
            # Chessboard pattern: (row + col) % 2 determines the color
            if (row_idx + col_idx) % 2 == 0:
                value = 0  # Black
            else:
                value = np.pi  # White
            
            # Apply to the specific macropixel
            assign_macropixel_value(pixel_matrix, row_idx, col_idx, value, 
                                  macropixel_rows, macropixel_cols)
    
    print(f"Chessboard pattern created with {numbers_to_compare} rows x {input_resolution_bits} columns")
    
    # Step 3: Generate final image
    print("\nGenerating final image...")
    
    # Convert phase values to binary image
    # 0 -> black (0), pi -> white (1)
    binary_image = (pixel_matrix == np.pi).astype(int)
    
    # Create plot subfolder if it doesn't exist
    plot_folder = "plots"
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
        print(f"Created folder: {plot_folder}")
    
    # Create 8-bit BMP outputs
    print("\nCreating 8-bit BMP outputs...")
    
    # First BMP: Adjusted dimensions - no unused pixels
    print("\n1. Creating BMP with adjusted dimensions (no unused pixels)...")
    used_matrix = pixel_matrix[:adjusted_rows, :adjusted_cols]
    binary_used = (used_matrix == np.pi).astype(int)
    bmp_image_adjusted = (binary_used * PI_WHITE_VALUE).astype(np.uint8)  # Use configurable white value
    
    pil_image_adjusted = Image.fromarray(bmp_image_adjusted, mode='L')
    bmp_output_filename_1 = os.path.join(plot_folder, f'slm_chessboard_{adjusted_rows}x{adjusted_cols}.bmp')
    pil_image_adjusted.save(bmp_output_filename_1, 'BMP')
    print(f"Adjusted BMP saved as: {bmp_output_filename_1}")
    print(f"Image size: {pil_image_adjusted.size[0]} x {pil_image_adjusted.size[1]} pixels")
    print(f"Color mapping: Black=0, White={PI_WHITE_VALUE}")
    
    # Second BMP: Original dimensions - unused pixels set to 0
    print("\n2. Creating BMP with original dimensions (unused pixels = 0)...")
    # Convert the full matrix to binary image
    binary_full = (pixel_matrix == np.pi).astype(int)
    # Unused pixels are already 0, but ensure they stay 0
    bmp_image_full = (binary_full * PI_WHITE_VALUE).astype(np.uint8)  # Use configurable white value
    
    pil_image_full = Image.fromarray(bmp_image_full, mode='L')
    bmp_output_filename_2 = os.path.join(plot_folder, f'slm_chessboard_{total_rows}x{total_cols}.bmp')
    pil_image_full.save(bmp_output_filename_2, 'BMP')
    print(f"Full BMP saved as: {bmp_output_filename_2}")
    print(f"Image size: {pil_image_full.size[0]} x {pil_image_full.size[1]} pixels")
    print(f"Color mapping: Black=0, White={PI_WHITE_VALUE}")
    print(f"Unused pixels (set to 0): {total_rows - adjusted_rows} rows, {total_cols - adjusted_cols} columns")
    
    # Show some statistics
    print(f"\nStatistics:")
    print(f"Original matrix size: {total_rows} x {total_cols}")
    print(f"Used matrix size: {adjusted_rows} x {adjusted_cols}")
    print(f"Total pixels in full image: {total_rows * total_cols}")
    print(f"Total used pixels: {adjusted_rows * adjusted_cols}")
    print(f"Unused pixels (black): {(total_rows * total_cols) - (adjusted_rows * adjusted_cols)}")
    print(f"Black pixels in used area (value=0): {np.sum(binary_used == 0)}")
    print(f"White pixels in used area (value=π): {np.sum(binary_used == 1)}")
    print(f"Unique phase values: {len(np.unique(used_matrix))}")
    print(f"Phase value range: {np.min(used_matrix):.3f} to {np.max(used_matrix):.3f}")
    print(f"Macropixel dimensions: {macropixel_rows} x {macropixel_cols} pixels each")
    
    return pixel_matrix, binary_image, used_matrix, binary_used

if __name__ == "__main__":
    # Change to the slm directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the algorithm
    phase_matrix, result_image, used_matrix, used_binary = process_slm_algorithm()
    
    print("\nSLM Algorithm completed successfully!")