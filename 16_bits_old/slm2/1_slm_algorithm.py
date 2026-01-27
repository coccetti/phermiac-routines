#!/usr/bin/env python3
"""
SLM (Spatial Light Modulator) Algorithm Implementation

This script implements the algorithm described in algorithm.txt:
- Creates a 1024x1280 pixel matrix divided into macropixels
- Reads binary data from CSV files and assigns phase values
- Generates 8-bit BMP files for SLM hardware
- Also creates PNG reference images with grid overlays

CONFIGURATION:
- Modify NUMBERS_TO_COMPARE (default: 24) to change number of macropixel rows
- Modify INPUT_RESOLUTION_BITS (default: 16) to change number of macropixel columns
- Modify TOTAL_ROWS (default: 1024) to change total pixel rows
- Modify TOTAL_COLS (default: 1280) to change total pixel columns
- Modify PI_WHITE_VALUE (default: 108) to change the gray value for white pixels in BMP
- Modify CERN01_FILE to specify the CSV file for CERN-01 data
- Modify CERN02_FILE to specify the CSV file for CERN-02 data

September, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from PIL import Image

# Configuration parameters - modify these to change the SLM pattern
NUMBERS_TO_COMPARE = 24     # Number of macropixel rows (height)
INPUT_RESOLUTION_BITS = 16  # Number of macropixel columns (width)
TOTAL_ROWS = 720           # 1024 Total number of pixel rows
TOTAL_COLS = 480           # 1272 Total number of pixel columns
PI_WHITE_VALUE = 108        # Gray value for white pixels in BMP (0-255)
TELESCOPE_01_FILE = "CERN-01_event1_nbit16.csv"  # CSV file for CERN-01 data
TELESCOPE_02_FILE = "CERN-02_sec1_nbit16.csv"    # CSV file for CERN-02 data

def create_slm_matrix():
    """
    Create the SLM pixel matrix and macropixel structure.
    
    Returns:
        tuple: (pixel_matrix, macropixel_rows, macropixel_cols)
    """
    # Use global configuration parameters
    input_resolution_bits = INPUT_RESOLUTION_BITS
    numbers_to_compare = NUMBERS_TO_COMPARE
    total_rows = TOTAL_ROWS
    total_cols = TOTAL_COLS
    
    # Create pixel matrix initialized to 0
    pixel_matrix = np.zeros((total_rows, total_cols), dtype=float)
    
    # Calculate macropixel dimensions
    macropixel_rows = total_rows // numbers_to_compare  # 1024 // 8 = 128
    macropixel_cols = total_cols // input_resolution_bits  # 1280 // 16 = 80
    
    print(f"Matrix size: {total_rows} x {total_cols}")
    print(f"Macropixel size: {macropixel_rows} x {macropixel_cols}")
    print(f"Number of macropixels: {numbers_to_compare} rows x {input_resolution_bits} columns")
    
    return pixel_matrix, macropixel_rows, macropixel_cols

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

def read_binary_data(filename):
    """
    Read binary data from a CSV file.
    
    Args:
        filename: Path to the CSV file
        
    Returns:
        str: Binary string from the file
    """
    try:
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            data = next(reader)[0]  # Read first row, first column
            return data
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None

def save_image_with_grid(matrix, macropixel_rows, macropixel_cols, numbers_to_compare, 
                        input_resolution_bits, filename, title, show=True):
    """
    Save and optionally display an image with macropixel grid and indices.
    
    Args:
        matrix: The pixel matrix
        macropixel_rows: Height of each macropixel in pixels
        macropixel_cols: Width of each macropixel in pixels
        numbers_to_compare: Number of macropixel rows
        input_resolution_bits: Number of macropixel columns
        filename: Output filename
        title: Title for the plot
        show: Whether to display the image on screen
    """
    # Convert phase values to binary image
    # 0 -> black (0), pi -> white (1)
    binary_image = (matrix == np.pi).astype(int)
    
    # Display the image with macropixel grid and indices
    plt.figure(figsize=(15, 10))
    plt.imshow(binary_image, cmap='gray', aspect='equal')
    plt.title(title)
    plt.xlabel('Pixel Columns (1280)')
    plt.ylabel('Pixel Rows (1024)')
    
    # Add macropixel grid lines
    for i in range(numbers_to_compare + 1):
        y_pos = i * macropixel_rows
        plt.axhline(y=y_pos, color='red', linewidth=0.5, alpha=0.7)
    
    for i in range(input_resolution_bits + 1):
        x_pos = i * macropixel_cols
        plt.axvline(x=x_pos, color='red', linewidth=0.5, alpha=0.7)
    
    # Add macropixel indices
    for row_idx in range(numbers_to_compare):
        for col_idx in range(input_resolution_bits):
            # Calculate center of macropixel
            center_y = (row_idx + 0.5) * macropixel_rows
            center_x = (col_idx + 0.5) * macropixel_cols
            
            # Add text with row,column indices
            plt.text(center_x, center_y, f'({row_idx},{col_idx})', 
                    ha='center', va='center', fontsize=6, 
                    color='red', weight='bold', alpha=0.8)
    
    # Save the image with grid and indices
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Image saved as: {filename}")
    
    if show:
        plt.show()
    else:
        plt.close()

def process_slm_algorithm():
    """
    Main function to process the SLM algorithm.
    """
    print("Starting SLM Algorithm Implementation")
    print("=" * 50)
    print(f"Matrix size: {TOTAL_ROWS} x {TOTAL_COLS} pixels")
    print(f"Macropixel grid: {INPUT_RESOLUTION_BITS} columns x {NUMBERS_TO_COMPARE} rows")
    print(f"White value: {PI_WHITE_VALUE} (0-255 range)")
    print(f"CSV files: {TELESCOPE_01_FILE}, {TELESCOPE_02_FILE}")
    
    # Use global configuration parameters
    input_resolution_bits = INPUT_RESOLUTION_BITS
    numbers_to_compare = NUMBERS_TO_COMPARE
    
    # Step 1: Create the pixel matrix and macropixel structure
    pixel_matrix, macropixel_rows, macropixel_cols = create_slm_matrix()
    
    # Step 2: Initialize all macropixels to 0 (already done in matrix creation)
    print("\nAll macropixels initialized to 0")
    
    # Step 3: Read CERN-01 CSV file
    cern01_file = TELESCOPE_01_FILE
    print(f"\nReading {cern01_file}...")
    cern01_data = read_binary_data(cern01_file)
    
    if cern01_data is None:
        return
    
    print(f"CERN-01 data: {cern01_data}")
    print(f"Length: {len(cern01_data)} bits")
    
    # Step 4: Process CERN-01 data - apply to all rows
    print("\nProcessing CERN-01 data (applying to all macropixel rows)...")
    for col_idx, digit in enumerate(cern01_data):
        if digit == '1':
            value = np.pi
        else:
            value = 0
        
        # Apply to all macropixel rows
        for row_idx in range(numbers_to_compare):
            assign_macropixel_value(pixel_matrix, row_idx, col_idx, value, 
                                  macropixel_rows, macropixel_cols)
    
    # Create plots folder if it doesn't exist
    plots_folder = "plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
        print(f"Created folder: {plots_folder}")
    
    # Step 4.5: Save and show image after CERN-01 processing
    print("\nSaving and showing image after CERN-01 processing...")
    png_filename_step4 = os.path.join(plots_folder, f'slm_step4_cern01_{pixel_matrix.shape[0]}x{pixel_matrix.shape[1]}_r{numbers_to_compare}_c{input_resolution_bits}.png')
    save_image_with_grid(pixel_matrix, macropixel_rows, macropixel_cols, 
                        numbers_to_compare, input_resolution_bits,
                        png_filename_step4, 
                        'SLM Algorithm - After CERN-01 Processing\n(Black=0, White=π)',
                        show=True)
    
    # Create 8-bit BMP after CERN-01 processing
    print("\nCreating 8-bit BMP after CERN-01 processing...")
    binary_image_step4 = (pixel_matrix == np.pi).astype(int)
    bmp_image_step4 = (binary_image_step4 * PI_WHITE_VALUE).astype(np.uint8)  # Use configurable white value
    pil_image_step4 = Image.fromarray(bmp_image_step4, mode='L')
    bmp_filename_step4 = os.path.join(plots_folder, f'slm_step4_cern01_{pixel_matrix.shape[0]}x{pixel_matrix.shape[1]}_r{numbers_to_compare}_c{input_resolution_bits}_w{PI_WHITE_VALUE}.bmp')
    pil_image_step4.save(bmp_filename_step4, 'BMP')
    print(f"8-bit BMP saved as: {bmp_filename_step4}")
    print(f"Image size: {pil_image_step4.size[0]} x {pil_image_step4.size[1]} pixels")
    print(f"Color mapping: Black=0, White={PI_WHITE_VALUE}")
    
    # Step 5: Read CERN-02 CSV file (first numbers_to_compare values)
    cern02_file = TELESCOPE_02_FILE
    print(f"\nReading first {numbers_to_compare} values from {cern02_file}...")
    
    try:
        with open(cern02_file, 'r') as file:
            reader = csv.reader(file)
            cern02_data = []
            for i, row in enumerate(reader):
                if i >= numbers_to_compare:  # Only read first numbers_to_compare rows
                    break
                cern02_data.append(row[0])
    except FileNotFoundError:
        print(f"Error: File {cern02_file} not found")
        return
    
    print(f"CERN-02 data (first {numbers_to_compare}): {cern02_data}")
    
    # Step 6: Process CERN-02 data - apply to specific rows
    print("\nProcessing CERN-02 data (applying to specific macropixel rows)...")
    for row_idx, binary_string in enumerate(cern02_data):
        for col_idx, digit in enumerate(binary_string):
            if digit == '1':
                value = np.pi
            else:
                value = 0
            
            # Add to existing value (modulo 2*pi)
            current_value = pixel_matrix[row_idx * macropixel_rows, col_idx * macropixel_cols]
            new_value = (current_value + value) % (2 * np.pi)
            
            # Apply to the specific macropixel
            assign_macropixel_value(pixel_matrix, row_idx, col_idx, new_value, 
                                  macropixel_rows, macropixel_cols)
    
    # Step 7: Generate final image
    print("\nGenerating final image...")
    
    # Convert phase values to binary image
    # 0 -> black (0), pi -> white (1)
    binary_image = (pixel_matrix == np.pi).astype(int)
    
    # Save and show the final image with grid and indices
    png_filename_final = os.path.join(plots_folder, f'slm_result_{pixel_matrix.shape[0]}x{pixel_matrix.shape[1]}_r{numbers_to_compare}_c{input_resolution_bits}.png')
    save_image_with_grid(pixel_matrix, macropixel_rows, macropixel_cols, 
                        numbers_to_compare, input_resolution_bits,
                        png_filename_final, 
                        'SLM Algorithm Result\n(Black=0, White=π)',
                        show=True)
    
    # Create 8-bit BMP files for final result
    print("\nCreating 8-bit BMP files for final result...")
    
    # Full size BMP
    bmp_image_full = (binary_image * PI_WHITE_VALUE).astype(np.uint8)  # Use configurable white value
    pil_image_full = Image.fromarray(bmp_image_full, mode='L')
    bmp_filename_final = os.path.join(plots_folder, f'slm_result_{pixel_matrix.shape[0]}x{pixel_matrix.shape[1]}_r{numbers_to_compare}_c{input_resolution_bits}_w{PI_WHITE_VALUE}.bmp')
    pil_image_full.save(bmp_filename_final, 'BMP')
    print(f"Full size BMP saved as: {bmp_filename_final}")
    print(f"Image size: {pil_image_full.size[0]} x {pil_image_full.size[1]} pixels")
    print(f"Color mapping: Black=0, White={PI_WHITE_VALUE}")
    
    # Create a clean image without axes, descriptions, or indices for PNG reference
    plt.figure(figsize=(12.8, 10.24), dpi=100)  # 1280x1024 pixels at 100 DPI
    plt.imshow(binary_image, cmap='gray', aspect='equal')
    plt.axis('off')  # Remove all axes and labels
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margins
    
    # Save the clean PNG reference image
    clean_output_filename = os.path.join(plots_folder, f'slm_clean_{pixel_matrix.shape[0]}x{pixel_matrix.shape[1]}_r{numbers_to_compare}_c{input_resolution_bits}.png')
    plt.savefig(clean_output_filename, dpi=100, bbox_inches='tight', 
                pad_inches=0, facecolor='white', edgecolor='none')
    print(f"Clean reference PNG saved as: {clean_output_filename}")
    
    # Close the clean plot to avoid showing it
    plt.close()
    
    # Show some statistics
    print(f"\nStatistics:")
    print(f"Total pixels: {pixel_matrix.size}")
    print(f"Black pixels (value=0): {np.sum(binary_image == 0)}")
    print(f"White pixels (value=π): {np.sum(binary_image == 1)}")
    print(f"Unique phase values: {len(np.unique(pixel_matrix))}")
    print(f"Phase value range: {np.min(pixel_matrix):.3f} to {np.max(pixel_matrix):.3f}")
    
    return pixel_matrix, binary_image

if __name__ == "__main__":
    # Change to the slm directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the algorithm
    phase_matrix, result_image = process_slm_algorithm()
    
    print("\nSLM Algorithm completed successfully!")