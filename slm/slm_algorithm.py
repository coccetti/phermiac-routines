#!/usr/bin/env python3
"""
SLM (Spatial Light Modulator) Algorithm Implementation

This script implements the algorithm described in algorithm.txt:
- Creates a 1024x1280 pixel matrix divided into macropixels
- Reads binary data from CSV files and assigns phase values
- Generates a final image based on phase values
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def create_slm_matrix():
    """
    Create the SLM pixel matrix and macropixel structure.
    
    Returns:
        tuple: (pixel_matrix, macropixel_rows, macropixel_cols)
    """
    # Define parameters
    input_resolution_bits = 16
    numbers_to_compare = 24
    total_rows = 1024
    total_cols = 1280
    
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

def process_slm_algorithm():
    """
    Main function to process the SLM algorithm.
    """
    print("Starting SLM Algorithm Implementation")
    print("=" * 50)
    
    # Define parameters
    input_resolution_bits = 16
    numbers_to_compare = 24
    
    # Step 1: Create the pixel matrix and macropixel structure
    pixel_matrix, macropixel_rows, macropixel_cols = create_slm_matrix()
    
    # Step 2: Initialize all macropixels to 0 (already done in matrix creation)
    print("\nAll macropixels initialized to 0")
    
    # Step 3: Read CERN-01_event1_nbit16.csv
    cern01_file = "CERN-01_event1_nbit16.csv"
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
    
    # Step 5: Read CERN-02_sec1_nbit16.csv (first numbers_to_compare values)
    cern02_file = "CERN-02_sec1_nbit16.csv"
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
    # 0 -> white (1), pi -> black (0)
    binary_image = (pixel_matrix == 0).astype(int)
    
    # Display the image with macropixel grid and indices
    plt.figure(figsize=(15, 10))
    plt.imshow(binary_image, cmap='gray', aspect='equal')
    plt.title('SLM Algorithm Result\n(White=0, Black=π)')
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
    
    # plt.colorbar(label='Pixel Value')
    
    # Save the image with grid and indices
    output_filename = 'slm_result.png'
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"Image saved as: {output_filename}")
    
    # Create a clean image without axes, descriptions, or indices
    plt.figure(figsize=(12.8, 10.24), dpi=100)  # 1280x1024 pixels at 100 DPI
    plt.imshow(binary_image, cmap='gray', aspect='equal')
    plt.axis('off')  # Remove all axes and labels
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margins
    
    # Save the clean image
    clean_output_filename = 'slm_clean.png'
    plt.savefig(clean_output_filename, dpi=100, bbox_inches='tight', 
                pad_inches=0, facecolor='white', edgecolor='none')
    print(f"Clean image saved as: {clean_output_filename}")
    
    # Close the clean plot to avoid showing it
    plt.close()
    
    # Show some statistics
    print(f"\nStatistics:")
    print(f"Total pixels: {pixel_matrix.size}")
    print(f"White pixels (value=0): {np.sum(binary_image == 1)}")
    print(f"Black pixels (value=π): {np.sum(binary_image == 0)}")
    print(f"Unique phase values: {len(np.unique(pixel_matrix))}")
    print(f"Phase value range: {np.min(pixel_matrix):.3f} to {np.max(pixel_matrix):.3f}")
    
    plt.show()
    
    return pixel_matrix, binary_image

if __name__ == "__main__":
    # Change to the slm directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the algorithm
    phase_matrix, result_image = process_slm_algorithm()
    
    print("\nSLM Algorithm completed successfully!")