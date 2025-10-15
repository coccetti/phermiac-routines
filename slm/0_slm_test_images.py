#!/usr/bin/env python3
"""
SLM (Spatial Light Modulator) Algorithm Implementation - Chessboard Pattern

This script creates a chessboard pattern for SLM testing:
- Creates a 1024x1280 pixel matrix divided into macropixels
- Generates an alternating black and white chessboard pattern
- Assigns phase values (0 for black, π for white) to macropixels
- Generates final images with and without grid overlays

September, 2025
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
    total_cols = 1272
    
    # Create pixel matrix initialized to 0
    pixel_matrix = np.zeros((total_rows, total_cols), dtype=float)
    
    # Calculate macropixel dimensions
    macropixel_rows = total_rows // numbers_to_compare  # 1024 // 24 = 42
    macropixel_cols = total_cols // input_resolution_bits  # 1272 // 16 = 79
    
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
    plt.xlabel('Pixel Columns (1272)')
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
    Main function to process the SLM algorithm with chessboard pattern.
    """
    print("Starting SLM Algorithm Implementation - Chessboard Pattern")
    print("=" * 60)
    
    # Define parameters
    input_resolution_bits = 16
    numbers_to_compare = 24
    
    # Step 1: Create the pixel matrix and macropixel structure
    pixel_matrix, macropixel_rows, macropixel_cols = create_slm_matrix()
    
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
    
    # Save and show the final image with grid and indices
    save_image_with_grid(pixel_matrix, macropixel_rows, macropixel_cols, 
                        numbers_to_compare, input_resolution_bits,
                        'slm_chessboard_result.png', 
                        'SLM Algorithm - Chessboard Pattern\n(Black=0, White=π)',
                        show=True)
    
    # Create a clean image without axes, descriptions, or indices
    plt.figure(figsize=(12.72, 10.24), dpi=100)  # 1272x1024 pixels at 100 DPI
    plt.imshow(binary_image, cmap='gray', aspect='equal')
    plt.axis('off')  # Remove all axes and labels
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margins
    
    # Save the clean image
    clean_output_filename = 'slm_chessboard_clean.png'
    plt.savefig(clean_output_filename, dpi=100, bbox_inches='tight', 
                pad_inches=0, facecolor='white', edgecolor='none')
    print(f"Clean chessboard image saved as: {clean_output_filename}")
    
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