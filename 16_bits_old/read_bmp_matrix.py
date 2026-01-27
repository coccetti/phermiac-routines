#!/usr/bin/env python3
"""
Script to read a BMP image file and print its pixel values as a matrix.
"""

import numpy as np
from PIL import Image
import sys
import os

def read_bmp_as_matrix(file_path):
    """
    Read a BMP file and return its pixel values as a matrix.
    
    Args:
        file_path (str): Path to the BMP file
        
    Returns:
        numpy.ndarray: Matrix containing pixel values
    """
    try:
        # Open the image using PIL
        img = Image.open(file_path)
        
        # Convert to grayscale if it's a color image
        if img.mode != 'L':
            img = img.convert('L')
        
        # Convert to numpy array
        matrix = np.array(img)
        
        return matrix
        
    except Exception as e:
        print(f"Error reading image: {e}")
        return None

def print_matrix(matrix, max_rows=20, max_cols=20):
    """
    Print the matrix with limited dimensions for readability.
    
    Args:
        matrix (numpy.ndarray): The matrix to print
        max_rows (int): Maximum number of rows to display
        max_cols (int): Maximum number of columns to display
    """
    if matrix is None:
        print("No matrix to display")
        return
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Data type: {matrix.dtype}")
    print(f"Min value: {np.min(matrix)}")
    print(f"Max value: {np.max(matrix)}")
    print()
    
    # Limit the display size
    rows_to_show = min(max_rows, matrix.shape[0])
    cols_to_show = min(max_cols, matrix.shape[1])
    
    print(f"Showing first {rows_to_show} rows and {cols_to_show} columns:")
    print("-" * (cols_to_show * 6))
    
    for i in range(rows_to_show):
        row_str = ""
        for j in range(cols_to_show):
            row_str += f"{matrix[i, j]:4d} "
        print(row_str)
    
    if matrix.shape[0] > max_rows or matrix.shape[1] > max_cols:
        print("...")
        print(f"(Matrix truncated - full size: {matrix.shape[0]} x {matrix.shape[1]})")

def main():
    # Default file path
    default_file = "slm2/CAL_LSH0905569_1064nm.bmp"
    
    # Check if file path is provided as command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = default_file
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found")
        print(f"Usage: python {sys.argv[0]} [file_path]")
        return
    
    print(f"Reading BMP file: {file_path}")
    print("=" * 50)
    
    # Read the BMP file
    matrix = read_bmp_as_matrix(file_path)
    
    if matrix is not None:
        # Print matrix information and values
        print_matrix(matrix)
        
        # Optionally save the matrix to a text file
        output_file = "bmp_matrix_values.txt"
        print(f"\nSaving full matrix to: {output_file}")
        np.savetxt(output_file, matrix, fmt='%d', delimiter='\t')
        print("Matrix saved successfully!")
    else:
        print("Failed to read the BMP file")

if __name__ == "__main__":
    main()
