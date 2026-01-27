#!/usr/bin/env python3
"""
Test script for SLM algorithm implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def test_slm_algorithm():
    """
    Test the SLM algorithm with a simplified version
    """
    print("Testing SLM Algorithm Implementation")
    print("=" * 50)
    
    # Create a smaller test matrix for verification
    test_rows = 32  # 4 macropixels * 8 rows each
    test_cols = 64  # 4 macropixels * 4 columns each (16 bits / 4 = 4 bits per macropixel)
    macropixel_rows = 8
    macropixel_cols = 4
    
    # Initialize matrix
    pixel_matrix = np.zeros((test_rows, test_cols), dtype=float)
    
    print(f"Test matrix size: {test_rows} x {test_cols}")
    print(f"Macropixel size: {macropixel_rows} x {macropixel_cols}")
    
    # Test data from CERN-01 (4 bits for test)
    cern01_data = "1000"
    print(f"CERN-01 data: {cern01_data}")
    
    # Apply CERN-01 data to all rows
    for col_idx, digit in enumerate(cern01_data):
        if digit == '1':
            value = np.pi
        else:
            value = 0
        
        # Apply to all 4 macropixel rows (for test)
        for row_idx in range(4):
            start_row = row_idx * macropixel_rows
            end_row = start_row + macropixel_rows
            start_col = col_idx * macropixel_cols
            end_col = start_col + macropixel_cols
            pixel_matrix[start_row:end_row, start_col:end_col] = value
    
    # Test data from CERN-02 (4 bits for test)
    cern02_data = ["0010", "0100", "0010", "1000"]
    print(f"CERN-02 data (first 4): {cern02_data}")
    
    # Apply CERN-02 data to specific rows
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
            start_row = row_idx * macropixel_rows
            end_row = start_row + macropixel_rows
            start_col = col_idx * macropixel_cols
            end_col = start_col + macropixel_cols
            pixel_matrix[start_row:end_row, start_col:end_col] = new_value
    
    # Generate binary image
    binary_image = (pixel_matrix == 0).astype(int)
    
    # Display results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Show phase values
    im1 = ax1.imshow(pixel_matrix, cmap='viridis', aspect='equal')
    ax1.set_title('Phase Values')
    ax1.set_xlabel('Columns')
    ax1.set_ylabel('Rows')
    
    # Add macropixel grid and indices to phase plot
    for i in range(5):  # 4 macropixels + 1
        y_pos = i * macropixel_rows
        ax1.axhline(y=y_pos, color='red', linewidth=0.5, alpha=0.7)
    
    for i in range(5):  # 4 macropixels + 1
        x_pos = i * macropixel_cols
        ax1.axvline(x=x_pos, color='red', linewidth=0.5, alpha=0.7)
    
    # Add macropixel indices to phase plot
    for row_idx in range(4):
        for col_idx in range(4):
            center_y = (row_idx + 0.5) * macropixel_rows
            center_x = (col_idx + 0.5) * macropixel_cols
            ax1.text(center_x, center_y, f'({row_idx},{col_idx})', 
                    ha='center', va='center', fontsize=8, 
                    color='white', weight='bold', alpha=0.9)
    
    plt.colorbar(im1, ax=ax1, label='Phase (radians)')
    
    # Show binary image
    im2 = ax2.imshow(binary_image, cmap='gray', aspect='equal')
    ax2.set_title('Binary Image\n(White=0, Black=π)')
    ax2.set_xlabel('Columns')
    ax2.set_ylabel('Rows')
    
    # Add macropixel grid and indices to binary plot
    for i in range(5):  # 4 macropixels + 1
        y_pos = i * macropixel_rows
        ax2.axhline(y=y_pos, color='red', linewidth=0.5, alpha=0.7)
    
    for i in range(5):  # 4 macropixels + 1
        x_pos = i * macropixel_cols
        ax2.axvline(x=x_pos, color='red', linewidth=0.5, alpha=0.7)
    
    # Add macropixel indices to binary plot
    for row_idx in range(4):
        for col_idx in range(4):
            center_y = (row_idx + 0.5) * macropixel_rows
            center_x = (col_idx + 0.5) * macropixel_cols
            ax2.text(center_x, center_y, f'({row_idx},{col_idx})', 
                    ha='center', va='center', fontsize=8, 
                    color='yellow', weight='bold', alpha=0.9)
    
    plt.colorbar(im2, ax=ax2, label='Pixel Value')
    
    plt.tight_layout()
    plt.savefig('test_slm_result.png', dpi=150, bbox_inches='tight')
    print("Test image saved as: test_slm_result.png")
    
    # Create a clean image without axes, descriptions, or indices
    plt.figure(figsize=(6.4, 3.2), dpi=100)  # 640x320 pixels at 100 DPI
    plt.imshow(binary_image, cmap='gray', aspect='equal')
    plt.axis('off')  # Remove all axes and labels
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove margins
    
    # Save the clean test image
    plt.savefig('test_slm_clean.png', dpi=100, bbox_inches='tight', 
                pad_inches=0, facecolor='white', edgecolor='none')
    print("Clean test image saved as: test_slm_clean.png")
    
    # Close the clean plot to avoid showing it
    plt.close()
    
    # Print statistics
    print(f"\nTest Statistics:")
    print(f"Total pixels: {pixel_matrix.size}")
    print(f"White pixels (value=0): {np.sum(binary_image == 1)}")
    print(f"Black pixels (value=π): {np.sum(binary_image == 0)}")
    print(f"Unique phase values: {len(np.unique(pixel_matrix))}")
    print(f"Phase value range: {np.min(pixel_matrix):.3f} to {np.max(pixel_matrix):.3f}")
    
    plt.show()
    
    return pixel_matrix, binary_image

if __name__ == "__main__":
    test_slm_algorithm()
    print("\nTest completed successfully!")
