# Read only the decimal part in the files CERN-01_1sec.csv and CERN-02_1sec.csv
# plot the histogram of the differences of time for each event of the first file with all the events of the second file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def extract_decimal_part(time_value):
    """Extract only the decimal part of a time value"""
    return time_value - int(time_value)

def main():
    # Read the CSV files
    data_dir = Path("08_bits/data")
    file1_path = data_dir / "CERN-01_1event.csv"
    file2_path = data_dir / "CERN-02_1sec.csv"
    
    # Read the data
    print("Reading CERN-01_1sec.csv...")
    data1 = pd.read_csv(file1_path, header=None, names=['time'])
    print(f"Loaded {len(data1)} events from CERN-01_1sec.csv")
    
    print("Reading CERN-02_1sec.csv...")
    data2 = pd.read_csv(file2_path, header=None, names=['time'])
    print(f"Loaded {len(data2)} events from CERN-02_1sec.csv")
    
    # Extract decimal parts
    print("Extracting decimal parts...")
    decimal1 = data1['time'].apply(extract_decimal_part)
    decimal2 = data2['time'].apply(extract_decimal_part)
    
    print(f"Decimal parts from file 1: min={decimal1.min():.6f}, max={decimal1.max():.6f}")
    print(f"Decimal parts from file 2: min={decimal2.min():.6f}, max={decimal2.max():.6f}")
    
    # Calculate all time differences
    print("Calculating time differences...")
    time_differences = []
    
    for dec1 in decimal1:
        for dec2 in decimal2:
            # Calculate difference between decimal parts
            diff = dec1 - dec2
            time_differences.append(diff)
    
    time_differences = np.array(time_differences)
    print(f"Calculated {len(time_differences)} time differences")
    print(f"Time differences: min={time_differences.min():.6f}, max={time_differences.max():.6f}")
    
    # save the time differences to a csv file
    output_dir = Path("08_bits/results")
    output_dir.mkdir(exist_ok=True)
    np.savetxt(output_dir / "time_differences.csv", time_differences, delimiter=',')
    print(f"Time differences saved to {output_dir / 'time_differences.csv'}")

    # Create histogram
    print("Creating histogram...")
    plt.figure(figsize=(12, 8))
    
    # Plot histogram
    plt.hist(time_differences, bins=10000, alpha=0.7, color='blue', edgecolor='black', range=(-1, 1))
    plt.xlabel('Time Difference (decimal part)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Time Differences Between CERN-01 and CERN-02 Events\n(Decimal Parts Only)')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_diff = np.mean(time_differences)
    std_diff = np.std(time_differences)
    plt.text(0.02, 0.98, f'Mean: {mean_diff:.6f}\nStd: {std_diff:.6f}\nTotal differences: {len(time_differences)}', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save the plot
    output_dir = Path("08_bits/results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "time_differences_histogram.png", dpi=300, bbox_inches='tight')
    print(f"Histogram saved to {output_dir / 'time_differences_histogram.png'}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Number of events in CERN-01: {len(data1)}")
    print(f"Number of events in CERN-02: {len(data2)}")
    print(f"Total time differences calculated: {len(time_differences)}")
    print(f"Mean time difference: {mean_diff:.6f}")
    print(f"Standard deviation: {std_diff:.6f}")
    print(f"Min time difference: {time_differences.min():.6f}")
    print(f"Max time difference: {time_differences.max():.6f}")

if __name__ == "__main__":
    main()

