#
# Coincidence analysis for CERN data
#
# This script reads the data from the csv file and finds the coincidences between the two files
# It then plots the time series and the coincidences
# It then plots the coincidence time differences
#
# The script is used to find the coincidence time differences between the two files

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def process_time_data(file_path):
    """Process time data from a CERN CSV file"""
    # Read only the Seconds and Nanoseconds columns
    data = pd.read_csv(file_path, usecols=['Seconds', 'Nanoseconds'])
    
    # Subtract 368143200 from all Seconds values
    data['Seconds'] = data['Seconds'] - 368143200
    
    # Pad Nanoseconds with leading zeros to ensure length of 9, then join with Seconds
    data['Time'] = data['Seconds'].astype(str) + '.' + data['Nanoseconds'].astype(str).str.zfill(9)
    
    # Convert to float for numerical operations
    data['Time'] = data['Time'].astype(float)
    
    return data['Time'].values

def find_coincidences(times1, times2, coincidence_window=1e-6):
    """Find coincidences between two time arrays within a given window"""
    coincidences = []
    
    for t1 in times1:
        # Find times in times2 that are within the coincidence window
        mask = np.abs(times2 - t1) <= coincidence_window
        if np.any(mask):
            coincident_times = times2[mask]
            for t2 in coincident_times:
                coincidences.append((t1, t2, t2 - t1))
    
    return coincidences

def plot_coincidences(times1, times2, coincidences, filename1, filename2):
    """Plot the time data and coincidences"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Time series
    ax1.plot(times1, np.ones_like(times1), 'b.', markersize=2, alpha=0.7, label=f'{filename1} ({len(times1)} events)')
    ax1.plot(times2, np.ones_like(times2) * 1.2, 'r.', markersize=2, alpha=0.7, label=f'{filename2} ({len(times2)} events)')
    
    # Mark coincidences
    if coincidences:
        coinc_times1 = [c[0] for c in coincidences]
        coinc_times2 = [c[1] for c in coincidences]
        ax1.plot(coinc_times1, np.ones_like(coinc_times1), 'go', markersize=4, label=f'Coincidences ({len(coincidences)})')
        ax1.plot(coinc_times2, np.ones_like(coinc_times2) * 1.2, 'go', markersize=4)
    
    ax1.set_ylabel('Detector')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_title('Time Series and Coincidences')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coincidence time differences (central 50 bins only)
    if coincidences:
        time_diffs = [c[2] for c in coincidences]
        # Compute histogram with 200 bins to find the center
        counts, bin_edges = np.histogram(time_diffs, bins=200)
        # Find the center bin index
        center_bin_idx = len(counts) // 2
        # Extract central 50 bins (25 bins on each side of center)
        start_bin = max(0, center_bin_idx - 25)
        end_bin = min(len(counts), center_bin_idx + 25)

        # bin_edges has length len(counts)+1, so for N bins we need N+1 edges
        edges = bin_edges[start_bin : end_bin + 1]
        x = edges[:-1]
        w = np.diff(edges)

        # Plot only the central bins
        ax2.bar(x, counts[start_bin:end_bin], width=w, alpha=0.7, color='green', edgecolor='black', align='edge')
        ax2.set_xlabel('Time Difference (seconds)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Coincidence Time Differences - Central 50 Bins (Total: {len(coincidences)})')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No coincidences found', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Coincidence Time Differences')
    
    plt.tight_layout()
    return fig

def main():
    # File paths
    file1 = '08_bits/data/CERN-01from2018-09-01to2018-09-02.csv'
    file2 = '08_bits/data/CERN-02from2018-09-01to2018-09-02.csv'
    
    print("Processing time data from both files...")
    
    # Process time data from both files
    times1 = process_time_data(file1)
    times2 = process_time_data(file2)
    
    print(f"File 1 ({file1}): {len(times1)} events")
    print(f"File 2 ({file2}): {len(times2)} events")
    
    # Find coincidences with different time windows
    coincidence_windows = [1e-6, 1e-5, 1e-4, 1e-3]  # 1μs, 10μs, 100μs, 1ms
    #
    # Use only one window for now
    # Important part of the program
    coincidence_windows = [1e-4]
    
    for window in coincidence_windows:
        print(f"\nSearching for coincidences with window: {window:.0e} seconds")
        coincidences = find_coincidences(times1, times2, window)
        print(f"Found {len(coincidences)} coincidences")
        
        if len(coincidences) > 0:
            # Plot coincidences
            fig = plot_coincidences(times1, times2, coincidences, 'CERN-01', 'CERN-02')
            
            # Save plot
            plot_filename = f'08_bits/results/coincidences_{window:.0e}s.png'
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {plot_filename}")
            
            # Save coincidence data
            coinc_df = pd.DataFrame(coincidences, columns=['Time1', 'Time2', 'TimeDiff'])
            csv_filename = f'08_bits/results/coincidences_{window:.0e}s.csv'
            coinc_df.to_csv(csv_filename, index=False)
            print(f"Coincidence data saved to {csv_filename}")
            
            plt.close(fig)
            
            # If we found coincidences, we can stop with this window
            break
    else:
        print("No coincidences found with any time window")
        
        # Still create a plot showing the time series
        fig = plot_coincidences(times1, times2, [], 'CERN-01', 'CERN-02')
        plot_filename = '08_bits/results/time_series_no_coincidences.png'
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Time series plot saved to {plot_filename}")
        plt.close(fig)

if __name__ == "__main__":
    main() 