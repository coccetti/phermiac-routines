import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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

def find_all_time_differences(times1, times2, max_window=1e-3):
    """Find all time differences between two time arrays within a maximum window"""
    time_diffs = []
    
    for t1 in times1:
        # Find all times in times2 that are within the maximum window
        mask = np.abs(times2 - t1) <= max_window
        if np.any(mask):
            diffs = times2[mask] - t1
            time_diffs.extend(diffs)
    
    return np.array(time_diffs)

def plot_time_differences_histogram(time_diffs, max_window=1e-3):
    """Plot histogram of time differences with Gaussian fit"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Full distribution of time differences
    ax1.hist(time_diffs, bins=100, alpha=0.7, color='blue', edgecolor='black', density=True, label='Time Differences')
    
    # Fit Gaussian to the data
    mu, sigma = stats.norm.fit(time_diffs)
    x = np.linspace(time_diffs.min(), time_diffs.max(), 1000)
    gaussian = stats.norm.pdf(x, mu, sigma)
    ax1.plot(x, gaussian, 'r-', linewidth=2, label=f'Gaussian Fit (μ={mu:.2e}, σ={sigma:.2e})')
    
    ax1.set_xlabel('Time Difference (seconds)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Distribution of Time Differences (Total: {len(time_diffs)} pairs, Window: ±{max_window:.0e}s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed view around zero (coincidence region)
    zoom_window = 1e-5  # 10μs zoom window
    zoom_mask = np.abs(time_diffs) <= zoom_window
    zoomed_diffs = time_diffs[zoom_mask]
    
    if len(zoomed_diffs) > 0:
        ax2.hist(zoomed_diffs, bins=50, alpha=0.7, color='green', edgecolor='black', density=True, label='Time Differences')
        
        # Fit Gaussian to zoomed data
        mu_zoom, sigma_zoom = stats.norm.fit(zoomed_diffs)
        x_zoom = np.linspace(zoomed_diffs.min(), zoomed_diffs.max(), 1000)
        gaussian_zoom = stats.norm.pdf(x_zoom, mu_zoom, sigma_zoom)
        ax2.plot(x_zoom, gaussian_zoom, 'r-', linewidth=2, label=f'Gaussian Fit (μ={mu_zoom:.2e}, σ={sigma_zoom:.2e})')
        
        ax2.set_xlabel('Time Difference (seconds)')
        ax2.set_ylabel('Density')
        ax2.set_title(f'Zoomed View around Zero (±{zoom_window:.0e}s, {len(zoomed_diffs)} events)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No events in zoom window', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'Zoomed View around Zero (±{zoom_window:.0e}s)')
    
    plt.tight_layout()
    return fig

def main():
    # File paths
    file1 = 'data/CERN-01from2018-09-01to2018-09-02.csv'
    file2 = 'data/CERN-02from2018-09-01to2018-09-02.csv'
    
    print("Processing time data from both files...")
    
    # Process time data from both files
    times1 = process_time_data(file1)
    times2 = process_time_data(file2)
    
    print(f"File 1 ({file1}): {len(times1)} events")
    print(f"File 2 ({file2}): {len(times2)} events")
    
    # Max window to capture the distribution
    max_window = 1e-3  # 1ms to capture the distribution
    
    print(f"\nAnalyzing time differences with max window: ±{max_window:.0e} seconds")
    
    # Find all time differences within the window
    time_diffs = find_all_time_differences(times1, times2, max_window)
    
    print(f"Found {len(time_diffs)} time difference pairs within ±{max_window:.0e}s")
    
    # Calculate statistics
    mean_diff = np.mean(time_diffs)
    std_diff = np.std(time_diffs)
    median_diff = np.median(time_diffs)
    
    print(f"Statistics:")
    print(f"  Mean: {mean_diff:.2e} seconds")
    print(f"  Standard Deviation: {std_diff:.2e} seconds")
    print(f"  Median: {median_diff:.2e} seconds")
    print(f"  Range: [{time_diffs.min():.2e}, {time_diffs.max():.2e}] seconds")
    
    # Plot histogram
    fig = plot_time_differences_histogram(time_diffs, max_window)
    
    # Save plot
    plot_filename = 'results/time_differences_histogram.png'
    fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_filename}")
    
    # Save time difference data
    diff_df = pd.DataFrame({'TimeDiff': time_diffs})
    csv_filename = 'results/time_differences_data.csv'
    diff_df.to_csv(csv_filename, index=False)
    print(f"Time difference data saved to {csv_filename}")
    
    plt.close(fig)

if __name__ == "__main__":
    main()
