import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('results/energy_analysis_results.csv')

# Create a figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Total Energy vs Event Index
ax1.plot(range(len(df)), df['total_energy'], 'bo', markersize=8, alpha=0.7)
ax1.set_xlabel('Event Index')
ax1.set_ylabel('Total Energy')
ax1.set_title('Total Energy vs Event Index')
ax1.grid(True, alpha=0.3)

# Plot 2: Histogram of Total Energy values
ax2.hist(df['total_energy'], bins=50, alpha=0.7, color='green', edgecolor='black')
ax2.set_xlabel('Total Energy')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Total Energy Values')
ax2.grid(True, alpha=0.3)

# Add some statistics
mean_energy = df['total_energy'].mean()
std_energy = df['total_energy'].std()
min_energy = df['total_energy'].min()
max_energy = df['total_energy'].max()

stats_text = f'Mean: {mean_energy:.2f}\nStd: {std_energy:.2f}\nMin: {min_energy:.2f}\nMax: {max_energy:.2f}'
ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, 
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('results/energy_analysis_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print(f"Energy Analysis Summary:")
print(f"Number of events: {len(df)}")
print(f"Mean energy: {mean_energy:.2f}")
print(f"Standard deviation: {std_energy:.2f}")
print(f"Minimum energy: {min_energy:.2f}")
print(f"Maximum energy: {max_energy:.2f}")
print(f"Energy range: {max_energy - min_energy:.2f}") 