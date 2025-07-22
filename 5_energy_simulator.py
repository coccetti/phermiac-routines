# read binary file and plot the histogram of the energy
# the binary file is a 8 or 16 bit binary string
# read each digit of the binary string and convert it to a number
# the number is the energy of the event
# plot the histogram of the energy

import numpy as np
import matplotlib.pyplot as plt
import cmath

# read the binary file
event_file_bin = 'data/time_differences_C01-event_C02-1sec_8bits.csv'
# Read the binary strings from the file
event_bin_strings = []
with open(event_file_bin, 'r') as f:
    for line in f:
        line = line.strip()
        if line:  # Skip empty lines
            event_bin_strings.append(line)

# Now event_bin_strings is a list of binary strings
print(event_bin_strings)

# Collect total energy for each event
all_total_energies = []

# Process each binary string
for binary_string in event_bin_strings:
    # Convert string to list of digits
    digits = list(binary_string)
    # print(f"Digits for {binary_string}: {digits}")
    # print(digits)

    # Convert digits to phase values (pi for 1, 0 for 0)
    phase_values = [np.pi if digit == '1' else 0 for digit in digits]
    print(f"Phase values for {binary_string}: {phase_values}")
    
    # Convert the phase values to a list of energy values
    energy_values = [cmath.exp(complex(0, phase)).real for phase in phase_values]
    print(f"Energy values for {binary_string}: {energy_values}")

    # Compute the total energy of the event (sum of energy values)
    # total_energy = sum(energy * (i + 1) for i, energy in enumerate(energy_values))
    total_energy = sum(energy_values)
    all_total_energies.append(total_energy)
    print(f"Total energy for {binary_string}: {total_energy}")

# Plot the histogram of the total energies
plt.figure(figsize=(10, 6))
plt.hist(all_total_energies, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Total Energy')
plt.ylabel('Frequency')
plt.title('Histogram of Total Energy per Event')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()



