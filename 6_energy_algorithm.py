import numpy as np
import cmath
import csv

# Read the line in the file data/CERN-01_1event_nbit16.csv and print it
with open('data/CERN-01_event1221_nbit16.csv', 'r') as file:
    event_01_string = file.readline().strip()
    # Convert the string of digits to a numpy array of integers
    event_01 = np.array([int(digit) for digit in event_01_string])
    print(event_01)

# Read all the lines in the file data/CERN-02_1sec_nbit16.csv
event_02_arrays = []
line_strings = []
with open('data/CERN-02_sec1221_nbit16.csv', 'r') as file:
    for line in file:
        line_string = line.strip()
        line_strings.append(line_string)
        # Convert the string of digits to a numpy array of integers
        event_02_array = np.array([int(digit) for digit in line_string])
        event_02_arrays.append(event_02_array)
        # print(event_02_array)
# print(f"Event_02_arrays: {event_02_arrays}")

# Process each binary string in the SLM
# Convert digits to phase values (pi for 1, 0 for 0)
event_01_phase = np.array([np.pi if digit == 1 else 0 for digit in event_01])
print(f"Phase values for event_01: {event_01_phase}")
# Convert the phase values to a list of energy values
event_01_energy = [cmath.exp(complex(0, phase)).real for phase in event_01_phase]
print(f"Energy values for event_01: {event_01_energy}")

# List to store total energy values
total_energies = []

# Process each event in the CERN-02 array
for event_02_array in event_02_arrays:
    # print(f"Event_02_array: {event_02_array}")
    # Convert digits to phase values (pi for 1, 0 for 0)
    event_02_phase = np.array([np.pi if digit == 1 else 0 for digit in event_02_array])
    # print(f"Phase values for event_02: {event_02_phase}")
    # #########################################################
    # Sum the phase values of event 01 and event 02
    # #########################################################
    summed_phase = event_01_phase + event_02_phase
    # print(f"Event_01_phase: {event_01_phase}")
    # print(f"Event_02_phase: {event_02_phase}")
    # print(f"Summed phase values: {summed_phase}")

    # #########################################################
    # Compute the energy of the event
    # #########################################################
    summed_energy = [cmath.exp(complex(0, phase)).real for phase in summed_phase]
    print(f"Energy values for summed phase: {summed_energy}")
    # Compute the total energy of the event (sum of energy values)
    # total_energy = sum(energy_values)
    # Calculate weighted energy sum with exponential weights
    total_energy = 0
    for i, energy in enumerate(summed_energy):
        # weight = 1.3 ** (16-i)  # Exponential weight factor
        weight = 2 ** (16-i)
        # weight = (16-i)**2
        # weight = 16-i # This works better
        # weight = 1
        weighted_energy = energy * weight
        total_energy += weighted_energy
        print(f"--------------------------------")
        print(f"Summed energy: {summed_energy}")
        print(f"i: {i}")
        print(f"Weight: {weight}")
        print(f"Energy: {energy}")
        print(f"Weighted energy: {weighted_energy}")
        print(f"Total energy: {total_energy}")
        print(f"--------------------------------")
    # total_energy = sum(energy for i, energy in enumerate(summed_energy))
    print(f"Total energy for event_02: {total_energy}")
    
    # Store the total energy for CSV output
    total_energies.append(total_energy)

# Save results to CSV file
with open('results/energy_analysis_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow(['line_string', 'total_energy'])
    
    # Write data rows
    for line_string, total_energy in zip(line_strings, total_energies):
        writer.writerow([line_string, total_energy])

print(f"Results saved to results/energy_analysis_results.csv")


