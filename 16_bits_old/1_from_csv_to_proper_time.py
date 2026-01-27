#
# From the standard CSV value to proper time value
#
# This script reads the data from the csv file and prints the seconds properly
# It then pads the nanoseconds with leading zeros to ensure length of 9, then joins with seconds
# It then saves the results to a csv file with a variation of the input filename


import pandas as pd

### Read the data from the csv file and print the seconds properly
# read only the Seconds and Nanoseconds columns from the csv file in the data folder
data = pd.read_csv('data/CERN-02from2018-09-01to2018-09-02.csv', usecols=['Seconds', 'Nanoseconds'])

# Subtract 368143200 from all Seconds values
data['Seconds'] = data['Seconds'] - 368143200

# print the data
print(data)

# Pad Nanoseconds with leading zeros to ensure length of 9, then join with Seconds
data['Time'] = data['Seconds'].astype(str) + '.' + data['Nanoseconds'].astype(str).str.zfill(9)

# print the data as string
print(data['Time'])

# Save only the Time column to a CSV file with a variation of the input filename
time_data = pd.DataFrame({'Time': data['Time']})
time_data.to_csv('data/CERN-02_processed.csv', index=False)
print("Results saved to data/CERN-02_processed.csv")

