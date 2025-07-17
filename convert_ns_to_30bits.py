#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read nanoseconds from a CERN-0*.csv file and convert each value to an n-bit binary string.
"""
import csv
import numpy as np

def read_nanoseconds(file_path):
    nanoseconds = []
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            nanoseconds.append(int(row['Nanoseconds']))
    return np.array(nanoseconds)

def convert_to_nbit_binary_string(number, n):
    return format(number, f'0{n}b')

if __name__ == "__main__":
    file_path = 'data/CERN-01.csv'
    nanoseconds = read_nanoseconds(file_path)
    n_bits = 30  # 30 is the number of bits for nanoseconds, \
                    # but it can be changed in the future \
                    # if we decide to take seconds into account
    nanoseconds_nbit = [convert_to_nbit_binary_string(ns, n_bits) for ns in nanoseconds]
    print("Test a number: ", nanoseconds_nbit[57])