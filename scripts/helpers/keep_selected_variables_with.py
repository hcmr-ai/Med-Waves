# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:21:50 2024

@author: Lily Oikonomou (HCMR)
"""

# ---this is to extract only the variables of interest
# required packages
import os

import xarray as xr

# Define input directory
input_dir = "/data2/ocean2/REANALYSIS/VAL"

# Generate output directory by appending '_reduced' to the input directory's name
output_dir = output_dir = "/data2/ntsolis/with_reduced"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

# Define variables to keep (significant wave height, wind speed, mean wave period)
variables_to_keep = ["WSPD", "VHM0", "VTM02", "WDIR", "VMDR"]
# variables_to_keep = ["VHM0", "WSPD", "VTM02", "WDIR", "VMDR"] #to keep directions as well

# Loop through each .nc file in the input directory
for filename in os.listdir(input_dir):
    if filename.startswith("WAVEAN2020") and filename.endswith(".nc"):
        # construct our full file path
        file_path = os.path.join(input_dir, filename)

        # load dataset
        with xr.open_dataset(file_path) as ds:
            # select only the required variables
            ds_selected = ds[variables_to_keep]
            # -------------------------------------------
            # save to the output directory with the same name as the original one
            output_file_path = os.path.join(output_dir, filename)
            ds_selected.to_netcdf(output_file_path)

            print(f"Saved selected variables from {filename} to {output_file_path}")
