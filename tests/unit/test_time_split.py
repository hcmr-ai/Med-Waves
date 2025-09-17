#!/usr/bin/env python3
"""
Test script to verify the time-based split works correctly with parquet files.
"""

import sys
from pathlib import Path
import glob

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_engineering.split import time_based_split, extract_date_from_filename

def test_time_split():
    """Test the time-based split function."""
    
    # Test with sample filenames
    test_files = [
        "/data/tsolis/AI_project/parquet/augmented_with_labels/hourly/WAVEAN20210101.parquet",
        "/data/tsolis/AI_project/parquet/augmented_with_labels/hourly/WAVEAN20211231.parquet",
        "/data/tsolis/AI_project/parquet/augmented_with_labels/hourly/WAVEAN20220101.parquet",
        "/data/tsolis/AI_project/parquet/augmented_with_labels/hourly/WAVEAN20221231.parquet",
        "/data/tsolis/AI_project/parquet/augmented_with_labels/hourly/WAVEAN20230101.parquet",
        "/data/tsolis/AI_project/parquet/augmented_with_labels/hourly/WAVEAN20231231.parquet",
    ]
    
    print("Testing time-based split...")
    print(f"Test files: {len(test_files)}")
    
    # Test date extraction
    print("\nTesting date extraction:")
    for f in test_files:
        try:
            date = extract_date_from_filename(f)
            print(f"  {Path(f).name} -> {date.strftime('%Y-%m-%d')}")
        except Exception as e:
            print(f"  ERROR: {f} -> {e}")
    
    # Test split
    print("\nTesting time-based split (train <= 2022, test >= 2023):")
    try:
        x_train, y_train, x_test, y_test = time_based_split(
            test_files, test_files, 
            train_end_year=2022, 
            test_start_year=2023
        )
        
        print(f"Train files: {len(x_train)}")
        for f in x_train:
            print(f"  {Path(f).name}")
        
        print(f"Test files: {len(x_test)}")
        for f in x_test:
            print(f"  {Path(f).name}")
    
    # Test debug mode
    print("\nTesting debug mode (1 train day, 1 test day):")
    try:
        x_train_debug, y_train_debug, x_test_debug, y_test_debug = time_based_split(
            test_files, test_files, 
            train_end_year=2022, 
            test_start_year=2023,
            debug_mode=True,
            debug_train_days=1,
            debug_test_days=1
        )
        
        print(f"Debug Train files: {len(x_train_debug)}")
        for f in x_train_debug:
            print(f"  {Path(f).name}")
        
        print(f"Debug Test files: {len(x_test_debug)}")
        for f in x_test_debug:
            print(f"  {Path(f).name}")
            
    except Exception as e:
        print(f"ERROR: {e}")

def test_real_files():
    """Test with real files if they exist."""
    data_dir = "/data/tsolis/AI_project/parquet/augmented_with_labels/hourly"
    
    if not Path(data_dir).exists():
        print(f"Data directory does not exist: {data_dir}")
        return
    
    # Get real files
    parquet_files = sorted(glob.glob(f"{data_dir}/WAVEAN*.parquet"))
    
    if not parquet_files:
        print(f"No parquet files found in {data_dir}")
        return
    
    print(f"\nFound {len(parquet_files)} real parquet files")
    print("First 5 files:")
    for f in parquet_files[:5]:
        print(f"  {Path(f).name}")
    
    print("Last 5 files:")
    for f in parquet_files[-5:]:
        print(f"  {Path(f).name}")
    
    # Test split with real files
    print("\nTesting time-based split with real files:")
    try:
        x_train, y_train, x_test, y_test = time_based_split(
            parquet_files, parquet_files, 
            train_end_year=2022, 
            test_start_year=2023
        )
        
        print(f"Train files: {len(x_train)}")
        print(f"Test files: {len(x_test)}")
        
        if x_train:
            print(f"Train period: {Path(x_train[0]).name} to {Path(x_train[-1]).name}")
        if x_test:
            print(f"Test period: {Path(x_test[0]).name} to {Path(x_test[-1]).name}")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_time_split()
    test_real_files()
