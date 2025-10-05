#!/usr/bin/env python3
"""
Analyze Wave Height Distribution Across Training Data

This script loads all training files and calculates the distribution of wave heights
to understand how many extreme waves we have and where they're located.
"""

import sys
import logging
import polars as pl
from pathlib import Path
from typing import Dict
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# from src.data_engineering.data_loader import DataLoader  # Not needed anymore
from src.commons.region_mapping import get_region_from_coordinates

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Wave height bins (same as in per_point_stratification.py)
WAVE_BINS = {
    "calm": [0.0, 1.0],
    "moderate": [1.0, 3.0], 
    "rough": [3.0, 6.0],
    "high": [6.0, 9.0],
    "extreme": [9.0, float('inf')]
}

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def analyze_single_file(file_path: str) -> Dict:
    """Analyze wave height distribution for a single file."""
    try:
        # Load the parquet file
        df = pl.read_parquet(file_path)
        
        if len(df) == 0:
            return {"file": file_path, "total_samples": 0, "bins": {}, "regions": {}}
        
        # Check what columns are available
        # print(f"Available columns: {df.columns}")
        
        # Try different possible column names for wave height
        wave_height_col = None
        for col_name in ["VHM0", "corrected_VHM0", "vhm0_y", "vhm0", "wave_height", "hs"]:
            if col_name in df.columns:
                wave_height_col = col_name
                break
        
        if wave_height_col is None:
            return {"file": file_path, "error": f"No wave height column found. Available: {df.columns}"}
        
        # print(f"Using wave height column: {wave_height_col}")
        
        # Get wave heights
        wave_heights = df[wave_height_col].to_numpy()
        total_samples = len(wave_heights)
        
        # Calculate bin counts
        bin_counts = {}
        for bin_name, (min_h, max_h) in WAVE_BINS.items():
            if bin_name == "extreme":
                mask = wave_heights >= min_h
            else:
                mask = (wave_heights >= min_h) & (wave_heights < max_h)
            bin_counts[bin_name] = int(np.sum(mask))
        
        # Calculate regional distribution (if coordinates available)
        region_counts = {}
        
        # Debug: Check what columns are available (only for first file)
        if "20210101" in file_path:  # Only print for first file
            print(f"DEBUG - Available columns in {Path(file_path).name}: {df.columns}")
        
        # Try different possible coordinate column names
        lat_col = None
        lon_col = None
        
        for lat_name in ["latitude", "lat", "LAT", "LATITUDE"]:
            if lat_name in df.columns:
                lat_col = lat_name
                break
                
        for lon_name in ["longitude", "lon", "LON", "LONGITUDE"]:
            if lon_name in df.columns:
                lon_col = lon_name
                break
        
        if lat_col and lon_col:
            coords = df.select([lat_col, lon_col]).to_numpy()
            regions = [get_region_from_coordinates(lat, lon) for lat, lon in coords]
            
            for region_id in [0, 1, 2]:  # Atlantic, Mediterranean, Eastern Med
                region_mask = np.array(regions) == region_id
                region_counts[f"region_{region_id}"] = int(np.sum(region_mask))
                
                # Also count extreme waves per region
                extreme_in_region = np.sum(region_mask & (wave_heights >= 9.0))
                region_counts[f"extreme_region_{region_id}"] = int(extreme_in_region)
        
        return {
            "file": file_path,
            "total_samples": total_samples,
            "bins": bin_counts,
            "regions": region_counts
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {e}")
        return {"file": file_path, "error": str(e)}

def main():
    """Main analysis function."""
    # Set AWS region to avoid polars warning
    os.environ["AWS_DEFAULT_REGION"] = "eu-central-1"
    
    # Load config
    config_path = "src/configs/config_per_point_stratified.yaml"
    config = load_config(config_path)
    
    # Get data configuration
    data_config = config["data"]
    source = data_config["source"]
    train_start_year = 2023
    train_end_year = 2023
    file_pattern = data_config["file_pattern"]
    
    logger.info(f"Analyzing wave distribution for {train_start_year}-{train_end_year}")
    logger.info(f"Pattern: {file_pattern}")
    
    # Generate list of training files directly
    train_files = []
    for year in range(train_start_year, train_end_year + 1):
        for month in range(1, 13):  # All months
            for day in range(1, 32):  # All days
                # Handle different month lengths
                if month in [4, 6, 9, 11] and day > 30:  # 30-day months
                    continue
                elif month == 2 and day > 28:  # February (assuming non-leap years)
                    continue
                
                filename = f"WAVEAN{year}{month:02d}{day:02d}.parquet"
                file_path = f"{source}year={year}/{filename}"
                train_files.append(file_path)
    
    logger.info(f"Found {len(train_files)} training files")
    
    # Analyze files in parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=8) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(analyze_single_file, file_path): file_path 
            for file_path in train_files
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_file), total=len(train_files), desc="Analyzing files"):
            result = future.result()
            all_results.append(result)
    
    # Aggregate results
    logger.info("Aggregating results...")
    
    total_samples = 0
    total_bins = {bin_name: 0 for bin_name in WAVE_BINS.keys()}
    total_regions = {f"region_{i}": 0 for i in range(3)}
    total_extreme_regions = {f"extreme_region_{i}": 0 for i in range(3)}
    
    files_with_errors = []
    
    for result in all_results:
        if "error" in result:
            files_with_errors.append(result)
            continue
            
        total_samples += result["total_samples"]
        
        # Aggregate bin counts
        for bin_name, count in result["bins"].items():
            total_bins[bin_name] += count
        
        # Aggregate regional counts
        for region_key, count in result["regions"].items():
            if region_key in total_regions:
                total_regions[region_key] += count
            elif region_key in total_extreme_regions:
                total_extreme_regions[region_key] += count
    
    # Print results
    print("\n" + "="*80)
    print("WAVE HEIGHT DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"Total files analyzed: {len(all_results)}")
    print(f"Files with errors: {len(files_with_errors)}")
    print(f"Total samples: {total_samples:,}")
    print()
    
    print("WAVE HEIGHT BIN DISTRIBUTION:")
    print("-" * 50)
    for bin_name, (min_h, max_h) in WAVE_BINS.items():
        count = total_bins[bin_name]
        percentage = (count / total_samples) * 100 if total_samples > 0 else 0
        
        if max_h == float('inf'):
            range_str = f"{min_h}+m"
        else:
            range_str = f"{min_h}-{max_h}m"
        
        print(f"{bin_name:12} ({range_str:>8}): {count:10,} samples ({percentage:5.1f}%)")
    
    print()
    print("REGIONAL DISTRIBUTION:")
    print("-" * 50)
    region_names = {0: "Atlantic", 1: "Mediterranean", 2: "Eastern Mediterranean"}
    
    for region_id in range(3):
        region_key = f"region_{region_id}"
        extreme_key = f"extreme_region_{region_id}"
        
        total_region = total_regions[region_key]
        extreme_region = total_extreme_regions[extreme_key]
        
        region_percentage = (total_region / total_samples) * 100 if total_samples > 0 else 0
        extreme_percentage = (extreme_region / total_bins["extreme"]) * 100 if total_bins["extreme"] > 0 else 0
        
        print(f"{region_names[region_id]:20}: {total_region:10,} samples ({region_percentage:5.1f}%)")
        print(f"  └─ Extreme waves: {extreme_region:10,} samples ({extreme_percentage:5.1f}% of all extreme)")
    
    print()
    print("EXTREME WAVE ANALYSIS (>9m):")
    print("-" * 50)
    print(f"Total extreme waves: {total_bins['extreme']:,}")
    print(f"Percentage of total: {(total_bins['extreme'] / total_samples) * 100:.3f}%")
    
    # Calculate potential loss scenarios
    print()
    print("POTENTIAL LOSS SCENARIOS (with 1M max_samples_per_file):")
    print("-" * 60)
    
    # Scenario 1: Current method (equal regional sampling)
    samples_per_region = 1000000 // 3  # 333,333 per region
    potential_loss = 0
    
    for region_id in range(3):
        region_key = f"region_{region_id}"
        extreme_key = f"extreme_region_{region_id}"
        
        total_region = total_regions[region_key]
        extreme_region = total_extreme_regions[extreme_key]
        
        if total_region > samples_per_region:
            # This region would be downsampled
            loss_ratio = (total_region - samples_per_region) / total_region
            extreme_loss = int(extreme_region * loss_ratio)
            potential_loss += extreme_loss
            
            print(f"Region {region_id} ({region_names[region_id]}):")
            print(f"  Total samples: {total_region:,} → {samples_per_region:,}")
            print(f"  Extreme waves: {extreme_region:,} → {extreme_region - extreme_loss:,} (lose {extreme_loss:,})")
    
    print(f"\nTotal extreme waves potentially lost: {potential_loss:,}")
    print(f"Percentage of extreme waves lost: {(potential_loss / total_bins['extreme']) * 100:.1f}%")
    
    # Save detailed results
    results_file = "wave_distribution_analysis.json"
    import json
    
    detailed_results = {
        "summary": {
            "total_files": len(all_results),
            "files_with_errors": len(files_with_errors),
            "total_samples": total_samples,
            "wave_bins": total_bins,
            "regions": total_regions,
            "extreme_by_region": total_extreme_regions
        },
        "per_file_results": all_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")

if __name__ == "__main__":
    main()
