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
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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

def create_distribution_plots(total_bins, total_regions, total_extreme_regions, 
                            total_year_month, total_year_month_bins, years_str):
    """Create comprehensive plots of wave height distribution."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Wave Height Bin Distribution (Pie Chart)
    ax1 = plt.subplot(3, 3, 1)
    bin_names = list(WAVE_BINS.keys())
    bin_counts = [total_bins[name] for name in bin_names]
    colors = ['#2E8B57', '#FFD700', '#FF8C00', '#FF4500', '#8B0000']  # Green to Dark Red
    
    wedges, texts, autotexts = ax1.pie(bin_counts, labels=bin_names, autopct='%1.1f%%', 
                                      colors=colors, startangle=90)
    ax1.set_title('Wave Height Distribution\n(Overall)', fontsize=14, fontweight='bold')
    
    # 2. Wave Height Bin Distribution (Bar Chart)
    ax2 = plt.subplot(3, 3, 2)
    bars = ax2.bar(bin_names, bin_counts, color=colors)
    ax2.set_title('Wave Height Distribution\n(Sample Counts)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Samples')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, bin_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=10)
    
    # 3. Regional Distribution
    ax3 = plt.subplot(3, 3, 3)
    region_names = ['Atlantic', 'Mediterranean', 'Eastern Mediterranean']
    region_counts = [total_regions[f'region_{i}'] for i in range(3)]
    extreme_counts = [total_extreme_regions[f'extreme_region_{i}'] for i in range(3)]
    
    x = np.arange(len(region_names))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, region_counts, width, label='Total Samples', alpha=0.8)
    bars2 = ax3.bar(x + width/2, extreme_counts, width, label='Extreme Waves (>9m)', 
                   color='red', alpha=0.8)
    
    ax3.set_title('Regional Distribution', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Samples')
    ax3.set_xticks(x)
    ax3.set_xticklabels(region_names, rotation=45, ha='right')
    ax3.legend()
    
    # 4. Year-Month Distribution (if available)
    ax4 = plt.subplot(3, 3, 4)
    if total_year_month:
        sorted_months = sorted(total_year_month.items())
        months = [item[0] for item in sorted_months]
        counts = [item[1] for item in sorted_months]
        
        ax4.plot(months, counts, marker='o', linewidth=2, markersize=6)
        ax4.set_title('Temporal Distribution\n(Samples per Month)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Number of Samples')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No temporal data available', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Temporal Distribution', fontsize=14, fontweight='bold')
    
    # 5. Extreme Waves by Region (Pie Chart)
    ax5 = plt.subplot(3, 3, 5)
    if sum(extreme_counts) > 0:
        wedges, texts, autotexts = ax5.pie(extreme_counts, labels=region_names, 
                                          autopct='%1.1f%%', startangle=90)
        ax5.set_title('Extreme Waves by Region\n(>9m waves)', fontsize=14, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No extreme waves found', ha='center', va='center', 
                transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Extreme Waves by Region', fontsize=14, fontweight='bold')
    
    # 6. Year-Month Bin Heatmap (if available)
    ax6 = plt.subplot(3, 3, 6)
    if total_year_month_bins:
        # Create a matrix for the heatmap
        year_month_groups = {}
        for key, count in total_year_month_bins.items():
            parts = key.split('_')
            if len(parts) >= 2:
                year_month = parts[0]
                bin_name = '_'.join(parts[1:])
                
                if year_month not in year_month_groups:
                    year_month_groups[year_month] = {}
                year_month_groups[year_month][bin_name] = count
        
        # Prepare data for heatmap
        months = sorted(year_month_groups.keys())
        bins = list(WAVE_BINS.keys())
        heatmap_data = []
        
        for month in months:
            row = []
            for bin_name in bins:
                count = year_month_groups[month].get(bin_name, 0)
                row.append(count)
            heatmap_data.append(row)
        
        heatmap_data = np.array(heatmap_data)
        
        # Create heatmap
        im = ax6.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax6.set_xticks(range(len(bins)))
        ax6.set_xticklabels(bins, rotation=45, ha='right')
        ax6.set_yticks(range(len(months)))
        ax6.set_yticklabels(months)
        ax6.set_title('Wave Height Distribution\nby Month', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax6)
        cbar.set_label('Sample Count')
    else:
        ax6.text(0.5, 0.5, 'No temporal-bin data available', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Wave Height Distribution by Month', fontsize=14, fontweight='bold')
    
    # 7. Extreme Wave Statistics
    ax7 = plt.subplot(3, 3, 7)
    total_extreme = total_bins['extreme']
    total_samples = sum(total_bins.values())
    
    stats_text = f"""
    Total Samples: {total_samples:,}
    Extreme Waves: {total_extreme:,}
    Extreme %: {(total_extreme/total_samples)*100:.3f}%
    
    Regional Extreme Distribution:
    • Atlantic: {extreme_counts[0]:,} ({(extreme_counts[0]/total_extreme)*100:.1f}%)
    • Mediterranean: {extreme_counts[1]:,} ({(extreme_counts[1]/total_extreme)*100:.1f}%)
    • Eastern Med: {extreme_counts[2]:,} ({(extreme_counts[2]/total_extreme)*100:.1f}%)
    """
    
    ax7.text(0.05, 0.95, stats_text, transform=ax7.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax7.set_title('Extreme Wave Statistics', fontsize=14, fontweight='bold')
    ax7.axis('off')
    
    # 8. Bin Distribution Comparison (Log Scale)
    ax8 = plt.subplot(3, 3, 8)
    log_counts = [max(1, count) for count in bin_counts]  # Avoid log(0)
    bars = ax8.bar(bin_names, log_counts, color=colors)
    ax8.set_yscale('log')
    ax8.set_title('Wave Height Distribution\n(Log Scale)', fontsize=14, fontweight='bold')
    ax8.set_ylabel('Number of Samples (log scale)')
    ax8.tick_params(axis='x', rotation=45)
    
    # 9. Summary Information
    ax9 = plt.subplot(3, 3, 9)
    summary_text = f"""
    Analysis Summary
    Years: {years_str}
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    Data Quality:
    • Total files processed
    • Null values filtered
    • Regional mapping applied
    • Temporal analysis included
    
    Key Insights:
    • Extreme waves are rare ({total_extreme/total_samples*100:.3f}%)
    • Regional distribution varies
    • Temporal patterns visible
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax9.set_title('Analysis Summary', fontsize=14, fontweight='bold')
    ax9.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plot_filename = f"wave_distribution_plots_{years_str}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_filename

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
        for col_name in ["vhm0_y","corrected_VHM0", "VHM0", "vhm0", "wave_height", "hs"]:
            if col_name in df.columns:
                wave_height_col = col_name
                break
        
        if wave_height_col is None:
            return {"file": file_path, "error": f"No wave height column found. Available: {df.columns}"}
        
        # print(f"Using wave height column: {wave_height_col}")
        
        # Filter out rows with null wave heights and get aligned data
        df_clean = df.filter(pl.col(wave_height_col).is_not_null())
        wave_heights = df_clean[wave_height_col].to_numpy()
        total_samples = len(wave_heights)
        
        # Extract year and month for temporal analysis
        year_month_counts = {}
        year_month_bin_counts = {}
        
        if "datetime" in df_clean.columns:
            # Extract year and month from datetime column
            df_clean = df_clean.with_columns([
                pl.col("datetime").dt.year().alias("year"),
                pl.col("datetime").dt.month().alias("month")
            ])
            
            # Count samples per year-month and per year-month-bin combination
            year_month_data = df_clean.select(["year", "month", wave_height_col]).to_numpy()
            for year, month, wave_height in year_month_data:
                # Ensure year and month are integers
                year_int = int(year) if not np.isnan(year) else None
                month_int = int(month) if not np.isnan(month) else None
                
                if year_int is not None and month_int is not None:
                    key = f"{year_int}-{month_int:02d}"
                    year_month_counts[key] = year_month_counts.get(key, 0) + 1
                    
                    # Determine which bin this wave height falls into
                    bin_name = None
                    for b_name, (min_h, max_h) in WAVE_BINS.items():
                        if b_name == "extreme":
                            if wave_height >= min_h:
                                bin_name = b_name
                                break
                        else:
                            if min_h <= wave_height < max_h:
                                bin_name = b_name
                                break
                    
                    if bin_name:
                        bin_key = f"{key}_{bin_name}"
                        year_month_bin_counts[bin_key] = year_month_bin_counts.get(bin_key, 0) + 1
                    
        elif "time" in df_clean.columns:
            # Try alternative time column
            df_clean = df_clean.with_columns([
                pl.col("time").dt.year().alias("year"),
                pl.col("time").dt.month().alias("month")
            ])
            
            year_month_data = df_clean.select(["year", "month", wave_height_col]).to_numpy()
            for year, month, wave_height in year_month_data:
                # Ensure year and month are integers
                year_int = int(year) if not np.isnan(year) else None
                month_int = int(month) if not np.isnan(month) else None
                
                if year_int is not None and month_int is not None:
                    key = f"{year_int}-{month_int:02d}"
                    year_month_counts[key] = year_month_counts.get(key, 0) + 1
                    
                    # Determine which bin this wave height falls into
                    bin_name = None
                    for b_name, (min_h, max_h) in WAVE_BINS.items():
                        if b_name == "extreme":
                            if wave_height >= min_h:
                                bin_name = b_name
                                break
                        else:
                            if min_h <= wave_height < max_h:
                                bin_name = b_name
                                break
                    
                    if bin_name:
                        bin_key = f"{key}_{bin_name}"
                        year_month_bin_counts[bin_key] = year_month_bin_counts.get(bin_key, 0) + 1
        
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
            # Use the cleaned dataframe to ensure coordinate alignment
            coords = df_clean.select([lat_col, lon_col]).to_numpy()
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
            "regions": region_counts,
            "year_month": year_month_counts,
            "year_month_bins": year_month_bin_counts
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {e}")
        return {"file": file_path, "error": str(e)}

def main():
    """Main analysis function."""
    # Set AWS region to avoid polars warning
    os.environ["AWS_DEFAULT_REGION"] = "eu-central-1"
    years_to_analyze = [2021, 2022]  # Change this list to analyze different years

    # Create output file for saving results
    years_str = '_'.join(map(str, years_to_analyze))
    output_file = f"wave_distribution_analysis_{years_str}.txt"
    output_f = open(output_file, 'w')
    
    # Redirect stdout to both console and file
    import sys
    original_stdout = sys.stdout
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                if not f.closed:
                    f.write(obj)
                    f.flush()
        def flush(self):
            for f in self.files:
                if not f.closed:
                    f.flush()
    
    sys.stdout = Tee(original_stdout, output_f)
    
    try:
        # Configuration
        s3_bucket = "medwav-dev-data"
        s3_prefix = "parquet/hourly/"
        
        logger.info(f"Analyzing wave distribution for years: {years_to_analyze}")
        logger.info(f"S3 bucket: {s3_bucket}")
        logger.info(f"S3 prefix: {s3_prefix}")
        
        # List actual files from S3 for the specified years
        from src.commons.aws.utils import list_s3_parquet_files
        
        # List parquet files for the specified years
        train_files = list_s3_parquet_files(s3_bucket, s3_prefix, None, None, None)
        
        # Filter files to only include the years we want to analyze
        filtered_files = []
        for file_path in train_files:
            # Extract year from file path (assuming format: year=YYYY/filename)
            if "year=" in file_path:
                year_str = file_path.split("year=")[1].split("/")[0]
                try:
                    year = int(year_str)
                    if year in years_to_analyze:
                        filtered_files.append(file_path)
                except ValueError:
                    continue
        
        train_files = filtered_files
        
        logger.info(f"Found {len(train_files)} training files")
        
        # Analyze files in parallel
        all_results = []
        with ProcessPoolExecutor(max_workers=64) as executor:
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
        total_year_month = {}
        total_year_month_bins = {}
        
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
            
            # Aggregate year-month counts
            for year_month_key, count in result.get("year_month", {}).items():
                total_year_month[year_month_key] = total_year_month.get(year_month_key, 0) + count
            
            # Aggregate year-month-bin counts
            for year_month_bin_key, count in result.get("year_month_bins", {}).items():
                total_year_month_bins[year_month_bin_key] = total_year_month_bins.get(year_month_bin_key, 0) + count
        
        # Print results
        print("\n" + "="*80)
        print("WAVE HEIGHT DISTRIBUTION ANALYSIS")
        print("="*80)
        print(f"Total files analyzed: {len(all_results)}")
        print(f"Files with errors: {len(files_with_errors)}")
        print(f"Total samples: {total_samples:,}")
        
        if total_samples == 0:
            print("\nWARNING: No valid samples found in any files!")
            print("This could indicate:")
            print("- All wave height columns contain null values")
            print("- Column names don't match expected patterns")
            print("- Files are empty or corrupted")
            print("\nCheck the error messages above for details.")
            return
        
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
        print("YEAR-MONTH DISTRIBUTION:")
        print("-" * 50)
        if total_year_month:
            # Sort by year-month for better readability
            sorted_year_months = sorted(total_year_month.items())
            for year_month, count in sorted_year_months:
                percentage = (count / total_samples) * 100 if total_samples > 0 else 0
                print(f"{year_month}: {count:10,} samples ({percentage:5.1f}%)")
        else:
            print("No temporal data available (datetime/time columns not found)")
        
        print()
        print("YEAR-MONTH BY WAVE HEIGHT BIN:")
        print("-" * 50)
        if total_year_month_bins:
            # Group by year-month and show bin breakdown
            year_month_groups = {}
            for key, count in total_year_month_bins.items():
                # Split the key properly: "2021-01_calm" -> year_month="2021-01", bin_name="calm"
                parts = key.split('_')
                if len(parts) >= 2:
                    year_month = parts[0]  # e.g., "2021-01"
                    bin_name = '_'.join(parts[1:])  # e.g., "calm", "extreme"
                    
                    if year_month not in year_month_groups:
                        year_month_groups[year_month] = {}
                    year_month_groups[year_month][bin_name] = count
            
            # Sort and display
            for year_month in sorted(year_month_groups.keys()):
                print(f"\n{year_month}:")
                total_for_month = sum(year_month_groups[year_month].values())
                for bin_name in WAVE_BINS.keys():
                    count = year_month_groups[year_month].get(bin_name, 0)
                    percentage = (count / total_for_month) * 100 if total_for_month > 0 else 0
                    print(f"  {bin_name:12}: {count:8,} samples ({percentage:5.1f}%)")
        else:
            print("No temporal-bin data available")
        
        print()
        print("EXTREME WAVE ANALYSIS (>9m):")
        print("-" * 50)
        print(f"Total extreme waves: {total_bins['extreme']:,}")
        if total_samples > 0:
            print(f"Percentage of total: {(total_bins['extreme'] / total_samples) * 100:.3f}%")
        else:
            print("Percentage of total: N/A (no samples found)")
        
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
        if total_bins['extreme'] > 0:
            print(f"Percentage of extreme waves lost: {(potential_loss / total_bins['extreme']) * 100:.1f}%")
        else:
            print("Percentage of extreme waves lost: N/A (no extreme waves found)")
        
        # Save detailed results
        results_file = f"wave_distribution_analysis_{years_str}.json"
        import json
        
        detailed_results = {
            "summary": {
                "total_files": len(all_results),
                "files_with_errors": len(files_with_errors),
                "total_samples": total_samples,
                "wave_bins": total_bins,
                "regions": total_regions,
                "extreme_by_region": total_extreme_regions,
                "year_month_distribution": total_year_month,
                "year_month_bin_distribution": total_year_month_bins
            },
            "per_file_results": all_results
        }
        
        with open(results_file, 'w') as json_f:
            json.dump(detailed_results, json_f, indent=2)
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Analysis output saved to: {output_file}")
        
        # Create and save plots
        print("Creating distribution plots...")
        plot_filename = create_distribution_plots(
            total_bins, total_regions, total_extreme_regions,
            total_year_month, total_year_month_bins, years_str
        )
        print(f"Distribution plots saved to: {plot_filename}")
    
    finally:
        # Restore original stdout and close file
        sys.stdout = original_stdout
        output_f.close()

if __name__ == "__main__":
    main()
