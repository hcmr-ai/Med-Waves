#!/usr/bin/env python3
"""
Regional Distribution Analysis Script

This script analyzes the distribution of data points between Mediterranean and Atlantic regions
to identify potential data imbalance issues that could cause overfitting or poor generalization.
"""

import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from typing import Dict, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "src/configs/config_full_dataset.yaml") -> Dict:
    """Load configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_data_files(data_path: str, file_pattern: str = "*.parquet") -> List[str]:
    """Get list of parquet files from data directory."""
    import glob
    pattern = os.path.join(data_path, file_pattern)
    files = glob.glob(pattern)
    files.sort()
    return files

def extract_date_from_filename(file_path: str) -> pd.Timestamp:
    """Extract date from filename like WAVEAN20210101.parquet."""
    filename = os.path.basename(file_path)
    # Extract date part (YYYYMMDD)
    date_str = filename.split('.')[0].replace('WAVEAN', '')
    return pd.to_datetime(date_str, format='%Y%m%d')

def load_sample_data(file_paths: List[str], max_files: int = 3) -> pl.DataFrame:
    """Load a sample of data files for analysis."""
    logger.info(f"Loading sample data from {min(max_files, len(file_paths))} files...")
    
    sample_files = file_paths[:max_files]
    dataframes = []
    
    for file_path in sample_files:
        try:
            df = pl.read_parquet(file_path)
            
            logger.info(f"Loaded {file_path}: {len(df)} samples")
            
            # Add file info
            date = extract_date_from_filename(file_path)
            df = df.with_columns([
                pl.lit(file_path).alias("file_path"),
                pl.lit(date).alias("date"),
                pl.lit(date.year).alias("year"),
                pl.lit(date.month).alias("month")
            ])
            dataframes.append(df)
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
            continue
    
    if not dataframes:
        raise ValueError("No data files could be loaded")
    
    # Combine all dataframes
    combined_df = pl.concat(dataframes)
    logger.info(f"Combined dataset: {combined_df.shape}")
    
    # Remove rows with NaN values
    original_size = len(combined_df)
    combined_df = combined_df.drop_nulls()
    final_size = len(combined_df)
    
    if original_size != final_size:
        logger.info(f"Removed {original_size - final_size:,} rows with NaN values ({((original_size - final_size) / original_size * 100):.1f}%)")
        logger.info(f"Final dataset: {combined_df.shape}")
    
    return combined_df

def classify_regions(df: pl.DataFrame) -> pl.DataFrame:
    """Apply regional classification based on longitude."""
    logger.info("Applying regional classification...")
    
    df = df.with_columns([
        # Regional classification (using integer IDs for performance)
        pl.when(pl.col("longitude") < -5)
        .then(pl.lit(0))  # atlantic
        .when(pl.col("longitude") > 30)
        .then(pl.lit(2))  # eastern_med
        .otherwise(pl.lit(1))  # mediterranean
        .alias("region"),
        
        # Additional regional features
        (pl.col("longitude") < -5).alias("atlantic_region"),
        (pl.col("longitude") > 30).alias("eastern_med_region"),
    ])
    
    return df

def analyze_regional_distribution(df: pl.DataFrame) -> Dict:
    """Analyze the distribution of data points across regions."""
    logger.info("Analyzing regional distribution...")
    
    # Basic counts
    region_counts = df.group_by("region").agg([
        pl.len().alias("count"),
        pl.col("latitude").mean().alias("mean_lat"),
        pl.col("longitude").mean().alias("mean_lon"),
        pl.col("latitude").std().alias("std_lat"),
        pl.col("longitude").std().alias("std_lon")
    ]).sort("count", descending=True)
    
    # Calculate percentages
    total_points = len(df)
    region_counts = region_counts.with_columns([
        (pl.col("count") / total_points * 100).alias("percentage")
    ])
    
    # Temporal analysis
    temporal_analysis = df.group_by(["region", "year", "month"]).agg([
        pl.len().alias("count")
    ]).sort(["region", "year", "month"])
    
    # Spatial analysis (grid-based)
    spatial_analysis = df.group_by(["region"]).agg([
        pl.col("latitude").min().alias("min_lat"),
        pl.col("latitude").max().alias("max_lat"),
        pl.col("longitude").min().alias("min_lon"),
        pl.col("longitude").max().alias("max_lon"),
        pl.col("latitude").n_unique().alias("unique_lats"),
        pl.col("longitude").n_unique().alias("unique_lons")
    ])
    
    return {
        "region_counts": region_counts,
        "temporal_analysis": temporal_analysis,
        "spatial_analysis": spatial_analysis,
        "total_points": total_points
    }

def create_visualizations(df: pl.DataFrame, analysis_results: Dict, output_dir: str = "regional_analysis"):
    """Create comprehensive visualizations of regional distribution."""
    logger.info("Creating visualizations...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Regional Distribution Bar Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    region_counts = analysis_results["region_counts"]
    regions = region_counts["region"].to_list()
    counts = region_counts["count"].to_list()
    percentages = region_counts["percentage"].to_list()
    
    # Count bar chart
    bars1 = ax1.bar(regions, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Data Points per Region', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Data Points')
    ax1.set_xlabel('Region')
    
    # Add count labels on bars
    for bar, count in zip(bars1, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Percentage pie chart
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    wedges, texts, autotexts = ax2.pie(percentages, labels=regions, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax2.set_title('Regional Distribution (%)', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    plt.tight_layout()
    plt.savefig(output_path / "regional_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Spatial Distribution Map
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot colored by region
    from src.commons.region_mapping import RegionMapper
    region_colors = {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c'}  # atlantic, mediterranean, eastern_med
    
    for region_id in regions:
        region_data = df.filter(pl.col("region") == region_id)
        if len(region_data) > 0:
            region_name = RegionMapper.get_display_name(region_id)
            ax.scatter(region_data["longitude"].to_list(), region_data["latitude"].to_list(),
                      c=region_colors[region_id], label=f'{region_name} ({len(region_data):,} points)',
                      alpha=0.6, s=1)
    
    # Add regional boundaries
    ax.axvline(x=-5, color='red', linestyle='--', alpha=0.7, label='Atlantic Boundary (lon=-5)')
    ax.axvline(x=30, color='red', linestyle='--', alpha=0.7, label='Eastern Med Boundary (lon=30)')
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Spatial Distribution of Data Points by Region', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "spatial_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Temporal Analysis
    temporal_data = analysis_results["temporal_analysis"]
    
    # Create temporal heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert to pandas for easier heatmap plotting
    temporal_pandas = temporal_data.to_pandas()
    
    # Pivot data for heatmap
    heatmap_data = temporal_pandas.pivot_table(index=["year", "month"], columns="region", values="count", fill_value=0)
    
    # Create year-month index
    heatmap_data.index = [f"{year}-{month:02d}" for year, month in heatmap_data.index]
    
    # Plot heatmap
    sns.heatmap(heatmap_data.T, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax)
    ax.set_title('Data Points by Region and Time Period', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year-Month')
    ax.set_ylabel('Region')
    
    plt.tight_layout()
    plt.savefig(output_path / "temporal_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Summary Statistics Table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create summary table
    summary_data = []
    for row in region_counts.iter_rows(named=True):
        summary_data.append([
            row['region'].title(),
            f"{row['count']:,}",
            f"{row['percentage']:.1f}%",
            f"{row['mean_lat']:.2f}",
            f"{row['mean_lon']:.2f}",
            f"{row['std_lat']:.2f}",
            f"{row['std_lon']:.2f}"
        ])
    
    table = ax.table(cellText=summary_data,
                    colLabels=['Region', 'Count', 'Percentage', 'Mean Lat', 'Mean Lon', 'Std Lat', 'Std Lon'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(summary_data) + 1):
        for j in range(len(summary_data[0])):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f1f1f2' if i % 2 == 0 else 'white')
    
    ax.set_title('Regional Distribution Summary', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path / "summary_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {output_path}")

def print_analysis_summary(analysis_results: Dict):
    """Print a comprehensive summary of the analysis."""
    logger.info("="*60)
    logger.info("REGIONAL DISTRIBUTION ANALYSIS SUMMARY")
    logger.info("="*60)
    
    region_counts = analysis_results["region_counts"]
    total_points = analysis_results["total_points"]
    
    logger.info(f"Total data points analyzed: {total_points:,}")
    logger.info("")
    
    logger.info("Regional Distribution:")
    for row in region_counts.iter_rows(named=True):
        logger.info(f"  {row['region'].title():15} {row['count']:>8,} points ({row['percentage']:>5.1f}%)")
    
    logger.info("")
    
    # Check for potential issues
    logger.info("Potential Issues:")
    
    # Check for imbalanced regions
    max_percentage = region_counts["percentage"].max()
    min_percentage = region_counts["percentage"].min()
    
    if max_percentage > 80:
        logger.warning(f"  ⚠️  One region dominates: {max_percentage:.1f}% of data")
    
    if min_percentage < 5:
        logger.warning(f"  ⚠️  One region is underrepresented: {min_percentage:.1f}% of data")
    
    # Check Atlantic specifically
    atlantic_data = region_counts.filter(pl.col("region") == 0)  # atlantic = 0
    if len(atlantic_data) > 0:
        atlantic_pct = atlantic_data["percentage"].item()
        if atlantic_pct < 10:
            logger.warning(f"  ⚠️  Atlantic region has only {atlantic_pct:.1f}% of data - this explains poor Atlantic performance!")
        else:
            logger.info(f"  ✅ Atlantic region has {atlantic_pct:.1f}% of data")
    
    logger.info("")
    logger.info("Recommendations:")
    
    if min_percentage < 5:
        logger.info("  • Consider stratified sampling to balance regions")
        logger.info("  • Collect more data from underrepresented regions")
        logger.info("  • Use regional scaling to handle imbalanced data")
    
    if max_percentage > 80:
        logger.info("  • Consider downsampling the dominant region")
        logger.info("  • Use class weights in model training")
    
    logger.info("  • Monitor regional performance separately during evaluation")
    logger.info("  • Consider region-specific models if imbalance is severe")
    
    logger.info("="*60)

def main():
    """Main analysis function."""
    logger.info("Starting Regional Distribution Analysis...")
    
    # Load configuration
    config = load_config()
    data_path = config["data"]["data_path"]
    file_pattern = config["data"]["file_pattern"]
    
    logger.info(f"Data path: {data_path}")
    logger.info(f"File pattern: {file_pattern}")
    
    # Get data files
    file_paths = get_data_files(data_path, file_pattern)
    logger.info(f"Found {len(file_paths)} data files")
    
    if len(file_paths) == 0:
        logger.error("No data files found!")
        return
    
    # Load sample data (limit to first 10 files for analysis)
    df = load_sample_data(file_paths, max_files=1)
    
    # Apply regional classification
    df = classify_regions(df)
    
    # Analyze distribution
    analysis_results = analyze_regional_distribution(df)
    
    # Create visualizations
    create_visualizations(df, analysis_results)
    
    # Print summary
    print_analysis_summary(analysis_results)
    
    logger.info("Analysis complete! Check the 'regional_analysis' folder for visualizations.")

if __name__ == "__main__":
    main()
