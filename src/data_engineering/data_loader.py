"""
DataLoader for Wave Height Bias Correction Research

This module provides the DataLoader class for handling data loading, file processing,
and basic sampling operations in a modular and reusable way.
"""

import glob
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
import polars as pl
from tqdm import tqdm


# DataLoader worker function (needs to be at module level for multiprocessing)
def _load_single_file_worker_for_dataloader(args):
    """Worker function for parallel file loading in DataLoader."""
    file_path, feature_config, sampling_manager = args
    
    try:
        # Import everything at the top to avoid import issues
        import sys
        import os
        import logging
        
        # Add the project root to the path to ensure imports work
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Now import our modules
        from src.features.helpers import extract_features_from_parquet
        # Sampling functions removed - now handled by SamplingManager
        
        logger = logging.getLogger(__name__)
        
        # Load the file
        df = extract_features_from_parquet(file_path, use_dask=False)
        
        # Create lag features BEFORE sampling to preserve temporal sequences
        use_lag_features = feature_config.get("lag_features", {}).get("enabled", False)
        if use_lag_features:
            logger.debug(f"Creating lag features for {file_path} before sampling...")
            lag_config = feature_config.get("lag_features", {}).get("lags", {})
            
            # Check if we have temporal data (time column)
            if "time" in df.columns or "timestamp" in df.columns:
                time_col = "time" if "time" in df.columns else "timestamp"
                
                # Sort by time to ensure proper lag calculation
                df = df.sort([time_col, "lat", "lon"])
                
                # Create lag features for each variable
                for variable, lags in lag_config.items():
                    if variable in df.columns:
                        for lag in lags:
                            lag_col_name = f"{variable}_lag_{lag}h"
                            # Create lag feature by shifting values within each location
                            df = df.with_columns([
                                pl.col(variable).shift(lag).over(["lat", "lon"]).alias(lag_col_name)
                            ])
                    else:
                        logger.warning(f"Variable {variable} not found in data for lag features")
            else:
                logger.warning("Lag features enabled but no time column found. Skipping lag features.")
        
        # Apply per-file sampling using SamplingManager
        if sampling_manager is not None:
            df = sampling_manager.apply_sampling(df)
        
        return (file_path, df, True)
        
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Error loading file {file_path}: {e}")
        import traceback
        logging.getLogger(__name__).warning(f"Traceback: {traceback.format_exc()}")
        return (file_path, None, False)


class DataLoader:
    """Handles data loading, file processing, and basic sampling."""
    
    def __init__(self, config: Dict[str, Any], sampling_manager=None):
        """Initialize DataLoader with configuration and optional SamplingManager."""
        self.data_config = config.get("data", {})
        self.feature_config = config.get("feature_block", {})
        self.sampling_manager = sampling_manager
        self.logger = logging.getLogger(__name__)
    
    def load_data(self, data_paths: Union[str, List[str]]) -> Tuple[pl.DataFrame, List[str]]:
        """
        Load and combine all data files from local or S3 paths.
        
        Args:
            data_paths: Single path or list of paths to data files/directories
            
        Returns:
            Tuple of (combined_dataframe, successful_files)
        """
        # Handle single path vs list of paths
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        
        # Expand paths to get all files
        all_files = []
        individual_files_count = 0
        directory_paths_count = 0
        
        for path in data_paths:
            # If path ends with .parquet, it's already an individual file
            if path.endswith('.parquet'):
                all_files.append(path)
                individual_files_count += 1
            else:
                # It's a directory path, need to expand it
                files = self._expand_data_path(path)
                all_files.extend(files)
                directory_paths_count += 1
        
        if individual_files_count > 0:
            self.logger.info(f"Using {individual_files_count} individual file(s) directly (no S3 listing needed)")
        if directory_paths_count > 0:
            self.logger.info(f"Expanding {directory_paths_count} directory path(s) to find files")
        
        self.logger.info(f"Loading data from {len(all_files)} files...")
        
        # Load files (parallel or sequential)
        successful_files = []
        all_dataframes = []
        
        parallel_loading = self.data_config.get("parallel_loading", True)
        if parallel_loading and len(all_files) > 1:
            max_workers = self.data_config.get("max_workers", 4)
            self.logger.info(f"Loading files in parallel with {max_workers} workers")
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_file = {
                    executor.submit(_load_single_file_worker_for_dataloader, (file_path, self.feature_config, self.sampling_manager)): file_path 
                    for file_path in all_files
                }
                
                # Process completed tasks
                for future in tqdm(as_completed(future_to_file), total=len(all_files), desc="Loading files"):
                    file_path, df, success = future.result()
                    
                    if success and df is not None:
                        successful_files.append(file_path)
                        all_dataframes.append(df)
                    else:
                        self.logger.warning(f"Failed to load file: {file_path}")
        else:
            # Sequential loading
            self.logger.info("Loading files sequentially")
            for file_path in tqdm(all_files, desc="Loading files"):
                file_path, df, success = _load_single_file_worker_for_dataloader((file_path, self.feature_config, self.sampling_manager))
                
                if success and df is not None:
                    successful_files.append(file_path)
                    all_dataframes.append(df)
                else:
                    self.logger.warning(f"Failed to load file: {file_path}")
        
        if not all_dataframes:
            raise ValueError("No data files were successfully loaded!")
        
        self.logger.info(f"Successfully loaded {len(successful_files)} out of {len(all_files)} files")
        
        # Combine all dataframes
        self.logger.info("Combining all dataframes...")
        combined_df = pl.concat(all_dataframes)
        self.logger.info(f"Combined dataframe shape: {combined_df.shape}")
        
        # Log total distribution across all files
        self._log_total_distribution_summary(combined_df, successful_files)
        
        return combined_df, successful_files
    
    def _log_total_distribution_summary(self, combined_df: pl.DataFrame, successful_files: List[str]) -> None:
        """Log the total wave height distribution across all training files."""
        if len(combined_df) == 0:
            return
        
        # Check if we have wave height data
        wave_height_col = None
        for col_name in ["vhm0_y", "VHM0", "corrected_VHM0", "vhm0", "wave_height", "hs"]:
            if col_name in combined_df.columns:
                wave_height_col = col_name
                break
        
        if wave_height_col is None:
            self.logger.warning("No wave height column found for distribution summary")
            return
        
        # Get wave height bins from configuration
        wave_bins = self._get_wave_height_bins_from_config()
        
        # Count samples in each bin
        total_samples = len(combined_df)
        self.logger.info("=" * 80)
        self.logger.info("TOTAL WAVE HEIGHT DISTRIBUTION AFTER PER-POINT SAMPLING")
        self.logger.info("=" * 80)
        self.logger.info(f"Total samples across {len(successful_files)} files: {total_samples:,}")
        self.logger.info("")
        
        bin_counts = {}
        for bin_name, (min_val, max_val) in wave_bins.items():
            if max_val == float('inf'):
                bin_df = combined_df.filter(pl.col(wave_height_col) >= min_val)
            else:
                bin_df = combined_df.filter(
                    (pl.col(wave_height_col) >= min_val) & (pl.col(wave_height_col) < max_val)
                )
            
            count = len(bin_df)
            percentage = (count / total_samples) * 100
            bin_counts[bin_name] = count
            
            if max_val == float('inf'):
                range_str = f"{min_val}+m"
            else:
                range_str = f"{min_val}-{max_val}m"
            
            self.logger.info(f"  {bin_name:12} ({range_str:>8}): {count:8,} samples ({percentage:5.1f}%)")
        
        # Log extreme wave statistics
        extreme_count = bin_counts.get("extreme", 0)
        if extreme_count > 0:
            extreme_df = combined_df.filter(pl.col(wave_height_col) >= 9.0)
            max_wave = float(extreme_df.select(pl.col(wave_height_col).max()).item())
            mean_extreme = float(extreme_df.select(pl.col(wave_height_col).mean()).item())
            
            self.logger.info("")
            self.logger.info(f"  Extreme wave statistics:")
            self.logger.info(f"    Maximum wave height: {max_wave:.2f}m")
            self.logger.info(f"    Mean extreme wave: {mean_extreme:.2f}m")
        
        # Log regional distribution if coordinates available
        lat_col = None
        lon_col = None
        for col_name in ["lat", "latitude", "LAT", "LATITUDE"]:
            if col_name in combined_df.columns:
                lat_col = col_name
                break
        for col_name in ["lon", "longitude", "LON", "LONGITUDE"]:
            if col_name in combined_df.columns:
                lon_col = col_name
                break
        
        if lat_col and lon_col:
            self.logger.info("")
            self.logger.info("  Regional distribution:")
            
            # Simple regional classification
            atlantic = combined_df.filter(pl.col(lon_col) < -6.0)
            mediterranean = combined_df.filter((pl.col(lon_col) >= -6.0) & (pl.col(lon_col) <= 25.0))
            eastern_med = combined_df.filter(pl.col(lon_col) > 25.0)
            
            for region_name, region_df in [("Atlantic", atlantic), ("Mediterranean", mediterranean), ("Eastern Med", eastern_med)]:
                count = len(region_df)
                percentage = (count / total_samples) * 100
                self.logger.info(f"    {region_name:15}: {count:8,} samples ({percentage:5.1f}%)")
        
        self.logger.info("=" * 80)
    
    def _get_wave_height_bins_from_config(self) -> Dict[str, Tuple[float, float]]:
        """Get wave height bins from configuration."""
        # Get stratification bins from feature config
        stratification_bins = self.feature_config.get("per_point_stratification_bins", {})
        
        if not stratification_bins:
            # Fallback to default bins if not configured
            self.logger.warning("No per_point_stratification_bins found in config, using default bins")
            return {
                "calm": (0.0, 1.0),
                "moderate": (1.0, 3.0), 
                "rough": (3.0, 6.0),
                "high": (6.0, 9.0),
                "extreme": (9.0, float('inf'))
            }
        
        # Convert config bins to the format expected by logging
        wave_bins = {}
        for bin_name, bin_config in stratification_bins.items():
            min_val, max_val = bin_config["range"]
            
            # Convert to float in case they come from YAML as strings
            min_val = float(min_val)
            if str(max_val) in [".inf", "float('inf')"] or max_val == float('inf'):
                max_val = float('inf')
            else:
                max_val = float(max_val)
            
            wave_bins[bin_name] = (min_val, max_val)
        
        return wave_bins
    
    def _expand_data_path(self, path: str) -> List[str]:
        """Expand data paths (S3, local, glob patterns)."""
        if path.startswith('s3://'):
            return self._expand_s3_path(path)
        else:
            return self._expand_local_path(path)
    
    def _expand_s3_path(self, s3_uri: str) -> List[str]:
        """Expand S3 path to list of parquet files."""
        self.logger.info(f"Listing S3 files from: {s3_uri}")
        
        try:
            from src.commons.aws.utils import list_s3_parquet_files
            
            # Parse S3 URI
            s3_path = s3_uri[5:]  # Remove 's3://'
            if '/' in s3_path:
                bucket, prefix = s3_path.split('/', 1)
            else:
                bucket = s3_path
                prefix = ""
            
            # Get S3 configuration
            s3_config = self.data_config.get("s3", {})
            aws_profile = s3_config.get("aws_profile", None)
            
            # Get year-based filtering configuration
            split_config = self.data_config.get("split", {})
            eval_months = split_config.get("eval_months", None)
            train_end_year = split_config.get("train_end_year", None)
            test_start_year = split_config.get("test_start_year", None)
            
            self.logger.info(f"Listing S3 files in bucket: {bucket}, prefix: {prefix}")
            self.logger.info(f"Year-based filtering: train_end_year={train_end_year}, test_start_year={test_start_year}")
            if eval_months:
                self.logger.info(f"Month filtering for test years: {eval_months}")
            
            # List parquet files with year-aware filtering
            files = list_s3_parquet_files(bucket, prefix, aws_profile, eval_months, train_end_year, test_start_year)
            
            self.logger.info(f"Found {len(files)} parquet files in S3")
            return files
        except Exception as e:
            self.logger.error(f"Error listing S3 files from {s3_uri}: {e}")
            return []
    
    def _expand_local_path(self, local_path: str) -> List[str]:
        """Expand local path to list of parquet files."""
        if Path(local_path).is_file():
            return [local_path]
        elif Path(local_path).is_dir():
            files = list(Path(local_path).glob("*.parquet"))
            return [str(f) for f in sorted(files)]
        else:
            # Try glob pattern
            files = glob.glob(local_path)
            parquet_files = [f for f in files if f.endswith('.parquet')]
            return sorted(parquet_files)