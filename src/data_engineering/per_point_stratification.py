"""
Per-Point Wave Height Stratification

This module implements per-location stratification based on wave height bins,
ensuring every location contributes samples across all wave conditions.
"""

import logging
import polars as pl
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PerPointStratification:
    """Per-location wave height stratification for balanced training data."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize per-point stratification with configuration."""
        self.config = config
        self.feature_config = config.get("feature_block", {})
        self.logger = logging.getLogger(__name__)
        
        # Global target samples per bin (not per location) - 6 bins for aggressive high wave bias
        self.global_targets = self.feature_config.get("per_point_stratification_bins", {
            "calm": {"range": [0.0, 1.0], "global_target_percentage": 0.15},      # 15% (0-1m) - reduced
            "moderate": {"range": [1.0, 3.0], "global_target_percentage": 0.25},  # 25% (1-3m) - reduced
            "rough": {"range": [3.0, 6.0], "global_target_percentage": 0.30},     # 30% (3-6m) - same
            "high": {"range": [6.0, 9.0], "global_target_percentage": 0.20},      # 20% (6-9m) - new bin for high waves
            "extreme": {"range": [9.0, ".inf"], "global_target_percentage": 0.02, "keep_all": True}, # 2% (>9m) - KEEP ALL SAMPLES
        })
        
        # Extract wave bins from global targets
        self.wave_bins = {name: config["range"] for name, config in self.global_targets.items()}
        
        # Maximum total samples per file
        self.max_samples_per_file = self.feature_config.get("max_samples_per_file", 1000000)
        
        # Debug logging
        self.logger.info(f"PerPointStratification initialized with max_samples_per_file: {self.max_samples_per_file:,}")
        
        # Sampling seed
        self.sampling_seed = self.feature_config.get("sampling_seed", 42)
    
    def apply_per_point_stratification(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply per-point wave height stratification with global quotas and geographic balance.
        
        This approach ensures we always get enough high wave samples while maintaining
        geographic representation across all regions.
        
        Args:
            df: Input dataframe with columns ['lat', 'lon', 'vhm0_y', 'region', ...]
            
        Returns:
            Stratified dataframe with balanced wave height distribution and geographic coverage
        """
        if len(df) == 0:
            return df
        
        original_size = len(df)
        self.logger.info(f"Applying per-point wave height stratification with global quotas: {original_size:,} samples")
        
        # Add region mapping if not present
        if "region" not in df.columns:
            df = self._add_region_mapping(df)
        
        # Get unique locations and regions
        unique_locations = df.select(["lat", "lon"]).unique()
        unique_regions = df.select("region").unique().to_series().to_list()
        n_locations = len(unique_locations)
        
        self.logger.info(f"Found {n_locations:,} unique locations across {len(unique_regions)} regions")
        
        # Step 1: Sample from each wave height bin with global targets
        sampled_dfs = []
        
        for bin_name, bin_config in self.global_targets.items():
            min_h, max_h = bin_config["range"]
            target_percentage = bin_config["global_target_percentage"]
            global_target = int(self.max_samples_per_file * target_percentage)
            
            # Filter for the current wave height bin
            # Convert to float in case they come from YAML as strings
            min_h = float(min_h)
            if str(max_h) in [".inf", "float('inf')"] or max_h == float('inf'):
                max_h = float('inf')
            else:
                max_h = float(max_h)
            
            if bin_name == "extreme":
                bin_df = df.filter(pl.col("vhm0_y") >= min_h)
            else:
                bin_df = df.filter(
                    (pl.col("vhm0_y") >= min_h) & (pl.col("vhm0_y") < max_h)
                )
            
            available_samples = len(bin_df)
            self.logger.info(f"  - {bin_name} bin: {available_samples:,} available, target: {global_target:,} ({target_percentage*100:.0f}%)")
            
            if available_samples > 0:
                # Check if this bin should keep ALL samples (ultra_extreme or extreme bins)
                keep_all_samples = (
                    bin_name == "extreme" or 
                    bin_name == "ultra_extreme" or 
                    bin_config.get("keep_all", False)
                )
                
                if keep_all_samples:
                    actual_samples = available_samples  # Keep all samples
                    sampled_bin = bin_df  # No sampling needed
                    self.logger.info(f"    → Keeping ALL {actual_samples:,} {bin_name} samples (rare conditions)")
                else:
                    actual_samples = min(global_target, available_samples)
                    sampled_bin = bin_df.sample(n=actual_samples, seed=self.sampling_seed + hash(bin_name) % (2**32 - 1))
                    self.logger.info(f"    → Sampled {actual_samples:,} samples")
                
                sampled_dfs.append(sampled_bin)
            else:
                self.logger.warning(f"    → No samples available in {bin_name} bin")
        
        if sampled_dfs:
            # Step 2: Combine all sampled data
            combined_df = pl.concat(sampled_dfs)
            
            # Step 3: Ensure geographic balance by region
            stratified_df = self._ensure_geographic_balance(combined_df, unique_regions)
            
            final_size = len(stratified_df)
            reduction_percent = ((original_size - final_size) / original_size) * 100
            
            self.logger.info(f"Per-point stratification completed: {original_size:,} → {final_size:,} samples ({reduction_percent:.1f}% reduction)")
            
            # Log final distribution
            self._log_wave_height_distribution(stratified_df)
            self._log_regional_distribution(stratified_df)
            
            # Drop region column after sampling is complete (region was only needed for sampling/balancing)
            if "region" in stratified_df.columns:
                stratified_df = stratified_df.drop("region")
                self.logger.info("Dropped 'region' column after sampling - region was only used for geographic balance")
            
            return stratified_df
        else:
            self.logger.warning("No stratified samples found, returning original data")
            # Drop region column even if no sampling was done
            if "region" in df.columns:
                df = df.drop("region")
                self.logger.info("Dropped 'region' column - region was only used for sampling")
            return df
    
    def _add_region_mapping(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add region mapping based on lat/lon coordinates."""
        from src.commons.region_mapping import get_region_from_coordinates
        
        def map_region_from_coords(row) -> int:
            """Map a row with lat/lon to region ID."""
            lat, lon = row["lat"], row["lon"]
            return get_region_from_coordinates(lat, lon)
        
        return df.with_columns([
            pl.struct(["lat", "lon"]).map_elements(map_region_from_coords, return_dtype=pl.Int32).alias("region")
        ])
    
    def _add_wave_height_bins(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add wave height bin classification to dataframe."""
        def classify_wave_height(wave_height: float) -> str:
            for bin_name, (min_val, max_val) in self.wave_bins.items():
                # Convert to float in case they come from YAML as strings
                min_val = float(min_val)
                if str(max_val) in [".inf", "float('inf')"] or max_val == float('inf'):
                    max_val = float('inf')
                else:
                    max_val = float(max_val)
                
                if min_val <= wave_height < max_val:
                    return bin_name
            return "extreme"  # Fallback for very high waves
        
        return df.with_columns([
            pl.col("vhm0_y").map_elements(classify_wave_height, return_dtype=pl.Utf8).alias("wave_height_bin")
        ])
    
    def _ensure_geographic_balance(self, df: pl.DataFrame, unique_regions: list) -> pl.DataFrame:
        """
        Ensure geographic balance while preserving ALL extreme waves.
        
        Args:
            df: Combined sampled DataFrame
            unique_regions: List of unique region IDs
            
        Returns:
            Geographically balanced DataFrame with all extreme waves preserved
        """
        if len(unique_regions) <= 1:
            # No need to balance if only one region
            return df
        
        # First, identify and preserve ALL extreme waves - they should never be dropped
        # Get extreme threshold from configuration
        extreme_min = float(self.wave_bins["high"][0])
        extreme_waves = df.filter(pl.col("vhm0_y") >= extreme_min)
        non_extreme_waves = df.filter(pl.col("vhm0_y") < extreme_min)
        
        extreme_count = len(extreme_waves)
        non_extreme_count = len(non_extreme_waves)
        
        self.logger.info(f"  - Preserving ALL {extreme_count:,} extreme waves (vhm0_y >= {extreme_min})")
        self.logger.info(f"  - Balancing {non_extreme_count:,} non-extreme waves across regions")
        
        # Calculate remaining quota for non-extreme waves
        remaining_quota = self.max_samples_per_file
        samples_per_region = remaining_quota // len(unique_regions)
        
        self.logger.info(f"  - Geographic balance: {samples_per_region:,} non-extreme samples per region")
        
        balanced_dfs = [extreme_waves]  # Always include all extreme waves
        
        for region in unique_regions:
            region_df = non_extreme_waves.filter(pl.col("region") == region)
            region_samples = len(region_df)
            
            if region_samples > 0:
                # Sample from this region's non-extreme waves
                actual_samples = min(samples_per_region, region_samples)
                sampled_region = region_df.sample(n=actual_samples, seed=self.sampling_seed + hash(region) % (2**32 - 1))
                balanced_dfs.append(sampled_region)
                self.logger.info(f"    → Region {region}: {actual_samples:,} non-extreme samples (from {region_samples:,} available)")
            else:
                self.logger.warning(f"    → Region {region}: No non-extreme samples available")
        
        if len(balanced_dfs) > 1:  # More than just extreme waves
            return pl.concat(balanced_dfs)
        else:
            return extreme_waves  # Only extreme waves available
    
    def _log_wave_height_distribution(self, df: pl.DataFrame) -> None:
        """Log the final wave height distribution."""
        if len(df) == 0:
            return
        
        # Add temporary bin classification for logging
        df_log = self._add_wave_height_bins(df)
        
        # Count samples in each bin
        bin_counts = df_log.group_by("wave_height_bin").count().sort("wave_height_bin")
        
        total_samples = len(df_log)
        self.logger.info("Final wave height distribution:")
        
        for row in bin_counts.iter_rows(named=True):
            bin_name = row["wave_height_bin"]
            count = row["count"]
            percentage = (count / total_samples) * 100
            
            # Get wave height range for this bin
            min_val, max_val = self.wave_bins[bin_name]
            if max_val == float('inf'):
                range_str = f"{min_val}+m"
            else:
                range_str = f"{min_val}-{max_val}m"
            
            self.logger.info(f"  {bin_name:12} ({range_str:>8}): {count:6,} samples ({percentage:5.1f}%)")
    
    def _log_regional_distribution(self, df: pl.DataFrame) -> None:
        """Log the final regional distribution."""
        if len(df) == 0 or "region" not in df.columns:
            return
        
        # Count samples in each region
        regional_counts = df.group_by("region").count().sort("region")
        
        total_samples = len(df)
        self.logger.info("Final regional distribution:")
        
        for row in regional_counts.iter_rows(named=True):
            region = row["region"]
            count = row["count"]
            percentage = (count / total_samples) * 100
            self.logger.info(f"  Region {region}: {count:6,} samples ({percentage:5.1f}%)")
    
    def get_stratification_info(self) -> Dict[str, Any]:
        """Get stratification configuration information."""
        return {
            "strategy": "per_point_wave_height_stratification_global_quotas",
            "wave_height_bins": self.wave_bins,
            "global_targets": self.global_targets,
            "max_samples_per_file": self.max_samples_per_file,
            "sampling_seed": self.sampling_seed
        }
    
    def log_stratification_config(self) -> None:
        """Log the stratification configuration."""
        info = self.get_stratification_info()
        
        self.logger.info("Per-point wave height stratification configuration (global quotas):")
        self.logger.info(f"  - Max samples per file: {info['max_samples_per_file']:,}")
        self.logger.info(f"  - Sampling seed: {info['sampling_seed']}")
        
        self.logger.info("  - Wave height bins and global targets:")
        for bin_name, (min_val, max_val) in self.wave_bins.items():
            target_percentage = self.global_targets[bin_name]["global_target_percentage"]
            target_samples = int(self.max_samples_per_file * target_percentage)
            if max_val == float('inf'):
                range_str = f"{min_val}+m"
            else:
                range_str = f"{min_val}-{max_val}m"
            self.logger.info(f"    {bin_name:12} ({range_str:>8}): {target_samples:6,} samples ({target_percentage*100:4.0f}%)")


def apply_per_point_stratification(df: pl.DataFrame, config: Dict[str, Any]) -> pl.DataFrame:
    """
    Convenience function to apply per-point wave height stratification.
    
    Args:
        df: Input dataframe
        config: Configuration dictionary (should include updated max_samples_per_file)
        
    Returns:
        Stratified dataframe
    """
    # Create stratifier with the passed config (which should have the correct max_samples_per_file)
    stratifier = PerPointStratification(config)
    return stratifier.apply_per_point_stratification(df)
