"""
SamplingManager for Wave Height Bias Correction Research

This module provides the SamplingManager class for handling different sampling strategies
including random, per-location, and temporal sampling operations.
"""

import logging
from typing import Dict, Any, Optional
import polars as pl

from src.data_engineering.split import (
    stratified_sample_by_location,
    random_sample_within_file,
    temporal_sample_within_file
)


class SamplingManager:
    """Handles different sampling strategies for data preprocessing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize SamplingManager with configuration."""
        self.feature_config = config.get("feature_block", {})
        self.logger = logging.getLogger(__name__)
    
    def apply_sampling(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply sampling strategy if configured.
        
        Args:
            df: Input dataframe to sample
            
        Returns:
            Sampled dataframe
        """
        max_samples = self.feature_config.get("max_samples_per_file", None)
        sampling_strategy = self.feature_config.get("sampling_strategy", "none")
        sampling_seed = self.feature_config.get("sampling_seed", 42)
        
        # Check if regional training is enabled
        regional_training = self.feature_config.get("regional_training", {})
        regional_training_enabled = regional_training.get("enabled", False)
        
        # If robust_stratified sampling is requested with regional training,
        # use single-region robust sampling instead of multi-region robust sampling
        if sampling_strategy == "robust_stratified" and regional_training_enabled:
            self.logger.info(
                "Regional training enabled with robust_stratified sampling. "
                "Using single-region robust sampling to maintain wave height stratification within selected region(s)."
            )
            sampling_strategy = "single_region_robust"
        
        original_size = len(df)
        
        if max_samples is None or sampling_strategy == "none":
            self.logger.debug(f"No sampling applied - using all {original_size:,} samples")
            return df
        
        self.logger.info(f"Applying {sampling_strategy} sampling: max {max_samples:,} samples from {original_size:,} original samples")
        
        if sampling_strategy == "per_location":
            sampled_df = self._apply_per_location_sampling(df, max_samples, sampling_seed)
        elif sampling_strategy == "temporal":
            sampled_df = self._apply_temporal_sampling(df, max_samples, sampling_seed)
        elif sampling_strategy == "random":
            sampled_df = self._apply_random_sampling(df, max_samples, sampling_seed)
        elif sampling_strategy == "robust_stratified":
            sampled_df = self._apply_robust_stratified_sampling(df, max_samples, sampling_seed)
        elif sampling_strategy == "single_region_robust":
            sampled_df = self._apply_single_region_robust_sampling(df, max_samples, sampling_seed)
        else:
            self.logger.warning(f"Unknown sampling strategy: {sampling_strategy}")
            return df
        
        final_size = len(sampled_df)
        reduction_percent = ((original_size - final_size) / original_size) * 100
        
        self.logger.info(f"Sampling completed: {original_size:,} → {final_size:,} samples ({reduction_percent:.1f}% reduction)")
        
        return sampled_df
    
    def _apply_per_location_sampling(self, df: pl.DataFrame, max_samples: int, sampling_seed: int) -> pl.DataFrame:
        """Apply per-location stratified sampling."""
        samples_per_location = self.feature_config.get("samples_per_location", 20)
        self.logger.info(f"  - Per-location sampling: {samples_per_location} samples per location")
        
        # Count unique locations for logging
        unique_locations = df.select(["lat", "lon"]).unique().height
        self.logger.info(f"  - Found {unique_locations:,} unique locations")
        
        sampled_df = stratified_sample_by_location(
            df, 
            max_samples_per_file=max_samples,
            samples_per_location=samples_per_location,
            seed=sampling_seed,
            location_cols=["lat", "lon"]
        )
        
        return sampled_df
    
    def _apply_temporal_sampling(self, df: pl.DataFrame, max_samples: int, sampling_seed: int) -> pl.DataFrame:
        """Apply temporal sampling."""
        samples_per_hour = self.feature_config.get("samples_per_hour", 100)
        self.logger.info(f"  - Temporal sampling: {samples_per_hour} samples per hour")
        
        # Count unique hours for logging
        if "time" in df.columns:
            unique_hours = df.select("time").unique().height
            self.logger.info(f"  - Found {unique_hours:,} unique time points")
        
        sampled_df = temporal_sample_within_file(
            df,
            max_samples_per_file=max_samples,
            samples_per_hour=samples_per_hour,
            seed=sampling_seed
        )
        
        return sampled_df
    
    def _apply_random_sampling(self, df: pl.DataFrame, max_samples: int, sampling_seed: int) -> pl.DataFrame:
        """Apply random sampling."""
        self.logger.info(f"  - Random sampling: selecting {max_samples:,} random samples")
        sampled_df = random_sample_within_file(
            df,
            max_samples_per_file=max_samples,
            seed=sampling_seed
        )
        
        return sampled_df
    
    def _apply_robust_stratified_sampling(self, df: pl.DataFrame, max_samples: int, sampling_seed: int) -> pl.DataFrame:
        """
        Apply robust stratified sampling with basin quotas and percentile-based wave height bins.
        
        Basin quotas per day (N≈40k):
        - Atlantic: 18k (45%)
        - Mediterranean: 18k (45%) 
        - East Med: 4k (10%)
        
        Within each basin (per day):
        - Low (≤q33 for that basin/day): 30%
        - Mid (q33–q66): 30%
        - High (>q66): 40%, with bias to rough/very-rough:
          - Rough (2.5–4.0 m): ~25%
          - Very-rough (>4.0 m): ~15% (or more if available)
        
        Hour coverage: within each (basin × bin), sample approximately uniformly across 24 hours.
        """
        self.logger.info("  - Robust stratified sampling: basin quotas with percentile-based wave height bins")
        
        # Define basin quotas (assuming max_samples ≈ 40k)
        basin_quotas = {
            "atlantic": int(max_samples * 0.45),      # 45%
            "mediterranean": int(max_samples * 0.45), # 45%
            "eastern_med": int(max_samples * 0.10)    # 10%
        }
        
        # Create basin classification
        df = df.with_columns([
            pl.when(pl.col("lon") < -5.0)
            .then(pl.lit("atlantic"))
            .when(pl.col("lon") <= 30.0)
            .then(pl.lit("mediterranean"))
            .otherwise(pl.lit("eastern_med"))
            .alias("basin")
        ])
        
        # Add hour column - REQUIRED for robust stratified sampling
        if "time" in df.columns:
            df = df.with_columns([
                pl.col("time").dt.hour().alias("hour")
            ])
            self.logger.debug("Using 'time' column for hourly coverage")
        elif "timestamp" in df.columns:
            df = df.with_columns([
                pl.col("timestamp").dt.hour().alias("hour")
            ])
            self.logger.debug("Using 'timestamp' column for hourly coverage")
        else:
            # Robust stratified sampling requires temporal information for hourly coverage
            raise ValueError(
                "Robust stratified sampling requires temporal information (time or timestamp column) "
                "for hourly coverage. No time/timestamp column found in data. "
                "Please ensure temporal data is preserved during data loading and feature engineering."
            )
        
        # Log time information availability
        time_col = "time" if "time" in df.columns else "timestamp"
        unique_hours = df[time_col].dt.hour().unique().sort()
        self.logger.info(f"  - Time information available: {len(unique_hours)} unique hours")
        
        # Log original distribution
        self._log_basin_distribution(df, "Original")
        
        # Apply basin-wise stratified sampling
        sampled_dfs = []
        
        for basin_name, basin_quota in basin_quotas.items():
            basin_df = df.filter(pl.col("basin") == basin_name)
            
            if len(basin_df) > 0:
                self.logger.info(f"  - Processing {basin_name}: {len(basin_df):,} samples → target {basin_quota:,}")
                
                # Apply percentile-based sampling within basin
                basin_sampled = self._sample_basin_with_percentiles(
                    basin_df, basin_quota, sampling_seed, basin_name
                )
                
                if len(basin_sampled) > 0:
                    sampled_dfs.append(basin_sampled)
                    self.logger.info(f"    {basin_name}: {len(basin_df):,} → {len(basin_sampled):,} samples")
        
        # Combine all sampled data
        if sampled_dfs:
            sampled_df = pl.concat(sampled_dfs)
            # Remove temporary columns
            sampled_df = sampled_df.drop(["basin", "hour"])
        else:
            self.logger.warning("No samples found for any basin, returning original data")
            sampled_df = df.drop(["basin", "hour"])
        
        # Log final distribution
        self._log_basin_distribution(sampled_df, "Sampled")
        
        self.logger.info(f"  - Robust stratified sampling: {len(df)} → {len(sampled_df)} samples")
        
        return sampled_df
    
    def _sample_basin_with_percentiles(self, basin_df: pl.DataFrame, basin_quota: int, sampling_seed: int, basin_name: str) -> pl.DataFrame:
        """
        Sample within a basin using percentile-based wave height bins and hourly coverage.
        
        Within each basin:
        - Low (≤q33): 30%
        - Mid (q33–q66): 30% 
        - High (>q66): 40%, with bias to rough/very-rough:
          - Rough (2.5–4.0 m): ~25%
          - Very-rough (>4.0 m): ~15%
        """
        if len(basin_df) == 0:
            return basin_df
        
        # Calculate percentiles for this basin
        q33 = basin_df["vhm0_y"].quantile(0.33)
        q66 = basin_df["vhm0_y"].quantile(0.66)
        
        self.logger.debug(f"    {basin_name} percentiles: q33={q33:.3f}, q66={q66:.3f}")
        
        # Define wave height bins within basin
        low_df = basin_df.filter(pl.col("vhm0_y") <= q33)
        mid_df = basin_df.filter((pl.col("vhm0_y") > q33) & (pl.col("vhm0_y") <= q66))
        high_df = basin_df.filter(pl.col("vhm0_y") > q66)
        
        # Calculate target samples for each bin
        low_target = int(basin_quota * 0.30)    # 30%
        mid_target = int(basin_quota * 0.30)    # 30%
        high_target = int(basin_quota * 0.40)   # 40%
        
        self.logger.debug(f"    {basin_name} targets: low={low_target}, mid={mid_target}, high={high_target}")
        
        # Sample from each bin
        sampled_dfs = []
        
        # Low bin (30%)
        if len(low_df) > 0:
            low_sampled = self._sample_with_hourly_coverage(
                low_df, low_target, sampling_seed, f"{basin_name}_low"
            )
            sampled_dfs.append(low_sampled)
        
        # Mid bin (30%)
        if len(mid_df) > 0:
            mid_sampled = self._sample_with_hourly_coverage(
                mid_df, mid_target, sampling_seed, f"{basin_name}_mid"
            )
            sampled_dfs.append(mid_sampled)
        
        # High bin (40%) with bias to rough/very-rough
        if len(high_df) > 0:
            high_sampled = self._sample_high_bin_with_rough_bias(
                high_df, high_target, sampling_seed, basin_name
            )
            sampled_dfs.append(high_sampled)
        
        # Combine all samples from this basin
        if sampled_dfs:
            return pl.concat(sampled_dfs)
        else:
            return basin_df.limit(0)  # Return empty dataframe with same schema
    
    def _sample_high_bin_with_rough_bias(self, high_df: pl.DataFrame, high_target: int, sampling_seed: int, basin_name: str) -> pl.DataFrame:
        """
        Sample from high wave height bin with bias toward rough/very-rough conditions.
        
        Target distribution within high bin:
        - Rough (2.5–4.0 m): ~25% of high_target
        - Very-rough (>4.0 m): ~15% of high_target  
        - Other high: remaining ~60%
        """
        rough_target = int(high_target * 0.25)      # 25%
        very_rough_target = int(high_target * 0.15) # 15%
        other_high_target = high_target - rough_target - very_rough_target  # ~60%
        
        sampled_dfs = []
        
        # Sample rough conditions (2.5-4.0 m)
        rough_df = high_df.filter((pl.col("vhm0_y") >= 2.5) & (pl.col("vhm0_y") <= 4.0))
        if len(rough_df) > 0:
            rough_sampled = self._sample_with_hourly_coverage(
                rough_df, rough_target, sampling_seed, f"{basin_name}_rough"
            )
            sampled_dfs.append(rough_sampled)
        
        # Sample very-rough conditions (>4.0 m)
        very_rough_df = high_df.filter(pl.col("vhm0_y") > 4.0)
        if len(very_rough_df) > 0:
            very_rough_sampled = self._sample_with_hourly_coverage(
                very_rough_df, very_rough_target, sampling_seed, f"{basin_name}_very_rough"
            )
            sampled_dfs.append(very_rough_sampled)
        
        # Sample remaining high conditions
        other_high_df = high_df.filter((pl.col("vhm0_y") > 2.0) & (pl.col("vhm0_y") < 2.5))
        if len(other_high_df) > 0:
            other_high_sampled = self._sample_with_hourly_coverage(
                other_high_df, other_high_target, sampling_seed, f"{basin_name}_other_high"
            )
            sampled_dfs.append(other_high_sampled)
        
        # If we don't have enough rough/very-rough, fill with other high conditions
        total_sampled = sum(len(df) for df in sampled_dfs)
        if total_sampled < high_target and len(high_df) > total_sampled:
            remaining_target = high_target - total_sampled
            
            # Get all sampled values to exclude them
            if sampled_dfs:
                try:
                    # Concatenate all sampled dataframes to get the values to exclude
                    all_sampled = pl.concat(sampled_dfs)
                    sampled_values = all_sampled.select("vhm0_y").to_series().to_list()
                    remaining_df = high_df.filter(~pl.col("vhm0_y").is_in(sampled_values))
                except Exception as e:
                    self.logger.warning(f"    Error filtering sampled values: {e}, using all remaining data")
                    remaining_df = high_df
            else:
                remaining_df = high_df
                
            if len(remaining_df) > 0:
                remaining_sampled = self._sample_with_hourly_coverage(
                    remaining_df, remaining_target, sampling_seed, f"{basin_name}_remaining"
                )
                sampled_dfs.append(remaining_sampled)
        
        if sampled_dfs:
            return pl.concat(sampled_dfs)
        else:
            return high_df.limit(0)
    
    def _sample_with_hourly_coverage(self, df: pl.DataFrame, target: int, sampling_seed: int, stratum_name: str) -> pl.DataFrame:
        """
        Sample with approximately uniform coverage across 24 hours.
        """
        if len(df) == 0 or target == 0:
            return df.limit(0)
        
        # If we have fewer samples than target, return all
        if len(df) <= target:
            return df
        
        # Get unique hours in this stratum
        unique_hours = df["hour"].unique().sort()
        n_hours = len(unique_hours)
        
        if n_hours == 0:
            # This should not happen since we require time information
            self.logger.warning(f"    {stratum_name}: No hour information found, using random sampling")
            return df.sample(n=target, seed=sampling_seed)
        
        # Calculate samples per hour
        samples_per_hour = target // n_hours
        remaining_samples = target % n_hours
        
        sampled_dfs = []
        total_sampled = 0
        
        for i, hour in enumerate(unique_hours):
            hour_df = df.filter(pl.col("hour") == hour)
            
            # Calculate target for this hour
            hour_target = samples_per_hour
            if i < remaining_samples:  # Distribute remaining samples
                hour_target += 1
            
            if len(hour_df) > 0:
                actual_target = min(hour_target, len(hour_df))
                hour_sampled = hour_df.sample(n=actual_target, seed=sampling_seed + hour)
                sampled_dfs.append(hour_sampled)
                total_sampled += actual_target
        
        if sampled_dfs:
            return pl.concat(sampled_dfs)
        else:
            return df.limit(0)
    
    def _calculate_target_samples_per_stratum(self, df: pl.DataFrame, max_samples: int) -> Dict[str, int]:
        """Calculate target number of samples per stratum."""
        # Count samples in each stratum
        stratum_counts = {}
        total_original = len(df)
        
        for wave_bin in ["calm", "moderate", "rough", "very_rough"]:
            for region_bin in ["atlantic", "mediterranean", "eastern_med"]:
                stratum_key = f"{region_bin}_{wave_bin}"
                count = len(df.filter(
                    (pl.col("wave_height_bin") == wave_bin) & 
                    (pl.col("region_bin") == region_bin)
                ))
                stratum_counts[stratum_key] = count
        
        # Calculate target samples per stratum
        target_samples_per_stratum = {}
        
        # Priority 1: Ensure minimum representation of high waves in Atlantic
        atlantic_high_waves = ["atlantic_rough", "atlantic_very_rough"]
        min_atlantic_high = max_samples // 20  # At least 5% for Atlantic high waves
        
        for stratum in atlantic_high_waves:
            if stratum_counts[stratum] > 0:
                target_samples_per_stratum[stratum] = min(
                    min_atlantic_high,
                    stratum_counts[stratum]
                )
        
        # Priority 2: Balance remaining samples across all strata
        remaining_samples = max_samples - sum(target_samples_per_stratum.values())
        non_zero_strata = {k: v for k, v in stratum_counts.items() if v > 0}
        
        if remaining_samples > 0 and non_zero_strata:
            # Proportional allocation of remaining samples
            total_available = sum(non_zero_strata.values())
            
            for stratum, count in non_zero_strata.items():
                if stratum not in target_samples_per_stratum:
                    proportional_allocation = int((count / total_available) * remaining_samples)
                    target_samples_per_stratum[stratum] = min(proportional_allocation, count)
        
        return target_samples_per_stratum
    
    def _log_basin_distribution(self, df: pl.DataFrame, stage: str):
        """Log the distribution of samples across basins and wave height ranges."""
        self.logger.info(f"  - {stage} distribution:")
        
        # Create temporary basin classification for logging
        df_log = df.with_columns([
            pl.when(pl.col("lon") < -5.0)
            .then(pl.lit("atlantic"))
            .when(pl.col("lon") <= 30.0)
            .then(pl.lit("mediterranean"))
            .otherwise(pl.lit("eastern_med"))
            .alias("basin")
        ])
        
        # Log basin distribution
        total_samples = len(df_log)
        for basin in ["atlantic", "mediterranean", "eastern_med"]:
            basin_df = df_log.filter(pl.col("basin") == basin)
            if len(basin_df) > 0:
                basin_percent = (len(basin_df) / total_samples) * 100
                
                # Calculate wave height statistics for this basin
                mean_wave = basin_df["vhm0_y"].mean()
                max_wave = basin_df["vhm0_y"].max()
                rough_count = len(basin_df.filter((pl.col("vhm0_y") >= 2.5) & (pl.col("vhm0_y") <= 4.0)))
                very_rough_count = len(basin_df.filter(pl.col("vhm0_y") > 4.0))
                
                self.logger.info(f"    {basin}: {len(basin_df):,} samples ({basin_percent:.1f}%) - "
                               f"mean: {mean_wave:.2f}m, max: {max_wave:.2f}m, "
                               f"rough: {rough_count:,}, very_rough: {very_rough_count:,}")
    
    def get_sampling_info(self) -> Dict[str, Any]:
        """Get current sampling configuration information."""
        max_samples = self.feature_config.get("max_samples_per_file", None)
        sampling_strategy = self.feature_config.get("sampling_strategy", "none")
        sampling_seed = self.feature_config.get("sampling_seed", 42)
        
        sampling_info = {
            "enabled": max_samples is not None and sampling_strategy != "none",
            "strategy": sampling_strategy,
            "max_samples_per_file": max_samples,
            "seed": sampling_seed
        }
        
        if sampling_strategy == "per_location":
            sampling_info["samples_per_location"] = self.feature_config.get("samples_per_location", 20)
        elif sampling_strategy == "temporal":
            sampling_info["samples_per_hour"] = self.feature_config.get("samples_per_hour", 100)
        elif sampling_strategy == "robust_stratified":
            sampling_info["basin_quotas"] = {
                "atlantic": 45,      # 45%
                "mediterranean": 45, # 45%
                "eastern_med": 10    # 10%
            }
            sampling_info["wave_height_distribution"] = {
                "low_percentile": 30,    # ≤q33: 30%
                "mid_percentile": 30,    # q33-q66: 30%
                "high_percentile": 40    # >q66: 40%
            }
            sampling_info["high_wave_bias"] = {
                "rough_2_5_4_0m": 25,    # 25% of high bin
                "very_rough_4_0m_plus": 15  # 15% of high bin
            }
            sampling_info["hourly_coverage"] = "uniform_across_24h"
        
        return sampling_info
    
    def log_sampling_config(self):
        """Log the current sampling configuration."""
        sampling_info = self.get_sampling_info()
        
        if sampling_info["enabled"]:
            self.logger.info(f"Sampling configuration:")
            self.logger.info(f"  - Strategy: {sampling_info['strategy']}")
            self.logger.info(f"  - Max samples per file: {sampling_info['max_samples_per_file']:,}")
            self.logger.info(f"  - Seed: {sampling_info['seed']}")
            
            if sampling_info["strategy"] == "per_location":
                self.logger.info(f"  - Samples per location: {sampling_info['samples_per_location']}")
            elif sampling_info["strategy"] == "temporal":
                self.logger.info(f"  - Samples per hour: {sampling_info['samples_per_hour']}")
            elif sampling_info["strategy"] == "robust_stratified":
                self.logger.info(f"  - Basin quotas: {sampling_info['basin_quotas']}")
                self.logger.info(f"  - Wave height distribution: {sampling_info['wave_height_distribution']}")
                self.logger.info(f"  - High wave bias: {sampling_info['high_wave_bias']}")
                self.logger.info(f"  - Hourly coverage: {sampling_info['hourly_coverage']}")
        else:
            self.logger.info("No sampling configured - using all available data")
    
    def estimate_sampling_reduction(self, total_files: int, total_samples: int) -> Dict[str, Any]:
        """
        Estimate sampling reduction for logging purposes.
        
        Args:
            total_files: Total number of files processed
            total_samples: Total samples after sampling
            
        Returns:
            Dictionary with sampling reduction information
        """
        sampling_info = self.get_sampling_info()
        
        if not sampling_info["enabled"]:
            return {"reduction_percent": 0, "estimated_original": total_samples}
        
        max_per_file = sampling_info["max_samples_per_file"]
        estimated_original = total_files * max_per_file
        reduction_percent = ((estimated_original - total_samples) / estimated_original) * 100
        
        return {
            "reduction_percent": reduction_percent,
            "estimated_original": estimated_original,
            "strategy": sampling_info["strategy"],
            "max_per_file": max_per_file
        }
    
    def _apply_single_region_robust_sampling(self, df: pl.DataFrame, max_samples: int, sampling_seed: int) -> pl.DataFrame:
        """
        Apply robust stratified sampling within a single region (for regional training).
        
        This maintains the same wave height stratification and temporal coverage benefits
        as robust_stratified sampling, but works within the selected region(s) only.
        
        Wave height bins (percentile-based):
        - Low (≤q33): 30%
        - Mid (q33–q66): 30% 
        - High (>q66): 40%, with bias to rough/very-rough:
          - Rough (2.5–4.0 m): ~25%
          - Very-rough (>4.0 m): ~15%
        
        Hourly coverage: Uniform sampling across 24 hours within each bin.
        """
        if len(df) == 0:
            return df
        
        # Add hour column - REQUIRED for single-region robust sampling
        if "time" in df.columns:
            df = df.with_columns([
                pl.col("time").dt.hour().alias("hour")
            ])
            self.logger.debug("Using 'time' column for hourly coverage")
        elif "timestamp" in df.columns:
            df = df.with_columns([
                pl.col("timestamp").dt.hour().alias("hour")
            ])
            self.logger.debug("Using 'timestamp' column for hourly coverage")
        else:
            # Single-region robust sampling requires temporal information for hourly coverage
            raise ValueError(
                "Single-region robust sampling requires temporal information (time or timestamp column) "
                "for hourly coverage. No time/timestamp column found in data. "
                "Please ensure temporal data is preserved during data loading and feature engineering."
            )
        
        self.logger.info(f"  - Single-region robust sampling: {len(df):,} samples → target {max_samples:,}")
        
        # Calculate percentiles for the entire dataset (single region)
        q33 = df["vhm0_y"].quantile(0.33)
        q66 = df["vhm0_y"].quantile(0.66)
        
        self.logger.info(f"  - Wave height percentiles: q33={q33:.3f}, q66={q66:.3f}")
        
        # Define wave height bins
        low_df = df.filter(pl.col("vhm0_y") <= q33)
        mid_df = df.filter((pl.col("vhm0_y") > q33) & (pl.col("vhm0_y") <= q66))
        high_df = df.filter(pl.col("vhm0_y") > q66)
        
        self.logger.info(f"  - Wave height bins: Low={len(low_df):,}, Mid={len(mid_df):,}, High={len(high_df):,}")
        
        # Calculate target samples per bin (same proportions as multi-region robust)
        low_target = int(max_samples * 0.30)  # 30%
        mid_target = int(max_samples * 0.30)  # 30%
        high_target = int(max_samples * 0.40)  # 40%
        
        self.logger.info(f"  - Target samples: Low={low_target:,}, Mid={mid_target:,}, High={high_target:,}")
        
        # Sample each bin with hourly coverage
        sampled_dfs = []
        
        # Sample low bin
        if len(low_df) > 0:
            low_sampled = self._sample_with_hourly_coverage(low_df, low_target, sampling_seed, "Low")
            if len(low_sampled) > 0:
                sampled_dfs.append(low_sampled)
                self.logger.info(f"    Low bin: {len(low_df):,} → {len(low_sampled):,} samples")
        
        # Sample mid bin
        if len(mid_df) > 0:
            mid_sampled = self._sample_with_hourly_coverage(mid_df, mid_target, sampling_seed, "Mid")
            if len(mid_sampled) > 0:
                sampled_dfs.append(mid_sampled)
                self.logger.info(f"    Mid bin: {len(mid_df):,} → {len(mid_sampled):,} samples")
        
        # Sample high bin with rough/very-rough bias
        if len(high_df) > 0:
            high_sampled = self._sample_high_bin_with_rough_bias(high_df, high_target, sampling_seed, sampled_dfs)
            if len(high_sampled) > 0:
                sampled_dfs.append(high_sampled)
                self.logger.info(f"    High bin: {len(high_df):,} → {len(high_sampled):,} samples")
        
        # Combine all sampled data
        if sampled_dfs:
            sampled_df = pl.concat(sampled_dfs)
            # Remove temporary hour column
            sampled_df = sampled_df.drop(["hour"])
        else:
            self.logger.warning("No samples found for any wave height bin, returning original data")
            sampled_df = df.drop(["hour"])
        
        # Log final wave height distribution
        self._log_wave_height_distribution(sampled_df, "Single-region sampled")
        
        self.logger.info(f"  - Single-region robust sampling: {len(df)} → {len(sampled_df)} samples")
        
        return sampled_df
    
    def _log_wave_height_distribution(self, df: pl.DataFrame, prefix: str) -> None:
        """Log wave height distribution for single-region sampling."""
        if len(df) == 0:
            return
        
        # Calculate percentiles
        q33 = df["vhm0_y"].quantile(0.33)
        q66 = df["vhm0_y"].quantile(0.66)
        
        # Count samples in each bin
        low_count = len(df.filter(pl.col("vhm0_y") <= q33))
        mid_count = len(df.filter((pl.col("vhm0_y") > q33) & (pl.col("vhm0_y") <= q66)))
        high_count = len(df.filter(pl.col("vhm0_y") > q66))
        
        # Count rough/very-rough in high bin
        rough_count = len(df.filter((pl.col("vhm0_y") > q66) & (pl.col("vhm0_y") >= 2.5) & (pl.col("vhm0_y") < 4.0)))
        very_rough_count = len(df.filter((pl.col("vhm0_y") > q66) & (pl.col("vhm0_y") >= 4.0)))
        
        total = len(df)
        
        self.logger.info(f"  - {prefix} wave height distribution:")
        self.logger.info(f"    Low (≤{q33:.3f}): {low_count:,} ({low_count/total*100:.1f}%)")
        self.logger.info(f"    Mid ({q33:.3f}-{q66:.3f}): {mid_count:,} ({mid_count/total*100:.1f}%)")
        self.logger.info(f"    High (>{q66:.3f}): {high_count:,} ({high_count/total*100:.1f}%)")
        self.logger.info(f"      Rough (2.5-4.0m): {rough_count:,} ({rough_count/total*100:.1f}%)")
        self.logger.info(f"      Very-rough (>4.0m): {very_rough_count:,} ({very_rough_count/total*100:.1f}%)")
