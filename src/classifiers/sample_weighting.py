"""
Sample weighting utilities for training data.

This module provides functionality to apply different types of sample weights
to training data, including regional weighting and wave height bin weighting.
"""

import numpy as np
from typing import Dict
import logging

from ..commons.region_mapping import RegionMapper

logger = logging.getLogger(__name__)


class SampleWeighting:
    """
    Handles sample weighting for training data.
    
    Supports multiple weighting strategies:
    - Regional weighting: Different weights for different geographical regions
    - Wave height bin weighting: Different weights based on wave height ranges
    - Combined weighting: Both regional and wave height weighting applied together
    """
    
    def __init__(self, feature_config: Dict):
        """
        Initialize the SampleWeighting class.
        
        Args:
            feature_config: Configuration dictionary containing weighting settings
        """
        self.feature_config = feature_config
        self.regional_config = feature_config.get("regional_weighting", {})
        self.wave_height_config = feature_config.get("wave_height_weighting", {})
    
    def apply_weights(self, y: np.ndarray, regions: np.ndarray = None) -> np.ndarray:
        """
        Apply sample weights based on configured weighting strategies.
        
        Args:
            y: Target vector (wave heights)
            regions: Region information for each sample (optional)
            
        Returns:
            Sample weights array
        """
        # Start with uniform weights
        sample_weights = np.ones(len(y))
        
        # Apply regional weighting if enabled
        if self.regional_config.get("enabled", False):
            regional_weights = self._apply_regional_weights(y, regions)
            sample_weights *= regional_weights
            logger.info("Applied regional weighting")
        
        # Apply wave height bin weighting if enabled
        if self.wave_height_config.get("enabled", False):
            wave_height_weights = self._apply_wave_height_weights(y)
            sample_weights *= wave_height_weights
            logger.info("Applied wave height bin weighting")
        
        # Log final weight statistics
        self._log_weight_statistics(sample_weights, y, regions)
        
        return sample_weights
    
    def _apply_regional_weights(self, y: np.ndarray, regions: np.ndarray) -> np.ndarray:
        """
        Apply regional weights to training data.
        
        Args:
            y: Target vector
            regions: Region information for each sample
            
        Returns:
            Regional weights array
        """
        if regions is None:
            logger.warning("Regional weighting enabled but no region information provided")
            return np.ones(len(y))
        
        weights_config = self.regional_config.get("weights", {})
        
        # Create weight array
        regional_weights = np.ones(len(y))
        
        # Apply weights based on region
        for region, weight in weights_config.items():
            region_mask = regions == region
            region_count = np.sum(region_mask)
            if region_count > 0:
                regional_weights[region_mask] = weight
                logger.info(f"Applied regional weight {weight} to {region_count:,} {region} samples")
        
        return regional_weights
    
    def _apply_wave_height_weights(self, y: np.ndarray) -> np.ndarray:
        """
        Apply wave height bin weights based on the specified formula.
        
        Weighting formula:
        w(Hs) = 1                        if Hs ≤ 3
              = 1 + 1.0*(Hs-3)           if 3 < Hs ≤ 4      → up to 2
              = 2 + 1.5*(Hs-4)           if 4 < Hs ≤ 6      → up to 5
              = 5 + 1.0*(Hs-6)           if 6 < Hs ≤ 8      → up to 7
              = 8                        if Hs > 8
        
        Args:
            y: Target vector (wave heights)
            
        Returns:
            Wave height weights array
        """
        wave_weights = np.ones(len(y))
        
        # Apply the piecewise linear weighting function
        # Hs ≤ 3: weight = 1
        mask_low = y <= 3.0
        wave_weights[mask_low] = 1.0
        
        # 3 < Hs ≤ 4: weight = 1 + 1.0*(Hs-3)
        mask_3_4 = (y > 3.0) & (y <= 4.0)
        wave_weights[mask_3_4] = 1.0 + 1.0 * (y[mask_3_4] - 3.0)
        
        # 4 < Hs ≤ 6: weight = 2 + 1.5*(Hs-4)
        mask_4_6 = (y > 4.0) & (y <= 6.0)
        wave_weights[mask_4_6] = 2.0 + 1.5 * (y[mask_4_6] - 4.0)
        
        # 6 < Hs ≤ 8: weight = 5 + 1.0*(Hs-6)
        mask_6_8 = (y > 6.0) & (y <= 8.0)
        wave_weights[mask_6_8] = 5.0 + 1.0 * (y[mask_6_8] - 6.0)
        
        # Hs > 8: weight = 8
        mask_high = y > 8.0
        wave_weights[mask_high] = 8.0
        
        # Log wave height bin statistics
        self._log_wave_height_bin_statistics(y, wave_weights)
        
        return wave_weights
    
    def _log_wave_height_bin_statistics(self, y: np.ndarray, wave_weights: np.ndarray) -> None:
        """Log statistics for wave height bins."""
        bins = [
            (0, 3, "≤ 3m"),
            (3, 4, "3-4m"),
            (4, 6, "4-6m"),
            (6, 8, "6-8m"),
            (8, np.inf, "> 8m")
        ]
        
        logger.info("Wave height bin weighting statistics:")
        for min_hs, max_hs, label in bins:
            if max_hs == np.inf:
                mask = y > min_hs
            else:
                mask = (y > min_hs) & (y <= max_hs)
            
            count = np.sum(mask)
            if count > 0:
                avg_weight = np.mean(wave_weights[mask])
                effective_count = np.sum(wave_weights[mask])
                logger.info(f"  {label}: {count:,} samples, avg weight: {avg_weight:.2f}, effective: {effective_count:,.0f}")
    
    def _log_weight_statistics(self, sample_weights: np.ndarray, y: np.ndarray, regions: np.ndarray = None) -> None:
        """Log comprehensive weight statistics."""
        total_weighted_samples = np.sum(sample_weights)
        logger.info(f"Total weighted samples: {total_weighted_samples:,.0f}")
        logger.info(f"Weight range: {np.min(sample_weights):.3f} - {np.max(sample_weights):.3f}")
        logger.info(f"Mean weight: {np.mean(sample_weights):.3f}")
        
        # Log regional effective sample counts if regions are available
        if regions is not None and self.regional_config.get("enabled", False):
            weights_config = self.regional_config.get("weights", {})
            logger.info("Regional effective sample counts:")
            for region_id in weights_config.keys():
                region_mask = regions == region_id
                if np.sum(region_mask) > 0:
                    region_count = np.sum(region_mask)
                    effective_count = np.sum(sample_weights[region_mask])
                    region_name = RegionMapper.get_display_name(region_id)
                    logger.info(f"  {region_name}: {region_count:,} samples → {effective_count:,.0f} effective samples")
    
    def get_weight_function_info(self) -> Dict[str, str]:
        """
        Get information about the weighting functions being used.
        
        Returns:
            Dictionary with weighting function descriptions
        """
        info = {}
        
        if self.regional_config.get("enabled", False):
            weights = self.regional_config.get("weights", {})
            info["regional"] = f"Regional weighting enabled with weights: {weights}"
        
        if self.wave_height_config.get("enabled", False):
            info["wave_height"] = (
                "Wave height bin weighting enabled with formula:\n"
                "w(Hs) = 1                        if Hs ≤ 3\n"
                "      = 1 + 1.0*(Hs-3)           if 3 < Hs ≤ 4\n"
                "      = 2 + 1.5*(Hs-4)           if 4 < Hs ≤ 6\n"
                "      = 5 + 1.0*(Hs-6)           if 6 < Hs ≤ 8\n"
                "      = 8                        if Hs > 8"
            )
        
        return info
