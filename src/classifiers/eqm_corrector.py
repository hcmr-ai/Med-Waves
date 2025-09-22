"""
Equivalent Quantile Mapping (EQM) corrector for bias correction.

EQM is a sophisticated bias correction method that maps model quantiles to observed quantiles,
preserving the quantile structure while correcting systematic biases.

Key advantages over simple EDCDF:
- Better handling of extreme values
- Smoother quantile mapping
- More robust extrapolation
- Preserves model variability structure
"""

import numpy as np
import polars as pl
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from typing import Dict, Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class EQMCorrector:
    """
    Equivalent Quantile Mapping (EQM) corrector.
    
    This implementation uses kernel density estimation for smoother CDFs
    and proper quantile mapping for bias correction.
    """
    
    def __init__(self, 
                 quantile_resolution: int = 1000,
                 extrapolation_method: str = "constant",
                 kde_bandwidth: Optional[float] = None):
        """
        Initialize EQM corrector.
        
        Args:
            quantile_resolution: Number of quantiles for CDF approximation
            extrapolation_method: Method for extrapolation ("constant", "linear", "nearest")
            kde_bandwidth: Bandwidth for kernel density estimation (auto if None)
        """
        self.quantile_resolution = quantile_resolution
        self.extrapolation_method = extrapolation_method
        self.kde_bandwidth = kde_bandwidth
        self.eqm_models = {}
        self.is_fitted = False
        
    def fit(self, 
            df: pl.DataFrame, 
            variables: List[str], 
            corrected_suffix: str = "corrected_",
            use_kde: bool = True) -> None:
        """
        Fit EQM corrector to training data.
        
        Args:
            df: Training DataFrame with model and observed data
            variables: List of variable names to correct
            corrected_suffix: Suffix for observed/corrected columns
            use_kde: Whether to use kernel density estimation for smoother CDFs
        """
        logger.info(f"Fitting EQM corrector for variables: {variables}")
        
        for var in variables:
            model_col = var
            obs_col = corrected_suffix + var
            
            if model_col not in df.columns or obs_col not in df.columns:
                logger.warning(f"Columns {model_col} or {obs_col} not found, skipping {var}")
                continue
                
            # Extract data and remove nulls
            valid_data = df.drop_nulls(subset=[model_col, obs_col])
            if len(valid_data) == 0:
                logger.warning(f"No valid data for {var}, skipping")
                continue
                
            model_values = valid_data[model_col].to_numpy()
            obs_values = valid_data[obs_col].to_numpy()
            
            logger.info(f"Fitting EQM for {var}: {len(model_values):,} samples")
            
            # Create EQM model
            eqm_model = self._create_eqm_model(
                model_values, obs_values, var, use_kde
            )
            self.eqm_models[var] = eqm_model
            
        self.is_fitted = True
        logger.info(f"EQM corrector fitted for {len(self.eqm_models)} variables")
    
    def _create_eqm_model(self, 
                         model_values: np.ndarray, 
                         obs_values: np.ndarray,
                         var_name: str,
                         use_kde: bool = True) -> Dict:
        """
        Create EQM model for a single variable.
        
        Args:
            model_values: Model predictions
            obs_values: Observed values
            var_name: Variable name for logging
            use_kde: Whether to use KDE for smoother CDFs
            
        Returns:
            Dictionary containing EQM model components
        """
        # Remove any remaining NaN or infinite values
        model_values = model_values[np.isfinite(model_values)]
        obs_values = obs_values[np.isfinite(obs_values)]
        
        if len(model_values) == 0 or len(obs_values) == 0:
            raise ValueError(f"No valid data for {var_name}")
        
        # Create quantile levels
        quantile_levels = np.linspace(0, 1, self.quantile_resolution)
        
        if use_kde:
            # Use kernel density estimation for smoother CDFs
            model_kde = gaussian_kde(model_values, bw_method=self.kde_bandwidth)
            obs_kde = gaussian_kde(obs_values, bw_method=self.kde_bandwidth)
            
            # Create evaluation points for KDE
            model_min, model_max = np.percentile(model_values, [0.1, 99.9])
            obs_min, obs_max = np.percentile(obs_values, [0.1, 99.9])
            
            # Extend range for better extrapolation
            model_range = model_max - model_min
            obs_range = obs_max - obs_min
            
            model_eval_points = np.linspace(
                model_min - 0.5 * model_range, 
                model_max + 0.5 * model_range, 
                1000
            )
            obs_eval_points = np.linspace(
                obs_min - 0.5 * obs_range, 
                obs_max + 0.5 * obs_range, 
                1000
            )
            
            # Compute CDFs using KDE
            model_cdf_values = np.cumsum(model_kde(model_eval_points))
            obs_cdf_values = np.cumsum(obs_kde(obs_eval_points))
            
            # Normalize CDFs
            model_cdf_values = model_cdf_values / model_cdf_values[-1]
            obs_cdf_values = obs_cdf_values / obs_cdf_values[-1]
            
            # Create quantile mappings
            model_quantiles = np.interp(quantile_levels, model_cdf_values, model_eval_points)
            obs_quantiles = np.interp(quantile_levels, obs_cdf_values, obs_eval_points)
            
        else:
            # Use empirical quantiles (simpler but less smooth)
            model_quantiles = np.percentile(model_values, quantile_levels * 100)
            obs_quantiles = np.percentile(obs_values, quantile_levels * 100)
        
        # Create interpolators for quantile mapping
        # Model to quantile mapping
        model_to_quantile = interp1d(
            model_quantiles, quantile_levels,
            kind='linear', 
            bounds_error=False, 
            fill_value=(0, 1) if self.extrapolation_method == "constant" else "extrapolate"
        )
        
        # Quantile to observed mapping
        quantile_to_obs = interp1d(
            quantile_levels, obs_quantiles,
            kind='linear',
            bounds_error=False,
            fill_value=(obs_quantiles[0], obs_quantiles[-1]) if self.extrapolation_method == "constant" else "extrapolate"
        )
        
        # Store model statistics for validation
        model_stats = {
            'mean': np.mean(model_values),
            'std': np.std(model_values),
            'min': np.min(model_values),
            'max': np.max(model_values),
            'q01': np.percentile(model_values, 1),
            'q99': np.percentile(model_values, 99)
        }
        
        obs_stats = {
            'mean': np.mean(obs_values),
            'std': np.std(obs_values),
            'min': np.min(obs_values),
            'max': np.max(obs_values),
            'q01': np.percentile(obs_values, 1),
            'q99': np.percentile(obs_values, 99)
        }
        
        logger.info(f"EQM model for {var_name}:")
        logger.info(f"  Model: mean={model_stats['mean']:.4f}, std={model_stats['std']:.4f}")
        logger.info(f"  Observed: mean={obs_stats['mean']:.4f}, std={obs_stats['std']:.4f}")
        
        return {
            'model_to_quantile': model_to_quantile,
            'quantile_to_obs': quantile_to_obs,
            'model_quantiles': model_quantiles,
            'obs_quantiles': obs_quantiles,
            'quantile_levels': quantile_levels,
            'model_stats': model_stats,
            'obs_stats': obs_stats,
            'use_kde': use_kde
        }
    
    def predict(self, df: pl.DataFrame, variables: Optional[List[str]] = None) -> pl.DataFrame:
        """
        Apply EQM correction to predictions.
        
        Args:
            df: DataFrame with model predictions
            variables: Variables to correct (if None, uses all fitted variables)
            
        Returns:
            DataFrame with corrected predictions
        """
        if not self.is_fitted:
            raise ValueError("EQM corrector must be fitted before prediction")
        
        if variables is None:
            variables = list(self.eqm_models.keys())
        
        result_df = df.clone()
        
        for var in variables:
            if var not in self.eqm_models:
                logger.warning(f"Variable {var} not found in fitted models, skipping")
                continue
                
            if var not in df.columns:
                logger.warning(f"Column {var} not found in input DataFrame, skipping")
                continue
            
            model_values = df[var].to_numpy()
            valid_mask = np.isfinite(model_values)
            
            if not np.any(valid_mask):
                logger.warning(f"No valid values for {var}, skipping")
                continue
            
            # Apply EQM correction
            corrected_values = np.full_like(model_values, np.nan)
            
            eqm_model = self.eqm_models[var]
            model_to_quantile = eqm_model['model_to_quantile']
            quantile_to_obs = eqm_model['quantile_to_obs']
            
            # Map model values to quantiles, then to observed values
            valid_model_values = model_values[valid_mask]
            quantiles = model_to_quantile(valid_model_values)
            corrected_valid = quantile_to_obs(quantiles)
            
            corrected_values[valid_mask] = corrected_valid
            
            # Add corrected column
            result_df = result_df.with_columns(
                pl.Series(name=f"eqm_corrected_{var}", values=corrected_values)
            )
            
            # Log correction statistics
            n_corrected = np.sum(valid_mask)
            bias_reduction = np.mean(valid_model_values) - np.mean(corrected_valid)
            logger.info(f"EQM correction for {var}: {n_corrected:,} values, bias reduction: {bias_reduction:.4f}")
        
        return result_df
    
    def get_correction_stats(self) -> Dict[str, Dict]:
        """
        Get statistics about the EQM correction models.
        
        Returns:
            Dictionary with correction statistics for each variable
        """
        if not self.is_fitted:
            return {}
        
        stats = {}
        for var, model in self.eqm_models.items():
            stats[var] = {
                'model_stats': model['model_stats'],
                'obs_stats': model['obs_stats'],
                'bias': model['model_stats']['mean'] - model['obs_stats']['mean'],
                'bias_relative': (model['model_stats']['mean'] - model['obs_stats']['mean']) / model['obs_stats']['mean'] * 100,
                'use_kde': model['use_kde']
            }
        
        return stats
    
    def save_model(self, filepath: str) -> None:
        """Save EQM model to file."""
        import joblib
        
        model_data = {
            'eqm_models': self.eqm_models,
            'quantile_resolution': self.quantile_resolution,
            'extrapolation_method': self.extrapolation_method,
            'kde_bandwidth': self.kde_bandwidth,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"EQM model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'EQMCorrector':
        """Load EQM model from file."""
        import joblib
        
        model_data = joblib.load(filepath)
        
        corrector = cls(
            quantile_resolution=model_data['quantile_resolution'],
            extrapolation_method=model_data['extrapolation_method'],
            kde_bandwidth=model_data['kde_bandwidth']
        )
        
        corrector.eqm_models = model_data['eqm_models']
        corrector.is_fitted = model_data['is_fitted']
        
        logger.info(f"EQM model loaded from {filepath}")
        return corrector
