import polars as pl
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class DeltaCorrector:
    """
    Simple bias correction using mean or median bias.
    
    This corrector calculates the bias between model predictions and observed values
    and applies a constant correction to all predictions.
    """
    
    def __init__(self, method: str = "mean_bias"):
        """
        Initialize Delta corrector.
        
        Args:
            method: Method for bias calculation ("mean_bias" or "median_bias")
        """
        self.method = method
        self.bias_per_variable = {}
        self.is_fitted = False
        
    def fit(self, df: pl.DataFrame, variables: List[str], corrected_suffix: str = "corrected_"):
        """
        Fit the delta corrector by calculating bias for each variable.
        
        Args:
            df: DataFrame with model and observed data
            variables: List of variable names to correct
            corrected_suffix: Suffix for observed/corrected columns
        """
        logger.info(f"Fitting Delta corrector for variables: {variables} (method: {self.method})")
        
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
            
            # Calculate bias based on method
            if self.method == "mean_bias":
                model_center = np.mean(model_values)
                obs_center = np.mean(obs_values)
                bias = model_center - obs_center
            elif self.method == "median_bias":
                model_center = np.median(model_values)
                obs_center = np.median(obs_values)
                bias = model_center - obs_center
            else:
                raise ValueError(f"Unknown bias method: {self.method}")
            
            self.bias_per_variable[var] = bias
            
            logger.info(f"Delta correction for {var}: bias={bias:.4f} "
                       f"(model_{self.method}={model_center:.4f}, obs_{self.method}={obs_center:.4f})")
        
        self.is_fitted = True
        logger.info(f"Delta corrector fitted for {len(self.bias_per_variable)} variables")
    
    def predict(self, df: pl.DataFrame, variables: List[str] = None) -> pl.DataFrame:
        """
        Apply delta correction to predictions.
        
        Args:
            df: DataFrame with model predictions
            variables: Variables to correct (if None, uses all fitted variables)
            
        Returns:
            DataFrame with corrected predictions
        """
        if not self.is_fitted:
            raise ValueError("Delta corrector must be fitted before prediction")
        
        if variables is None:
            variables = list(self.bias_per_variable.keys())
        
        result_df = df.clone()
        
        for var in variables:
            if var not in self.bias_per_variable:
                logger.warning(f"Variable {var} not found in fitted model, skipping")
                continue
                
            if var not in df.columns:
                logger.warning(f"Column {var} not found in input DataFrame, skipping")
                continue
            
            bias = self.bias_per_variable[var]
            
            # Apply bias correction
            result_df = result_df.with_columns(
                (pl.col(var) - bias).alias(f"delta_corrected_{var}")
            )
            
            logger.info(f"Applied delta correction to {var}: bias={bias:.4f}")
        
        return result_df
    
    def get_correction_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about the delta correction.
        
        Returns:
            Dictionary with correction statistics for each variable
        """
        if not self.is_fitted:
            return {}
        
        stats = {}
        for var, bias in self.bias_per_variable.items():
            stats[var] = {
                'bias': bias,
                'method': self.method,
                'correction_type': 'constant_bias'
            }
        
        return stats
    
    def save_model(self, filepath: str) -> None:
        """Save Delta model to file."""
        import joblib
        
        model_data = {
            'bias_per_variable': self.bias_per_variable,
            'method': self.method,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Delta model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'DeltaCorrector':
        """Load Delta model from file."""
        import joblib
        
        model_data = joblib.load(filepath)
        
        corrector = cls(method=model_data['method'])
        corrector.bias_per_variable = model_data['bias_per_variable']
        corrector.is_fitted = model_data['is_fitted']
        
        logger.info(f"Delta model loaded from {filepath}")
        return corrector
