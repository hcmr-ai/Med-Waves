from pathlib import Path
from typing import Any, Dict
import pandas as pd


class TrainingMetricsStorage:
    """Disk-based storage for training metrics to avoid memory overload."""
    
    def __init__(self, save_path: str, max_history: int = 1000):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.max_history = max_history
        self.current_batch = 0
        
    def _get_current_batch_from_files(self) -> int:
        """Get the current batch number by looking at existing files."""
        # Look for both batch_metrics and validation_metrics files
        batch_files = list(self.save_path.glob("batch_metrics_*.csv"))
        val_files = list(self.save_path.glob("validation_metrics_*.csv"))
        
        all_files = batch_files + val_files
        if not all_files:
            return 0
        
        # Extract batch numbers from filenames and find the maximum
        batch_numbers = []
        for file_path in all_files:
            filename = file_path.stem
            if filename.startswith("batch_metrics_") or filename.startswith("validation_metrics_"):
                try:
                    batch_num = int(filename.split("_")[-1])
                    batch_numbers.append(batch_num)
                except ValueError:
                    continue
        
        return max(batch_numbers) + 1 if batch_numbers else 0
        
    def save_batch_metrics(self, metrics: Dict[str, Any]):
        """Save batch metrics to disk efficiently."""
        df = pd.DataFrame([metrics])
        file_path = self.save_path / f"batch_metrics_{self.current_batch:06d}.csv"
        df.to_csv(file_path)
        
        # Cleanup old files
        if self.current_batch > self.max_history:
            old_file = self.save_path / f"batch_metrics_{self.current_batch - self.max_history:06d}.parquet"
            if old_file.exists():
                old_file.unlink()
        
        self.current_batch += 1
    
    def save_validation_metrics(self, metrics: Dict[str, Any]):
        """Save validation metrics to disk."""
        df = pd.DataFrame([metrics])
        file_path = self.save_path / f"validation_metrics_{metrics['batch']:06d}.csv"
        df.to_csv(file_path, index=False)
    
    def save_feature_importance(self, importance_data: Dict[str, Any]):
        """Save feature importance data to disk."""
        df = pd.DataFrame([importance_data])
        file_path = self.save_path / f"feature_importance_{importance_data['batch']:06d}.csv"
        df.to_csv(file_path, index=False)
    
    def load_recent_metrics(self, n_batches: int) -> pd.DataFrame:
        """Load recent batch metrics from disk."""
        if self.current_batch == 0:
            return pd.DataFrame()
        
        # Since current_batch is incremented after saving, we need to look for files from 0 to current_batch-1
        start_batch = max(0, self.current_batch - n_batches)
        metrics_list = []
        
        for batch_idx in range(start_batch, self.current_batch):
            file_path = self.save_path / f"batch_metrics_{batch_idx:06d}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                metrics_list.append(df)
        
        if metrics_list:
            return pd.concat(metrics_list, ignore_index=True)
        return pd.DataFrame()
    
    def load_validation_metrics(self, n_batches: int) -> pd.DataFrame:
        """Load recent validation metrics from disk."""
        if self.current_batch == 0:
            return pd.DataFrame()
        
        start_batch = max(0, self.current_batch - n_batches)
        metrics_list = []
        
        for batch_idx in range(start_batch, self.current_batch):
            file_path = self.save_path / f"validation_metrics_{batch_idx:06d}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                metrics_list.append(df)
        
        if metrics_list:
            return pd.concat(metrics_list, ignore_index=True)
        return pd.DataFrame()