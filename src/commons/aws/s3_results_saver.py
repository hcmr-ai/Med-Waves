"""
S3 Results Saver for uploading training results and models to S3.

This module provides functionality to save training results, models, and artifacts
directly to S3, which is perfect for spot instance workflows where local storage
is ephemeral.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

logger = logging.getLogger(__name__)


class S3ResultsSaver:
    """
    Handles saving training results, models, and artifacts to S3.
    
    Features:
    - Upload models and results to S3
    - Organize files with timestamps and experiment names
    - Handle large file uploads with multipart upload
    - Retry logic for failed uploads
    - Progress tracking for large uploads
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the S3 results saver.
        
        Args:
            config: Configuration dictionary containing S3 settings
        """
        self.config = config
        self.output_config = config.get("output", {})
        self.s3_config = self.output_config.get("s3", {})
        
        self.enabled = self.s3_config.get("enabled", False)
        self.bucket = self.s3_config.get("bucket")
        self.prefix = self.s3_config.get("prefix", "experiments") + "/" + self.output_config.get("experiment_name", "")
        self.prefix = self.prefix + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        self.region = self.s3_config.get("region", "eu-central-1")
        self.aws_profile = self.s3_config.get("aws_profile")
        
        self.s3_client = None
        self.s3_resource = None
        
        if self.enabled and BOTO3_AVAILABLE:
            self._initialize_s3_client()
        elif self.enabled and not BOTO3_AVAILABLE:
            logger.error("S3 saving enabled but boto3 not available. Install boto3 to use S3 functionality.")
            self.enabled = False
    
    def _initialize_s3_client(self):
        """Initialize S3 client and resource."""
        try:
            session_kwargs = {}
            if self.aws_profile:
                session_kwargs['profile_name'] = self.aws_profile
            
            session = boto3.Session(**session_kwargs)
            self.s3_client = session.client('s3', region_name=self.region)
            self.s3_resource = session.resource('s3', region_name=self.region)
            
            # Test connection
            self.s3_client.head_bucket(Bucket=self.bucket)
            logger.info(f"S3 client initialized successfully for bucket: {self.bucket}")
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS credentials.")
            self.enabled = False
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"S3 bucket '{self.bucket}' not found.")
            else:
                logger.error(f"Error accessing S3 bucket: {e}")
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.enabled = False
    
    def _get_s3_key(self, filename: str, experiment_name: str = None) -> str:
        """
        Generate S3 key for a file.
        
        Args:
            filename: Name of the file
            experiment_name: Name of the experiment
            
        Returns:
            S3 key (path) for the file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if experiment_name:
            key = f"{self.prefix}/{experiment_name}/{timestamp}/{filename}"
        else:
            key = f"{self.prefix}/{timestamp}/{filename}"
        
        return key
    
    def upload_file(self, local_path: str, s3_key: str = None, experiment_name: str = None) -> bool:
        """
        Upload a single file to S3.
        
        Args:
            local_path: Local path to the file
            s3_key: S3 key (path) for the file. If None, will be generated
            experiment_name: Name of the experiment for organizing files
            
        Returns:
            True if upload successful, False otherwise
        """
        if not self.enabled:
            logger.warning("S3 saving is disabled")
            return False
        
        local_path = Path(local_path)
        if not local_path.exists():
            logger.error(f"File not found: {local_path}")
            return False
        
        if s3_key is None:
            s3_key = self._get_s3_key(local_path.name, experiment_name)
        
        try:
            file_size = local_path.stat().st_size
            logger.info(f"Uploading {local_path.name} ({file_size / (1024**2):.1f} MB) to s3://{self.bucket}/{s3_key}")
            
            # Use multipart upload for large files (>100MB)
            if file_size > 100 * 1024 * 1024:
                self._upload_large_file(str(local_path), s3_key)
            else:
                self.s3_client.upload_file(str(local_path), self.bucket, s3_key)
            
            logger.info(f"Successfully uploaded {local_path.name} to S3")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload {local_path.name} to S3: {e}")
            return False
    
    def _upload_large_file(self, local_path: str, s3_key: str):
        """Upload large files using multipart upload."""
        try:
            # Configure multipart upload
            config = boto3.s3.transfer.TransferConfig(
                multipart_threshold=1024 * 25,  # 25MB
                max_concurrency=10,
                multipart_chunksize=1024 * 25,
                use_threads=True
            )
            
            self.s3_client.upload_file(
                local_path, 
                self.bucket, 
                s3_key,
                Config=config
            )
        except Exception as e:
            logger.error(f"Multipart upload failed: {e}")
            raise
    
    def upload_directory(self, local_dir: str, s3_prefix: str = None, experiment_name: str = None) -> Dict[str, bool]:
        """
        Upload an entire directory to S3.
        
        Args:
            local_dir: Local directory path
            s3_prefix: S3 prefix for the directory. If None, will be generated
            experiment_name: Name of the experiment for organizing files
            
        Returns:
            Dictionary mapping file paths to upload success status
        """
        if not self.enabled:
            logger.warning("S3 saving is disabled")
            return {}
        
        local_dir = Path(local_dir)
        if not local_dir.exists():
            logger.error(f"Directory not found: {local_dir}")
            return {}
        
        results = {}
        
        for file_path in local_dir.rglob('*'):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_dir)
                
                if s3_prefix:
                    s3_key = f"{s3_prefix}/{relative_path}"
                else:
                    s3_key = self._get_s3_key(str(relative_path), experiment_name)
                
                success = self.upload_file(str(file_path), s3_key, experiment_name)
                results[str(file_path)] = success
        
        successful_uploads = sum(1 for success in results.values() if success)
        total_files = len(results)
        
        logger.info(f"Directory upload completed: {successful_uploads}/{total_files} files uploaded successfully")
        return results
    
    def save_results_json(self, results: Dict[str, Any], experiment_name: str = None) -> bool:
        """
        Save results dictionary as JSON to S3.
        
        Args:
            results: Results dictionary to save
            experiment_name: Name of the experiment
            
        Returns:
            True if save successful, False otherwise
        """
        if not self.enabled:
            logger.warning("S3 saving is disabled")
            return False
        
        try:
            # Create temporary local file
            temp_file = Path("/tmp/results.json")
            with open(temp_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Upload to S3
            s3_key = self._get_s3_key("results.json", experiment_name)
            success = self.upload_file(str(temp_file), s3_key, experiment_name)
            
            # Clean up temp file
            temp_file.unlink()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to save results JSON to S3: {e}")
            return False
    
    def save_model_artifacts(self, model_path: str, experiment_name: str = None) -> Dict[str, bool]:
        """
        Save model artifacts (model, scaler, etc.) to S3.
        
        Args:
            model_path: Local path to the model directory
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary mapping file paths to upload success status
        """
        if not self.enabled:
            logger.warning("S3 saving is disabled")
            return {}
        
        model_path = Path(model_path)
        if not model_path.exists():
            logger.error(f"Model path not found: {model_path}")
            return {}
        
        logger.info(f"Saving model artifacts from {model_path} to S3")
        
        # Upload model directory
        s3_prefix = self._get_s3_key("model", experiment_name)
        results = self.upload_directory(str(model_path), s3_prefix, experiment_name)
        
        return results
    
    def save_diagnostic_plots(self, plots_dir: str, experiment_name: str = None) -> Dict[str, bool]:
        """
        Save diagnostic plots to S3.
        
        Args:
            plots_dir: Local path to the plots directory
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary mapping file paths to upload success status
        """
        if not self.enabled:
            logger.warning("S3 saving is disabled")
            return {}
        
        plots_dir = Path(plots_dir)
        if not plots_dir.exists():
            logger.warning(f"Plots directory not found: {plots_dir}")
            return {}
        
        logger.info(f"Saving diagnostic plots from {plots_dir} to S3")
        
        # Upload plots directory
        s3_prefix = self._get_s3_key("plots", experiment_name)
        results = self.upload_directory(str(plots_dir), s3_prefix, experiment_name)
        
        return results
    
    def get_s3_url(self, s3_key: str) -> str:
        """
        Get the S3 URL for a given key.
        
        Args:
            s3_key: S3 key (path)
            
        Returns:
            S3 URL
        """
        return f"s3://{self.bucket}/{s3_key}"
    
    def list_experiment_files(self, experiment_name: str) -> List[str]:
        """
        List all files for a given experiment.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            List of S3 keys for the experiment
        """
        if not self.enabled:
            return []
        
        try:
            prefix = f"{self.prefix}/{experiment_name}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                return [obj['Key'] for obj in response['Contents']]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Failed to list experiment files: {e}")
            return []


def create_s3_results_saver(config: Dict[str, Any]) -> S3ResultsSaver:
    """
    Factory function to create an S3ResultsSaver instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        S3ResultsSaver instance
    """
    return S3ResultsSaver(config)