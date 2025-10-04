"""
S3 Model Loader for loading trained models and preprocessing components from S3.

This module provides functionality to load models, scalers, feature selectors,
and other preprocessing components directly from S3, enabling cloud-native
model evaluation workflows.
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import joblib

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

logger = logging.getLogger(__name__)


class S3ModelLoader:
    """
    Handles loading trained models and preprocessing components from S3.
    
    Features:
    - Load models, scalers, feature selectors from S3
    - Support for both S3 URIs and local paths
    - Automatic temporary file management
    - Error handling and validation
    - Progress tracking for large downloads
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the S3 model loader.
        
        Args:
            config: Configuration dictionary containing S3 settings
        """
        self.config = config or {}
        self.s3_config = self.config.get("s3", {})
        
        self.enabled = self.s3_config.get("enabled", False)
        self.bucket = self.s3_config.get("bucket")
        self.region = self.s3_config.get("region", "us-east-1")
        self.aws_profile = self.s3_config.get("aws_profile")
        
        self.s3_client = None
        self.temp_dir = None
        
        if self.enabled and BOTO3_AVAILABLE:
            self._initialize_s3_client()
        elif self.enabled and not BOTO3_AVAILABLE:
            logger.error("S3 loading enabled but boto3 not available. Install boto3 to use S3 functionality.")
            self.enabled = False
    
    def _initialize_s3_client(self):
        """Initialize S3 client."""
        try:
            session_kwargs = {}
            if self.aws_profile:
                session_kwargs['profile_name'] = self.aws_profile
            
            session = boto3.Session(**session_kwargs)
            self.s3_client = session.client('s3', region_name=self.region)
            
            # Test connection
            if self.bucket:
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
    
    def _is_s3_path(self, path: Union[str, Path]) -> bool:
        """Check if the path is an S3 URI."""
        path_str = str(path)
        return path_str.startswith('s3://')
    
    def _parse_s3_uri(self, s3_uri: str) -> tuple:
        """Parse S3 URI into bucket and key."""
        if not s3_uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        
        # Remove s3:// prefix
        path = s3_uri[5:]
        
        # Split into bucket and key
        parts = path.split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ""
        
        return bucket, key
    
    def _download_from_s3(self, s3_uri: str, local_path: Path) -> bool:
        """Download a file from S3 to local path."""
        try:
            bucket, key = self._parse_s3_uri(s3_uri)
            
            # Create parent directories if they don't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            self.s3_client.download_file(bucket, key, str(local_path))
            logger.info(f"Downloaded {s3_uri} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {s3_uri}: {e}")
            return False
    
    def _ensure_local_file(self, file_path: Union[str, Path]) -> Optional[Path]:
        """Ensure file is available locally, downloading from S3 if necessary."""
        # Handle S3 paths properly - don't convert to Path if it's an S3 URL
        if isinstance(file_path, str) and file_path.startswith('s3://'):
            file_path_str = file_path
        else:
            file_path = Path(file_path)
            file_path_str = str(file_path)
        
        if not self._is_s3_path(file_path_str):
            # Local path - check if exists
            if file_path.exists():
                return file_path
            else:
                logger.error(f"Local file not found: {file_path}")
                return None
        
        # S3 path - download to temporary location
        if not self.enabled:
            logger.error("S3 loading not enabled")
            return None
        
        # Create temporary directory if not exists
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix="s3_models_"))
            logger.info(f"Created temporary directory: {self.temp_dir}")
        
        # Create local path in temp directory
        # Extract filename from S3 URI
        filename = file_path_str.split('/')[-1]
        local_path = self.temp_dir / filename
        
        # Download if not already cached
        if not local_path.exists():
            if not self._download_from_s3(file_path_str, local_path):
                return None
        
        return local_path
    
    def load_model(self, model_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load model and all preprocessing components.
        
        Args:
            model_path: Path to model directory (local or S3)
            
        Returns:
            Dictionary containing loaded components
        """
        logger.info(f"Loading model from: {model_path}")
        
        components = {}
        # Handle S3 paths properly - don't convert to Path if it's an S3 URL
        if isinstance(model_path, str) and model_path.startswith('s3://'):
            model_path_str = model_path
        else:
            model_path = Path(model_path)
            model_path_str = str(model_path)
        
        # Define component files to load
        component_files = {
            'model': 'model.pkl',
            'scaler': 'scaler.pkl',
            'feature_selector': 'feature_selector.pkl',
            'dimension_reducer': 'dimension_reducer.pkl',
            'feature_names': 'feature_names.pkl',
            'selected_features': 'selected_features.pkl',
            'training_history': 'training_history.pkl'
        }
        
        # Load each component
        for component_name, filename in component_files.items():
            if isinstance(model_path, str) and model_path.startswith('s3://'):
                # For S3 paths, construct the full S3 URI
                component_path = f"{model_path_str.rstrip('/')}/{filename}"
            else:
                # For local paths, use Path operations
                component_path = model_path / filename
            
            # Ensure file is available locally
            local_path = self._ensure_local_file(component_path)
            if local_path is None:
                logger.warning(f"Could not load {component_name} from {component_path}")
                continue
            
            try:
                component = joblib.load(local_path)
                components[component_name] = component
                logger.info(f"Loaded {component_name} from {local_path}")
            except Exception as e:
                logger.error(f"Failed to load {component_name}: {e}")
        
        logger.info(f"✅ Loaded {len(components)} components successfully")
        return components
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            shutil.rmtree(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            self.temp_dir = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup temporary files."""
        self.cleanup()
