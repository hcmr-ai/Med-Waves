import logging
import glob
import os
import s3fs
from lightning.pytorch.callbacks import Callback

logger = logging.getLogger(__name__)

class S3CheckpointSyncCallback(Callback):
    """Callback to sync checkpoints to S3 for spot instance safety"""

    def __init__(self, s3_dir: str, local_dir: str, sync_frequency: int = 5):
        super().__init__()
        self.s3_dir = s3_dir
        self.local_dir = local_dir
        self.sync_frequency = sync_frequency
        self.fs = s3fs.S3FileSystem()

    def on_train_epoch_end(self, trainer, pl_module):
        """Sync checkpoints to S3 every N epochs"""
        if trainer.current_epoch % self.sync_frequency == 0:
            self._sync_to_s3()

    def on_train_end(self, trainer, pl_module):
        """Final sync at training end"""
        self._sync_to_s3()

    def _sync_to_s3(self):
        """Sync all checkpoint files to S3"""
        try:

            # Get all checkpoint files
            checkpoint_files = glob.glob(f"{self.local_dir}/*.ckpt")

            for file_path in checkpoint_files:
                filename = os.path.basename(file_path)
                s3_path = f"{self.s3_dir}/{filename}"

                # Check if file needs to be uploaded
                should_upload = False
                
                try:
                    # Check if file exists in S3
                    s3_info = self.fs.info(s3_path)
                    
                    # Get local file info
                    local_size = os.path.getsize(file_path)
                    local_mtime = os.path.getmtime(file_path)
                    
                    # Compare size and modification time
                    s3_size = s3_info.get('size', 0)
                    s3_mtime = s3_info.get('LastModified', 0)
                    
                    # Upload if size differs or local file is newer
                    if local_size != s3_size or local_mtime > s3_mtime.timestamp() if hasattr(s3_mtime, 'timestamp') else True:
                        should_upload = True
                    else:
                        logger.debug(f"Skipping {filename} - already up to date in S3")
                        
                except FileNotFoundError:
                    # File doesn't exist in S3, need to upload
                    should_upload = True
                    logger.info(f"New checkpoint detected: {filename}")

                # Upload if needed
                if should_upload:
                    self.fs.put(file_path, s3_path)
                    logger.info(f"Synced checkpoint to S3: {s3_path}")

        except Exception as e:
            logger.warning(f"Failed to sync checkpoints to S3: {e}")
