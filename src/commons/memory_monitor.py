"""
Memory monitoring utilities for tracking memory usage across the application.

This module provides comprehensive memory monitoring capabilities including:
- Process memory tracking (RSS, VMS)
- System memory monitoring
- Comet ML integration
- Configurable logging levels
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    from comet_ml import Experiment
    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """
    Comprehensive memory monitoring utility.
    
    Provides detailed memory tracking including process memory (RSS/VMS),
    system memory, and integration with Comet ML for experiment tracking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the memory monitor.
        
        Args:
            config: Configuration dictionary for memory monitoring settings
        """
        self.config = config or {}
        self.memory_config = self.config.get("memory_monitoring", {})
        self.enabled = self.memory_config.get("enabled", True)
        self.log_detailed = self.memory_config.get("log_detailed_memory", True)
        self.log_system = self.memory_config.get("log_system_memory", True)
        self.log_to_comet = self.memory_config.get("log_to_comet", True)
        
        # Store experiment reference for Comet ML logging
        self.experiment: Optional[Experiment] = None
        
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available - memory monitoring disabled")
            self.enabled = False
    
    def set_experiment(self, experiment: Optional[Experiment]):
        """Set the Comet ML experiment for logging."""
        self.experiment = experiment
    
    def get_process_memory_info(self) -> Dict[str, float]:
        """
        Get detailed process memory information.
        
        Returns:
            Dictionary containing memory metrics in GB
        """
        if not PSUTIL_AVAILABLE or not self.enabled:
            return {}
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            return {
                "rss_gb": memory_info.rss / (1024**3),  # Resident Set Size (physical memory)
                "vms_gb": memory_info.vms / (1024**3),  # Virtual Memory Size (allocated memory)
                "memory_percent": memory_percent,
                "num_threads": process.num_threads(),
                "cpu_percent": process.cpu_percent()
            }
        except Exception as e:
            logger.warning(f"Could not get process memory info: {e}")
            return {}
    
    def get_system_memory_info(self) -> Dict[str, float]:
        """
        Get system-wide memory information.
        
        Returns:
            Dictionary containing system memory metrics in GB
        """
        if not PSUTIL_AVAILABLE or not self.enabled:
            return {}
        
        try:
            system_memory = psutil.virtual_memory()
            return {
                "total_gb": system_memory.total / (1024**3),
                "available_gb": system_memory.available / (1024**3),
                "used_gb": system_memory.used / (1024**3),
                "percent_used": system_memory.percent,
                "free_gb": system_memory.free / (1024**3)
            }
        except Exception as e:
            logger.warning(f"Could not get system memory info: {e}")
            return {}
    
    def log_memory_usage(self, stage: str, log_level: int = logging.INFO):
        """
        Log memory usage at a specific stage.
        
        Args:
            stage: Name of the stage (e.g., "before_loading", "after_training")
            log_level: Logging level for the output
        """
        if not self.enabled:
            return
        
        process_info = self.get_process_memory_info()
        if not process_info:
            return
        
        # Log process memory
        if self.log_detailed:
            logger.log(log_level, f"Memory usage at {stage}:")
            logger.log(log_level, f"  - RSS (Physical): {process_info['rss_gb']:.2f} GB")
            logger.log(log_level, f"  - VMS (Virtual): {process_info['vms_gb']:.2f} GB")
            logger.log(log_level, f"  - Memory %: {process_info['memory_percent']:.1f}%")
            logger.log(log_level, f"  - Threads: {process_info['num_threads']}")
        else:
            logger.log(log_level, f"Memory usage at {stage}: {process_info['rss_gb']:.2f} GB (RSS)")
        
        # Log to Comet ML if enabled
        if self.log_to_comet and self.experiment and COMET_AVAILABLE:
            self._log_to_comet(stage, process_info, {})
    
    def log_system_memory(self, stage: str = "system", log_level: int = logging.INFO):
        """
        Log system-wide memory information.
        
        Args:
            stage: Name of the stage for logging
            log_level: Logging level for the output
        """
        if not self.enabled or not self.log_system:
            return
        
        system_info = self.get_system_memory_info()
        if not system_info:
            return
        
        logger.log(log_level, f"System Memory Info ({stage}):")
        logger.log(log_level, f"  - Total: {system_info['total_gb']:.2f} GB")
        logger.log(log_level, f"  - Used: {system_info['used_gb']:.2f} GB ({system_info['percent_used']:.1f}%)")
        logger.log(log_level, f"  - Available: {system_info['available_gb']:.2f} GB")
        logger.log(log_level, f"  - Free: {system_info['free_gb']:.2f} GB")
        
        # Log to Comet ML if enabled
        if self.log_to_comet and self.experiment and COMET_AVAILABLE:
            self._log_to_comet(stage, {}, system_info)
    
    def log_comprehensive_memory(self, stage: str, log_level: int = logging.INFO):
        """
        Log both process and system memory information.
        
        Args:
            stage: Name of the stage for logging
            log_level: Logging level for the output
        """
        self.log_memory_usage(stage, log_level)
        self.log_system_memory(stage, log_level)
    
    def _log_to_comet(self, stage: str, process_info: Dict[str, float], system_info: Dict[str, float]):
        """Log memory metrics to Comet ML experiment."""
        if not self.experiment or not COMET_AVAILABLE:
            return
        
        try:
            # Log process metrics
            for key, value in process_info.items():
                self.experiment.log_metric(f"memory_{key}_{stage}", value)
            
            # Log system metrics
            for key, value in system_info.items():
                self.experiment.log_metric(f"system_{key}_{stage}", value)
                
        except Exception as e:
            logger.warning(f"Could not log memory metrics to Comet ML: {e}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive memory summary.
        
        Returns:
            Dictionary containing all memory information
        """
        return {
            "process": self.get_process_memory_info(),
            "system": self.get_system_memory_info(),
            "monitoring_enabled": self.enabled,
            "psutil_available": PSUTIL_AVAILABLE,
            "comet_available": COMET_AVAILABLE
        }
    
    def log_memory_summary(self, stage: str = "summary"):
        """Log a comprehensive memory summary."""
        summary = self.get_memory_summary()
        
        logger.info(f"=== Memory Summary ({stage}) ===")
        logger.info(f"Monitoring enabled: {summary['monitoring_enabled']}")
        logger.info(f"psutil available: {summary['psutil_available']}")
        logger.info(f"Comet ML available: {summary['comet_available']}")
        
        if summary['process']:
            proc = summary['process']
            logger.info(f"Process Memory: {proc['rss_gb']:.2f} GB RSS, {proc['vms_gb']:.2f} GB VMS")
        
        if summary['system']:
            sys = summary['system']
            logger.info(f"System Memory: {sys['used_gb']:.2f} GB used of {sys['total_gb']:.2f} GB total")


class MemoryTracker:
    """
    Context manager for tracking memory usage during specific operations.
    
    Usage:
        with MemoryTracker("data_loading", monitor):
            # Your code here
            pass
    """
    
    def __init__(self, operation_name: str, monitor: MemoryMonitor):
        """
        Initialize the memory tracker.
        
        Args:
            operation_name: Name of the operation being tracked
            monitor: MemoryMonitor instance to use for tracking
        """
        self.operation_name = operation_name
        self.monitor = monitor
        self.start_memory = None
    
    def __enter__(self):
        """Start memory tracking."""
        if self.monitor.enabled:
            self.start_memory = self.monitor.get_process_memory_info()
            self.monitor.log_memory_usage(f"{self.operation_name}_start")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End memory tracking and log differences."""
        if self.monitor.enabled and self.start_memory:
            end_memory = self.monitor.get_process_memory_info()
            
            if end_memory and self.start_memory:
                rss_diff = end_memory['rss_gb'] - self.start_memory['rss_gb']
                vms_diff = end_memory['vms_gb'] - self.start_memory['vms_gb']
                
                logger.info(f"Memory change during {self.operation_name}:")
                logger.info(f"  - RSS: {rss_diff:+.2f} GB")
                logger.info(f"  - VMS: {vms_diff:+.2f} GB")
            
            self.monitor.log_memory_usage(f"{self.operation_name}_end")


def create_memory_monitor(config: Optional[Dict[str, Any]] = None) -> MemoryMonitor:
    """
    Factory function to create a MemoryMonitor instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MemoryMonitor instance
    """
    return MemoryMonitor(config)


def log_memory_usage(stage: str, config: Optional[Dict[str, Any]] = None, 
                    experiment: Optional[Experiment] = None):
    """
    Convenience function to log memory usage.
    
    Args:
        stage: Stage name for logging
        config: Configuration dictionary
        experiment: Comet ML experiment for logging
    """
    monitor = create_memory_monitor(config)
    if experiment:
        monitor.set_experiment(experiment)
    monitor.log_memory_usage(stage)
