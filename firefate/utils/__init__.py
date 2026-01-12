"""
Utilities Module

This module provides utility functions and helper classes for FIREFate analysis,
including file I/O, logging, and common operations.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
from pathlib import Path


def setup_logger(
    name: str = "firefate",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger for the package.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to write logs to
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class FileHandler:
    """Handle file I/O operations."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize FileHandler.
        
        Args:
            base_dir: Base directory for file operations
        """
        self.base_dir = base_dir or os.getcwd()
    
    def read_json(self, filename: str) -> Dict[str, Any]:
        """
        Read JSON file.
        
        Args:
            filename: Path to JSON file
            
        Returns:
            Parsed JSON data
        """
        filepath = os.path.join(self.base_dir, filename)
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def write_json(self, data: Dict[str, Any], filename: str, indent: int = 2):
        """
        Write data to JSON file.
        
        Args:
            data: Data to write
            filename: Output filename
            indent: JSON indentation
        """
        filepath = os.path.join(self.base_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
    
    def ensure_dir(self, directory: str) -> str:
        """
        Ensure directory exists, create if it doesn't.
        
        Args:
            directory: Directory path
            
        Returns:
            Absolute directory path
        """
        dirpath = os.path.join(self.base_dir, directory)
        os.makedirs(dirpath, exist_ok=True)
        return dirpath
    
    def list_files(self, directory: str, pattern: str = "*") -> List[str]:
        """
        List files in directory matching pattern.
        
        Args:
            directory: Directory to search
            pattern: File pattern to match
            
        Returns:
            List of matching file paths
        """
        dirpath = Path(self.base_dir) / directory
        return [str(p) for p in dirpath.glob(pattern)]


class ConfigManager:
    """Manage configuration settings."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = {}
        
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
    
    def load_config(self, config_file: str):
        """
        Load configuration from file.
        
        Args:
            config_file: Path to configuration file
        """
        with open(config_file, 'r') as f:
            self.config = json.load(f)
    
    def save_config(self, config_file: Optional[str] = None):
        """
        Save configuration to file.
        
        Args:
            config_file: Path to save configuration (uses self.config_file if not provided)
        """
        output_file = config_file or self.config_file
        if not output_file:
            raise ValueError("No config file specified")
        
        with open(output_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 0
) -> bool:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        True if valid
        
    Raises:
        ValueError if validation fails
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, minimum {min_rows} required")
    
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    return True


def merge_dataframes(
    dfs: List[pd.DataFrame],
    on: Optional[str] = None,
    how: str = "inner"
) -> pd.DataFrame:
    """
    Merge multiple DataFrames.
    
    Args:
        dfs: List of DataFrames to merge
        on: Column to merge on (if None, uses index)
        how: Type of merge ('inner', 'outer', 'left', 'right')
        
    Returns:
        Merged DataFrame
    """
    if not dfs:
        raise ValueError("No DataFrames provided")
    
    if len(dfs) == 1:
        return dfs[0]
    
    result = dfs[0]
    for df in dfs[1:]:
        if on:
            result = pd.merge(result, df, on=on, how=how)
        else:
            result = pd.merge(result, df, left_index=True, right_index=True, how=how)
    
    return result


def compute_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        predictions: Predicted values
        ground_truth: Ground truth values
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        "accuracy": accuracy_score(ground_truth, predictions),
        "precision": precision_score(ground_truth, predictions, average='weighted', zero_division=0),
        "recall": recall_score(ground_truth, predictions, average='weighted', zero_division=0),
        "f1": f1_score(ground_truth, predictions, average='weighted', zero_division=0),
    }
    
    return metrics


__all__ = [
    "setup_logger",
    "FileHandler",
    "ConfigManager",
    "validate_dataframe",
    "merge_dataframes",
    "compute_metrics",
]
