"""
Data Processing Module

This module provides functions for data loading, preprocessing,
and transformation for regulatory fate determination analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List


class DataLoader:
    """Load and validate input data for FIREFate analysis."""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize DataLoader.
        
        Args:
            data_path: Path to the data file (optional)
        """
        self.data_path = data_path
        self.data = None
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            Loaded DataFrame
        """
        self.data = pd.read_csv(file_path, **kwargs)
        return self.data
    
    def load_expression_matrix(self, file_path: str, transpose: bool = False) -> pd.DataFrame:
        """
        Load gene expression matrix.
        
        Args:
            file_path: Path to expression matrix file
            transpose: Whether to transpose the matrix
            
        Returns:
            Expression matrix DataFrame
        """
        data = pd.read_csv(file_path, index_col=0)
        if transpose:
            data = data.T
        self.data = data
        return self.data


class DataPreprocessor:
    """Preprocess and normalize data for analysis."""
    
    def __init__(self):
        """Initialize DataPreprocessor."""
        pass
    
    def normalize(self, data: pd.DataFrame, method: str = "zscore") -> pd.DataFrame:
        """
        Normalize data using specified method.
        
        Args:
            data: Input DataFrame
            method: Normalization method ('zscore', 'minmax', 'log')
            
        Returns:
            Normalized DataFrame
        """
        if method == "zscore":
            return (data - data.mean()) / data.std()
        elif method == "minmax":
            return (data - data.min()) / (data.max() - data.min())
        elif method == "log":
            return np.log1p(data)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def filter_low_variance(self, data: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
        """
        Filter out features with low variance.
        
        Args:
            data: Input DataFrame
            threshold: Variance threshold
            
        Returns:
            Filtered DataFrame
        """
        variances = data.var()
        return data.loc[:, variances > threshold]
    
    def handle_missing_values(
        self, 
        data: pd.DataFrame, 
        method: str = "drop",
        fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in data.
        
        Args:
            data: Input DataFrame
            method: Method to handle missing values ('drop', 'fill', 'interpolate')
            fill_value: Value to fill missing entries (if method is 'fill')
            
        Returns:
            DataFrame with handled missing values
        """
        if method == "drop":
            return data.dropna()
        elif method == "fill":
            if fill_value is None:
                fill_value = 0
            return data.fillna(fill_value)
        elif method == "interpolate":
            return data.interpolate()
        else:
            raise ValueError(f"Unknown method: {method}")


def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    Convenience function to load data.
    
    Args:
        file_path: Path to data file
        **kwargs: Additional arguments for data loading
        
    Returns:
        Loaded DataFrame
    """
    loader = DataLoader()
    return loader.load_csv(file_path, **kwargs)


def preprocess_data(
    data: pd.DataFrame,
    normalize: bool = True,
    filter_variance: bool = True,
    handle_missing: str = "drop"
) -> pd.DataFrame:
    """
    Convenience function to preprocess data.
    
    Args:
        data: Input DataFrame
        normalize: Whether to normalize data
        filter_variance: Whether to filter low variance features
        handle_missing: Method to handle missing values
        
    Returns:
        Preprocessed DataFrame
    """
    preprocessor = DataPreprocessor()
    
    # Handle missing values
    data = preprocessor.handle_missing_values(data, method=handle_missing)
    
    # Filter low variance
    if filter_variance:
        data = preprocessor.filter_low_variance(data)
    
    # Normalize
    if normalize:
        data = preprocessor.normalize(data)
    
    return data


__all__ = [
    "DataLoader",
    "DataPreprocessor",
    "load_data",
    "preprocess_data",
]
