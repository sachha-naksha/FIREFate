"""
Basic tests for FIREFate package
"""

import pytest
import numpy as np
import pandas as pd

# Test imports
def test_imports():
    """Test that all modules can be imported."""
    import firefate
    from firefate.data_processing import DataLoader, DataPreprocessor
    from firefate.analysis import DimensionalityReducer, ClusterAnalyzer
    from firefate.visualization import Plotter
    from firefate.utils import setup_logger, FileHandler
    
    assert firefate.__version__ == "0.1.0"


def test_data_loader():
    """Test DataLoader functionality."""
    from firefate.data_processing import DataLoader
    
    loader = DataLoader()
    assert loader.data is None


def test_data_preprocessor():
    """Test DataPreprocessor functionality."""
    from firefate.data_processing import DataPreprocessor
    
    preprocessor = DataPreprocessor()
    
    # Create sample data
    data = pd.DataFrame(np.random.randn(10, 5))
    
    # Test normalization
    normalized = preprocessor.normalize(data, method="zscore")
    assert normalized.shape == data.shape
    
    # Test variance filtering
    filtered = preprocessor.filter_low_variance(data, threshold=0.0)
    assert filtered.shape[1] <= data.shape[1]


def test_pca():
    """Test PCA analysis."""
    from firefate.analysis import perform_pca
    
    # Create sample data
    data = pd.DataFrame(np.random.randn(50, 10))
    
    # Perform PCA
    transformed, variance = perform_pca(data, n_components=2)
    
    assert transformed.shape == (50, 2)
    assert len(variance) == 2
    assert np.sum(variance) <= 1.0


def test_clustering():
    """Test clustering functionality."""
    from firefate.analysis import cluster_data
    
    # Create sample data
    data = np.random.randn(30, 5)
    
    # Perform clustering
    labels = cluster_data(data, n_clusters=3)
    
    assert len(labels) == 30
    assert len(np.unique(labels)) <= 3


def test_logger():
    """Test logger setup."""
    from firefate.utils import setup_logger
    
    logger = setup_logger(name="test_logger")
    assert logger is not None
    assert logger.name == "test_logger"


def test_file_handler():
    """Test FileHandler."""
    from firefate.utils import FileHandler
    
    handler = FileHandler()
    assert handler.base_dir is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
