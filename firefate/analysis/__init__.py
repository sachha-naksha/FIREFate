"""
Analysis Module

This module provides core analysis functions for regulatory fate determination,
including clustering, dimensionality reduction, and statistical analysis.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats


class DimensionalityReducer:
    """Perform dimensionality reduction on high-dimensional data."""
    
    def __init__(self, method: str = "pca", n_components: int = 2):
        """
        Initialize DimensionalityReducer.
        
        Args:
            method: Reduction method ('pca', 'tsne', 'umap')
            n_components: Number of components to reduce to
        """
        self.method = method
        self.n_components = n_components
        self.model = None
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform data.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Reduced data array
        """
        if self.method == "pca":
            self.model = PCA(n_components=self.n_components)
            return self.model.fit_transform(data)
        else:
            raise ValueError(f"Method {self.method} not implemented yet")
    
    def get_explained_variance(self) -> Optional[np.ndarray]:
        """
        Get explained variance ratio for PCA.
        
        Returns:
            Explained variance ratio array
        """
        if self.method == "pca" and self.model is not None:
            return self.model.explained_variance_ratio_
        return None


class ClusterAnalyzer:
    """Perform clustering analysis on data."""
    
    def __init__(self, n_clusters: int = 3, method: str = "kmeans"):
        """
        Initialize ClusterAnalyzer.
        
        Args:
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'hierarchical')
        """
        self.n_clusters = n_clusters
        self.method = method
        self.model = None
        self.labels_ = None
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Fit clustering model to data.
        
        Args:
            data: Input data
            
        Returns:
            Cluster labels
        """
        if self.method == "kmeans":
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
            self.labels_ = self.model.fit_predict(data)
            return self.labels_
        else:
            raise ValueError(f"Method {self.method} not implemented yet")
    
    def predict(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            data: Input data
            
        Returns:
            Predicted cluster labels
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        return self.model.predict(data)


class StatisticalAnalyzer:
    """Perform statistical analysis on data."""
    
    def __init__(self):
        """Initialize StatisticalAnalyzer."""
        pass
    
    def differential_analysis(
        self,
        group1: pd.DataFrame,
        group2: pd.DataFrame,
        test: str = "ttest"
    ) -> pd.DataFrame:
        """
        Perform differential analysis between two groups.
        
        Args:
            group1: First group data
            group2: Second group data
            test: Statistical test to use ('ttest', 'mannwhitneyu')
            
        Returns:
            DataFrame with test statistics and p-values
        """
        results = []
        
        for feature in group1.columns:
            if test == "ttest":
                statistic, pvalue = stats.ttest_ind(
                    group1[feature].dropna(),
                    group2[feature].dropna()
                )
            elif test == "mannwhitneyu":
                statistic, pvalue = stats.mannwhitneyu(
                    group1[feature].dropna(),
                    group2[feature].dropna()
                )
            else:
                raise ValueError(f"Unknown test: {test}")
            
            results.append({
                "feature": feature,
                "statistic": statistic,
                "pvalue": pvalue,
                "mean_group1": group1[feature].mean(),
                "mean_group2": group2[feature].mean(),
            })
        
        return pd.DataFrame(results)
    
    def correlation_analysis(
        self,
        data: pd.DataFrame,
        method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Compute correlation matrix.
        
        Args:
            data: Input DataFrame
            method: Correlation method ('pearson', 'spearman')
            
        Returns:
            Correlation matrix
        """
        return data.corr(method=method)


def perform_pca(data: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to perform PCA.
    
    Args:
        data: Input DataFrame
        n_components: Number of components
        
    Returns:
        Tuple of (transformed data, explained variance)
    """
    reducer = DimensionalityReducer(method="pca", n_components=n_components)
    transformed = reducer.fit_transform(data)
    variance = reducer.get_explained_variance()
    return transformed, variance


def cluster_data(data: Union[pd.DataFrame, np.ndarray], n_clusters: int = 3) -> np.ndarray:
    """
    Convenience function to cluster data.
    
    Args:
        data: Input data
        n_clusters: Number of clusters
        
    Returns:
        Cluster labels
    """
    analyzer = ClusterAnalyzer(n_clusters=n_clusters)
    return analyzer.fit(data)


__all__ = [
    "DimensionalityReducer",
    "ClusterAnalyzer",
    "StatisticalAnalyzer",
    "perform_pca",
    "cluster_data",
]
