"""
Visualization Module

This module provides visualization functions for regulatory fate determination analysis,
including plots for dimensionality reduction, clustering, and expression patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List, Tuple


class Plotter:
    """Base plotter class for creating visualizations."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8), style: str = "default"):
        """
        Initialize Plotter.
        
        Args:
            figsize: Figure size (width, height)
            style: Matplotlib style
        """
        self.figsize = figsize
        self.style = style
        if style != "default":
            plt.style.use(style)
    
    def save_figure(self, filename: str, dpi: int = 300, bbox_inches: str = "tight"):
        """
        Save current figure to file.
        
        Args:
            filename: Output filename
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box setting
        """
        plt.savefig(filename, dpi=dpi, bbox_inches=bbox_inches)


class DimensionalityPlotter(Plotter):
    """Create plots for dimensionality reduction results."""
    
    def plot_pca(
        self,
        transformed_data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        explained_variance: Optional[np.ndarray] = None,
        title: str = "PCA Plot"
    ):
        """
        Plot PCA results.
        
        Args:
            transformed_data: PCA-transformed data
            labels: Cluster or group labels
            explained_variance: Explained variance ratios
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if labels is not None:
            scatter = ax.scatter(
                transformed_data[:, 0],
                transformed_data[:, 1],
                c=labels,
                cmap="viridis",
                alpha=0.6,
                edgecolors="black",
                linewidth=0.5
            )
            plt.colorbar(scatter, ax=ax, label="Cluster")
        else:
            ax.scatter(
                transformed_data[:, 0],
                transformed_data[:, 1],
                alpha=0.6,
                edgecolors="black",
                linewidth=0.5
            )
        
        xlabel = "PC1"
        ylabel = "PC2"
        
        if explained_variance is not None and len(explained_variance) >= 2:
            xlabel += f" ({explained_variance[0]:.2%} variance)"
            ylabel += f" ({explained_variance[1]:.2%} variance)"
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def plot_scree(
        self,
        explained_variance: np.ndarray,
        title: str = "Scree Plot"
    ):
        """
        Plot scree plot for explained variance.
        
        Args:
            explained_variance: Explained variance ratios
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        n_components = len(explained_variance)
        x = range(1, n_components + 1)
        
        ax.bar(x, explained_variance, alpha=0.7, edgecolor="black")
        ax.plot(x, np.cumsum(explained_variance), 'ro-', linewidth=2, markersize=6)
        
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(x)
        
        return fig, ax


class ExpressionPlotter(Plotter):
    """Create plots for expression data."""
    
    def plot_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Expression Heatmap",
        cmap: str = "RdBu_r",
        cluster_rows: bool = True,
        cluster_cols: bool = True
    ):
        """
        Plot expression heatmap.
        
        Args:
            data: Expression data DataFrame
            title: Plot title
            cmap: Colormap
            cluster_rows: Whether to cluster rows
            cluster_cols: Whether to cluster columns
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(
            data,
            cmap=cmap,
            center=0,
            robust=True,
            cbar_kws={"label": "Expression"},
            ax=ax
        )
        
        ax.set_title(title)
        
        return fig, ax
    
    def plot_boxplot(
        self,
        data: pd.DataFrame,
        groupby: Optional[pd.Series] = None,
        title: str = "Expression Distribution"
    ):
        """
        Plot expression distribution as boxplot.
        
        Args:
            data: Expression data
            groupby: Grouping variable
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if groupby is not None:
            data_to_plot = []
            labels = []
            for group in groupby.unique():
                data_to_plot.append(data[groupby == group].values.flatten())
                labels.append(str(group))
            
            ax.boxplot(data_to_plot, labels=labels)
            ax.set_xlabel("Group")
        else:
            ax.boxplot(data.values)
        
        ax.set_ylabel("Expression")
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        return fig, ax


class StatisticalPlotter(Plotter):
    """Create plots for statistical analysis results."""
    
    def plot_volcano(
        self,
        results: pd.DataFrame,
        pvalue_col: str = "pvalue",
        fc_col: str = "fold_change",
        pvalue_threshold: float = 0.05,
        fc_threshold: float = 1.5,
        title: str = "Volcano Plot"
    ):
        """
        Plot volcano plot for differential analysis.
        
        Args:
            results: Results DataFrame with p-values and fold changes
            pvalue_col: Column name for p-values
            fc_col: Column name for fold changes
            pvalue_threshold: P-value threshold for significance
            fc_threshold: Fold change threshold for significance
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate -log10(pvalue)
        neg_log_pvalue = -np.log10(results[pvalue_col])
        
        # Determine colors based on significance
        colors = []
        for _, row in results.iterrows():
            if row[pvalue_col] < pvalue_threshold and abs(row[fc_col]) > fc_threshold:
                colors.append("red" if row[fc_col] > 0 else "blue")
            else:
                colors.append("gray")
        
        ax.scatter(results[fc_col], neg_log_pvalue, c=colors, alpha=0.6)
        
        ax.axhline(-np.log10(pvalue_threshold), color='black', linestyle='--', linewidth=1)
        ax.axvline(fc_threshold, color='black', linestyle='--', linewidth=1)
        ax.axvline(-fc_threshold, color='black', linestyle='--', linewidth=1)
        
        ax.set_xlabel("Fold Change")
        ax.set_ylabel("-log10(p-value)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig, ax


def plot_pca_results(
    transformed_data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    explained_variance: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Convenience function to plot PCA results.
    
    Args:
        transformed_data: PCA-transformed data
        labels: Cluster or group labels
        explained_variance: Explained variance ratios
        save_path: Path to save figure (optional)
    """
    plotter = DimensionalityPlotter()
    plotter.plot_pca(transformed_data, labels, explained_variance)
    
    if save_path:
        plotter.save_figure(save_path)
    
    plt.show()


def plot_heatmap(
    data: pd.DataFrame,
    save_path: Optional[str] = None,
    **kwargs
):
    """
    Convenience function to plot expression heatmap.
    
    Args:
        data: Expression data DataFrame
        save_path: Path to save figure (optional)
        **kwargs: Additional arguments for plot_heatmap
    """
    plotter = ExpressionPlotter()
    plotter.plot_heatmap(data, **kwargs)
    
    if save_path:
        plotter.save_figure(save_path)
    
    plt.show()


__all__ = [
    "Plotter",
    "DimensionalityPlotter",
    "ExpressionPlotter",
    "StatisticalPlotter",
    "plot_pca_results",
    "plot_heatmap",
]
