"""
FIREFate Example Usage

This example demonstrates how to use the FIREFate package for
regulatory fate determination analysis.
"""

import numpy as np
import pandas as pd

# Import FIREFate modules
from firefate.data_processing import DataLoader, DataPreprocessor, preprocess_data
from firefate.analysis import perform_pca, cluster_data, StatisticalAnalyzer
from firefate.visualization import plot_pca_results, plot_heatmap
from firefate.utils import setup_logger, FileHandler

# Set up logging
logger = setup_logger(name="firefate_example", level=20)  # INFO level

logger.info("Starting FIREFate analysis example")

# Generate synthetic data for demonstration
np.random.seed(42)
n_samples = 100
n_features = 50

# Create synthetic expression data
data = pd.DataFrame(
    np.random.randn(n_samples, n_features),
    columns=[f"gene_{i}" for i in range(n_features)],
    index=[f"sample_{i}" for i in range(n_samples)]
)

logger.info(f"Generated synthetic data: {data.shape}")

# 1. Data Preprocessing
logger.info("Preprocessing data...")
preprocessor = DataPreprocessor()

# Normalize data
normalized_data = preprocessor.normalize(data, method="zscore")
logger.info("Data normalized using z-score")

# Filter low variance features
filtered_data = preprocessor.filter_low_variance(normalized_data, threshold=0.1)
logger.info(f"Filtered data shape: {filtered_data.shape}")

# 2. Dimensionality Reduction
logger.info("Performing PCA...")
pca_data, explained_variance = perform_pca(filtered_data, n_components=2)
logger.info(f"PCA explained variance: {explained_variance}")

# 3. Clustering
logger.info("Clustering data...")
labels = cluster_data(filtered_data, n_clusters=3)
logger.info(f"Clustering completed. Found {len(np.unique(labels))} clusters")

# 4. Statistical Analysis
logger.info("Performing statistical analysis...")
analyzer = StatisticalAnalyzer()

# Split data into two groups for differential analysis
group1 = filtered_data.iloc[:50]
group2 = filtered_data.iloc[50:]

diff_results = analyzer.differential_analysis(group1, group2, test="ttest")
significant = diff_results[diff_results["pvalue"] < 0.05]
logger.info(f"Found {len(significant)} significantly different features")

# Compute correlation matrix
corr_matrix = analyzer.correlation_analysis(filtered_data.T)
logger.info(f"Computed correlation matrix: {corr_matrix.shape}")

# 5. Visualization
logger.info("Creating visualizations...")

# Plot PCA results
try:
    plot_pca_results(
        pca_data,
        labels=labels,
        explained_variance=explained_variance,
        save_path="examples/pca_plot.png"
    )
    logger.info("PCA plot saved to examples/pca_plot.png")
except Exception as e:
    logger.warning(f"Could not save PCA plot: {e}")

# Plot heatmap of top varying features
try:
    top_features = filtered_data.var().nlargest(20).index
    plot_heatmap(
        filtered_data[top_features].T,
        save_path="examples/heatmap.png",
        title="Top 20 Variable Features"
    )
    logger.info("Heatmap saved to examples/heatmap.png")
except Exception as e:
    logger.warning(f"Could not save heatmap: {e}")

# 6. Save results
logger.info("Saving results...")
file_handler = FileHandler(base_dir="examples")

# Save differential analysis results
try:
    diff_results.to_csv("examples/differential_analysis.csv", index=False)
    logger.info("Differential analysis results saved")
except Exception as e:
    logger.warning(f"Could not save results: {e}")

# Save cluster assignments
cluster_df = pd.DataFrame({
    "sample": filtered_data.index,
    "cluster": labels
})
try:
    cluster_df.to_csv("examples/cluster_assignments.csv", index=False)
    logger.info("Cluster assignments saved")
except Exception as e:
    logger.warning(f"Could not save cluster assignments: {e}")

logger.info("FIREFate analysis example completed!")

print("\n" + "="*60)
print("FIREFate Analysis Complete!")
print("="*60)
print(f"\nData shape: {data.shape}")
print(f"Filtered shape: {filtered_data.shape}")
print(f"Number of clusters: {len(np.unique(labels))}")
print(f"Significant features: {len(significant)}")
print(f"PCA variance explained: {sum(explained_variance):.2%}")
print("\nResults saved to 'examples/' directory")
print("="*60)
