# FIREFate

**Functional and Interpretable Regulatory Encoding of Fate determination**

FIREFate is a comprehensive Python package for regulatory fate determination analysis. It provides multiple modules encompassing downstream tasks accomplished by stitching together multiple other packages and custom scripts.

## Features

- **Data Processing**: Load, preprocess, and normalize data
- **Analysis**: Perform dimensionality reduction, clustering, and statistical analysis
- **Visualization**: Create publication-ready plots and figures
- **Utilities**: Helper functions for file I/O, logging, and configuration management

## Installation

### From PyPI (when published)

```bash
pip install firefate
```

### From source

```bash
git clone https://github.com/sachha-naksha/FIREFate-.git
cd FIREFate-
pip install -e .
```

## Quick Start

```python
import firefate
from firefate.data_processing import load_data, preprocess_data
from firefate.analysis import perform_pca, cluster_data
from firefate.visualization import plot_pca_results

# Load and preprocess data
data = load_data("data.csv")
processed_data = preprocess_data(data, normalize=True)

# Perform PCA
pca_data, variance = perform_pca(processed_data, n_components=2)

# Cluster data
labels = cluster_data(processed_data, n_clusters=3)

# Visualize results
plot_pca_results(pca_data, labels=labels, explained_variance=variance)
```

## Modules

### Data Processing (`firefate.data_processing`)

- `DataLoader`: Load data from various formats
- `DataPreprocessor`: Normalize and clean data
- Convenience functions: `load_data()`, `preprocess_data()`

### Analysis (`firefate.analysis`)

- `DimensionalityReducer`: PCA, t-SNE, UMAP
- `ClusterAnalyzer`: K-means, hierarchical clustering
- `StatisticalAnalyzer`: Differential analysis, correlation
- Convenience functions: `perform_pca()`, `cluster_data()`

### Visualization (`firefate.visualization`)

- `DimensionalityPlotter`: PCA plots, scree plots
- `ExpressionPlotter`: Heatmaps, boxplots
- `StatisticalPlotter`: Volcano plots
- Convenience functions: `plot_pca_results()`, `plot_heatmap()`

### Utilities (`firefate.utils`)

- `FileHandler`: File I/O operations
- `ConfigManager`: Configuration management
- `setup_logger()`: Logging setup
- Helper functions: `validate_dataframe()`, `merge_dataframes()`, `compute_metrics()`

## Requirements

- Python >= 3.8
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0

## Development

### Install development dependencies

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
pytest
```

### Format code

```bash
black firefate/
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Author

Akanksha Sachan

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
