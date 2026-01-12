"""
FIREFate - Functional and Interpretable Regulatory Encoding of Fate determination

A comprehensive package for regulatory fate determination analysis,
encompassing data processing, analysis, visualization, and utilities.
"""

__version__ = "0.1.0"
__author__ = "Akanksha Sachan"
__license__ = "MIT"

from firefate import data_processing
from firefate import analysis
from firefate import visualization
from firefate import utils

__all__ = [
    "data_processing",
    "analysis",
    "visualization",
    "utils",
]
