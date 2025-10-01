"""
Quadtree benchmarking package.

This package provides comprehensive benchmarking capabilities for various quadtree
implementations, including performance comparison, visualization, and analysis.
"""

from .engines import Engine, get_engines
from .plotting import PlotManager
from .runner import BenchmarkConfig, BenchmarkRunner

__version__ = "1.0.0"
__all__ = ["BenchmarkConfig", "BenchmarkRunner", "Engine", "PlotManager", "get_engines"]
