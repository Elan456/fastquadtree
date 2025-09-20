"""
Quadtree benchmarking package.

This package provides comprehensive benchmarking capabilities for various quadtree
implementations, including performance comparison, visualization, and analysis.
"""

from .engines import Engine, get_engines
from .runner import BenchmarkRunner, BenchmarkConfig
from .plotting import PlotManager

__version__ = "1.0.0"
__all__ = ["Engine", "get_engines", "BenchmarkRunner", "BenchmarkConfig", "PlotManager"]