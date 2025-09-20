#!/usr/bin/env python3
"""
Entry point script for running quadtree benchmarks.

This script can be run directly or imported as a module.
"""

import sys
from pathlib import Path

# Add the benchmarks directory to Python path for imports
benchmark_dir = Path(__file__).parent
if str(benchmark_dir) not in sys.path:
    sys.path.insert(0, str(benchmark_dir))

# Now we can import the package
from quadtree_bench.main import main, run_quick_benchmark

if __name__ == "__main__":
    # Check if user wants quick benchmark
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_benchmark()
    else:
        main()