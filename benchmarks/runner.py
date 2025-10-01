#!/usr/bin/env python3
"""
Entry point script for running quadtree benchmarks.

This script can be run directly or imported as a module.
"""

import sys

from quadtree_bench.main import main, run_quick_benchmark

if __name__ == "__main__":
    # Check if user wants quick benchmark
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_benchmark()
    else:
        main()
