#!/usr/bin/env python3
"""
Main entry point for quadtree benchmarking.

This script provides a command-line interface for running comprehensive
quadtree performance benchmarks with customizable parameters.
"""

import argparse
from pathlib import Path

from .engines import get_engines
from .plotting import PlotManager
from .runner import BenchmarkConfig, BenchmarkRunner


def main():
    """Main entry point for benchmark execution."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive quadtree performance benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Benchmark parameters
    parser.add_argument(
        "--max-points",
        type=int,
        default=128,
        help="Maximum points per node before splitting",
    )
    parser.add_argument("--max-depth", type=int, default=16, help="Maximum tree depth")
    parser.add_argument(
        "--n-queries", type=int, default=500, help="Number of queries per experiment"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeat runs (median will be taken)",
    )
    parser.add_argument(
        "--max-experiment-points",
        type=int,
        default=250_000,
        help="Maximum number of points in largest experiment",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible results"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="assets",
        help="Directory to save output images",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="quadtree_bench",
        help="Prefix for output filenames",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save plots to files"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Don't show plots in browser"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print text summary, skip plotting",
    )

    # Bounds configuration
    parser.add_argument(
        "--bounds",
        nargs=4,
        type=int,
        default=[0, 0, 1000, 1000],
        metavar=("MIN_X", "MIN_Y", "MAX_X", "MAX_Y"),
        help="Bounding box for quadtrees",
    )

    args = parser.parse_args()

    # Create configuration
    config = BenchmarkConfig(
        bounds=tuple(args.bounds),
        max_points=args.max_points,
        max_depth=args.max_depth,
        n_queries=args.n_queries,
        repeats=args.repeats,
        rng_seed=args.seed,
        max_experiment_points=args.max_experiment_points,
    )

    print("Quadtree Benchmark Suite")
    print("=" * 50)
    print("Configuration:")
    print(f"  Bounds: {config.bounds}")
    print(f"  Max points per node: {config.max_points}")
    print(f"  Max depth: {config.max_depth}")
    print(f"  Queries per experiment: {config.n_queries}")
    print(f"  Repeats: {config.repeats}")
    print(f"  Experiments: {config.experiments}")
    print(f"  Total experiments: {len(config.experiments)}")
    print()

    # Get available engines
    engines = get_engines(config.bounds, config.max_points, config.max_depth)
    print(f"Available engines: {list(engines.keys())}")
    print()

    # Run benchmark
    runner = BenchmarkRunner(config)
    print("Starting benchmark...")
    results = runner.run_benchmark(engines)
    print("Benchmark completed!")
    print()

    # Print summary
    runner.print_summary(results)

    # Create and handle plots
    if not args.summary_only:
        plot_manager = PlotManager(results)
        time_fig, throughput_fig = plot_manager.create_all_plots()

        # Save plots if requested
        if not args.no_save:
            # Ensure output directory exists
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            plot_manager.save_plots(
                time_fig, throughput_fig, args.output_prefix, args.output_dir
            )

        # Show plots if requested
        if not args.no_show:
            plot_manager.show_plots(time_fig, throughput_fig)


def run_quick_benchmark():
    """Run a quick benchmark with default settings."""
    config = BenchmarkConfig()
    config.experiments = [100, 500, 1000, 5000]  # Smaller experiment set
    config.repeats = 1  # Single run for speed

    engines = get_engines(config.bounds, config.max_points, config.max_depth)
    runner = BenchmarkRunner(config)

    print("Running quick benchmark...")
    results = runner.run_benchmark(engines)
    runner.print_summary(results)

    return results


def run_custom_benchmark(
    bounds=(0, 0, 1000, 1000), max_points=20, max_depth=10, experiments=None
):
    """
    Run a custom benchmark with specified parameters.

    Args:
        bounds: Quadtree bounding box
        max_points: Maximum points per node
        max_depth: Maximum tree depth
        experiments: List of point counts to test

    Returns:
        Benchmark results dictionary
    """
    config = BenchmarkConfig(bounds=bounds, max_points=max_points, max_depth=max_depth)

    if experiments:
        config.experiments = experiments

    engines = get_engines(bounds, max_points, max_depth)
    runner = BenchmarkRunner(config)

    return runner.run_benchmark(engines)


if __name__ == "__main__":
    main()
