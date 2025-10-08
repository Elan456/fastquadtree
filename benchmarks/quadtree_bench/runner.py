"""
Benchmark runner for quadtree performance testing.

This module handles the execution of benchmarks, data generation,
and result collection for performance analysis.
"""

import gc
import math
import random
import statistics as stats
from dataclasses import dataclass
from time import perf_counter as now
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from .engines import Engine


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    bounds: Tuple[int, int, int, int] = (0, 0, 1000, 1000)
    max_points: int = 64  # node capacity where supported
    max_depth: int = 1_000  # depth cap for fairness where supported
    n_queries: int = 100  # queries per experiment
    repeats: int = 3  # median over repeats
    rng_seed: int = 42  # random seed for reproducibility
    max_experiment_points: int = 100_000

    def __post_init__(self):
        """Generate experiment point sizes."""
        self.experiments = [2, 4, 8, 16]
        while self.experiments[-1] < self.max_experiment_points:
            self.experiments.append(int(self.experiments[-1] * 2))
        self.experiments[-1] = min(self.experiments[-1], self.max_experiment_points)


class BenchmarkRunner:
    """Handles execution of quadtree performance benchmarks."""

    def __init__(self, config: BenchmarkConfig | None = None):
        """Initialize with configuration."""
        self.config = config or BenchmarkConfig()
        self.rng = random.Random(self.config.rng_seed)

    def generate_points(
        self, n: int, rng: random.Random | None = None
    ) -> List[Tuple[int, int]]:
        """Generate n random points within bounds."""
        if rng is None:
            rng = self.rng
        x_min, y_min, x_max, y_max = self.config.bounds
        return [
            (rng.randint(x_min, x_max - 1), rng.randint(y_min, y_max - 1))
            for _ in range(n)
        ]

    def generate_queries(
        self, m: int, rng: random.Random | None = None
    ) -> List[Tuple[int, int, int, int]]:
        """Generate m random rectangular queries within bounds."""
        if rng is None:
            rng = self.rng
        x_min, y_min, x_max, y_max = self.config.bounds
        queries = []
        for _ in range(m):
            x = rng.randint(x_min, x_max)
            y = rng.randint(y_min, y_max)
            w = rng.randint(0, x_max - x) // rng.randint(1, 8)
            h = rng.randint(0, y_max - y) // rng.randint(1, 8)
            queries.append((x, y, x + w, y + h))
        return queries

    def benchmark_engine_once(
        self,
        engine: Engine,
        points: List[Tuple[int, int]],
        queries: List[Tuple[int, int, int, int]],
    ) -> Tuple[float, float]:
        """Run a single benchmark iteration for an engine."""
        # Separate build vs query timing
        t0 = now()
        tree = engine.build(points)
        t_build = now() - t0

        t0 = now()
        engine.query(tree, queries)
        t_query = now() - t0

        return t_build, t_query

    def median_or_nan(self, vals: List[float]) -> float:
        """Calculate median, returning NaN for empty/invalid data."""
        cleaned = [x for x in vals if isinstance(x, (int, float)) and not math.isnan(x)]
        return stats.median(cleaned) if cleaned else math.nan

    def _print_experiment_summary(
        self, n: int, results: Dict[str, Any], exp_idx: int
    ) -> None:
        """Print a summary of results for the current experiment."""

        def fmt(x):
            return f"{x:.3f}" if not math.isnan(x) else "nan"

        # Get the results for this experiment (last index)
        total = results["total"]
        build = results["build"]
        query = results["query"]

        # Find the fastest engine for this experiment
        valid_engines = [
            (name, total[name][exp_idx])
            for name in total
            if not math.isnan(total[name][exp_idx])
        ]

        if not valid_engines:
            return

        fastest = min(valid_engines, key=lambda x: x[1])

        print(f"\n  📊 Results for {n:,} points:")
        print(f"     Fastest: {fastest[0]} ({fmt(fastest[1])}s total)")

        # Show top 3 performers
        sorted_engines = sorted(valid_engines, key=lambda x: x[1])[:3]
        for rank, (name, time) in enumerate(sorted_engines, 1):
            b = build[name][exp_idx]
            q = query[name][exp_idx]
            print(
                f"     {rank}. {name:15} build={fmt(b)}s, query={fmt(q)}s, total={fmt(time)}s"
            )
        print()

    def run_benchmark(self, engines: Dict[str, Engine]) -> Dict[str, Any]:
        """
        Run complete benchmark suite.

        Args:
            engines: Dictionary of engine name -> Engine instance

        Returns:
            Dictionary containing benchmark results
        """
        # Warmup on a small set to JIT caches, etc.
        print("Warming up engines...")
        warmup_points = self.generate_points(2_000)
        warmup_queries = self.generate_queries(self.config.n_queries)
        for engine in engines.values():
            self.benchmark_engine_once(engine, warmup_points, warmup_queries)

        # Initialize result containers
        results = {
            "total": {name: [] for name in engines},
            "build": {name: [] for name in engines},
            "query": {name: [] for name in engines},
            "insert_rate": {name: [] for name in engines},
            "query_rate": {name: [] for name in engines},
        }

        # Run experiments
        print(
            f"\nRunning {len(self.config.experiments)} experiments with {len(engines)} engines..."
        )
        experiment_bar = tqdm(
            self.config.experiments, desc="Experiments", unit="exp", position=0
        )

        for exp_idx, n in enumerate(experiment_bar):
            experiment_bar.set_description(
                f"Experiment {exp_idx + 1}/{len(self.config.experiments)}"
            )
            experiment_bar.set_postfix({"points": f"{n:,}"})

            # Generate data for this experiment
            exp_rng = random.Random(10_000 + n)
            points = self.generate_points(n, exp_rng)
            queries = self.generate_queries(self.config.n_queries, exp_rng)

            # Collect results across repeats
            engine_times = {name: {"build": [], "query": []} for name in engines}

            # Progress bar for engines x repeats
            total_iterations = len(engines) * self.config.repeats
            engine_bar = tqdm(
                total=total_iterations,
                desc="  Testing engines",
                unit="run",
                position=1,
                leave=False,
            )

            for repeat in range(self.config.repeats):
                gc.disable()

                # Benchmark each engine
                for name, engine in engines.items():
                    engine_bar.set_description(
                        f"  {name} (repeat {repeat + 1}/{self.config.repeats})"
                    )

                    try:
                        build_time, query_time = self.benchmark_engine_once(
                            engine, points, queries
                        )
                    except Exception as e:  # noqa: BLE001
                        # Mark as failed for this repeat
                        print(
                            f"  {name} (repeat {repeat + 1}/{self.config.repeats}) failed: {e}"
                        )
                        build_time, query_time = math.nan, math.nan

                    engine_times[name]["build"].append(build_time)
                    engine_times[name]["query"].append(query_time)

                    engine_bar.update(1)

                gc.enable()

            engine_bar.close()

            # Calculate medians and derived metrics
            for name in engines:
                build_median = self.median_or_nan(engine_times[name]["build"])
                query_median = self.median_or_nan(engine_times[name]["query"])
                total_median = (
                    build_median + query_median
                    if not math.isnan(build_median) and not math.isnan(query_median)
                    else math.nan
                )

                results["build"][name].append(build_median)
                results["query"][name].append(query_median)
                results["total"][name].append(total_median)

                # Calculate rates
                insert_rate = (
                    (n / build_median) if build_median and build_median > 0 else 0.0
                )
                query_rate = (
                    (self.config.n_queries / query_median)
                    if query_median and query_median > 0
                    else 0.0
                )

                results["insert_rate"][name].append(insert_rate)
                results["query_rate"][name].append(query_rate)

            # Print intermediate results for this experiment
            self._print_experiment_summary(n, results, exp_idx)

        experiment_bar.close()

        # Add metadata to results
        results["engines"] = engines  # pyright: ignore[reportArgumentType]
        results["config"] = self.config  # pyright: ignore[reportArgumentType]

        return results

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print markdown summary of benchmark results."""
        total = results["total"]
        build = results["build"]
        query = results["query"]
        config = results["config"]

        # Use largest dataset for summary
        i = len(config.experiments) - 1

        def fmt(x):
            return f"{x:.3f}" if x is not None and not math.isnan(x) else "nan"

        print("\n### Summary (largest dataset, PyQtree baseline)")
        print(
            f"- Points: **{config.experiments[i]:,}**, Queries: **{config.n_queries}**"
        )

        # Find fastest and show key results
        ranked = sorted(
            total.keys(),
            key=lambda n: total[n][i] if not math.isnan(total[n][i]) else float("inf"),
        )
        best = ranked[0]
        pyqt_total = total.get("PyQtree", [math.nan])[i]

        print(f"- Fastest total: **{best}** at **{fmt(total[best][i])} s**")

        # Results table
        print("\n| Library | Build (s) | Query (s) | Total (s) | Speed vs PyQtree |")
        print("|---|---:|---:|---:|---:|")

        def rel_speed(name: str) -> str:
            t = total[name][i]
            if math.isnan(pyqt_total) or math.isnan(t) or t <= 0:
                return "n/a"
            return f"{(pyqt_total / t):.2f}×"  # noqa: RUF001

        for name in ranked:
            b = build.get(name, [math.nan])[i] if name in build else math.nan
            q = query[name][i]
            t = total[name][i]
            print(f"| {name:12} | {fmt(b)} | {fmt(q)} | {fmt(t)} | {rel_speed(name)} |")

        print("")

        # Config table
        print("#### Benchmark Configuration")
        print("| Parameter | Value |")
        print("|---|---:|")
        print(f"| Bounds | {config.bounds} |")
        print(f"| Max points per node | {config.max_points} |")
        print(f"| Max depth | {config.max_depth} |")
        print(f"| Queries per experiment | {config.n_queries} |")
