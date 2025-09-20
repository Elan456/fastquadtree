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
from typing import Dict, List, Tuple, Any

from tqdm import tqdm

from .engines import Engine


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    bounds: Tuple[int, int, int, int] = (0, 0, 1000, 1000)
    max_points: int = 20           # node capacity where supported
    max_depth: int = 10            # depth cap for fairness where supported
    n_queries: int = 500           # queries per experiment
    repeats: int = 3               # median over repeats
    rng_seed: int = 42             # random seed for reproducibility
    max_experiment_points: int = 500_000
    
    def __post_init__(self):
        """Generate experiment point sizes."""
        self.experiments = [2, 4, 8, 16]
        while self.experiments[-1] < self.max_experiment_points:
            self.experiments.append(int(self.experiments[-1] * 1.5))
        if self.experiments[-1] > self.max_experiment_points:
            self.experiments[-1] = self.max_experiment_points


class BenchmarkRunner:
    """Handles execution of quadtree performance benchmarks."""
    
    def __init__(self, config: BenchmarkConfig | None = None):
        """Initialize with configuration."""
        self.config = config or BenchmarkConfig()
        self.rng = random.Random(self.config.rng_seed)
    
    def generate_points(self, n: int, rng: random.Random | None = None) -> List[Tuple[int, int]]:
        """Generate n random points within bounds."""
        if rng is None:
            rng = self.rng
        x_min, y_min, x_max, y_max = self.config.bounds
        return [(rng.randint(x_min, x_max - 1), rng.randint(y_min, y_max - 1)) for _ in range(n)]
    
    def generate_queries(self, m: int, rng: random.Random | None = None) -> List[Tuple[int, int, int, int]]:
        """Generate m random rectangular queries within bounds."""
        if rng is None:
            rng = self.rng
        x_min, y_min, x_max, y_max = self.config.bounds
        queries = []
        for _ in range(m):
            x = rng.randint(x_min, x_max)
            y = rng.randint(y_min, y_max)
            w = rng.randint(0, x_max - x)
            h = rng.randint(0, y_max - y)
            queries.append((x, y, x + w, y + h))
        return queries
    
    def benchmark_engine_once(self, 
                             engine: Engine,
                             points: List[Tuple[int, int]],
                             queries: List[Tuple[int, int, int, int]]) -> Tuple[float, float]:
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
    
    def run_benchmark(self, engines: Dict[str, Engine]) -> Dict[str, Any]:
        """
        Run complete benchmark suite.
        
        Args:
            engines: Dictionary of engine name -> Engine instance
            
        Returns:
            Dictionary containing benchmark results
        """
        # Warmup on a small set to JIT caches, etc.
        warmup_points = self.generate_points(2_000)
        warmup_queries = self.generate_queries(self.config.n_queries)
        for engine in engines.values():
            try:
                self.benchmark_engine_once(engine, warmup_points, warmup_queries)
            except Exception:
                pass  # Ignore warmup failures
        
        # Initialize result containers
        results = {
            "total": {name: [] for name in engines},
            "build": {name: [] for name in engines},
            "query": {name: [] for name in engines},
            "insert_rate": {name: [] for name in engines},
            "query_rate": {name: [] for name in engines},
        }
        
        # Run experiments
        iterator = tqdm(self.config.experiments, desc="Experiments", unit="points")
        for n in iterator:
            iterator.set_postfix({"points": n})
            
            # Generate data for this experiment
            exp_rng = random.Random(10_000 + n)
            points = self.generate_points(n, exp_rng)
            queries = self.generate_queries(self.config.n_queries, exp_rng)
            
            # Collect results across repeats
            engine_times = {name: {"build": [], "query": []} for name in engines}
            
            for repeat in range(self.config.repeats):
                gc.disable()
                
                # Benchmark each engine
                for name, engine in engines.items():
                    try:
                        build_time, query_time = self.benchmark_engine_once(engine, points, queries)
                    except Exception:
                        # Mark as failed for this repeat
                        build_time, query_time = math.nan, math.nan
                    
                    engine_times[name]["build"].append(build_time)
                    engine_times[name]["query"].append(query_time)
                
                gc.enable()
            
            # Calculate medians and derived metrics
            for name in engines:
                build_median = self.median_or_nan(engine_times[name]["build"])
                query_median = self.median_or_nan(engine_times[name]["query"])
                total_median = build_median + query_median if not math.isnan(build_median) and not math.isnan(query_median) else math.nan
                
                results["build"][name].append(build_median)
                results["query"][name].append(query_median)
                results["total"][name].append(total_median)
                
                # Calculate rates
                insert_rate = (n / build_median) if build_median and build_median > 0 else 0.0
                query_rate = (self.config.n_queries / query_median) if query_median and query_median > 0 else 0.0
                
                results["insert_rate"][name].append(insert_rate)
                results["query_rate"][name].append(query_rate)
        
        # Add metadata to results
        results["engines"] = engines
        results["config"] = self.config
        
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
        print(f"- Points: **{config.experiments[i]:,}**, Queries: **{config.n_queries}**")
        print("--------------------")
        
        # Find fastest and show key results
        ranked = sorted(total.keys(), key=lambda n: total[n][i] if not math.isnan(total[n][i]) else float('inf'))
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
            return f"{(pyqt_total / t):.2f}×"
        
        for name in ranked:
            b = build.get(name, [math.nan])[i] if name in build else math.nan
            q = query[name][i]
            t = total[name][i]
            print(f"| {name:12} | {fmt(b)} | {fmt(q)} | {fmt(t)} | {rel_speed(name)} |")
        
        print("")