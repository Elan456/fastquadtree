import math
import random
from dataclasses import dataclass
from statistics import median

from tqdm import tqdm

from .engines import _create_fastquadtree_np_engine
from .runner import BenchmarkConfig, BenchmarkRunner

# --------- helpers ---------


def _latin_hypercube(n_samples: int, dims: int, rng: random.Random):
    """Return n_samples points in [0,1]^dims using a simple Latin Hypercube."""
    # One stratified coordinate per dim
    cut = [
        [(i + rng.random()) / n_samples for i in range(n_samples)] for _ in range(dims)
    ]
    # Shuffle each dimension independently
    for d in range(dims):
        rng.shuffle(cut[d])
    # Zip to points
    return [[cut[d][i] for d in range(dims)] for i in range(n_samples)]


def _round_to_choice(x: float, choices: list[int]) -> int:
    return min(choices, key=lambda c: abs(c - x))


def _powers_of_two_like(low: int, high: int) -> list[int]:
    """Generate a dense set skewed toward powers of two inside [low, high]."""
    s = set()
    # all powers of two in range
    p = 1
    while p < low:
        p <<= 1
    while p <= high:
        s.add(p)
        # also neighbors around each power of two for finesse
        for k in (p // 2, p // 4, p + p // 2, p + p // 4):
            if low <= k <= high:
                s.add(k)
        p <<= 1
    # fill any large gaps up to a decent resolution
    step = max(1, (high - low) // 24)
    for v in range(low, high + 1, step):
        s.add(v)
    return sorted(s)


def _median_of_means(times: list[float], groups: int = 3) -> float:
    """Robust aggregate: split into groups, average each, take median of those means."""
    if not times:
        return float("inf")
    g = max(1, min(groups, len(times)))
    chunk = math.ceil(len(times) / g)
    means = []
    for i in range(0, len(times), chunk):
        seg = times[i : i + chunk]
        means.append(sum(seg) / len(seg))
    return median(means)


@dataclass(frozen=True)
class QTConfig:
    max_points_per_leaf: int
    max_depth: int


# --------- main optimizer ---------


def optimize(bounds, *, rng_seed: int = 42):
    """
    Hyperband-style successive halving to find good (max_points_per_leaf, max_depth)
    for ~100k points. Returns (best_max_points_per_leaf, best_max_depth).
    """
    rng = random.Random(rng_seed)

    # Search space
    mp_choices = _powers_of_two_like(4, 256)  # leaf capacity
    md_choices = list(range(6, 25))  # depth 6..24

    # Multi-fidelity budgets: cheap -> medium -> full
    # You can tune these if your machine is faster/slower.
    budgets = [
        {"max_experiment_points": 25_000, "n_queries": 200, "repeats": 1},
        {"max_experiment_points": 50_000, "n_queries": 500, "repeats": 2},
        {"max_experiment_points": 100_000, "n_queries": 1000, "repeats": 3},
    ]

    # Successive halving parameters
    eta = 3  # keep roughly top 1/eta each round
    n0 = 60  # initial number of candidates to try; adjust for your machine

    # Latin Hypercube sample over [0,1]^2, then map to discrete choices
    lhs = _latin_hypercube(n0, 2, rng)
    candidates = []
    for u, v in lhs:
        mp = _round_to_choice(
            u * (mp_choices[-1] - mp_choices[0]) + mp_choices[0], mp_choices
        )
        md = _round_to_choice(
            v * (md_choices[-1] - md_choices[0]) + md_choices[0], md_choices
        )
        candidates.append(QTConfig(mp, md))
    # De-duplicate in case rounding collided
    candidates = list(dict.fromkeys(candidates))

    best_cfg = None
    best_score = float("inf")

    for stage, budget in enumerate(budgets):
        stage_label = f"Stage {stage + 1}/{len(budgets)}"
        scores = []

        bar = tqdm(
            candidates, desc=f"{stage_label} evaluating {len(candidates)} configs"
        )
        for cfg in bar:
            # Build a config per budget
            bc = BenchmarkConfig(
                bounds=bounds,
                max_points=cfg.max_points_per_leaf,
                max_depth=cfg.max_depth,
                n_queries=budget["n_queries"],
                repeats=budget["repeats"],
                rng_seed=rng_seed,
                max_experiment_points=budget["max_experiment_points"],
                verbose=False,
            )
            # Pin to the fidelity we want
            bc.experiments = [budget["max_experiment_points"]]

            # Run a few short tries to reduce noise
            run_times = []
            tries = 2 if stage < len(budgets) - 1 else 3
            for _ in range(tries):
                runner = BenchmarkRunner(bc)
                results = runner.run_benchmark(
                    {
                        "fastquadtree": _create_fastquadtree_np_engine(
                            bounds, cfg.max_points_per_leaf, cfg.max_depth
                        )
                    }
                )
                t = results["total"]["fastquadtree"][0]
                run_times.append(t)

            score = _median_of_means(run_times)
            scores.append((score, cfg))

            # Track best so far for user feedback
            if score < best_score:
                best_score = score
                best_cfg = cfg
            if best_cfg is not None:
                bar.set_description(
                    f"{stage_label}: best mp={best_cfg.max_points_per_leaf}, md={best_cfg.max_depth}, time={best_score:.4f}s"
                )

        # Prune to top ~1/eta for next stage, but always keep at least 6
        scores.sort(key=lambda x: x[0])
        keep = max(6, math.ceil(len(scores) / eta))
        candidates = [cfg for _, cfg in scores[:keep]]

    # Final answer at full budget
    if best_cfg is not None:
        print(
            f"Optimized max_points: {best_cfg.max_points_per_leaf}, max_depth: {best_cfg.max_depth} (time ~{best_score:.4f}s)"
        )
        return best_cfg.max_points_per_leaf, best_cfg.max_depth
    raise RuntimeError("No best configuration found")
