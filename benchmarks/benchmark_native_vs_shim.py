# benchmarks/bench_native_vs_shim.py
from __future__ import annotations

import argparse
import gc
import random
import statistics as stats
from time import perf_counter as now

from pyqtree import Index as PyQTreeIndex
from system_info_collector import collect_system_info, format_system_info_markdown_lite
from tqdm import tqdm

from fastquadtree import QuadTree as ShimQuadTree
from fastquadtree._native import QuadTree as NativeQuadTree
from fastquadtree.pyqtree import Index as FQTIndex

BOUNDS = (0.0, 0.0, 1000.0, 1000.0)
CAPACITY = 64
MAX_DEPTH = 10
SEED = 42


def gen_points(n: int, rng: random.Random):
    return [(rng.randint(0, 999), rng.randint(0, 999)) for _ in range(n)]


def gen_queries(m: int, rng: random.Random):
    qs = []
    for _ in range(m):
        x = rng.randint(0, 1000)
        y = rng.randint(0, 1000)
        w = rng.randint(0, 1000 - x)
        h = rng.randint(0, 1000 - y)
        qs.append((x, y, x + w, y + h))
    return qs


def bench_native(points, queries):
    t0 = now()
    qt = NativeQuadTree(BOUNDS, CAPACITY, max_depth=MAX_DEPTH)
    for i, p in enumerate(points):
        qt.insert(i, p)
    t_build = now() - t0

    t0 = now()
    for q in queries:
        _ = qt.query(q)
    t_query = now() - t0
    return t_build, t_query


def bench_shim(points, queries, *, track_objects: bool, with_objs: bool):
    # track_objects controls the map. with_objs decides if we actually store objects.
    t0 = now()
    qt = ShimQuadTree(
        BOUNDS, CAPACITY, max_depth=MAX_DEPTH, track_objects=track_objects
    )
    if with_objs:
        for i, p in enumerate(points):
            qt.insert(p, obj=i)  # store a tiny object
    else:
        for _, p in enumerate(points):
            qt.insert(p)
    t_build = now() - t0

    t0 = now()
    for q in queries:
        _ = qt.query(q, as_items=track_objects)  # tuples path for speed
    t_query = now() - t0
    return t_build, t_query


def bench_pyqtree(points, queries, fqt: bool):
    """
    Benchmarks the pyqtree compatibility shim vs the original pyqtree.

    Set fqt to True to use the shim, False to use the original pyqtree.
    """
    t0 = now()
    qt = (
        FQTIndex(bbox=BOUNDS, max_items=CAPACITY, max_depth=MAX_DEPTH)
        if fqt
        else PyQTreeIndex(bbox=BOUNDS, max_items=CAPACITY, max_depth=MAX_DEPTH)
    )
    for i, p in enumerate(points):
        box = (p[0], p[1], p[0], p[1])
        qt.insert(i, box)
    t_build = now() - t0

    t0 = now()
    for q in queries:
        _ = qt.intersect(q)
    t_query = now() - t0
    return t_build, t_query


def median_times(fn, points, queries, repeats: int, desc: str = "Running"):
    """Run benchmark multiple times and return median times."""
    builds, queries_t = [], []
    for _ in tqdm(range(repeats), desc=desc, unit="run"):
        gc.disable()
        b, q = fn(points, queries)
        gc.enable()
        builds.append(b)
        queries_t.append(q)
    return stats.median(builds), stats.median(queries_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=500_000)
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    print("Native vs Shim Benchmark")
    print("=" * 50)
    print("Configuration:")
    print(f"  Points: {args.points:,}")
    print(f"  Queries: {args.queries}")
    print(f"  Repeats: {args.repeats}")
    print()

    rng = random.Random(SEED)
    points = gen_points(args.points, rng)
    queries = gen_queries(args.queries, rng)

    # Warmup to load modules
    print("Warming up...")
    _ = bench_native(points[:1000], queries[:50])
    _ = bench_shim(points[:1000], queries[:50], track_objects=False, with_objs=False)
    print()

    print("Running benchmarks...")
    n_build, n_query = median_times(
        lambda pts, qs: bench_native(pts, qs),
        points,
        queries,
        args.repeats,
        desc="Native",
    )
    s_build_no_map, s_query_no_map = median_times(
        lambda pts, qs: bench_shim(pts, qs, track_objects=False, with_objs=False),
        points,
        queries,
        args.repeats,
        desc="Shim (no tracking)",
    )
    s_build_map, s_query_map = median_times(
        lambda pts, qs: bench_shim(pts, qs, track_objects=True, with_objs=True),
        points,
        queries,
        args.repeats,
        desc="Shim (tracking)",
    )
    p_build, p_query = median_times(
        lambda pts, qs: bench_pyqtree(pts, qs, fqt=False),
        points,
        queries,
        args.repeats,
        desc="pyqtree (original)",
    )
    fqt_build, fqt_query = median_times(
        lambda pts, qs: bench_pyqtree(pts, qs, fqt=True),
        points,
        queries,
        args.repeats,
        desc="pyqtree (FQT shim)",
    )
    print()

    def fmt(x):
        return f"{x:.3f}"

    md = f"""
## Native vs Shim

### Configuration
- Points: {args.points:,}
- Queries: {args.queries}
- Repeats: {args.repeats}

### Results

| Variant | Build | Query | Total |
|---|---:|---:|---:|
| Native | {fmt(n_build)} | {fmt(n_query)} | {fmt(n_build + n_query)} |
| Shim (no tracking) | {fmt(s_build_no_map)} | {fmt(s_query_no_map)} | {fmt(s_build_no_map + s_query_no_map)} |
| Shim (tracking) | {fmt(s_build_map)} | {fmt(s_query_map)} | {fmt(s_build_map + s_query_map)} |

### Summary

Using the shim with object tracking increases build time by {fmt(s_build_map / n_build)}x and query time by {fmt(s_query_map / n_query)}x.
**Total slowdown = {fmt((s_build_map + s_query_map) / (n_build + n_query))}x.**

Adding the object map tends to only impact the build time, not the query time.

## pyqtree drop-in shim performance gains

### Configuration
- Points: {args.points:,}
- Queries: {args.queries}
- Repeats: {args.repeats}

### Results

| Variant | Build | Query | Total |
|---|---:|---:|---:|
| pyqtree (fastquadtree) | {fmt(fqt_build)} | {fmt(fqt_query)} | {fmt(fqt_build + fqt_query)} |
| pyqtree (original) | {fmt(p_build)} | {fmt(p_query)} | {fmt(p_build + p_query)} |

### Summary

If you directly replace pyqtree with the drop-in `fastquadtree.pyqtree.Index` shim, you get a build time of {fmt(fqt_build)}s and query time of {fmt(fqt_query)}s.
This is a **total speedup of {fmt((p_build + p_query) / (fqt_build + fqt_query))}x** compared to the original pyqtree and requires no code changes.

"""
    print(md.strip())

    info = collect_system_info()
    print(format_system_info_markdown_lite(info))


if __name__ == "__main__":
    main()
