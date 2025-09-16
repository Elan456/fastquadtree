# benchmarks/bench_native_vs_shim.py
from __future__ import annotations

import argparse
import gc
import random
import statistics as stats
from time import perf_counter as now
from tqdm import tqdm 

from quadtree_rs._native import QuadTree as NativeQuadTree
from quadtree_rs import QuadTree as ShimQuadTree


BOUNDS = (0.0, 0.0, 1000.0, 1000.0)
CAPACITY = 20
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
    qt = ShimQuadTree(BOUNDS, CAPACITY, max_depth=MAX_DEPTH, track_objects=track_objects)
    if with_objs:
        for i, p in enumerate(points):
            qt.insert(p, id=i, obj=i)  # store a tiny object
    else:
        for i, p in enumerate(points):
            qt.insert(p, id=i)
    t_build = now() - t0

    t0 = now()
    for q in queries:
        _ = qt.query(q)  # tuples path for speed
    t_query = now() - t0
    return t_build, t_query


def median_times(fn, points, queries, repeats: int):
    builds, queries_t = [], []
    for _ in tqdm(range(repeats)):
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

    rng = random.Random(SEED)
    points = gen_points(args.points, rng)
    queries = gen_queries(args.queries, rng)

    # Warmup to load modules
    _ = bench_native(points[:1000], queries[:50])
    _ = bench_shim(points[:1000], queries[:50], track_objects=False, with_objs=False)

    n_build, n_query = median_times(lambda pts, qs: bench_native(pts, qs), points, queries, args.repeats)
    s_build_no_map, s_query_no_map = median_times(
        lambda pts, qs: bench_shim(pts, qs, track_objects=False, with_objs=False),
        points, queries, args.repeats
    )
    s_build_map, s_query_map = median_times(
        lambda pts, qs: bench_shim(pts, qs, track_objects=True, with_objs=True),
        points, queries, args.repeats
    )

    def fmt(x): return f"{x:.3f}"

    md = f"""
### Native vs Shim

**Setup**
- Points: {args.points:,}
- Queries: {args.queries}
- Repeats: {args.repeats}

**Timing (seconds)**

| Variant | Build | Query | Total |
|---|---:|---:|---:|
| Native | {fmt(n_build)} | {fmt(n_query)} | {fmt(n_build + n_query)} |
| Shim (no map) | {fmt(s_build_no_map)} | {fmt(s_query_no_map)} | {fmt(s_build_no_map + s_query_no_map)} |
| Shim (track+objs) | {fmt(s_build_map)} | {fmt(s_query_map)} | {fmt(s_build_map + s_query_map)} |

**Overhead vs Native**

- No map: build {s_build_no_map / n_build:.2f}x, query {s_query_no_map / n_query:.2f}x, total {(s_build_no_map + s_query_no_map) / (n_build + n_query):.2f}x  
- Track + objs: build {s_build_map / n_build:.2f}x, query {s_query_map / n_query:.2f}x, total {(s_build_map + s_query_map) / (n_build + n_query):.2f}x
"""
    print(md.strip())

if __name__ == "__main__":
    main()
