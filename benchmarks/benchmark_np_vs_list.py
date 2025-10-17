#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import statistics as stats
from time import perf_counter as now

import numpy as np
from system_info_collector import (
    collect_system_info,
    format_system_info_markdown_lite,
)

from fastquadtree import QuadTree as ShimQuadTree  # high level wrapper with insert_many

BOUNDS: tuple[float, float, float, float] = (0.0, 0.0, 1000.0, 1000.0)
CAPACITY = 64
MAX_DEPTH = 10
SEED = 12345


def gen_np_points(n: int, dtype: np.dtype) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    # Generate within bounds. Use dtype the user requested.
    if np.issubdtype(dtype, np.integer):
        arr = rng.integers(low=0, high=1000, size=(n, 2), dtype=dtype)
    else:
        arr = rng.random(size=(n, 2), dtype=dtype) * 1000.0
    return arr


def _build_tree_np(points_np: np.ndarray, track_objects: bool) -> float:
    t0 = now()
    qt = ShimQuadTree(
        BOUNDS, CAPACITY, max_depth=MAX_DEPTH, track_objects=track_objects
    )
    # Direct NumPy path
    inserted = qt.insert_many(points_np)
    dt = now() - t0
    assert (
        inserted == points_np.shape[0]
    ), f"Inserted {inserted} != {points_np.shape[0]}"
    return dt


def _build_tree_list(
    points_list: list[tuple[float, float]], track_objects: bool
) -> float:
    t0 = now()
    qt = ShimQuadTree(
        BOUNDS, CAPACITY, max_depth=MAX_DEPTH, track_objects=track_objects
    )
    inserted = qt.insert_many(points_list)
    dt = now() - t0
    assert inserted == len(points_list), f"Inserted {inserted} != {len(points_list)}"
    return dt


def bench_np_direct(points_np: np.ndarray, repeats: int, track_objects: bool) -> float:
    times = []
    for _ in range(repeats):
        gc.disable()
        times.append(_build_tree_np(points_np, track_objects))
        gc.enable()
    return stats.median(times)


def bench_list_from_np(
    points_np: np.ndarray, repeats: int, track_objects: bool, include_conversion: bool
) -> float:
    times = []
    if not include_conversion:
        # Convert once up front so measured time is insert only
        points_list = [tuple(map(float, row)) for row in points_np]
    for _ in range(repeats):
        gc.disable()
        if include_conversion:
            # Count conversion cost
            t0 = now()
            points_list = [tuple(map(float, row)) for row in points_np]
            convert_time = now() - t0
            build_time = _build_tree_list(points_list, track_objects)  # type: ignore
            times.append(convert_time + build_time)
        else:
            # Insert only
            times.append(_build_tree_list(points_list, track_objects))  # pyright: ignore[reportPossiblyUnboundVariable, reportArgumentType]
        gc.enable()
    return stats.median(times)


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark: NumPy insert vs Python list insert"
    )
    ap.add_argument("--points", type=int, default=500_000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float64", "int32", "int64"],
        help="Dtype for generated NumPy points",
    )
    ap.add_argument(
        "--track-objects",
        action="store_true",
        help="Enable object tracking in the shim to include store overhead",
    )
    args = ap.parse_args()

    dtype_map = {
        "float32": np.float32,
        "float64": np.float64,
        "int32": np.int32,
        "int64": np.int64,
    }
    dtype = dtype_map[args.dtype]

    print("NumPy vs list insert benchmark")
    print("=" * 50)
    print("Configuration:")
    print(f"  Points: {args.points:,}")
    print(f"  Repeats: {args.repeats}")
    print(f"  Dtype: {args.dtype}")
    print(f"  Track objects: {args.track_objects}")
    print()

    # Data
    pts_np = gen_np_points(args.points, dtype=dtype)

    # Warmup
    _ = bench_np_direct(pts_np[:10_000], repeats=1, track_objects=args.track_objects)
    _ = bench_list_from_np(
        pts_np[:10_000],
        repeats=1,
        track_objects=args.track_objects,
        include_conversion=False,
    )
    _ = bench_list_from_np(
        pts_np[:10_000],
        repeats=1,
        track_objects=args.track_objects,
        include_conversion=True,
    )

    # Actual runs
    t_np = bench_np_direct(pts_np, args.repeats, args.track_objects)
    t_list_insert_only = bench_list_from_np(
        pts_np, args.repeats, args.track_objects, include_conversion=False
    )
    t_list_with_convert = bench_list_from_np(
        pts_np, args.repeats, args.track_objects, include_conversion=True
    )

    def fmt(x: float) -> str:
        if x < 1e-3:
            return f"{x * 1e6:.1f} µs"
        if x < 1:
            return f"{x * 1e3:.1f} ms"
        return f"{x:.3f} s"

    print("Results (median of repeats)")
    print()
    print("| Variant | Build time |")
    print("|---|---:|")
    print(f"| NumPy array direct | {fmt(t_np)} |")
    print(f"| Python list insert only | {fmt(t_list_insert_only)} |")
    print(f"| Python list including conversion | {fmt(t_list_with_convert)} |")

    if collect_system_info and format_system_info_markdown_lite:
        info = collect_system_info()
        print()
        print(format_system_info_markdown_lite(info))


if __name__ == "__main__":
    main()
