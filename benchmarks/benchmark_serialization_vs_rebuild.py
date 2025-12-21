"""
Benchmark: serialization/deserialization vs rebuild from NumPy array
"""

from __future__ import annotations

import gc
import statistics as stats
from pathlib import Path
from time import perf_counter as pc

import numpy as np

from fastquadtree import QuadTree

CAPACITY = 64
MAX_DEPTH = 10
N = 1_000_000
REPEATS = 7


def timeit(fn, repeat=REPEATS):
    # warmup
    fn()
    times = []
    gc.disable()
    try:
        for _ in range(repeat):
            t0 = pc()
            fn()
            times.append(pc() - t0)
    finally:
        gc.enable()
    return times


def make_points():
    rng = np.random.default_rng(42)  # seed
    # shape (N, 2), float32 to match native expectations
    return rng.uniform(0.0, 1000.0, size=(N, 2)).astype(np.float32)


def build_original(pts: np.ndarray) -> QuadTree:
    qt = QuadTree((0, 0, 1000, 1000), capacity=CAPACITY, max_depth=MAX_DEPTH)
    qt.insert_many_np(pts)
    return qt


def main():
    pts = make_points()
    original_qt = build_original(pts)

    # correctness baseline
    base_count = len(original_qt)
    assert base_count == N

    # serialize timing
    ser_times = timeit(lambda: original_qt.to_bytes())
    qt_bytes = original_qt.to_bytes()
    print(f"Serialized size: {len(qt_bytes):,} bytes")

    # write once for the file path
    fname = "quadtree_serialization.bin"
    with Path(fname).open("wb") as f:
        f.write(qt_bytes)

    def rebuild_points():
        qt = QuadTree((0, 0, 1000, 1000), capacity=CAPACITY, max_depth=MAX_DEPTH)
        qt.insert_many_np(pts)
        assert len(qt) == base_count
        _ = qt.query((100, 100, 200, 200))
        return qt

    def rebuild_from_mem():
        qt = QuadTree.from_bytes(qt_bytes)
        assert len(qt) == base_count
        _ = qt.query((100, 100, 200, 200))
        return qt

    def rebuild_from_file():
        with Path(fname).open("rb") as f:
            data = f.read()
        qt = QuadTree.from_bytes(data)
        assert len(qt) == base_count
        _ = qt.query((100, 100, 200, 200))
        return qt

    t_points = timeit(rebuild_points)
    t_mem = timeit(rebuild_from_mem)
    t_file = timeit(rebuild_from_file)

    Path(fname).unlink(missing_ok=True)

    def show(label, arr):
        print(
            f"{label:<28} mean={stats.mean(arr):.6f}s  stdev={stats.pstdev(arr):.6f}s "
        )

    show("serialize to bytes", ser_times)
    show("rebuild from points", t_points)
    show("rebuild from bytes", t_mem)
    show("rebuild from file", t_file)

    # ----- Markdown summary -----
    def fmt(x: float) -> str:
        return f"{x:.6f}"

    m_ser = stats.mean(ser_times)
    m_pts = stats.mean(t_points)
    m_mem = stats.mean(t_mem)
    m_file = stats.mean(t_file)

    s_ser = stats.pstdev(ser_times)
    s_pts = stats.pstdev(t_points)
    s_mem = stats.pstdev(t_mem)
    s_file = stats.pstdev(t_file)

    speedup_mem = m_pts / m_mem if m_mem > 0 else float("inf")
    speedup_file = m_pts / m_file if m_file > 0 else float("inf")

    md = f"""
## Serialization vs Rebuild

### Configuration
- Points: {N:,}
- Capacity: {CAPACITY}
- Max depth: {MAX_DEPTH}
- Repeats: {REPEATS}

### Results

| Variant | Mean (s) | Stdev (s) |
|---|---:|---:|
| Serialize to bytes | {fmt(m_ser)} | {fmt(s_ser)} |
| Rebuild from points | {fmt(m_pts)} | {fmt(s_pts)} |
| Rebuild from bytes | {fmt(m_mem)} | {fmt(s_mem)} |
| Rebuild from file | {fmt(m_file)} | {fmt(s_file)} |

### Summary

- Rebuild from bytes is **{fmt(speedup_mem)}x** faster than reinserting points.
- Rebuild from file is **{fmt(speedup_file)}x** faster than reinserting points.
- Serialized blob size is **{len(qt_bytes):,} bytes**.
"""
    print(md.strip())


if __name__ == "__main__":
    main()
