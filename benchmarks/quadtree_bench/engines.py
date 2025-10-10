"""
Engine adapters for various quadtree implementations.

This module provides a unified interface for different quadtree libraries,
allowing fair comparison of their performance characteristics.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from pyqtree import Index as PyQTree  # Pyqtree

# Built-in engines (always available in this repo)
from pyquadtree.quadtree import QuadTree as EPyQuadTree  # e-pyquadtree
from shapely import box as shp_box, points  # Shapely 2.x
from shapely.strtree import STRtree

from fastquadtree import QuadTree as RustQuadTree  # fastquadtree


class Engine:
    """
    Adapter interface for each quadtree implementation.

    Provides a unified interface for building trees and executing queries,
    allowing fair performance comparison across different libraries.
    """

    def __init__(
        self,
        name: str,
        color: str,
        build_fn: Callable[[List[Tuple[int, int]]], Any],
        query_fn: Callable[[Any, List[Tuple[int, int, int, int]]], None],
    ):
        """
        Initialize an engine adapter.

        Args:
            name: Display name for the engine
            color: Color for plotting
            build_fn: Function to build tree from points
            query_fn: Function to execute queries on tree
        """
        self.name = name
        self.color = color
        self._build = build_fn
        self._query = query_fn

    def build(self, points: List[Tuple[int, int]]) -> Any:
        """Build a tree from the given points."""
        return self._build(points)

    def query(self, tree: Any, queries: List[Tuple[int, int, int, int]]) -> None:
        """Execute queries on the tree."""
        return self._query(tree, queries)


def _create_e_pyquadtree_engine(
    bounds: Tuple[int, int, int, int], max_points: int, max_depth: int
) -> Engine:
    """Create engine adapter for e-pyquadtree."""

    def build(points):
        qt = EPyQuadTree(bounds, max_points, max_depth)
        for p in points:
            qt.add(None, p)
        return qt

    def query(qt, queries):
        for q in queries:
            _ = qt.query(q)

    return Engine(
        "e-pyquadtree",
        "#1f77b4",
        build,
        query,  # display name  # color (blue)
    )


def _create_pyqtree_engine(
    bounds: Tuple[int, int, int, int], max_points: int, max_depth: int
) -> Engine:
    """Create engine adapter for PyQtree."""

    def build(points):
        qt = PyQTree(bbox=bounds, max_items=max_points, max_depth=max_depth)
        for x, y in points:
            qt.insert(None, bbox=(x, y, x + 1, y + 1))
        return qt

    def query(qt, queries):
        for q in queries:
            _ = list(qt.intersect(q))

    return Engine("PyQtree", "#2ca02c", build, query)  # display name  # color (green)


def _create_fastquadtree_engine(
    bounds: Tuple[int, int, int, int], max_points: int, max_depth: int
) -> Engine:
    """Create engine adapter for fastquadtree."""

    def build(points):
        qt = RustQuadTree(bounds, max_points, max_depth=max_depth)
        qt.insert_many(points)
        return qt

    def query(qt, queries):
        for q in queries:
            _ = qt.query(q)

    return Engine(
        "fastquadtree",
        "#ff7f0e",
        build,
        query,  # display name  # color (orange)
    )


def _create_quads_engine(
    bounds: Tuple[int, int, int, int], max_points: int, max_depth: int
) -> Optional[Engine]:
    """Create engine adapter for quads library (optional)."""
    try:
        import quads as qd
    except ImportError:
        return None

    def build(points):
        (xmin, ymin, xmax, ymax) = bounds
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        tree = qd.QuadTree((cx, cy), w, h, capacity=max_points)
        for p in points:
            tree.insert(p)  # accepts tuple or qd.Point
        return tree

    def query(tree, queries):
        import quads as qd

        for xmin, ymin, xmax, ymax in queries:
            bb = qd.BoundingBox(min_x=xmin, min_y=ymin, max_x=xmax, max_y=ymax)
            _ = tree.within_bb(bb)

    return Engine("quads", "#8c564b", build, query)  # display name  # color (brown)


def _create_nontree_engine(
    bounds: Tuple[int, int, int, int], max_points: int, max_depth: int
) -> Optional[Engine]:
    """Create engine adapter for nontree library (optional)."""
    try:
        from nontree.TreeMap import TreeMap
    except ImportError:
        return None

    def build(points):
        (xmin, ymin, xmax, ymax) = bounds
        w = xmax - xmin
        h = ymax - ymin
        tm = TreeMap(
            (xmin, ymin, w, h), mode=4, bucket=max_points, lvl=max_depth
        )  # 4 => QuadTree
        # Store a tiny payload to match API; value is irrelevant
        for p in points:
            tm[p] = 1
        return tm

    def query(tm: TreeMap, queries):
        for xmin, ymin, xmax, ymax in queries:
            _ = tm.get_rect((xmin, ymin, xmax - xmin, ymax - ymin))

    return Engine(
        "nontree-QuadTree",
        "#17becf",
        build,
        query,  # display name  # color (cyan)
    )


def _create_brute_force_engine(
    bounds: Tuple[int, int, int, int], max_points: int, max_depth: int
) -> Engine:
    """Create engine adapter for brute force search (always available)."""

    def build(points):
        # Append each item as if they were being added separately
        out = []
        for p in points:
            out.append(p)  # noqa: PERF402
        return out

    def query(points, queries):
        for q in queries:
            # Brute force search through all points
            _ = [p for p in points if q[0] <= p[0] <= q[2] and q[1] <= p[1] <= q[3]]

    return Engine(
        "Brute force",
        "#9467bd",
        build,
        query,  # display name  # color (purple)
    )


def _create_rtree_engine(
    bounds: Tuple[int, int, int, int], max_points: int, max_depth: int
) -> Optional[Engine]:
    """Create engine adapter for rtree library (optional)."""
    try:
        from rtree import index as rindex
    except ImportError:
        return None

    def build(points):
        # Tune for typical in-memory use. Keep nodes modest to increase fanout.
        p = rindex.Property()
        p.dimension = 2
        p.variant = rindex.RT_Star
        cap = max(16, min(128, max_points))  # tie to your per-node capacity range
        p.leaf_capacity = cap
        p.index_capacity = cap
        p.fill_factor = 0.7

        # Bulk stream loading is the fastest way to build
        # Keep the same 1x1 bbox convention used elsewhere for fairness
        stream = ((i, (x, y, x + 1, y + 1), None) for i, (x, y) in enumerate(points))
        return rindex.Index(stream, properties=p)

    def query(idx, queries):
        # Do not materialize results into a list, just consume the generator
        # to make query overhead comparable to engines that return iterables.
        for q in queries:
            for _ in idx.intersection(q, objects=False):
                pass

    return Engine("Rtree", "#e377c2", build, query)


def _create_strtree_engine(
    bounds: Tuple[int, int, int, int], max_points: int, max_depth: int
) -> Optional[Engine]:
    """Create engine adapter for Shapely STRtree (optional)."""

    def build(points_list: List[Tuple[int, int]]):
        # Build geometries efficiently

        xs = np.fromiter(
            (x for x, _ in points_list), dtype="float32", count=len(points_list)
        )
        ys = np.fromiter(
            (y for _, y in points_list), dtype="float32", count=len(points_list)
        )
        geoms = points(xs, ys)  # vectorized Point creation
        assert type(geoms) is np.ndarray
        tree = STRtree(geoms, node_capacity=max_points)
        # Keep geoms alive next to the tree
        return (tree, geoms)

    def query(built, queries: List[Tuple[int, int, int, int]]):
        tree, _geoms = built
        for xmin, ymin, xmax, ymax in queries:
            window = shp_box(xmin, ymin, xmax, ymax)
            # Shapely 2.x returns ndarray of indices for a single geometry
            res = tree.query(window)
            # Consume results without materializing to keep parity with other engines
            if hasattr(res, "__iter__"):
                for _ in res:
                    pass

    return Engine("Shapely STRtree", "#7f7f7f", build, query)


def get_engines(
    bounds: Tuple[int, int, int, int] = (0, 0, 1000, 1000),
    max_points: int = 20,
    max_depth: int = 10,
) -> Dict[str, Engine]:
    """
    Get all available engine adapters.

    Args:
        bounds: Bounding box for quadtrees (min_x, min_y, max_x, max_y)
        max_points: Maximum points per node before splitting
        max_depth: Maximum tree depth

    Returns:
        Dictionary mapping engine names to Engine instances
    """
    # Always available engines
    engines = {
        "fastquadtree": _create_fastquadtree_engine(bounds, max_points, max_depth),
        "e-pyquadtree": _create_e_pyquadtree_engine(bounds, max_points, max_depth),
        "PyQtree": _create_pyqtree_engine(bounds, max_points, max_depth),
        #    "Brute force": _create_brute_force_engine(bounds, max_points, max_depth),  # Brute force doesn't scale well on the graphs so omit it from the main set
    }

    # Optional engines (only include if import succeeded)
    optional_engines = [
        _create_quads_engine,
        _create_nontree_engine,
        _create_rtree_engine,
        _create_strtree_engine,
    ]

    for engine_creator in optional_engines:
        engine = engine_creator(bounds, max_points, max_depth)
        if engine is not None:
            engines[engine.name] = engine

    return engines
