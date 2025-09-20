"""
Engine adapters for various quadtree implementations.

This module provides a unified interface for different quadtree libraries,
allowing fair comparison of their performance characteristics.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

# Built-in engines (always available in this repo)
from pyquadtree.quadtree import QuadTree as EPyQuadTree          # e-pyquadtree
from pyqtree import Index as PyQTree                             # Pyqtree
from quadtree_rs import QuadTree as RustQuadTree                 # quadtree-rs


class Engine:
    """
    Adapter interface for each quadtree implementation.
    
    Provides a unified interface for building trees and executing queries,
    allowing fair performance comparison across different libraries.
    """
    
    def __init__(self, 
                 name: str, 
                 color: str,
                 build_fn: Callable[[List[Tuple[int, int]]], Any],
                 query_fn: Callable[[Any, List[Tuple[int, int, int, int]]], None]):
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


def _create_e_pyquadtree_engine(bounds: Tuple[int, int, int, int], 
                               max_points: int, 
                               max_depth: int) -> Engine:
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
        "e-pyquadtree",  # display name
        "#1f77b4",       # color (blue)
        build, 
        query
    )


def _create_pyqtree_engine(bounds: Tuple[int, int, int, int], 
                          max_points: int, 
                          max_depth: int) -> Engine:
    """Create engine adapter for PyQtree."""
    def build(points):
        qt = PyQTree(bbox=bounds, max_items=max_points, max_depth=max_depth)
        for x, y in points:
            qt.insert(None, bbox=(x, y, x + 1, y + 1))
        return qt
    
    def query(qt, queries):
        for q in queries:
            _ = list(qt.intersect(q))
    
    return Engine(
        "PyQtree",  # display name
        "#2ca02c",  # color (green)
        build, 
        query
    )


def _create_quadtree_rs_engine(bounds: Tuple[int, int, int, int], 
                              max_points: int, 
                              max_depth: int) -> Engine:
    """Create engine adapter for quadtree-rs."""
    def build(points):
        qt = RustQuadTree(bounds, max_points, max_depth=max_depth)
        for p in points:
            qt.insert(p)
        return qt
    
    def query(qt, queries):
        for q in queries:
            _ = qt.query(q)
    
    return Engine(
        "quadtree-rs",  # display name
        "#ff7f0e",      # color (orange)
        build, 
        query
    )


def _create_quads_engine(bounds: Tuple[int, int, int, int], 
                        max_points: int, 
                        max_depth: int) -> Optional[Engine]:
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
        for (xmin, ymin, xmax, ymax) in queries:
            bb = qd.BoundingBox(min_x=xmin, min_y=ymin, max_x=xmax, max_y=ymax)
            _ = tree.within_bb(bb)
    
    return Engine(
        "quads",    # display name
        "#8c564b",  # color (brown)
        build, 
        query
    )


def _create_nontree_engine(bounds: Tuple[int, int, int, int], 
                          max_points: int, 
                          max_depth: int) -> Optional[Engine]:
    """Create engine adapter for nontree library (optional)."""
    try:
        from nontree.TreeMap import TreeMap
    except ImportError:
        return None
    
    def build(points):
        (xmin, ymin, xmax, ymax) = bounds
        w = xmax - xmin
        h = ymax - ymin
        tm = TreeMap((xmin, ymin, w, h), mode=4, bucket=max_points, lvl=max_depth)  # 4 => QuadTree
        # Store a tiny payload to match API; value is irrelevant
        for p in points:
            tm[p] = 1
        return tm
    
    def query(tm: TreeMap, queries):
        for (xmin, ymin, xmax, ymax) in queries:
            _ = tm.get_rect((xmin, ymin, xmax - xmin, ymax - ymin))
    
    return Engine(
        "nontree-QuadTree",  # display name
        "#17becf",           # color (cyan)
        build, 
        query
    )


def _create_brute_force_engine(bounds: Tuple[int, int, int, int], 
                              max_points: int, 
                              max_depth: int) -> Engine:
    """Create engine adapter for brute force search (always available)."""
    def build(points):
        # Append each item as if they were being added separately
        out = []
        for p in points:
            out.append(p)
        return out
    
    def query(points, queries):
        for q in queries:
            # Brute force search through all points
            _ = [p for p in points if q[0] <= p[0] <= q[2] and q[1] <= p[1] <= q[3]]
    
    return Engine(
        "Brute force",  # display name
        "#9467bd",      # color (purple)
        build, 
        query
    )

def _create_rtree_engine(bounds: Tuple[int, int, int, int], 
                         max_points: int, 
                         max_depth: int) -> Optional[Engine]:
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
        idx = rindex.Index(stream, properties=p)
        return idx

    def query(idx, queries):
        # Do not materialize results into a list, just consume the generator
        # to make query overhead comparable to engines that return iterables.
        for q in queries:
            for _ in idx.intersection(q, objects=False):
                pass

    return Engine(
        "Rtree",
        "#e377c2",
        build,
        query
    )


def get_engines(bounds: Tuple[int, int, int, int] = (0, 0, 1000, 1000),
               max_points: int = 20,
               max_depth: int = 10) -> Dict[str, Engine]:
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
        "quadtree-rs": _create_quadtree_rs_engine(bounds, max_points, max_depth),
        "e-pyquadtree": _create_e_pyquadtree_engine(bounds, max_points, max_depth),
        "PyQtree": _create_pyqtree_engine(bounds, max_points, max_depth),
        "Brute force": _create_brute_force_engine(bounds, max_points, max_depth),
    }
    
    # Optional engines (only include if import succeeded)
    optional_engines = [
        _create_quads_engine,
        _create_nontree_engine,
        _create_rtree_engine,
    ]
    
    for engine_creator in optional_engines:
        engine = engine_creator(bounds, max_points, max_depth)
        if engine is not None:
            engines[engine.name] = engine
    
    return engines