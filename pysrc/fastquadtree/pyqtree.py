"""
Compatibility layer providing the pyqtree interface.

This module provides a drop-in replacement for the pyqtree package with
significantly improved performance through a Rust-backed implementation.
"""

from __future__ import annotations

from collections.abc import Iterable
from operator import itemgetter
from typing import Any, SupportsFloat, Tuple

from ._native import RectQuadTree

Point = Tuple[SupportsFloat, SupportsFloat]  # only for type hints in docstrings

# Default parameters from pyqtree
MAX_ITEMS = 10
MAX_DEPTH = 20


# Helper to gather objects by ids in chunks
# Performance improvement over list comprehension for large result sets
# 2.945 median query time --> 2.030 median query time (500k items, 500 queries)
def gather_objs(objs, ids, chunk=2048):
    out = []
    for i in range(0, len(ids), chunk):
        getter = itemgetter(*ids[i : i + chunk])
        vals = getter(objs)  # tuple or single object
        if isinstance(vals, tuple):
            out.extend(vals)
        else:
            out.append(vals)
    return out


class Index:
    """
    Pyqtree-compatible spatial index with improved performance.

    This class provides the same interface as pyqtree.Index but is backed by
    a high-performance Rust implementation. Based on benchmarks, this provides
    an overall performance boost of approximately 6.5x compared to the original
    pure-Python implementation.

    For new projects not requiring pyqtree compatibility, consider using
    `RectQuadTreeObjects` for more control and better performance.

    Args:
        bbox: Coordinate system bounding box as (xmin, ymin, xmax, ymax).
        x: X center coordinate (alternative to bbox).
        y: Y center coordinate (alternative to bbox).
        width: Width from center (alternative to bbox).
        height: Height from center (alternative to bbox).
        max_items: Maximum items per quad before splitting (default: 10).
        max_depth: Maximum nesting levels (default: 20).

    Note:
        Either `bbox` or all of (`x`, `y`, `width`, `height`) must be provided.

    Example:
        ```python
        from fastquadtree.pyqtree import Index

        spindex = Index(bbox=(0, 0, 100, 100))
        spindex.insert('duck', (50, 30, 53, 60))
        spindex.insert('cookie', (10, 20, 15, 25))
        spindex.insert('python', (40, 50, 95, 90))
        results = spindex.intersect((51, 51, 86, 86))
        print(sorted(results))  # ['duck', 'python']
        ```
    """

    __slots__ = ("_free", "_item_to_id", "_objects", "_qt")

    def __init__(
        self,
        bbox: Iterable[SupportsFloat] | None = None,
        x: float | int | None = None,
        y: float | int | None = None,
        width: float | int | None = None,
        height: float | int | None = None,
        max_items: int = MAX_ITEMS,
        max_depth: int = MAX_DEPTH,
    ):
        """
        Initialize the spatial index.

        Specify either a bounding box or center point with dimensions.

        Args:
            bbox: Bounding box as (xmin, ymin, xmax, ymax).
            x: X center coordinate (alternative to bbox).
            y: Y center coordinate (alternative to bbox).
            width: Distance from x center to edges (alternative to bbox).
            height: Distance from y center to edges (alternative to bbox).
            max_items: Maximum items per quad before splitting (default: 10).
            max_depth: Maximum nesting levels (default: 20).

        Raises:
            ValueError: If neither bbox nor (x, y, width, height) are provided.
        """
        if bbox is not None:
            min_x, min_y, max_x, max_y = bbox
            self._qt = RectQuadTree(
                (min_x, min_y, max_x, max_y), max_items, max_depth=max_depth
            )

        elif (
            x is not None and y is not None and width is not None and height is not None
        ):
            self._qt = RectQuadTree(
                (x - width / 2, y - height / 2, x + width / 2, y + height / 2),
                max_items,
                max_depth=max_depth,
            )

        else:
            raise ValueError(
                "Either the bbox argument must be set, or the x, y, width, and height arguments must be set"
            )

        self._objects = []
        self._free = []
        self._item_to_id = {}

    def insert(self, item: Any, bbox: Iterable[SupportsFloat]):
        """
        Insert an item with its bounding box.

        Args:
            item: Item to insert (will be returned by intersect queries).
            bbox: Spatial bounding box as (xmin, ymin, xmax, ymax).
        """
        if type(bbox) is not tuple:  # Handle non-tuple input
            bbox = tuple(bbox)

        if self._free:
            rid = self._free.pop()
            self._objects[rid] = item
        else:
            rid = len(self._objects)
            self._objects.append(item)
        self._qt.insert(rid, bbox)
        self._item_to_id[id(item)] = rid

    def remove(self, item: Any, bbox: Iterable[SupportsFloat]):
        """
        Remove an item from the index.

        Args:
            item: Item to remove (must match the inserted item).
            bbox: Bounding box as (xmin, ymin, xmax, ymax) (must match insertion).

        Note:
            Both parameters must exactly match those used during insertion.
        """
        if type(bbox) is not tuple:  # Handle non-tuple input
            bbox = tuple(bbox)

        rid = self._item_to_id.pop(id(item))
        self._qt.delete(rid, bbox)
        self._objects[rid] = None
        self._free.append(rid)

    def intersect(self, bbox: Iterable[SupportsFloat]) -> list:
        """
        Query items that intersect with a bounding box.

        Args:
            bbox: Query bounding box as (xmin, ymin, xmax, ymax).

        Returns:
            List of items whose bounding boxes intersect the query box.
        """
        if type(bbox) is not tuple:  # Handle non-tuple input
            bbox = tuple(bbox)
        result = self._qt.query_ids(bbox)
        # result = [id1, id2, ...]
        return gather_objs(self._objects, result)
