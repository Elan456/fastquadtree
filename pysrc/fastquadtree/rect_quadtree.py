# rect_quadtree.py
from __future__ import annotations

from typing import Any, Literal, Tuple, overload

from ._base_quadtree import Bounds, _BaseQuadTree
from ._item import RectItem
from ._native import RectQuadTree as _RustRectQuadTree  # native rect tree

_IdRect = Tuple[int, float, float, float, float]
Point = Tuple[float, float]  # only for type hints in docstrings


class RectQuadTree(_BaseQuadTree[Bounds, _IdRect, RectItem]):
    """
    Rectangle version of the quadtree. All geometries are axis-aligned rectangles. (min_x, min_y, max_x, max_y)
    High-level Python wrapper over the Rust quadtree engine.

    Performance characteristics:
        Inserts: average O(log n) <br>
        Rect queries: average O(log n + k) where k is matches returned <br>
        Nearest neighbor: average O(log n) <br>

    Thread-safety:
        Instances are not thread-safe. Use external synchronization if you
        mutate the same tree from multiple threads.

    Args:
        bounds: World bounds as (min_x, min_y, max_x, max_y).
        capacity: Max number of points per node before splitting.
        max_depth: Optional max tree depth. If omitted, engine decides.
        track_objects: Enable id <-> object mapping inside Python.

    Raises:
        ValueError: If parameters are invalid or inserts are out of bounds.
    """

    def __init__(
        self,
        bounds: Bounds,
        capacity: int,
        *,
        max_depth: int | None = None,
        track_objects: bool = False,
    ):
        super().__init__(
            bounds,
            capacity,
            max_depth=max_depth,
            track_objects=track_objects,
        )

    @overload
    def query(
        self, rect: Bounds, *, as_items: Literal[False] = ...
    ) -> list[_IdRect]: ...
    @overload
    def query(self, rect: Bounds, *, as_items: Literal[True]) -> list[RectItem]: ...
    def query(
        self, rect: Bounds, *, as_items: bool = False
    ) -> list[_IdRect] | list[RectItem]:
        """
        Query the tree for all items that intersect the given rectangle.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).
            as_items: If True, return Item wrappers. If False, return raw tuples.

        Returns:
            If as_items is False: list of (id, x0, y0, x1, y1) tuples.
            If as_items is True: list of Item objects.
        """
        if not as_items:
            return self._native.query(rect)
        if self._store is None:
            raise ValueError("Cannot return results as items with track_objects=False")
        return self._store.get_many_by_ids(self._native.query_ids(rect))

    def _new_native(self, bounds: Bounds, capacity: int, max_depth: int | None) -> Any:
        if max_depth is None:
            return _RustRectQuadTree(bounds, capacity)
        return _RustRectQuadTree(bounds, capacity, max_depth=max_depth)

    def _make_item(self, id_: int, geom: Bounds, obj: Any | None) -> RectItem:
        return RectItem(id_, geom, obj)
