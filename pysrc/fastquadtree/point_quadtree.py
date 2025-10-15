# point_quadtree.py
from __future__ import annotations

from typing import Any, Literal, Tuple, overload

from ._base_quadtree import Bounds, _BaseQuadTree
from ._item import Point, PointItem
from ._native import QuadTree as _RustQuadTree  # native point tree

_IdCoord = Tuple[int, float, float]


class QuadTree(_BaseQuadTree[Point, _IdCoord, PointItem]):
    """
    Point version of the quadtree. All geometries are 2D points (x, y).
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
    ) -> list[_IdCoord]: ...
    @overload
    def query(self, rect: Bounds, *, as_items: Literal[True]) -> list[PointItem]: ...
    def query(
        self, rect: Bounds, *, as_items: bool = False
    ) -> list[PointItem] | list[_IdCoord]:
        """
        Return all points inside an axis-aligned rectangle.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).
            as_items: If True, return Item wrappers. If False, return raw tuples.

        Returns:
            If as_items is False: list of (id, x, y) tuples.
            If as_items is True: list of Item objects.
        """
        if not as_items:
            return self._native.query(rect)
        if self._store is None:
            raise ValueError("Cannot return results as items with track_objects=False")
        return self._store.get_many_by_ids(self._native.query_ids(rect))

    @overload
    def nearest_neighbor(
        self, xy: Point, *, as_item: Literal[False] = ...
    ) -> _IdCoord | None: ...
    @overload
    def nearest_neighbor(
        self, xy: Point, *, as_item: Literal[True]
    ) -> PointItem | None: ...
    def nearest_neighbor(
        self, xy: Point, *, as_item: bool = False
    ) -> PointItem | _IdCoord | None:
        """
        Return the single nearest neighbor to the query point.

        Args:
            xy: Query point (x, y).
            as_item: If True, return Item. If False, return (id, x, y).

        Returns:
            The nearest neighbor or None if the tree is empty.
        """
        t = self._native.nearest_neighbor(xy)
        if t is None or not as_item:
            return t
        if self._store is None:
            raise ValueError("Cannot return result as item with track_objects=False")
        id_, _x, _y = t
        it = self._store.by_id(id_)
        if it is None:
            raise RuntimeError("Internal error: missing tracked item")
        return it

    @overload
    def nearest_neighbors(
        self, xy: Point, k: int, *, as_items: Literal[False] = ...
    ) -> list[_IdCoord]: ...
    @overload
    def nearest_neighbors(
        self, xy: Point, k: int, *, as_items: Literal[True]
    ) -> list[PointItem]: ...
    def nearest_neighbors(self, xy: Point, k: int, *, as_items: bool = False):
        """
        Return the k nearest neighbors to the query point in order of increasing distance.

        Args:
            xy: Query point (x, y).
            k: Number of neighbors to return.
            as_items: If True, return Item wrappers. If False, return raw tuples.
        Returns:
            If as_items is False: list of (id, x, y) tuples.
            If as_items is True: list of Item objects.
        """
        raw = self._native.nearest_neighbors(xy, k)
        if not as_items:
            return raw
        if self._store is None:
            raise ValueError("Cannot return results as items with track_objects=False")
        out: list[PointItem] = []
        for id_, _x, _y in raw:
            it = self._store.by_id(id_)
            if it is None:
                raise RuntimeError("Internal error: missing tracked item")
            out.append(it)
        return out

    def _new_native(self, bounds: Bounds, capacity: int, max_depth: int | None) -> Any:
        if max_depth is None:
            return _RustQuadTree(bounds, capacity)
        return _RustQuadTree(bounds, capacity, max_depth=max_depth)

    def _make_item(self, id_: int, geom: Point, obj: Any | None) -> PointItem:
        return PointItem(id_, geom, obj)
