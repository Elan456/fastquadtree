# point_quadtree.py
from __future__ import annotations

from typing import Any, Literal, Tuple, overload

from ._base_quadtree import Bounds, _BaseQuadTree
from ._item import Point, PointItem
from ._native import QuadTree as _RustQuadTree  # native point tree

_IdCoord = Tuple[int, float, float]


class QuadTree(_BaseQuadTree[Point, _IdCoord, PointItem]):
    """
    High-level Python wrapper over the Rust quadtree engine.

    The quadtree stores points with integer IDs. You may attach an arbitrary
    Python object per ID when object tracking is enabled.

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
        start_id: Starting auto-assigned id when you omit id on insert.

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
        start_id: int = 1,
    ):
        super().__init__(
            bounds,
            capacity,
            max_depth=max_depth,
            track_objects=track_objects,
            start_id=start_id,
        )

    def insert(self, xy: Point, *, id_: int | None = None, obj: Any = None) -> int:
        return self._insert_common(xy, id_=id_, obj=obj)

    def insert_many_points(self, points: list[Point]) -> int:
        return self._insert_many_common(points)

    def delete(self, id_: int, xy: Point) -> bool:
        return self._delete_exact(id_, xy)

    @overload
    def query(
        self, rect: Bounds, *, as_items: Literal[False] = ...
    ) -> list[_IdCoord]: ...
    @overload
    def query(self, rect: Bounds, *, as_items: Literal[True]) -> list[PointItem]: ...
    def query(self, rect: Bounds, *, as_items: bool = False):
        raw = self._native.query(rect)
        if not as_items:
            return raw
        if self._items is None:
            raise ValueError("Cannot return results as items with track_objects=False")
        out: list[PointItem] = []
        for id_, _x, _y in raw:
            it = self._items.by_id(id_)
            if it is None:
                raise RuntimeError(
                    f"Internal error: id {id_} present in native tree but missing from tracker."
                )
            out.append(it)
        return out

    def nearest_neighbor(self, xy: Point, *, as_item: bool = False):
        t = self._native.nearest_neighbor(xy)
        if t is None or not as_item:
            return t
        if self._items is None:
            raise ValueError("Cannot return result as item with track_objects=False")
        id_, _x, _y = t
        it = self._items.by_id(id_)
        if it is None:
            raise RuntimeError("Internal error: missing tracked item")
        return it

    def nearest_neighbors(self, xy: Point, k: int, *, as_items: bool = False):
        raw = self._native.nearest_neighbors(xy, k)
        if not as_items:
            return raw
        if self._items is None:
            raise ValueError("Cannot return results as items with track_objects=False")
        out: list[PointItem] = []
        for id_, _x, _y in raw:
            it = self._items.by_id(id_)
            if it is None:
                raise RuntimeError("Internal error: missing tracked item")
            out.append(it)
        return out

    def get_all_node_boundaries(self) -> list[Bounds]:
        return self._native.get_all_node_boundaries()

    def _new_native(self, bounds: Bounds, capacity: int, max_depth: int | None) -> Any:
        if max_depth is None:
            return _RustQuadTree(bounds, capacity)
        return _RustQuadTree(bounds, capacity, max_depth=max_depth)

    def _make_item(self, id_: int, geom: Point, obj: Any | None) -> PointItem:
        return PointItem(id_, geom, obj)

    NativeQuadTree = _RustQuadTree
