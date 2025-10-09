# point_quadtree.py
from __future__ import annotations

from typing import Any, Literal, Tuple, overload

from ._base_quadtree import Bounds, _BaseQuadTree
from ._item import Point, PointItem
from ._native import QuadTree as _RustQuadTree  # native point tree

_IdCoord = Tuple[int, float, float]


class QuadTree(_BaseQuadTree[Point, _IdCoord, PointItem]):
    """
    High-level Python wrapper over the Rust point QuadTree.
    """

    # ---- native hooks ----

    def _new_native(self, bounds: Bounds, capacity: int, max_depth: int | None) -> Any:
        if max_depth is None:
            return _RustQuadTree(bounds, capacity)
        return _RustQuadTree(bounds, capacity, max_depth=max_depth)

    def _native_insert(self, id_: int, geom: Point) -> bool:
        return self._native.insert(id_, geom)

    def _native_insert_many(self, start_id: int, geoms: list[Point]) -> int:
        return self._native.insert_many_points(start_id, geoms)

    def _native_delete(self, id_: int, geom: Point) -> bool:
        return self._native.delete(id_, geom)

    def _native_query(self, rect: Bounds) -> list[_IdCoord]:
        return self._native.query(rect)

    def _native_count(self) -> int:
        return self._native.count_items()

    def _make_item(self, id_: int, geom: Point, obj: Any | None) -> PointItem:
        return PointItem(id_, geom, obj)

    # ---- public API identical to before ----

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
        raw = self._native_query(rect)
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

    # Nearest neighbor features remain point-only
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

    # Convenience passthrough kept for compatibility
    def get_all_node_boundaries(self) -> list[Bounds]:
        return self._native.get_all_node_boundaries()

    # Power users
    NativeQuadTree = _RustQuadTree
