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
    High-level Python wrapper over the Rust RectQuadTree.
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

    @overload
    def query(
        self, rect: Bounds, *, as_items: Literal[False] = ...
    ) -> list[_IdRect]: ...
    @overload
    def query(self, rect: Bounds, *, as_items: Literal[True]) -> list[RectItem]: ...
    def query(self, rect: Bounds, *, as_items: bool = False):
        """
        Query the tree for all items that intersect the given rectangle.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).
            as_items: If True, return Item wrappers. If False, return raw tuples.

        Returns:
            If as_items is False: list of (id, x0, y0, x1, y1) tuples.
            If as_items is True: list of Item objects.
        """
        raw = self._native.query(rect)
        if not as_items:
            return raw
        if self._items is None:
            # Build RectItem without objects
            return [
                RectItem(id_, (x0, y0, x1, y1), None) for (id_, x0, y0, x1, y1) in raw
            ]
        out: list[RectItem] = []
        for id_, _x0, _y0, _x1, _y1 in raw:
            it = self._items.by_id(id_)
            if it is None:
                raise RuntimeError(
                    f"Internal error: id {id_} present in native tree but missing from tracker."
                )
            out.append(it)
        return out

    def _new_native(self, bounds: Bounds, capacity: int, max_depth: int | None) -> Any:
        if max_depth is None:
            return _RustRectQuadTree(bounds, capacity)
        return _RustRectQuadTree(bounds, capacity, max_depth=max_depth)

    def _make_item(self, id_: int, geom: Bounds, obj: Any | None) -> RectItem:
        return RectItem(id_, geom, obj)
