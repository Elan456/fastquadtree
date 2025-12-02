# rect_quadtree.py
from __future__ import annotations

from typing import Any, Literal, SupportsFloat, Tuple, overload

from ._base_quadtree import Bounds, _BaseQuadTree
from ._item import Point, RectItem
from ._native import (
    RectQuadTree as RectQuadTreeF32,
    RectQuadTreeF64,
    RectQuadTreeI32,
    RectQuadTreeI64,
)

_IdRect = Tuple[int, SupportsFloat, SupportsFloat, SupportsFloat, SupportsFloat]

DTYPE_MAP = {
    "f32": RectQuadTreeF32,
    "f64": RectQuadTreeF64,
    "i32": RectQuadTreeI32,
    "i64": RectQuadTreeI64,
}


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
        dtype: Data type for coordinates and ids in the native engine. Default is 'f32'. Options are 'f32', 'f64', 'i32', 'i64'.

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
        dtype: str = "f32",
    ):
        super().__init__(
            bounds,
            capacity,
            max_depth=max_depth,
            track_objects=track_objects,
            dtype=dtype,
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

        Example:
            ```python
            results = rqt.query((10.0, 10.0, 20.0, 20.0), as_items=True)
            for item in results:
                print(f"Found rect id={item.id_} at {item.geom} with obj={item.obj}")
            ```
        """
        if not as_items:
            return self._native.query(rect)
        if self._store is None:
            raise ValueError("Cannot return results as items with track_objects=False")
        return self._store.get_many_by_ids(self._native.query_ids(rect))

    def query_np(self, rect: Bounds) -> tuple[Any, Any]:
        """
        Return all points inside an axis-aligned rectangle as NumPy arrays.
        The first array is an array of IDs, and the second is a corresponding array of rectangle coordinates.

        Requirements:
            NumPy must be installed to use this method.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).

        Returns:
            Tuple of (ids, locations) where:
                ids:       NDArray[np.uint64] with shape (N,)
                locations: NDArray[np.floating] with shape (N, 4)

        Example:
            ```python
            ids, locations = rqt.query_np((10.0, 10.0, 20.0, 20.0))
            for id_, (x0, y0, x1, y1) in zip(ids, locations):
                print(f"Found rect id={id_} at ({x0}, {y0}, {x1}, {y1})")
            ```
        """

        return self._native.query_np(rect)

    @overload
    def nearest_neighbor(
        self, xy: Point, *, as_item: Literal[False] = ...
    ) -> _IdRect | None: ...
    @overload
    def nearest_neighbor(
        self, xy: Point, *, as_item: Literal[True]
    ) -> RectItem | None: ...
    def nearest_neighbor(
        self, xy: Point, *, as_item: bool = False
    ) -> RectItem | _IdRect | None:
        """
        Return the single nearest neighbor to the query point.
        Utilizes euclidean distance to the nearest edge of rectangles.

        Args:
            xy: Query point (x, y).
            as_item: If True, return Item. If False, return (id, x0, y0, x1, y1).

        Returns:
            The nearest neighbor or None if the tree is empty.

        Example:
            ```python
            nn = rqt.nearest_neighbor((15.0, 15.0), as_item=True)
            if nn is not None:
                print(f"Nearest rect id={nn.id_} at {nn.geom} with obj={nn.obj}")
            else:
                print("No rectangles in the tree.")
            ```
        """
        t = self._native.nearest_neighbor(xy)
        if t is None or not as_item:
            return t
        if self._store is None:
            raise ValueError("Cannot return result as item with track_objects=False")
        id_, _x0, _y0, _x1, _y1 = t
        it = self._store.by_id(id_)
        if it is None:
            raise RuntimeError("Internal error: missing tracked item")
        return it

    @overload
    def nearest_neighbors(
        self, xy: Point, k: int, *, as_items: Literal[False] = ...
    ) -> list[_IdRect]: ...
    @overload
    def nearest_neighbors(
        self, xy: Point, k: int, *, as_items: Literal[True]
    ) -> list[RectItem]: ...
    def nearest_neighbors(
        self, xy: Point, k: int, *, as_items: bool = False
    ) -> list[RectItem] | list[_IdRect]:
        """
        Return the k nearest neighbors to the query point in order of increasing distance.
        Utilizes euclidean distance to the nearest edge of rectangles.

        Args:
            xy: Query point (x, y).
            k: Number of neighbors to return.
            as_items: If True, return Item wrappers. If False, return raw tuples.

        Returns:
            If as_items is False: list of (id, x0, y0, x1, y1) tuples. <br>
            If as_items is True: list of Item objects. <br>

        Example:
            ```python
            # Gets the 3 nearest rectangles to point (15.0, 15.0)
            results = rqt.nearest_neighbors((15.0, 15.0), 3, as_items=True)
            for item in results:
                print(f"Found rect id={item.id_} at {item.geom} with obj={item.obj}")
            ```
        """
        raw = self._native.nearest_neighbors(xy, k)
        if not as_items:
            return raw
        if self._store is None:
            raise ValueError("Cannot return results as items with track_objects=False")
        out: list[RectItem] = []
        for id_, _x0, _y0, _x1, _y1 in raw:
            it = self._store.by_id(id_)
            if it is None:
                raise RuntimeError("Internal error: missing tracked item")
            out.append(it)
        return out

    def _new_native(self, bounds: Bounds, capacity: int, max_depth: int | None) -> Any:
        """Create the native engine instance."""
        rust_cls = DTYPE_MAP.get(self._dtype)
        if rust_cls is None:
            raise TypeError(f"Unsupported dtype: {self._dtype}")
        return rust_cls(bounds, capacity, max_depth)

    @classmethod
    def _new_native_from_bytes(cls, data: bytes, dtype: str = "f32") -> Any:
        """Create a new native engine instance from serialized bytes."""
        rust_cls = DTYPE_MAP.get(dtype)
        if rust_cls is None:
            raise TypeError(f"Unsupported dtype: {dtype}")
        return rust_cls.from_bytes(data)

    @staticmethod
    def _make_item(id_: int, geom: Bounds, obj: Any | None) -> RectItem:
        return RectItem(id_, geom, obj)
