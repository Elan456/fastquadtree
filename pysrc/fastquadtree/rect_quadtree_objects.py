# rect_quadtree_objects.py
"""RectQuadTreeObjects - Rectangle quadtree with Python object association."""

from __future__ import annotations

from typing import Any

from ._base_quadtree_objects import _BaseQuadTreeObjects
from ._common import Bounds
from ._item import RectItem
from ._native import (
    RectQuadTree as RectQuadTreeF32,
    RectQuadTreeF64,
    RectQuadTreeI32,
    RectQuadTreeI64,
)

DTYPE_MAP = {
    "f32": RectQuadTreeF32,
    "f64": RectQuadTreeF64,
    "i32": RectQuadTreeI32,
    "i64": RectQuadTreeI64,
}


class RectQuadTreeObjects(_BaseQuadTreeObjects[Bounds, RectItem]):
    """
    Rectangle quadtree with Python object association.

    This class provides spatial indexing for axis-aligned rectangles with the
    ability to associate arbitrary Python objects with each rectangle. IDs are
    managed internally using dense allocation for efficient object lookup.

    Performance characteristics:
        Inserts: average O(log n)
        Rect queries: average O(log n + k) where k is matches returned
        Nearest neighbor: average O(log n)

    Thread-safety:
        Instances are not thread-safe. Use external synchronization if you
        mutate the same tree from multiple threads.

    Args:
        bounds: World bounds as (min_x, min_y, max_x, max_y).
        capacity: Max number of rectangles per node before splitting.
        max_depth: Optional max tree depth. If omitted, engine decides.
        dtype: Data type for coordinates ('f32', 'f64', 'i32', 'i64'). Default is 'f32'.

    Raises:
        ValueError: If parameters are invalid or inserts are out of bounds.

    Example:
        ```python
        rqt = RectQuadTreeObjects((0.0, 0.0, 100.0, 100.0), capacity=10)
        id_ = rqt.insert((10.0, 20.0, 30.0, 40.0), obj="my data")
        results = rqt.query((5.0, 5.0, 35.0, 35.0))
        for item in results:
            print(f"Rect {item.id_} at ({item.min_x}, {item.min_y}, {item.max_x}, {item.max_y})")
        ```
    """

    def _new_native(self, bounds: Bounds, capacity: int, max_depth: int | None) -> Any:
        """Create the native engine instance."""
        rust_cls = DTYPE_MAP.get(self._dtype)
        if rust_cls is None:
            raise TypeError(f"Unsupported dtype: {self._dtype}")
        return rust_cls(bounds, capacity, max_depth)

    @classmethod
    def _new_native_from_bytes(cls, data: bytes, dtype: str) -> Any:
        """Create the native engine instance from serialized bytes."""
        rust_cls = DTYPE_MAP.get(dtype)
        if rust_cls is None:
            raise TypeError(f"Unsupported dtype: {dtype}")
        return rust_cls.from_bytes(data)

    @staticmethod
    def _make_item(id_: int, geom: Bounds, obj: Any | None) -> RectItem:
        """Build a RectItem from id, geometry, and optional object."""
        return RectItem(id_, geom, obj)

    @staticmethod
    def _extract_coords_from_geom(geom: Bounds) -> tuple:
        """Extract coordinate tuple from rectangle geometry."""
        return geom

    # ---- Rectangle-specific deletion ----

    def delete_at(self, min_x: float, min_y: float, max_x: float, max_y: float) -> bool:
        """
        Delete one item at the given rectangle coordinates.

        If multiple items exist at the same coordinates, deletes the one with the lowest ID.

        Args:
            min_x: Min x coordinate.
            min_y: Min y coordinate.
            max_x: Max x coordinate.
            max_y: Max y coordinate.

        Returns:
            True if an item was found and deleted.

        Example:
            ```python
            rqt.insert((5.0, 5.0, 10.0, 10.0))
            ok = rqt.delete_at(5.0, 5.0, 10.0, 10.0)
            assert ok is True
            ```
        """
        # Query for overlapping rectangles
        rect = (min_x, min_y, max_x, max_y)
        candidates = self._native.query(rect)

        # Find all exact matches
        matches = [
            (id_, rmin_x, rmin_y, rmax_x, rmax_y)
            for id_, rmin_x, rmin_y, rmax_x, rmax_y in candidates
            if rmin_x == min_x
            and rmin_y == min_y
            and rmax_x == max_x
            and rmax_y == max_y
        ]
        if not matches:
            return False

        # Delete the one with the lowest ID
        min_id = min(id_ for id_, _, _, _, _ in matches)
        return self.delete(min_id)

    # ---- Rectangle-specific update ----

    def update(
        self,
        id_: int,
        new_min_x: float,
        new_min_y: float,
        new_max_x: float,
        new_max_y: float,
    ) -> bool:
        """
        Move an existing rectangle to a new location.

        This is efficient because the old coordinates are stored with the object.

        Args:
            id_: The ID of the rectangle to move.
            new_min_x: New min x coordinate.
            new_min_y: New min y coordinate.
            new_max_x: New max x coordinate.
            new_max_y: New max y coordinate.

        Returns:
            True if the update succeeded.

        Raises:
            ValueError: If new coordinates are outside bounds.

        Example:
            ```python
            i = rqt.insert((1.0, 1.0, 2.0, 2.0))
            ok = rqt.update(i, 3.0, 3.0, 4.0, 4.0)
            assert ok is True
            ```
        """
        item = self._store.by_id(id_)
        if item is None:
            return False

        old_rect = item.geom
        new_rect = (new_min_x, new_min_y, new_max_x, new_max_y)

        # Use base class _update_geom to handle deletion and insertion
        if not self._update_geom(id_, old_rect, new_rect):
            return False

        # Update stored item
        self._store.add(RectItem(id_, new_rect, item.obj))
        return True

    def update_by_object(
        self,
        obj: Any,
        new_min_x: float,
        new_min_y: float,
        new_max_x: float,
        new_max_y: float,
    ) -> bool:
        """
        Move an existing rectangle to a new location by object reference.

        If multiple items have this object, updates the one with the lowest ID.

        Args:
            obj: The Python object to search for.
            new_min_x: New min x coordinate.
            new_min_y: New min y coordinate.
            new_max_x: New max x coordinate.
            new_max_y: New max y coordinate.

        Returns:
            True if the update succeeded.

        Raises:
            ValueError: If new coordinates are outside bounds.

        Example:
            ```python
            my_obj = {"data": "example"}
            rqt.insert((1.0, 1.0, 2.0, 2.0), obj=my_obj)
            ok = rqt.update_by_object(my_obj, 3.0, 3.0, 4.0, 4.0)
            assert ok is True
            ```
        """
        item = self._store.by_obj(obj)
        if item is None:
            return False

        return self.update(item.id_, new_min_x, new_min_y, new_max_x, new_max_y)
