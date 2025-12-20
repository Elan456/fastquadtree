"""Rectangle quadtree with Python object association."""

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
    Spatial index for axis-aligned rectangles with Python object association.

    This class provides fast spatial indexing for rectangles while allowing you to
    associate arbitrary Python objects with each rectangle. IDs are managed internally
    using dense allocation for efficient lookup.

    Args:
        bounds: World bounds as (min_x, min_y, max_x, max_y).
        capacity: Maximum rectangles per node before splitting.
        max_depth: Optional maximum tree depth (uses engine default if not specified).
        dtype: Coordinate data type ('f32', 'f64', 'i32', 'i64'). Default: 'f32'.

    Performance:
        - Inserts: O(log n) average
        - Queries: O(log n + k) average, where k is the number of matches
        - Nearest neighbor: O(log n) average

    Thread Safety:
        Not thread-safe. Use external synchronization for concurrent access.

    Raises:
        ValueError: If parameters are invalid or geometry is outside bounds.

    Example:
        ```python
        rqt = RectQuadTreeObjects((0.0, 0.0, 100.0, 100.0), capacity=10)
        rect_id = rqt.insert((10.0, 20.0, 30.0, 40.0), obj="my data")
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
        Delete a rectangle at specific coordinates.

        If multiple rectangles exist at the same coordinates, deletes the one with the lowest ID.

        Args:
            min_x: Minimum X coordinate.
            min_y: Minimum Y coordinate.
            max_x: Maximum X coordinate.
            max_y: Maximum Y coordinate.

        Returns:
            True if a rectangle was found and deleted, False otherwise.

        Example:
            ```python
            rqt.insert((5.0, 5.0, 10.0, 10.0))
            success = rqt.delete_at(5.0, 5.0, 10.0, 10.0)
            assert success is True
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
        Move a rectangle to new coordinates.

        This is efficient because old coordinates are retrieved from internal storage.

        Args:
            id_: ID of the rectangle to move.
            new_min_x: New minimum X coordinate.
            new_min_y: New minimum Y coordinate.
            new_max_x: New maximum X coordinate.
            new_max_y: New maximum Y coordinate.

        Returns:
            True if the update succeeded, False if the ID was not found.

        Raises:
            ValueError: If new coordinates are outside tree bounds.

        Example:
            ```python
            rect_id = rqt.insert((1.0, 1.0, 2.0, 2.0))
            success = rqt.update(rect_id, 3.0, 3.0, 4.0, 4.0)
            assert success is True
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
        Move a rectangle to new coordinates by finding it via its associated object.

        If multiple items have the same object, updates the one with the lowest ID.

        Args:
            obj: Python object to search for (by identity).
            new_min_x: New minimum X coordinate.
            new_min_y: New minimum Y coordinate.
            new_max_x: New maximum X coordinate.
            new_max_y: New maximum Y coordinate.

        Returns:
            True if the update succeeded, False if object was not found.

        Raises:
            ValueError: If new coordinates are outside tree bounds.

        Example:
            ```python
            my_obj = {"data": "example"}
            rqt.insert((1.0, 1.0, 2.0, 2.0), obj=my_obj)
            success = rqt.update_by_object(my_obj, 3.0, 3.0, 4.0, 4.0)
            assert success is True
            ```
        """
        item = self._store.by_obj(obj)
        if item is None:
            return False

        return self.update(item.id_, new_min_x, new_min_y, new_max_x, new_max_y)
