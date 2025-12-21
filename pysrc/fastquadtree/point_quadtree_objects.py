"""Point quadtree with Python object association."""

from __future__ import annotations

from typing import Any

from ._base_quadtree_objects import _BaseQuadTreeObjects
from ._common import Bounds
from ._item import Point, PointItem
from ._native import QuadTree as QuadTreeF32, QuadTreeF64, QuadTreeI32, QuadTreeI64

DTYPE_MAP = {
    "f32": QuadTreeF32,
    "f64": QuadTreeF64,
    "i32": QuadTreeI32,
    "i64": QuadTreeI64,
}


class QuadTreeObjects(_BaseQuadTreeObjects[Point, PointItem]):
    """
    Spatial index for 2D points with Python object association.

    This class provides fast spatial indexing for points while allowing you to
    associate arbitrary Python objects with each point. IDs are managed internally
    using dense allocation for efficient lookup.

    Args:
        bounds: World bounds as (min_x, min_y, max_x, max_y).
        capacity: Maximum points per node before splitting.
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
        qt = QuadTreeObjects((0.0, 0.0, 100.0, 100.0), capacity=10)
        id_ = qt.insert((10.0, 20.0), obj="my data")
        results = qt.query((5.0, 5.0, 25.0, 25.0))
        for item in results:
            print(f"Point {item.id_} at ({item.x}, {item.y}) with obj={item.obj}")
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
    def _make_item(id_: int, geom: Point, obj: Any | None) -> PointItem:
        """Build a PointItem from id, geometry, and optional object."""
        return PointItem(id_, geom, obj)

    @staticmethod
    def _extract_coords_from_geom(geom: Point) -> tuple:
        """Extract coordinate tuple from point geometry."""
        return geom

    # ---- Point-specific deletion ----

    def delete_at(self, x: float, y: float) -> bool:
        """
        Delete an item at specific coordinates.

        If multiple items exist at the same point, deletes the one with the lowest ID.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if an item was found and deleted, False otherwise.

        Example:
            ```python
            qt.insert((5.0, 5.0))
            success = qt.delete_at(5.0, 5.0)
            assert success is True
            ```
        """
        # Query a tiny rect around the point
        eps = 1 if self._dtype[0] == "i" else 1e-5
        rect = (x - eps, y - eps, x + eps, y + eps)
        candidates = self._native.query(rect)

        # Find all exact matches
        matches = [(id_, px, py) for id_, px, py in candidates if px == x and py == y]
        if not matches:
            return False

        # Delete the one with the lowest ID
        min_id = min(id_ for id_, _, _ in matches)
        return self.delete(min_id)

    # ---- Point-specific update ----

    def update(self, id_: int, new_x: float, new_y: float) -> bool:
        """
        Move a point to new coordinates.

        This is efficient because old coordinates are retrieved from internal storage.

        Args:
            id_: ID of the point to move.
            new_x: New X coordinate.
            new_y: New Y coordinate.

        Returns:
            True if the update succeeded, False if the ID was not found.

        Raises:
            ValueError: If new coordinates are outside tree bounds.

        Example:
            ```python
            point_id = qt.insert((1.0, 1.0))
            success = qt.update(point_id, 2.0, 2.0)
            assert success is True
            ```
        """
        item = self._store.by_id(id_)
        if item is None:
            return False

        old_point = item.geom
        new_point = (new_x, new_y)

        # Use base class _update_geom to handle deletion and insertion
        if not self._update_geom(id_, old_point, new_point):
            return False

        # Update stored item
        self._store.add(PointItem(id_, new_point, item.obj))
        return True

    def update_by_object(self, obj: Any, new_x: float, new_y: float) -> bool:
        """
        Move a point to new coordinates by finding it via its associated object.

        If multiple items have the same object, updates the one with the lowest ID.

        Args:
            obj: Python object to search for (by identity).
            new_x: New X coordinate.
            new_y: New Y coordinate.

        Returns:
            True if the update succeeded, False if object was not found.

        Raises:
            ValueError: If new coordinates are outside tree bounds.

        Example:
            ```python
            my_obj = {"data": "example"}
            qt.insert((1.0, 1.0), obj=my_obj)
            success = qt.update_by_object(my_obj, 2.0, 2.0)
            assert success is True
            ```
        """
        item = self._store.by_obj(obj)
        if item is None:
            return False

        return self.update(item.id_, new_x, new_y)
