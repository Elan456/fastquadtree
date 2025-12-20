# point_quadtree_objects.py
"""QuadTreeObjects - Point quadtree with Python object association."""

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
    Point quadtree with Python object association.

    This class provides spatial indexing for 2D points with the ability to
    associate arbitrary Python objects with each point. IDs are managed
    internally using dense allocation for efficient object lookup.

    Performance characteristics:
        Inserts: average O(log n)
        Rect queries: average O(log n + k) where k is matches returned
        Nearest neighbor: average O(log n)

    Thread-safety:
        Instances are not thread-safe. Use external synchronization if you
        mutate the same tree from multiple threads.

    Args:
        bounds: World bounds as (min_x, min_y, max_x, max_y).
        capacity: Max number of points per node before splitting.
        max_depth: Optional max tree depth. If omitted, engine decides.
        dtype: Data type for coordinates ('f32', 'f64', 'i32', 'i64'). Default is 'f32'.

    Raises:
        ValueError: If parameters are invalid or inserts are out of bounds.

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
        Delete one item at the given coordinates.

        If multiple items exist at (x, y), deletes the one with the lowest ID.

        Args:
            x: X coordinate.
            y: Y coordinate.

        Returns:
            True if an item was found and deleted.

        Example:
            ```python
            qt.insert((5.0, 5.0))
            ok = qt.delete_at(5.0, 5.0)
            assert ok is True
            ```
        """
        # Query a tiny rect around the point
        eps = 1e-9
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
        Move an existing point to a new location.

        This is efficient because the old coordinates are stored with the object.

        Args:
            id_: The ID of the point to move.
            new_x: New x coordinate.
            new_y: New y coordinate.

        Returns:
            True if the update succeeded.

        Raises:
            ValueError: If new coordinates are outside bounds.

        Example:
            ```python
            i = qt.insert((1.0, 1.0))
            ok = qt.update(i, 2.0, 2.0)
            assert ok is True
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
        Move an existing point to a new location by object reference.

        If multiple items have this object, updates the one with the lowest ID.

        Args:
            obj: The Python object to search for.
            new_x: New x coordinate.
            new_y: New y coordinate.

        Returns:
            True if the update succeeded.

        Raises:
            ValueError: If new coordinates are outside bounds.

        Example:
            ```python
            my_obj = {"data": "example"}
            qt.insert((1.0, 1.0), obj=my_obj)
            ok = qt.update_by_object(my_obj, 2.0, 2.0)
            assert ok is True
            ```
        """
        item = self._store.by_obj(obj)
        if item is None:
            return False

        return self.update(item.id_, new_x, new_y)
