# rect_quadtree_objects.py
"""RectQuadTreeObjects - Rectangle quadtree with Python object association."""

from __future__ import annotations

from typing import Any

from ._base_quadtree import Bounds
from ._base_quadtree_objects import _BaseQuadTreeObjects
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

    def delete_at(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """
        Delete one item at the given rectangle coordinates.

        If multiple items exist at the same coordinates, deletes the one with the lowest ID.

        Args:
            x0: Min x coordinate.
            y0: Min y coordinate.
            x1: Max x coordinate.
            y1: Max y coordinate.

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
        rect = (x0, y0, x1, y1)
        candidates = self._native.query(rect)

        # Find all exact matches
        matches = [
            (id_, rx0, ry0, rx1, ry1)
            for id_, rx0, ry0, rx1, ry1 in candidates
            if rx0 == x0 and ry0 == y0 and rx1 == x1 and ry1 == y1
        ]
        if not matches:
            return False

        # Delete the one with the lowest ID
        min_id = min(id_ for id_, _, _, _, _ in matches)
        return self.delete(min_id)

    # ---- Rectangle-specific update ----

    def update(
        self, id_: int, new_x0: float, new_y0: float, new_x1: float, new_y1: float
    ) -> bool:
        """
        Move an existing rectangle to a new location.

        This is efficient because the old coordinates are stored with the object.

        Args:
            id_: The ID of the rectangle to move.
            new_x0: New min x coordinate.
            new_y0: New min y coordinate.
            new_x1: New max x coordinate.
            new_y1: New max y coordinate.

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

        # Delete from old position
        if not self._native.delete(id_, old_rect):
            return False

        # Insert at new position
        new_rect = (new_x0, new_y0, new_x1, new_y1)
        if not self._native.insert(id_, new_rect):
            # Rollback: reinsert at old position
            self._native.insert(id_, old_rect)
            bx0, by0, bx1, by1 = self._bounds
            raise ValueError(
                f"New rectangle {new_rect!r} is outside bounds ({bx0}, {by0}, {bx1}, {by1})"
            )

        # Update stored item
        self._store.add(RectItem(id_, new_rect, item.obj))
        return True
