# rect_quadtree.py
"""RectQuadTree - High-performance rectangle spatial index without object association."""

from __future__ import annotations

from typing import Any

from ._base_quadtree import _BaseQuadTree
from ._common import Bounds, Point
from ._native import (
    RectQuadTree as RectQuadTreeF32,
    RectQuadTreeF64,
    RectQuadTreeI32,
    RectQuadTreeI64,
)

_IdRect = tuple[int, float, float, float, float]

DTYPE_MAP = {
    "f32": RectQuadTreeF32,
    "f64": RectQuadTreeF64,
    "i32": RectQuadTreeI32,
    "i64": RectQuadTreeI64,
}


class RectQuadTree(_BaseQuadTree[Bounds]):
    """
    High-performance spatial index for axis-aligned rectangles.

    This class provides fast spatial indexing without Python object association.
    Rectangles are stored with integer IDs that you can use to correlate with
    external data structures. For object association, see RectQuadTreeObjects.

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
        rqt = RectQuadTree((0.0, 0.0, 100.0, 100.0), capacity=10)
        id_ = rqt.insert((10.0, 20.0, 30.0, 40.0))
        results = rqt.query((5.0, 5.0, 35.0, 35.0))
        for id_, min_x, min_y, max_x, max_y in results:
            print(f"Rect {id_} at ({min_x}, {min_y}, {max_x}, {max_y})")
        ```
    """

    # ---- Native engine factory methods ----

    def _new_native(
        self, bounds: Bounds, capacity: int, max_depth: int | None, dtype: str
    ) -> Any:
        """Create the native engine instance."""
        rust_cls = DTYPE_MAP.get(dtype)
        if rust_cls is None:
            raise TypeError(f"Unsupported dtype: {dtype}")
        return rust_cls(bounds, capacity, max_depth)

    @classmethod
    def _new_native_from_bytes(cls, data: bytes, dtype: str) -> Any:
        """Create the native engine instance from serialized bytes."""
        rust_cls = DTYPE_MAP.get(dtype)
        if rust_cls is None:
            raise TypeError(f"Unsupported dtype: {dtype}")
        return rust_cls.from_bytes(data)

    # ---- Queries ----

    def query(self, rect: Bounds) -> list[_IdRect]:
        """
        Return all rectangles that intersect the query rectangle.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).

        Returns:
            List of (id, min_x, min_y, max_x, max_y) tuples.

        Example:
            ```python
            results = rqt.query((10.0, 10.0, 20.0, 20.0))
            for id_, min_x, min_y, max_x, max_y in results:
                print(f"Found rect id={id_} at ({min_x}, {min_y}, {max_x}, {max_y})")
            ```
        """
        return self._native.query(rect)

    def query_np(self, rect: Bounds) -> tuple[Any, Any]:
        """
        Return all rectangles that intersect the query rectangle as NumPy arrays.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).

        Returns:
            Tuple of (ids, coords) where:
                ids: NDArray[np.int64] with shape (N,)
                coords: NDArray with shape (N, 4) and dtype matching tree

        Raises:
            ImportError: If NumPy is not installed.

        Example:
            ```python
            ids, coords = rqt.query_np((10.0, 10.0, 20.0, 20.0))
            for id_, (min_x, min_y, max_x, max_y) in zip(ids, coords):
                print(f"Found rect id={id_} at ({min_x}, {min_y}, {max_x}, {max_y})")
            ```
        """
        return self._native.query_np(rect)

    def nearest_neighbor(self, point: Point) -> _IdRect | None:
        """
        Return the single nearest rectangle to the query point.

        Uses Euclidean distance to the nearest edge of rectangles.

        Args:
            point: Query point (x, y).

        Returns:
            Tuple of (id, min_x, min_y, max_x, max_y) or None if the tree is empty.

        Example:
            ```python
            nn = rqt.nearest_neighbor((15.0, 15.0))
            if nn is not None:
                id_, min_x, min_y, max_x, max_y = nn
                print(f"Nearest: {id_} at ({min_x}, {min_y}, {max_x}, {max_y})")
            ```
        """
        return self._native.nearest_neighbor(point)

    def nearest_neighbor_np(self, point: Point) -> tuple[int, Any] | None:
        """
        Return the single nearest rectangle as NumPy array.

        Args:
            point: Query point (x, y).

        Returns:
            Tuple of (id, coords) or None if tree is empty, where coords is ndarray shape (4,).

        Raises:
            ImportError: If NumPy is not installed.
        """
        return self._native.nearest_neighbor_np(point)

    def nearest_neighbors(self, point: Point, k: int) -> list[_IdRect]:
        """
        Return the k nearest rectangles to the query point.

        Uses Euclidean distance to the nearest edge of rectangles.

        Args:
            point: Query point (x, y).
            k: Number of neighbors to return.

        Returns:
            List of (id, min_x, min_y, max_x, max_y) tuples in order of increasing distance.

        Example:
            ```python
            neighbors = rqt.nearest_neighbors((15.0, 15.0), k=5)
            for id_, min_x, min_y, max_x, max_y in neighbors:
                print(f"Neighbor {id_} at ({min_x}, {min_y}, {max_x}, {max_y})")
            ```
        """
        return self._native.nearest_neighbors(point, k)

    def nearest_neighbors_np(self, point: Point, k: int) -> tuple[Any, Any]:
        """
        Return the k nearest rectangles as NumPy arrays.

        Args:
            point: Query point (x, y).
            k: Number of neighbors to return.

        Returns:
            Tuple of (ids, coords) where:
                ids: NDArray[np.int64] with shape (k,)
                coords: NDArray with shape (k, 4) and dtype matching tree

        Raises:
            ImportError: If NumPy is not installed.
        """
        return self._native.nearest_neighbors_np(point, k)

    # ---- Deletion ----
    def delete(
        self, id_: int, min_x: float, min_y: float, max_x: float, max_y: float
    ) -> bool:
        return self._delete_geom(id_, (min_x, min_y, max_x, max_y))

    def delete_tuple(self, t: _IdRect) -> bool:
        id_, min_x, min_y, max_x, max_y = t
        return self._delete_geom(id_, (min_x, min_y, max_x, max_y))

    # ---- Utilities ----
    def __contains__(self, rect: Bounds) -> bool:
        """
        Check if any item exists at the given rectangle coordinates.

        Args:
            rect: Rectangle as (min_x, min_y, max_x, max_y).

        Returns:
            True if at least one item exists at these exact coordinates.

        Example:
            ```python
            rqt.insert((10.0, 20.0, 30.0, 40.0))
            assert (10.0, 20.0, 30.0, 40.0) in rqt
            assert (5.0, 5.0, 10.0, 10.0) not in rqt
            ```
        """
        min_x, min_y, max_x, max_y = rect
        candidates = self._native.query(rect)
        return any(
            rmin_x == min_x and rmin_y == min_y and rmax_x == max_x and rmax_y == max_y
            for _, rmin_x, rmin_y, rmax_x, rmax_y in candidates
        )

    def __iter__(self):
        """
        Iterate over all (id, min_x, min_y, max_x, max_y) tuples in the tree.

        Example:
            ```python
            for id_, min_x, min_y, max_x, max_y in rqt:
                print(f"ID {id_} at ({min_x}, {min_y}, {max_x}, {max_y})")
            ```
        """
        # Query the entire bounds to get all items
        all_items = self._native.query(self._bounds)
        return iter(all_items)
