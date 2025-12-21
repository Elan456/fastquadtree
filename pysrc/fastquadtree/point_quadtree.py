"""High-performance point spatial index without object association."""

from __future__ import annotations

from typing import Any

from ._base_quadtree import _BaseQuadTree
from ._common import Bounds, Point
from ._native import QuadTree as QuadTreeF32, QuadTreeF64, QuadTreeI32, QuadTreeI64

_IdCoord = tuple[int, float, float]

DTYPE_MAP = {
    "f32": QuadTreeF32,
    "f64": QuadTreeF64,
    "i32": QuadTreeI32,
    "i64": QuadTreeI64,
}


class QuadTree(_BaseQuadTree[Point]):
    """
    Spatial index for 2D points without object association.

    This class provides fast spatial indexing for points using integer IDs that you
    can correlate with external data structures. For automatic object association,
    see `QuadTreeObjects`.

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
        qt = QuadTree((0.0, 0.0, 100.0, 100.0), capacity=10)
        point_id = qt.insert((10.0, 20.0))
        results = qt.query((5.0, 5.0, 25.0, 25.0))
        for id_, x, y in results:
            print(f"Point {id_} at ({x}, {y})")
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

    def query(self, rect: Bounds) -> list[_IdCoord]:
        """
        Find all points within a rectangular region.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).

        Returns:
            List of (id, x, y) tuples for points inside the rectangle.

        Example:
            ```python
            results = qt.query((10.0, 10.0, 20.0, 20.0))
            for id_, x, y in results:
                print(f"Point {id_} at ({x}, {y})")
            ```
        """
        return self._native.query(rect)

    def query_np(self, rect: Bounds) -> tuple[Any, Any]:
        """
        Find all points within a rectangular region, returning NumPy arrays.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).

        Returns:
            Tuple of (ids, coords) where:
                - ids: NDArray[np.int64] with shape (N,)
                - coords: NDArray with shape (N, 2) and dtype matching the tree

        Raises:
            ImportError: If NumPy is not installed.

        Example:
            ```python
            ids, coords = qt.query_np((10.0, 10.0, 20.0, 20.0))
            for id_, (x, y) in zip(ids, coords):
                print(f"Point {id_} at ({x}, {y})")
            ```
        """
        return self._native.query_np(rect)

    def nearest_neighbor(self, point: Point) -> _IdCoord | None:
        """
        Return the single nearest neighbor to the query point.

        Args:
            point: Query point (x, y).

        Returns:
            Tuple of (id, x, y) or None if the tree is empty.

        Example:
            ```python
            nn = qt.nearest_neighbor((15.0, 15.0))
            if nn is not None:
                id_, x, y = nn
                print(f"Nearest: {id_} at ({x}, {y})")
            ```
        """
        return self._native.nearest_neighbor(point)

    def nearest_neighbor_np(self, point: Point) -> tuple[int, Any] | None:
        """
        Return the single nearest neighbor as NumPy array.

        Args:
            point: Query point (x, y).

        Returns:
            Tuple of (id, coords) or None if tree is empty, where:
                id: int (uint64)
                coords: NDArray with shape (2,) and dtype matching tree

        Raises:
            ImportError: If NumPy is not installed.
        """
        return self._native.nearest_neighbor_np(point)

    def nearest_neighbors(self, point: Point, k: int) -> list[_IdCoord]:
        """
        Return the k nearest neighbors to the query point.

        Args:
            point: Query point (x, y).
            k: Number of neighbors to return.

        Returns:
            List of (id, x, y) tuples in order of increasing distance.

        Example:
            ```python
            neighbors = qt.nearest_neighbors((15.0, 15.0), k=5)
            for id_, x, y in neighbors:
                print(f"Neighbor {id_} at ({x}, {y})")
            ```
        """
        return self._native.nearest_neighbors(point, k)

    def nearest_neighbors_np(self, point: Point, k: int) -> tuple[Any, Any]:
        """
        Return the k nearest neighbors as NumPy arrays.

        Args:
            point: Query point (x, y).
            k: Number of neighbors to return.

        Returns:
            Tuple of (ids, coords) where:
                ids: NDArray[np.uint64] with shape (k,)
                coords: NDArray with shape (k, 2) and dtype matching tree

        Raises:
            ImportError: If NumPy is not installed.
        """
        return self._native.nearest_neighbors_np(point, k)

    # ---- Deletion ----
    def delete(self, id_: int, x: float, y: float) -> bool:
        """
        Remove a point from the quadtree.

        Coordinates must be provided since this class doesn't store geometry internally.

        Args:
            id_: ID of the point to delete.
            x: X coordinate of the point.
            y: Y coordinate of the point.

        Returns:
            True if the point was found and deleted, False if not found.

        Example:
            ```python
            point_id = qt.insert((10.0, 20.0))
            success = qt.delete(point_id, 10.0, 20.0)
            assert success is True
            ```
        """
        return self._delete_geom(id_, (x, y))

    def delete_tuple(self, t: _IdCoord) -> bool:
        """
        Remove a point from the quadtree using a tuple.

        This is a convenience method that accepts the point data as a single tuple,
        typically from query results.

        Args:
            t: Tuple of (id, x, y) representing the point to delete.

        Returns:
            True if the point was found and deleted, False if not found.

        Example:
            ```python
            point_id = qt.insert((10.0, 20.0))
            success = qt.delete_tuple((point_id, 10.0, 20.0))
            assert success is True
            ```
        """
        id_, x, y = t
        return self._delete_geom(id_, (x, y))

    # ---- Mutation ----

    def update(
        self, id_: int, old_x: float, old_y: float, new_x: float, new_y: float
    ) -> bool:
        """
        Move a point to new coordinates.

        Old coordinates must be provided since this class doesn't store geometry internally.

        Args:
            id_: ID of the point to move.
            old_x: Current X coordinate.
            old_y: Current Y coordinate.
            new_x: New X coordinate.
            new_y: New Y coordinate.

        Returns:
            True if the update succeeded, False if the point was not found.

        Raises:
            ValueError: If new coordinates are outside tree bounds.

        Example:
            ```python
            point_id = qt.insert((1.0, 1.0))
            success = qt.update(point_id, 1.0, 1.0, 2.0, 2.0)
            assert success is True
            ```
        """
        old_point = (old_x, old_y)
        new_point = (new_x, new_y)
        return self._update_geom(id_, old_point, new_point)

    def update_tuple(self, id_: int, old_point: Point, new_point: Point) -> bool:
        """
        Move an existing point to a new location using tuple geometry.

        This is a convenience method that accepts geometry as tuples,
        reducing the number of parameters compared to update().

        Args:
            id_: The ID of the point to move.
            old_point: Current point as (x, y).
            new_point: New point as (x, y).

        Returns:
            True if the update succeeded.

        Raises:
            ValueError: If new coordinates are outside bounds.

        Example:
            ```python
            i = qt.insert((1.0, 1.0))
            ok = qt.update_tuple(i, (1.0, 1.0), (2.0, 2.0))
            assert ok is True
            ```
        """
        return self._update_geom(id_, old_point, new_point)

    # ---- Utilities ----

    def __contains__(self, point: Point) -> bool:
        """
        Check if any item exists at the given point coordinates.

        Args:
            point: Point as (x, y).

        Returns:
            True if at least one item exists at these coordinates.

        Example:
            ```python
            qt.insert((10.0, 20.0))
            assert (10.0, 20.0) in qt
            assert (5.0, 5.0) not in qt
            ```
        """
        x, y = point
        eps = 1 if self._dtype[0] == "i" else 1e-5
        rect = (x - eps, y - eps, x + eps, y + eps)
        candidates = self._native.query(rect)
        return any(px == x and py == y for _, px, py in candidates)

    def __iter__(self):
        """
        Iterate over all (id, x, y) tuples in the tree.

        Example:
            ```python
            for id_, x, y in qt:
                print(f"ID {id_} at ({x}, {y})")
            ```
        """
        # Query the entire bounds to get all items
        all_items = self._native.query(self._bounds)
        return iter(all_items)
