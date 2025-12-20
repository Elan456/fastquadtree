"""High-performance rectangle spatial index without object association."""

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
    Spatial index for axis-aligned rectangles without object association.

    This class provides fast spatial indexing for rectangles using integer IDs that
    you can correlate with external data structures. For automatic object association,
    see `RectQuadTreeObjects`.

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
        rqt = RectQuadTree((0.0, 0.0, 100.0, 100.0), capacity=10)
        rect_id = rqt.insert((10.0, 20.0, 30.0, 40.0))
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
        Find all rectangles that intersect with a query rectangle.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).

        Returns:
            List of (id, min_x, min_y, max_x, max_y) tuples for intersecting rectangles.

        Example:
            ```python
            results = rqt.query((10.0, 10.0, 20.0, 20.0))
            for id_, min_x, min_y, max_x, max_y in results:
                print(f"Rect {id_} at ({min_x}, {min_y}, {max_x}, {max_y})")
            ```
        """
        return self._native.query(rect)

    def query_np(self, rect: Bounds) -> tuple[Any, Any]:
        """
        Find intersecting rectangles, returning NumPy arrays.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).

        Returns:
            Tuple of (ids, coords) where:
                - ids: NDArray[np.int64] with shape (N,)
                - coords: NDArray with shape (N, 4) and dtype matching the tree

        Raises:
            ImportError: If NumPy is not installed.

        Example:
            ```python
            ids, coords = rqt.query_np((10.0, 10.0, 20.0, 20.0))
            for id_, (min_x, min_y, max_x, max_y) in zip(ids, coords):
                print(f"Rect {id_} at ({min_x}, {min_y}, {max_x}, {max_y})")
            ```
        """
        return self._native.query_np(rect)

    def nearest_neighbor(self, point: Point) -> _IdRect | None:
        """
        Find the nearest rectangle to a query point.

        Distance is measured as Euclidean distance to the nearest edge of each rectangle.

        Args:
            point: Query point as (x, y).

        Returns:
            Tuple of (id, min_x, min_y, max_x, max_y), or None if tree is empty.

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
            Tuple of (id, coords) or None if tree is empty, where:
                id: int (uint64)
                coords: NDArray with shape (4,) and dtype matching tree

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
                ids: NDArray[np.uint64] with shape (k,)
                coords: NDArray with shape (k, 4) and dtype matching tree

        Raises:
            ImportError: If NumPy is not installed.
        """
        return self._native.nearest_neighbors_np(point, k)

    # ---- Deletion ----
    def delete(
        self, id_: int, min_x: float, min_y: float, max_x: float, max_y: float
    ) -> bool:
        """
        Remove a rectangle from the quadtree.

        Coordinates must be provided since this class doesn't store geometry internally.

        Args:
            id_: ID of the rectangle to delete.
            min_x: Minimum x coordinate of the rectangle.
            min_y: Minimum y coordinate of the rectangle.
            max_x: Maximum x coordinate of the rectangle.
            max_y: Maximum y coordinate of the rectangle.

        Returns:
            True if the rectangle was found and deleted, False if not found.

        Example:
            ```python
            rect_id = rqt.insert((10.0, 20.0, 30.0, 40.0))
            success = rqt.delete(rect_id, 10.0, 20.0, 30.0, 40.0)
            assert success is True
            ```
        """
        return self._delete_geom(id_, (min_x, min_y, max_x, max_y))

    def delete_tuple(self, t: _IdRect) -> bool:
        """
        Remove a rectangle from the quadtree using a tuple.

        This is a convenience method that accepts the rectangle data as a single tuple,
        typically from query results.

        Args:
            t: Tuple of (id, min_x, min_y, max_x, max_y) representing the rectangle to delete.

        Returns:
            True if the rectangle was found and deleted, False if not found.

        Example:
            ```python
            rect_id = rqt.insert((10.0, 20.0, 30.0, 40.0))
            success = rqt.delete_tuple((rect_id, 10.0, 20.0, 30.0, 40.0))
            assert success is True
            ```
        """
        id_, min_x, min_y, max_x, max_y = t
        return self._delete_geom(id_, (min_x, min_y, max_x, max_y))

    # ---- Mutation ----

    def update(
        self,
        id_: int,
        old_min_x: float,
        old_min_y: float,
        old_max_x: float,
        old_max_y: float,
        new_min_x: float,
        new_min_y: float,
        new_max_x: float,
        new_max_y: float,
    ) -> bool:
        """
        Move an existing rectangle to a new location.

        Old geometry is required because this class doesn't store it.

        Args:
            id_: The ID of the rectangle to move.
            old_min_x: Current min x coordinate.
            old_min_y: Current min y coordinate.
            old_max_x: Current max x coordinate.
            old_max_y: Current max y coordinate.
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
            ok = rqt.update(i, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0)
            assert ok is True
            ```
        """
        old_rect = (old_min_x, old_min_y, old_max_x, old_max_y)
        new_rect = (new_min_x, new_min_y, new_max_x, new_max_y)
        return self._update_geom(id_, old_rect, new_rect)

    def update_tuple(self, id_: int, old_rect: Bounds, new_rect: Bounds) -> bool:
        """
        Move an existing rectangle to a new location using tuple geometry.

        This is a convenience method that accepts geometry as tuples,
        reducing the number of parameters compared to update().

        Args:
            id_: The ID of the rectangle to move.
            old_rect: Current rectangle as (min_x, min_y, max_x, max_y).
            new_rect: New rectangle as (min_x, min_y, max_x, max_y).

        Returns:
            True if the update succeeded.

        Raises:
            ValueError: If new coordinates are outside bounds.

        Example:
            ```python
            i = rqt.insert((1.0, 1.0, 2.0, 2.0))
            ok = rqt.update_tuple(i, (1.0, 1.0, 2.0, 2.0), (3.0, 3.0, 4.0, 4.0))
            assert ok is True
            ```
        """
        return self._update_geom(id_, old_rect, new_rect)

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
