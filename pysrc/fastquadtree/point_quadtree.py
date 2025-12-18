# point_quadtree.py
"""QuadTree - High-performance point spatial index without object association."""

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
    High-performance spatial index for 2D points.

    This class provides fast spatial indexing without Python object association.
    Points are stored with integer IDs that you can use to correlate with external
    data structures. For object association, see QuadTreeObjects.

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
        qt = QuadTree((0.0, 0.0, 100.0, 100.0), capacity=10)
        id_ = qt.insert((10.0, 20.0))
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
        Return all points inside an axis-aligned rectangle.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).

        Returns:
            List of (id, x, y) tuples.

        Example:
            ```python
            results = qt.query((10.0, 10.0, 20.0, 20.0))
            for id_, x, y in results:
                print(f"Found point id={id_} at ({x}, {y})")
            ```
        """
        return self._native.query(rect)

    def query_np(self, rect: Bounds) -> tuple[Any, Any]:
        """
        Return all points inside an axis-aligned rectangle as NumPy arrays.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).

        Returns:
            Tuple of (ids, coords) where:
                ids: NDArray[np.int64] with shape (N,)
                coords: NDArray with shape (N, 2) and dtype matching tree

        Raises:
            ImportError: If NumPy is not installed.

        Example:
            ```python
            ids, coords = qt.query_np((10.0, 10.0, 20.0, 20.0))
            for id_, (x, y) in zip(ids, coords):
                print(f"Found point id={id_} at ({x}, {y})")
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
            Tuple of (id, coords) or None if tree is empty, where coords is ndarray shape (2,).

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
                ids: NDArray[np.int64] with shape (k,)
                coords: NDArray with shape (k, 2) and dtype matching tree

        Raises:
            ImportError: If NumPy is not installed.
        """
        return self._native.nearest_neighbors_np(point, k)

    # ---- Deletion ----
    def delete(self, id_: int, x: float, y: float) -> bool:
        return self._delete_geom(id_, (x, y))

    def delete_tuple(self, t: _IdCoord) -> bool:
        id_, x, y = t
        return self._delete_geom(id_, (x, y))

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
        eps = 1e-9
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
