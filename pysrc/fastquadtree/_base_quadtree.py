# _base_quadtree.py
"""Base class for QuadTree and RectQuadTree without object tracking (v2.0 API)."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from ._common import (
    Bounds,
    _is_np_array,
    validate_bounds,
    validate_np_dtype,
)
from ._insert_result import InsertResult

# Generic parameters
G = TypeVar("G")  # geometry type, e.g. Point or Bounds


class _BaseQuadTree(Generic[G], ABC):
    """
    Shared logic for QuadTree and RectQuadTree without object tracking.

    This base class implements the core functionality for spatial indexing
    without Python object association. Concrete subclasses must implement:
      - _new_native(bounds, capacity, max_depth, dtype)
      - _new_native_from_bytes(data, dtype)
    """

    __slots__ = (
        "_bounds",
        "_capacity",
        "_count",
        "_dtype",
        "_max_depth",
        "_native",
        "_next_id",
    )

    # ---- Required hooks for subclasses ----

    @abstractmethod
    def _new_native(
        self, bounds: Bounds, capacity: int, max_depth: int | None, dtype: str
    ) -> Any:
        """Create the native engine instance."""

    @classmethod
    @abstractmethod
    def _new_native_from_bytes(cls, data: bytes, dtype: str) -> Any:
        """Create the native engine instance from serialized bytes."""

    # ---- Initialization ----

    def __init__(
        self,
        bounds: Bounds,
        capacity: int,
        *,
        max_depth: int | None = None,
        dtype: str = "f32",
    ):
        self._bounds = validate_bounds(bounds)
        self._capacity = capacity
        self._max_depth = max_depth
        self._dtype = dtype

        self._native = self._new_native(self._bounds, capacity, max_depth, dtype)

        self._next_id = 0
        self._count = 0

    # ---- Insertion ----

    def insert(self, geom: G, id_: int | None = None) -> int:
        """
        Insert a single geometry.

        IDs are auto-assigned by default. You can optionally provide a custom ID
        to correlate with external data structures.

        Warning: Mixing auto-assigned and custom IDs is dangerous. The quadtree
        does not track which IDs have been used. If you provide a custom ID that
        collides with an auto-assigned ID, both entries will exist with the same
        ID, leading to undefined behavior. Users who provide custom IDs are
        responsible for ensuring uniqueness.

        Args:
            geom: Geometry (Point or Bounds).
            id_: Optional custom ID. If None, auto-assigns the next ID.

        Returns:
            The ID used for this geometry.

        Raises:
            ValueError: If geometry is outside the tree bounds.
        """
        if id_ is None:
            id_ = self._next_id
            self._next_id += 1

        if not self._native.insert(id_, geom):
            bx0, by0, bx1, by1 = self._bounds
            raise ValueError(
                f"Geometry {geom!r} is outside bounds ({bx0}, {by0}, {bx1}, {by1})"
            )

        self._count += 1
        return id_

    def insert_many(self, geoms: list[G]) -> InsertResult:
        """
        Bulk insert geometries with auto-assigned contiguous IDs.

        Custom IDs are not supported for bulk insertion. Use single insert()
        calls if you need custom IDs.

        Args:
            geoms: List of geometries.

        Returns:
            InsertResult with count, start_id, and end_id.

        Raises:
            ValueError: If any geometry is outside bounds.
        """
        if len(geoms) == 0:
            return InsertResult(
                count=0, start_id=self._next_id, end_id=self._next_id - 1
            )

        start_id = self._next_id
        last_id = self._native.insert_many(start_id, geoms)
        num = last_id - start_id + 1

        if num < len(geoms):
            raise ValueError("One or more geometries are outside tree bounds")

        self._next_id = last_id + 1
        self._count += num
        return InsertResult(count=num, start_id=start_id, end_id=last_id)

    def insert_many_np(self, geoms: Any) -> InsertResult:
        """
        Bulk insert geometries from NumPy array with auto-assigned contiguous IDs.

        Args:
            geoms: NumPy array with dtype matching the tree's dtype.

        Returns:
            InsertResult with count, start_id, and end_id.

        Raises:
            TypeError: If geoms is not a NumPy array or dtype doesn't match.
            ValueError: If any geometry is outside bounds.
            ImportError: If NumPy is not installed.
        """
        if not _is_np_array(geoms):
            raise TypeError("insert_many_np requires a NumPy array")

        import numpy as np

        if not isinstance(geoms, np.ndarray):
            raise TypeError("insert_many_np requires a NumPy array")

        if geoms.size == 0:
            return InsertResult(
                count=0, start_id=self._next_id, end_id=self._next_id - 1
            )

        validate_np_dtype(geoms, self._dtype)

        start_id = self._next_id
        last_id = self._native.insert_many_np(start_id, geoms)
        num = last_id - start_id + 1

        if num < len(geoms):
            raise ValueError("One or more geometries are outside tree bounds")

        self._next_id = last_id + 1
        self._count += num
        return InsertResult(count=num, start_id=start_id, end_id=last_id)

    # ---- Deletion ----

    def delete(self, id_: int, geom: G) -> bool:
        """
        Delete an item by ID and exact geometry.

        Geometry is required because non-Objects classes don't store it.

        Args:
            id_: The ID of the item to delete.
            geom: The exact geometry of the item.

        Returns:
            True if the item was found and deleted.
        """
        deleted = self._native.delete(id_, geom)
        if deleted:
            self._count -= 1
        return deleted

    def clear(self) -> None:
        """
        Empty the tree in place, preserving bounds, capacity, and max_depth.
        """
        self._native = self._new_native(
            self._bounds, self._capacity, self._max_depth, self._dtype
        )
        self._count = 0
        self._next_id = 0

    # ---- Mutation ----

    def update(self, id_: int, old_geom: G, new_geom: G) -> bool:
        """
        Move an existing item to a new location.

        Old geometry is required because non-Objects classes don't store it.

        Args:
            id_: The ID of the item to move.
            old_geom: Current geometry.
            new_geom: New geometry.

        Returns:
            True if the update succeeded.

        Raises:
            ValueError: If new geometry is outside bounds.
        """
        # Delete from old position
        if not self._native.delete(id_, old_geom):
            return False

        # Insert at new position
        if not self._native.insert(id_, new_geom):
            # Rollback: reinsert at old position
            self._native.insert(id_, old_geom)
            bx0, by0, bx1, by1 = self._bounds
            raise ValueError(
                f"New geometry {new_geom!r} is outside bounds ({bx0}, {by0}, {bx1}, {by1})"
            )

        return True

    # ---- Utilities ----

    def __len__(self) -> int:
        """Return the number of items in the tree."""
        return self._count

    def get_all_node_boundaries(self) -> list[Bounds]:
        """
        Return all node boundaries in the tree. Useful for visualization.
        """
        return self._native.get_all_node_boundaries()

    def get_inner_max_depth(self) -> int:
        """
        Return the maximum depth of the quadtree.

        Useful if you constructed with max_depth=None.
        """
        return self._native.get_max_depth()

    # ---- Serialization ----

    def to_bytes(self) -> bytes:
        """
        Serialize the quadtree to bytes.

        Returns:
            Bytes representing the serialized quadtree.
        """
        core_bytes = self._native.to_bytes()

        data = {
            "core": core_bytes,
            "bounds": self._bounds,
            "capacity": self._capacity,
            "max_depth": self._max_depth,
            "dtype": self._dtype,
            "next_id": self._next_id,
            "count": self._count,
        }

        return pickle.dumps(data)

    @classmethod
    def from_bytes(cls, data: bytes):
        """
        Deserialize a quadtree from bytes.

        Args:
            data: Bytes from to_bytes().

        Returns:
            A new instance.
        """
        in_dict = pickle.loads(data)

        dtype = in_dict["dtype"]

        qt = cls.__new__(cls)
        qt._native = cls._new_native_from_bytes(in_dict["core"], dtype)
        qt._bounds = in_dict["bounds"]
        qt._capacity = in_dict["capacity"]
        qt._max_depth = in_dict["max_depth"]
        qt._dtype = dtype
        qt._next_id = in_dict["next_id"]
        qt._count = in_dict["count"]

        return qt
