# _base_quadtree.py
"""Base class for QuadTree and RectQuadTree without object tracking (v2.0 API)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence, TypeVar

from ._common import (
    SERIALIZATION_FORMAT_VERSION,
    Bounds,
    QuadTreeDType,
    SerializationError,
    _is_np_array,
    build_container,
    parse_container,
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
    def _new_native_from_bytes(cls, data: bytes, dtype: QuadTreeDType) -> Any:
        """Create the native engine instance from serialized bytes."""

    # ---- Initialization ----

    def __init__(
        self,
        bounds: Bounds,
        capacity: int,
        *,
        max_depth: int | None = None,
        dtype: QuadTreeDType = "f32",
    ):
        self._bounds = validate_bounds(bounds)
        self._capacity = capacity
        self._max_depth = max_depth
        self._dtype: QuadTreeDType = dtype

        self._native = self._new_native(self._bounds, capacity, max_depth, dtype)

        self._next_id: int = 0
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
            min_x, min_y, max_x, max_y = self._bounds
            raise ValueError(
                f"Geometry {geom!r} is outside bounds ({min_x}, {min_y}, {max_x}, {max_y})"
            )

        self._count += 1
        return id_

    def insert_many(self, geoms: Sequence[G]) -> InsertResult:
        """
        Bulk insert geometries with auto-assigned contiguous IDs. <br>
        IDs start at 0 and increment by 1, so they will be aligned with the indexes of the input list if the tree started empty. <br>

        Custom IDs are not supported for bulk insertion. Use single insert()
        calls if you need custom IDs.

        Args:
            geoms: Sequence of geometries.

        Returns:
            InsertResult with count, start_id, and end_id.

        Raises:
            TypeError: If geoms is a NumPy array (use insert_many_np instead).
            ValueError: If any geometry is outside bounds.

        Example:
            ```python
            # Point Quadtree Example:

            points = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
            qt = QuadTree(bounds=(0.0, 0.0, 10.0, 10.0), capacity=16)
            result = qt.insert_many(points) # Each point's ID corresponds to its index in the points list
            print(result)  # InsertResult(count=3, start_id=0, end_id=2)
            ```
        """
        if _is_np_array(geoms):
            raise TypeError(
                "NumPy arrays are not supported by insert_many. "
                "Use insert_many_np() for NumPy arrays."
            )

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

    def _delete_geom(self, id_: int, geom: G) -> bool:
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

    def _update_geom(self, id_: int, old_geom: G, new_geom: G) -> bool:
        """
        Update an item's geometry by moving it from old_geom to new_geom.

        This is an internal helper used by subclass update() methods.

        Args:
            id_: The ID of the item to update.
            old_geom: The old geometry.
            new_geom: The new geometry.

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
            min_x, min_y, max_x, max_y = self._bounds
            raise ValueError(
                f"New geometry {new_geom!r} is outside bounds ({min_x}, {min_y}, {max_x}, {max_y})"
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
        core_bytes = self._native.to_bytes()

        flags = 0
        if self._max_depth is not None:
            flags |= 1  # max_depth_present

        return build_container(
            fmt_ver=SERIALIZATION_FORMAT_VERSION,
            dtype=self._dtype,  # type: ignore[arg-type]
            flags=flags,
            capacity=self._capacity,
            max_depth=self._max_depth,
            next_id=self._next_id,
            count=self._count,
            bounds=self._bounds,
            core=core_bytes,
            extra_sections=None,  # reserved for Objects trees
        )

    @classmethod
    def from_bytes(cls, data: bytes):
        parsed = parse_container(data)

        fmt_ver = parsed["fmt_ver"]
        if fmt_ver > SERIALIZATION_FORMAT_VERSION:
            raise SerializationError(
                f"Unsupported serialization format version {fmt_ver}; "
                f"this package supports up to {SERIALIZATION_FORMAT_VERSION}"
            )

        dtype = parsed["dtype"]
        core = parsed["core"]

        qt = cls.__new__(cls)
        qt._dtype = dtype
        qt._bounds = parsed["bounds"]
        qt._capacity = parsed["capacity"]
        qt._max_depth = parsed["max_depth"]
        qt._next_id = parsed["next_id"]
        qt._count = parsed["count"]
        qt._native = cls._new_native_from_bytes(core, dtype)

        return qt
