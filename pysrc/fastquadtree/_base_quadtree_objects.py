# _base_quadtree_objects.py
"""Base class for QuadTree and RectQuadTree with object tracking."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from ._common import (
    Bounds,
    Point,
    _is_np_array,
    validate_bounds,
    validate_np_dtype,
)
from ._insert_result import InsertResult
from ._item import Item
from ._obj_store import ObjStore

# Generic parameters
G = TypeVar("G")  # geometry type, e.g. Point or Bounds
ItemType = TypeVar("ItemType", bound=Item)  # e.g. PointItem or RectItem


class _BaseQuadTreeObjects(Generic[G, ItemType], ABC):
    """
    Shared logic for QuadTree and RectQuadTree variants with object tracking.

    Concrete subclasses must implement:
      - _new_native(bounds, capacity, max_depth)
      - _new_native_from_bytes(data, dtype)
      - _make_item(id_, geom, obj)
      - _extract_coords_from_geom(geom) -> tuple for exact coordinate matching
    """

    __slots__ = (
        "_bounds",
        "_capacity",
        "_count",
        "_dtype",
        "_max_depth",
        "_native",
        "_store",
    )

    # ---- Required hooks for subclasses ----

    @abstractmethod
    def _new_native(self, bounds: Bounds, capacity: int, max_depth: int | None) -> Any:
        """Create the native engine instance."""

    @classmethod
    @abstractmethod
    def _new_native_from_bytes(cls, data: bytes, dtype: str) -> Any:
        """Create the native engine instance from serialized bytes."""

    @staticmethod
    @abstractmethod
    def _make_item(id_: int, geom: G, obj: Any | None) -> ItemType:
        """Build an ItemType from id, geometry, and optional object."""

    @staticmethod
    @abstractmethod
    def _extract_coords_from_geom(geom: G) -> tuple:
        """Extract coordinate tuple from geometry for exact matching."""

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

        self._native = self._new_native(self._bounds, capacity, max_depth)
        self._store: ObjStore[ItemType] = ObjStore()
        self._count = 0

    # ---- Insertion ----

    def insert(self, geom: G, obj: Any = None) -> int:
        """
        Insert a single geometry with an optional associated object.

        IDs are auto-assigned using dense allocation for efficient object lookup.
        Custom IDs are not supported in Objects classes.

        Args:
            geom: Geometry (Point or Bounds).
            obj: Optional Python object to associate with this geometry.

        Returns:
            The auto-assigned ID.

        Raises:
            ValueError: If geometry is outside the tree bounds.
        """
        rid = self._store.alloc_id()

        if not self._native.insert(rid, geom):
            bx0, by0, bx1, by1 = self._bounds
            raise ValueError(
                f"Geometry {geom!r} is outside bounds ({bx0}, {by0}, {bx1}, {by1})"
            )

        self._store.add(self._make_item(rid, geom, obj))
        self._count += 1
        return rid

    def insert_many(
        self, geoms: list[G], objs: list[Any] | None = None
    ) -> InsertResult:
        """
        Bulk insert geometries with auto-assigned contiguous IDs.

        Args:
            geoms: List of geometries.
            objs: Optional list of Python objects aligned with geoms.

        Returns:
            InsertResult with count, start_id, and end_id.

        Raises:
            ValueError: If any geometry is outside bounds or objs length doesn't match.
        """
        if len(geoms) == 0:
            start_id = len(self._store._arr)
            return InsertResult(count=0, start_id=start_id, end_id=start_id - 1)

        start_id = len(self._store._arr)
        last_id = self._native.insert_many(start_id, geoms)
        num = last_id - start_id + 1

        if num < len(geoms):
            raise ValueError("One or more geometries are outside tree bounds")

        # Add items to the store
        add = self._store.add
        mk = self._make_item
        if objs is None:
            for off, geom in enumerate(geoms):
                add(mk(start_id + off, geom, None))
        else:
            if len(objs) != len(geoms):
                raise ValueError("objs length must match geoms length")
            for off, (geom, obj) in enumerate(zip(geoms, objs)):
                add(mk(start_id + off, geom, obj))

        self._count += num
        return InsertResult(count=num, start_id=start_id, end_id=last_id)

    def insert_many_np(self, geoms: Any, objs: list[Any] | None = None) -> InsertResult:
        """
        Bulk insert geometries from NumPy array with auto-assigned contiguous IDs.

        Args:
            geoms: NumPy array with dtype matching the tree's dtype.
            objs: Optional list of Python objects aligned with geoms.

        Returns:
            InsertResult with count, start_id, and end_id.

        Raises:
            TypeError: If geoms is not a NumPy array or dtype doesn't match.
            ValueError: If any geometry is outside bounds or objs length doesn't match.
            ImportError: If NumPy is not installed.
        """
        if not _is_np_array(geoms):
            raise TypeError("insert_many_np requires a NumPy array")

        import numpy as np

        if not isinstance(geoms, np.ndarray):
            raise TypeError("insert_many_np requires a NumPy array")

        if geoms.size == 0:
            start_id = len(self._store._arr)
            return InsertResult(count=0, start_id=start_id, end_id=start_id - 1)

        validate_np_dtype(geoms, self._dtype)

        start_id = len(self._store._arr)
        last_id = self._native.insert_many_np(start_id, geoms)
        num = last_id - start_id + 1

        if num < len(geoms):
            raise ValueError("One or more geometries are outside tree bounds")

        # Convert to Python list for storage
        geoms_list = geoms.tolist()

        # Add items to the store
        add = self._store.add
        mk = self._make_item
        if objs is None:
            for off, geom in enumerate(geoms_list):
                add(mk(start_id + off, tuple(geom), None))
        else:
            if len(objs) != len(geoms_list):
                raise ValueError("objs length must match geoms length")
            for off, (geom, obj) in enumerate(zip(geoms_list, objs)):
                add(mk(start_id + off, tuple(geom), obj))

        self._count += num
        return InsertResult(count=num, start_id=start_id, end_id=last_id)

    # ---- Queries ----

    def query(self, rect: Bounds) -> list[ItemType]:
        """
        Return all items that intersect/contain the query rectangle.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).

        Returns:
            List of Item objects.
        """
        return self._store.get_many_by_ids(self._native.query_ids(rect))

    def query_ids(self, rect: Bounds) -> list[int]:
        """
        Return IDs of all items that intersect/contain the query rectangle.

        Fast path that only returns IDs without fetching items.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).

        Returns:
            List of integer IDs.
        """
        return self._native.query_ids(rect)

    def query_np(self, rect: Bounds) -> tuple[Any, Any]:
        """
        Return all items as NumPy arrays.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).

        Returns:
            Tuple of (ids, coords) where ids is NDArray[np.int64] and coords matches tree dtype.

        Raises:
            ImportError: If NumPy is not installed.
        """
        return self._native.query_np(rect)

    def nearest_neighbor(self, point: Point) -> ItemType | None:
        """
        Return the single nearest neighbor to the query point.

        Args:
            point: Query point (x, y).

        Returns:
            Item or None if the tree is empty.
        """
        t = self._native.nearest_neighbor(point)
        if t is None:
            return None
        id_ = t[0]
        it = self._store.by_id(id_)
        if it is None:
            raise RuntimeError("Internal error: missing tracked item")
        return it

    def nearest_neighbor_np(self, point: Point) -> tuple[int, Any] | None:
        """
        Return the single nearest neighbor as NumPy array.

        Args:
            point: Query point (x, y).

        Returns:
            Tuple of (id, coords) or None if tree is empty.

        Raises:
            ImportError: If NumPy is not installed.
        """
        return self._native.nearest_neighbor_np(point)

    def nearest_neighbors(self, point: Point, k: int) -> list[ItemType]:
        """
        Return the k nearest neighbors to the query point.

        Args:
            point: Query point (x, y).
            k: Number of neighbors to return.

        Returns:
            List of Item objects in order of increasing distance.
        """
        raw = self._native.nearest_neighbors(point, k)
        out: list[ItemType] = []
        for item_tuple in raw:
            id_ = item_tuple[0]
            it = self._store.by_id(id_)
            if it is None:
                raise RuntimeError("Internal error: missing tracked item")
            out.append(it)
        return out

    def nearest_neighbors_np(self, point: Point, k: int) -> tuple[Any, Any]:
        """
        Return the k nearest neighbors as NumPy arrays.

        Args:
            point: Query point (x, y).
            k: Number of neighbors to return.

        Returns:
            Tuple of (ids, coords) as NumPy arrays.

        Raises:
            ImportError: If NumPy is not installed.
        """
        return self._native.nearest_neighbors_np(point, k)

    # ---- Deletion ----

    def delete(self, id_: int) -> bool:
        """
        Delete an item by ID alone.

        Args:
            id_: The ID of the item to delete.

        Returns:
            True if the item was found and deleted.
        """
        item = self._store.by_id(id_)
        if item is None:
            return False

        deleted = self._native.delete(id_, item.geom)
        if deleted:
            self._count -= 1
            self._store.pop_id(id_)
        return deleted

    def delete_by_object(self, obj: Any) -> int:
        """
        Delete all items with the given object (by identity, not equality).

        Args:
            obj: The Python object to search for.

        Returns:
            Number of items deleted.
        """
        deleted_count = 0
        while True:
            it = self._store.by_obj(obj)
            if it is None:
                break
            if self.delete(it.id_):
                deleted_count += 1
            else:
                break
        return deleted_count

    def delete_one_by_object(self, obj: Any) -> bool:
        """
        Delete one item with the given object (by identity).

        If multiple items have this object, deletes the one with the lowest ID.

        Args:
            obj: The Python object to search for.

        Returns:
            True if an item was deleted.
        """
        it = self._store.by_obj(obj)
        if it is None:
            return False
        return self.delete(it.id_)

    def clear(self) -> None:
        """Empty the tree in place, preserving bounds, capacity, and max_depth."""
        self._native = self._new_native(self._bounds, self._capacity, self._max_depth)
        self._count = 0
        self._store.clear()

    # ---- Object Management ----

    def get(self, id_: int) -> Any | None:
        """
        Return the object associated with the given ID.

        Args:
            id_: The ID to look up.

        Returns:
            The associated object or None if not found.
        """
        item = self._store.by_id(id_)
        return None if item is None else item.obj

    def attach(self, id_: int, obj: Any) -> None:
        """
        Attach or replace the Python object for an existing ID.

        Args:
            id_: The ID of the item.
            obj: The Python object to attach.

        Raises:
            KeyError: If the ID is not found.
        """
        it = self._store.by_id(id_)
        if it is None:
            raise KeyError(f"ID {id_} not found in quadtree")
        # Preserve geometry from existing item
        self._store.add(self._make_item(id_, it.geom, obj))  # type: ignore[arg-type]

    def get_all_objects(self) -> list[Any]:
        """Return all tracked Python objects in the tree."""
        return [item.obj for item in self._store.items() if item.obj is not None]

    def get_all_items(self) -> list[ItemType]:
        """Return all Item wrappers in the tree."""
        return list(self._store.items())

    # ---- Utilities ----

    def __len__(self) -> int:
        """Return the number of items in the tree."""
        return self._count

    def __contains__(self, geom: G) -> bool:
        """
        Check if any item exists at the given coordinates.

        Args:
            geom: Geometry to check.

        Returns:
            True if at least one item exists at these exact coordinates.
        """
        # Subclasses can override this for geometry-specific logic
        coords = self._extract_coords_from_geom(geom)
        # Query for candidates and check for exact match
        # For points: rect is a tiny box around the point
        # For rects: rect is the exact rect
        if len(coords) == 2:
            # Point
            x, y = coords
            eps = 1e-9
            rect = (x - eps, y - eps, x + eps, y + eps)
            candidates = self._native.query(rect)
            return any(item_coords[1:3] == coords for item_coords in candidates)
        # Rect
        rect = coords
        candidates = self._native.query(rect)
        return any(item_coords[1:] == coords for item_coords in candidates)

    def __iter__(self):
        """Iterate over all Item objects in the tree."""
        return iter(self._store.items())

    def get_all_node_boundaries(self) -> list[Bounds]:
        """Return all node boundaries in the tree. Useful for visualization."""
        return self._native.get_all_node_boundaries()

    def get_inner_max_depth(self) -> int:
        """
        Return the maximum depth of the quadtree.

        Useful if you constructed with max_depth=None.
        """
        return self._native.get_max_depth()

    # ---- Serialization ----

    def to_bytes(self, include_objects: bool = False) -> bytes:
        """
        Serialize the quadtree to bytes.

        Object serialization is explicit and off by default for safety.

        Args:
            include_objects: If True, serialize Python objects using pickle (unsafe for untrusted data).

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
            "count": self._count,
            "include_objects": include_objects,
        }

        if include_objects:
            data["store"] = self._store.to_dict()
        else:
            # Store geometry and IDs but not objects
            items_data = [
                {"id": item.id_, "geom": item.geom, "obj": None}
                for item in self._store.items()
            ]
            data["store"] = {"items": items_data}

        return pickle.dumps(data)

    @classmethod
    def from_bytes(cls, data: bytes, allow_objects: bool = False):
        """
        Deserialize a quadtree from bytes.

        Args:
            data: Bytes from to_bytes().
            allow_objects: If True, allow loading pickled Python objects (unsafe for untrusted data).

        Returns:
            A new instance.

        Raises:
            ValueError: If allow_objects=False but data contains objects.
        """
        in_dict = pickle.loads(data)

        if in_dict.get("include_objects", False) and not allow_objects:
            raise ValueError(
                "Serialized data contains Python objects but allow_objects=False. "
                "Set allow_objects=True to load objects (unsafe for untrusted data)."
            )

        dtype = in_dict["dtype"]

        qt = cls.__new__(cls)
        qt._native = cls._new_native_from_bytes(in_dict["core"], dtype)
        qt._bounds = in_dict["bounds"]
        qt._capacity = in_dict["capacity"]
        qt._max_depth = in_dict["max_depth"]
        qt._dtype = dtype
        qt._count = in_dict["count"]

        # Restore store - use _make_item as factory
        qt._store = ObjStore.from_dict(in_dict["store"], cls._make_item)

        return qt
