# _base_quadtree_objects.py
"""Base class for QuadTree and RectQuadTree with object tracking."""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence, TypeVar

from ._common import (
    SECTION_ITEMS,
    SECTION_OBJECTS,
    SERIALIZATION_FORMAT_VERSION,
    Bounds,
    Point,
    QuadTreeDType,
    SerializationError,
    _is_np_array,
    build_container,
    parse_container,
    validate_bounds,
    validate_np_dtype,
)
from ._insert_result import InsertResult
from ._item import Item
from ._obj_store import ObjStore

# Generic parameters
G = TypeVar("G")  # geometry type, e.g. Point or Bounds
ItemType = TypeVar("ItemType", bound=Item)  # e.g. PointItem or RectItem


def _is_point_geom(geom: Any) -> bool:
    # Points: (x, y)
    return isinstance(geom, tuple) and len(geom) == 2


def _is_rect_geom(geom: Any) -> bool:
    # Rects: (min_x, min_y, max_x, max_y)
    return isinstance(geom, tuple) and len(geom) == 4


def _encode_items_section(items: list[ItemType]) -> bytes:
    """
    Encode items (id + geom) into a safe payload.

    This is intentionally NOT pickle. It's a simple validated format using only:
      - int IDs
      - geometry tuples of length 2 or 4
    """
    import struct

    # Format:
    #   kind[u8] = 0 (points) or 1 (rects)
    #   reserved[u8]
    #   n[u32]
    #   repeated:
    #     id[u64]
    #     geom (2 or 4) as float64 (portable, independent of tree dtype)
    #
    # Note: we store geometry as float64 to keep it stable and easy.
    # The tree dtype still lives in the container header and is used by native core.
    if not items:
        # default to points kind=0 (doesn't matter when empty)
        return struct.pack("<BBI", 0, 0, 0)

    first_geom = items[0].geom
    if _is_point_geom(first_geom):
        kind = 0
        geom_len = 2
        pack_geom = struct.Struct("<2d").pack
    elif _is_rect_geom(first_geom):
        kind = 1
        geom_len = 4
        pack_geom = struct.Struct("<4d").pack
    else:
        raise SerializationError("Unsupported geometry in items section")

    # Validate all items match the same geometry type
    for it in items:
        g = it.geom
        if kind == 0 and not _is_point_geom(g):
            raise SerializationError("Mixed geometry kinds in items section")
        if kind == 1 and not _is_rect_geom(g):
            raise SerializationError("Mixed geometry kinds in items section")

    n = len(items)
    if n > 0xFFFFFFFF:
        raise SerializationError("Too many items to serialize")

    out = bytearray()
    out += struct.pack("<BBI", kind, 0, n)

    pack_id = struct.Struct("<Q").pack
    if geom_len == 2:
        for it in items:
            x, y = it.geom  # type: ignore[misc]
            out += pack_id(int(it.id_))
            out += pack_geom(float(x), float(y))
    else:
        for it in items:
            min_x, min_y, max_x, max_y = it.geom  # type: ignore[misc]
            out += pack_id(int(it.id_))
            out += pack_geom(float(min_x), float(min_y), float(max_x), float(max_y))

    return bytes(out)


def _decode_items_section(payload: bytes) -> list[tuple[int, tuple]]:
    """
    Decode the safe items section into (id, geom) pairs.
    """
    import struct

    buf = memoryview(payload)
    if len(buf) < 6:
        raise SerializationError("Items section too short")

    kind, _reserved, n = struct.unpack_from("<BBI", buf, 0)
    off = 6

    out: list[tuple[int, tuple]] = []
    if n == 0:
        return out

    if kind == 0:
        stride = 8 + 16  # id(u64) + 2*float64
        need = off + n * stride
        if len(buf) < need:
            raise SerializationError("Items section truncated (points)")
        for _ in range(n):
            (id_,) = struct.unpack_from("<Q", buf, off)
            off += 8
            x, y = struct.unpack_from("<2d", buf, off)
            off += 16
            out.append((int(id_), (float(x), float(y))))
        return out

    if kind == 1:
        stride = 8 + 32  # id(u64) + 4*float64
        need = off + n * stride
        if len(buf) < need:
            raise SerializationError("Items section truncated (rects)")
        for _ in range(n):
            (id_,) = struct.unpack_from("<Q", buf, off)
            off += 8
            min_x, min_y, max_x, max_y = struct.unpack_from("<4d", buf, off)
            off += 32
            out.append(
                (int(id_), (float(min_x), float(min_y), float(max_x), float(max_y)))
            )
        return out

    raise SerializationError(f"Unknown items section kind: {kind}")


def _encode_objects_section(store: ObjStore[ItemType]) -> bytes:
    """
    Encode objects (id -> obj) as a pickle payload (unsafe).
    Only used when include_objects=True.
    """
    # Keep it minimal: serialize (id, obj) pairs for ids that currently exist.
    # This avoids depending on ObjStore internal representation.
    pairs: list[tuple[int, Any]] = [(int(it.id_), it.obj) for it in store.items()]
    return pickle.dumps(pairs, protocol=pickle.HIGHEST_PROTOCOL)


def _decode_objects_section(payload: bytes) -> list[tuple[int, Any]]:
    """
    Decode pickled objects section. Caller must have checked allow_objects=True.
    """
    pairs = pickle.loads(payload)
    if not isinstance(pairs, list):
        raise SerializationError("Objects section malformed")
    out: list[tuple[int, Any]] = []
    for p in pairs:
        if not (isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], int)):
            raise SerializationError("Objects section contains invalid entries")
        out.append((int(p[0]), p[1]))
    return out


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
    def _new_native_from_bytes(cls, data: bytes, dtype: QuadTreeDType) -> Any:
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
        dtype: QuadTreeDType = "f32",
    ):
        self._bounds = validate_bounds(bounds)
        self._capacity = capacity
        self._max_depth = max_depth
        self._dtype: QuadTreeDType = dtype

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
            # Return the allocated id to the free-list to avoid id gaps
            self._store._free.append(rid)
            min_x, min_y, max_x, max_y = self._bounds
            raise ValueError(
                f"Geometry {geom!r} is outside bounds ({min_x}, {min_y}, {max_x}, {max_y})"
            )

        self._store.add(self._make_item(rid, geom, obj))
        self._count += 1
        return rid

    def insert_many(
        self, geoms: Sequence[G], objs: list[Any] | None = None
    ) -> InsertResult:
        """
        Bulk insert geometries with auto-assigned contiguous IDs.

        Note:
            For performance, this method always appends new items and does not reuse
            IDs from the free-list created by deletions. Use insert() to fill holes.

        Args:
            geoms: Sequence of geometries.
            objs: Optional list of Python objects aligned with geoms.

        Returns:
            InsertResult with count, start_id, and end_id.

        Raises:
            TypeError: If geoms is a NumPy array (use insert_many_np instead).
            ValueError: If any geometry is outside bounds or objs length doesn't match.
        """
        if _is_np_array(geoms):
            raise TypeError(
                "NumPy arrays are not supported by insert_many. "
                "Use insert_many_np() for NumPy arrays."
            )

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

        Note:
            For performance, this method always appends new items and does not reuse
            IDs from the free-list created by deletions. Use insert() to fill holes.

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
                add(mk(start_id + off, tuple(geom), None))  # type: ignore
        else:
            if len(objs) != len(geoms_list):
                raise ValueError("objs length must match geoms length")
            for off, (geom, obj) in enumerate(zip(geoms_list, objs)):
                add(mk(start_id + off, tuple(geom), obj))  # type: ignore

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
        items = self._store.by_obj_all(obj)
        deleted_count = 0
        for item in items:
            if self.delete(item.id_):
                deleted_count += 1
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
            eps = 1 if self._dtype[0] == "i" else 1e-5
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

        Safety:
          - include_objects=False (default): safe to load from untrusted data (no pickle executed)
          - include_objects=True: includes a pickle section; unsafe for untrusted data

        Args:
            include_objects: If True, serialize Python objects using pickle (unsafe).

        Returns:
            Bytes representing the serialized quadtree.
        """
        core_bytes = self._native.to_bytes()

        flags = 0
        if self._max_depth is not None:
            flags |= 1  # max_depth_present

        # Always store items (id + geom) safely.
        items_payload = _encode_items_section(list(self._store.items()))
        sections: list[tuple[int, bytes]] = [(SECTION_ITEMS, items_payload)]

        # Optionally store Python objects (unsafe).
        if include_objects:
            sections.append((SECTION_OBJECTS, _encode_objects_section(self._store)))

        return build_container(
            fmt_ver=SERIALIZATION_FORMAT_VERSION,
            dtype=self._dtype,  # type: ignore[arg-type]
            flags=flags,
            capacity=self._capacity,
            max_depth=self._max_depth,
            next_id=0,  # unused for Objects trees (dense IDs live in store)
            count=self._count,
            bounds=self._bounds,
            core=core_bytes,
            extra_sections=sections,
        )

    @classmethod
    def from_bytes(cls, data: bytes, allow_objects: bool = False):
        """
        Deserialize a quadtree from bytes.

        Args:
            data: Bytes from to_bytes().
            allow_objects: If True, load pickled Python objects (unsafe).
                          If False (default), object payloads are silently ignored.

        Returns:
            A new instance.

        Note:
            Object deserialization uses pickle-like semantics. Never load
            serialized data from untrusted sources with allow_objects=True.
        """
        parsed = parse_container(data)

        fmt_ver = parsed["fmt_ver"]
        if fmt_ver > SERIALIZATION_FORMAT_VERSION:
            raise SerializationError(
                f"Unsupported serialization format version {fmt_ver}; "
                f"this package supports up to {SERIALIZATION_FORMAT_VERSION}"
            )

        dtype = parsed["dtype"]
        core = parsed["core"]

        # Pull out sections
        sections: list[tuple[int, bytes]] = parsed["sections"]
        items_section = None
        objects_section = None
        for stype, payload in sections:
            if stype == SECTION_ITEMS:
                items_section = payload
            elif stype == SECTION_OBJECTS:
                objects_section = payload

        if items_section is None:
            raise SerializationError("Missing required items section")

        # Decode items safely (id + geom)
        id_geom_pairs = _decode_items_section(items_section)

        # Decode objects (unsafe) if present AND allowed
        # If objects_section exists but allow_objects=False
        id_to_obj: dict[int, Any] = {}
        if objects_section is not None and allow_objects:
            # Rewrite as dict comprehension
            id_to_obj = dict(_decode_objects_section(objects_section))

        # Construct instance without __init__
        qt = cls.__new__(cls)
        qt._dtype = dtype
        qt._bounds = parsed["bounds"]
        qt._capacity = parsed["capacity"]
        qt._max_depth = parsed["max_depth"]
        qt._count = parsed["count"]
        qt._native = cls._new_native_from_bytes(core, dtype)

        # Rebuild store from decoded ids/geoms (+ optional objects)
        store: ObjStore[ItemType] = ObjStore()
        add = store.add
        mk = cls._make_item

        for id_, geom in id_geom_pairs:
            obj = id_to_obj.get(id_)
            add(mk(id_, geom, obj), handle_out_of_order=True)  # type: ignore[arg-type]

        # Populate free-list from None holes created during deserialization.
        # This ensures holes are reusable for future inserts, matching the
        # behavior of the original tree before serialization.
        # Note: store._arr[i] is None checks for empty slots, NOT items with obj=None.
        # Items with obj=None are valid Item objects, empty slots are literally None.
        for i in range(len(store._arr)):
            if store._arr[i] is None:
                store._free.append(i)

        qt._store = store

        return qt
