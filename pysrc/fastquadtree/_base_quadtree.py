# _abc_quadtree.py
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Sequence,
    SupportsFloat,
    Tuple,
    TypeVar,
    overload,
)

from ._item import Item  # base class for PointItem and RectItem
from ._obj_store import ObjStore

if TYPE_CHECKING:
    from typing import Self  # Only in Python 3.11+

    from numpy.typing import NDArray

Bounds = Tuple[SupportsFloat, SupportsFloat, SupportsFloat, SupportsFloat]

# Generic parameters
G = TypeVar("G")  # geometry type, e.g. Point or Bounds
HitT = TypeVar("HitT")  # raw native tuple, e.g. (id,x,y) or (id,x0,y0,x1,y1)
ItemType = TypeVar("ItemType", bound=Item)  # e.g. PointItem or RectItem

# Quadtree dtype to numpy dtype mapping
QUADTREE_DTYPE_TO_NP_DTYPE = {
    "f32": "float32",
    "f64": "float64",
    "i32": "int32",
    "i64": "int64",
}


def _is_np_array(x: Any) -> bool:
    mod = getattr(x.__class__, "__module__", "")
    return mod.startswith("numpy") and hasattr(x, "ndim") and hasattr(x, "shape")


class _BaseQuadTree(Generic[G, HitT, ItemType], ABC):
    """
    Shared logic for Python QuadTree wrappers over native Rust engines.

    Concrete subclasses must implement:
      - _new_native(bounds, capacity, max_depth)
      - _make_item(id_, geom, obj)
    """

    __slots__ = (
        "_bounds",
        "_capacity",
        "_count",
        "_dtype",
        "_max_depth",
        "_native",
        "_next_id",
        "_store",
        "_track_objects",
    )

    # ---- required native hooks ----

    @abstractmethod
    def _new_native(self, bounds: Bounds, capacity: int, max_depth: int | None) -> Any:
        """Create the native engine instance."""

    @classmethod
    def _new_native_from_bytes(cls, data: bytes, dtype: str) -> Any:
        """Create the native engine instance from serialized bytes."""

    @staticmethod
    @abstractmethod
    def _make_item(id_: int, geom: G, obj: Any | None) -> ItemType:
        """Build an ItemType from id, geometry, and optional object."""

    # ---- ctor ----

    def __init__(
        self,
        bounds: Bounds,
        capacity: int,
        *,
        max_depth: int | None = None,
        track_objects: bool = False,
        dtype: str = "f32",
    ):
        # Handle some bounds validation and list --> tuple conversion
        if type(bounds) is not tuple:
            bounds = tuple(bounds)  # pyright: ignore[reportAssignmentType]
        if len(bounds) != 4:
            raise ValueError(
                "bounds must be a tuple of four numeric values (x min, y min, x max, y max)"
            )

        self._bounds = bounds

        self._max_depth = max_depth
        self._capacity = capacity
        self._dtype = dtype
        self._native = self._new_native(self._bounds, self._capacity, self._max_depth)

        self._track_objects = bool(track_objects)
        self._store: ObjStore[ItemType] | None = ObjStore() if track_objects else None

        # Auto ids when not using ObjStore.free slots
        self._next_id = 0
        self._count = 0

    # ---- serialization ----

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the quadtree to a dict suitable for JSON or other serialization.

        Returns:
            Includes a binary 'core' key for the native engine state, plus other metadata such as bounds and capacity and the obj store if tracking is enabled.

        Example:
            ```python
            state = qt.to_dict()
            assert "core" in state and "bounds" in state
            ```
        """

        core_bytes = self._native.to_bytes()

        return {
            "core": core_bytes,
            "store": self._store.to_dict() if self._store is not None else None,
            "bounds": self._bounds,
            "capacity": self._capacity,
            "max_depth": self._max_depth,
            "track_objects": self._track_objects,
            "next_id": self._next_id,
            "count": self._count,
        }

    def to_bytes(self) -> bytes:
        """
        Serialize the quadtree to bytes.

        Returns:
            Bytes representing the serialized quadtree. Can be saved as a file or loaded with `from_bytes()`.

        Example:
            ```python
            blob = qt.to_bytes()
            with open("tree.fqt", "wb") as f:
                f.write(blob)
            ```
        """
        return pickle.dumps(self.to_dict())

    @classmethod
    def from_bytes(cls, data: bytes, dtype: str = "f32") -> Self:
        """
        Deserialize a quadtree from bytes. Specifiy the dtype if the original tree that was serialized used a non-default dtype.

        Args:
            data: Bytes representing the serialized quadtree from `to_bytes()`.
            dtype: The data type used in the native engine ('f32', 'f64', 'i32', 'i64') when saved to bytes.

        Returns:
            A new quadtree instance with the same state as when serialized.

        Example:
            ```python
            blob = qt.to_bytes()
            qt2 = type(qt).from_bytes(blob)
            assert qt2.count_items() == qt.count_items()
            ```
        """
        in_dict = pickle.loads(data)
        core_bytes = in_dict["core"]
        store_dict = in_dict["store"]

        qt = cls.__new__(cls)  # type: ignore[call-arg]
        try:
            qt._native = cls._new_native_from_bytes(core_bytes, dtype=dtype)
        except ValueError as ve:
            raise ValueError(
                "Failed to deserialize quadtree native core. "
                "This may be due to a dtype mismatch. "
                "Ensure the dtype used in from_bytes() matches the original tree. "
                "Error details: " + str(ve)
            ) from ve

        if store_dict is not None:
            qt._store = ObjStore.from_dict(store_dict, qt._make_item)
        else:
            qt._store = None

        # Extract bounds, capacity, max_depth from native
        qt._bounds = in_dict["bounds"]
        qt._capacity = in_dict["capacity"]
        qt._max_depth = in_dict["max_depth"]
        qt._next_id = in_dict["next_id"]
        qt._count = in_dict["count"]
        qt._track_objects = in_dict["track_objects"]

        return qt

    # ---- internal helper ----

    def _ids_to_objects(self, ids: Iterable[int]) -> list[Any]:
        """Map ids -> Python objects via ObjStore in a batched way."""
        if self._store is None:
            raise ValueError("Cannot map ids to objects when track_objects=False")
        return self._store.get_many_objects(list(ids))

    # ---- shared API ----

    def insert(self, geom: G, *, obj: Any | None = None) -> int:
        """
        Insert a single item.

        Args:
            geom: Point (x, y) or Rect (x0, y0, x1, y1) depending on quadtree type.
            obj: Optional Python object to associate with id if tracking is enabled.

        Returns:
            The id used for this insert.

        Raises:
            ValueError: If geometry is outside the tree bounds.

        Example:
            ```python
            id0 = point_qt.insert((10.0, 20.0))  # for point trees
            id1 = rect_qt.insert((0.0, 0.0, 5.0, 5.0), obj="box")  # for rect trees
            assert isinstance(id0, int) and isinstance(id1, int)
            ```
        """
        if self._store is not None:
            # Reuse a dense free slot if available, else append
            rid = self._store.alloc_id()
        else:
            rid = self._next_id
            self._next_id += 1

        if not self._native.insert(rid, geom):
            bx0, by0, bx1, by1 = self._bounds
            raise ValueError(
                f"Geometry {geom!r} is outside bounds ({bx0}, {by0}, {bx1}, {by1})"
            )

        if self._store is not None:
            self._store.add(self._make_item(rid, geom, obj))

        self._count += 1
        return rid

    @overload
    def insert_many(self, geoms: Sequence[G], objs: list[Any] | None = None) -> int: ...
    @overload
    def insert_many(
        self, geoms: NDArray[Any], objs: list[Any] | None = None
    ) -> int: ...
    def insert_many(
        self, geoms: NDArray[Any] | Sequence[G], objs: list[Any] | None = None
    ) -> int:
        """
        Bulk insert with auto-assigned contiguous ids. Faster than inserting one-by-one.<br>
        Can accept either a Python sequence of geometries or a NumPy array of shape (N,2) or (N,4) with a dtype that matches the quadtree's dtype.

        If tracking is enabled, the objects will be bulk stored internally.
        If no objects are provided, the items will have obj=None (if tracking).

        Args:
            geoms: List of geometries.
            objs: Optional list of Python objects aligned with geoms.

        Returns:
            Number of items inserted.

        Raises:
            ValueError: If any geometry is outside bounds.

        Example:
            ```python
            n = qt.insert_many([(1.0, 1.0), (2.0, 2.0)])
            assert n == 2

            import numpy as np
            arr = np.array([[3.0, 3.0], [4.0, 4.0]], dtype=np.float32)
            n2 = qt.insert_many(arr)
            assert n2 == 2
            ```
        """
        if type(geoms) is list and len(geoms) == 0:
            return 0

        if _is_np_array(geoms):
            import numpy as _np
        else:
            _np = None

        # Early return if the numpy array is empty
        if _np is not None and isinstance(geoms, _np.ndarray):
            if geoms.size == 0:
                return 0

            # Check if dtype matches quadtree dtype
            expected_np_dtype = QUADTREE_DTYPE_TO_NP_DTYPE.get(self._dtype)
            if geoms.dtype != expected_np_dtype:
                raise TypeError(
                    f"Numpy array dtype {geoms.dtype} does not match quadtree dtype {self._dtype}"
                )

        if self._store is None:
            # Simple contiguous path with native bulk insert
            start_id = self._next_id

            if _np is not None:
                last_id = self._native.insert_many_np(start_id, geoms)
            else:
                last_id = self._native.insert_many(start_id, geoms)
            num = last_id - start_id + 1
            if num < len(geoms):
                raise ValueError("One or more items are outside tree bounds")
            self._next_id = last_id + 1
            self._count += num
            return num

        # With tracking enabled:
        start_id = len(self._store._arr)  # contiguous tail position
        if _np is not None:
            last_id = self._native.insert_many_np(start_id, geoms)
        else:
            last_id = self._native.insert_many(start_id, geoms)
        num = last_id - start_id + 1
        if num < len(geoms):
            raise ValueError("One or more items are outside tree bounds")

        # Add items to the store in one pass
        if objs is None:
            for off, geom in enumerate(geoms):
                id_ = start_id + off
                self._store.add(self._make_item(id_, geom, None))
        else:
            if len(objs) != len(geoms):
                raise ValueError("objs length must match geoms length")
            for off, (geom, o) in enumerate(zip(geoms, objs)):
                id_ = start_id + off
                self._store.add(self._make_item(id_, geom, o))

        # Keep _next_id monotonic for the non-tracking path
        self._next_id = max(self._next_id, last_id + 1)

        self._count += num
        return num

    def delete(self, id_: int, geom: G) -> bool:
        """
        Delete an item by id and exact geometry.

        Args:
            id_: The id of the item to delete.
            geom: The geometry of the item to delete.

        Returns:
            True if the item was found and deleted.

        Example:
            ```python
            i = qt.insert((1.0, 2.0))
            ok = qt.delete(i, (1.0, 2.0))
            assert ok is True
            ```
        """
        deleted = self._native.delete(id_, geom)
        if deleted:
            self._count -= 1
            if self._store is not None:
                self._store.pop_id(id_)
        return deleted

    def attach(self, id_: int, obj: Any) -> None:
        """
        Attach or replace the Python object for an existing id.
        Tracking must be enabled.

        Args:
            id_: The id of the item to attach the object to.
            obj: The Python object to attach.

        Example:
            ```python
            i = qt.insert((2.0, 3.0), obj=None)
            qt.attach(i, {"meta": 123})
            assert qt.get(i) == {"meta": 123}
            ```
        """
        if self._store is None:
            raise ValueError("Cannot attach objects when track_objects=False")
        it = self._store.by_id(id_)
        if it is None:
            raise KeyError(f"Id {id_} not found in quadtree")
        # Preserve geometry from existing item
        self._store.add(self._make_item(id_, it.geom, obj))  # type: ignore[attr-defined]

    def delete_by_object(self, obj: Any) -> bool:
        """
        Delete an item by Python object identity. Tracking must be enabled.

        Args:
            obj: The Python object to delete.

        Example:
            ```python
            i = qt.insert((3.0, 4.0), obj="tag")
            ok = qt.delete_by_object("tag")
            assert ok is True
            ```
        """
        if self._store is None:
            raise ValueError("Cannot delete by object when track_objects=False")
        it = self._store.by_obj(obj)
        if it is None:
            return False
        return self.delete(it.id_, it.geom)  # type: ignore[arg-type]

    def clear(self) -> None:
        """
        Empty the tree in place, preserving bounds, capacity, and max_depth.

        If tracking is enabled, the id -> object mapping is also cleared.
        The ids are reset to start at zero again.

        Example:
            ```python
            _ = qt.insert((5.0, 6.0))
            qt.clear()
            assert qt.count_items() == 0 and len(qt) == 0
            ```
        """
        self._native = self._new_native(self._bounds, self._capacity, self._max_depth)
        self._count = 0
        if self._store is not None:
            self._store.clear()
        self._next_id = 0

    def get_all_objects(self) -> list[Any]:
        """
        Return all tracked Python objects in the tree.

        Example:
            ```python
            _ = qt.insert((7.0, 8.0), obj="a")
            _ = qt.insert((9.0, 1.0), obj="b")
            objs = qt.get_all_objects()
            assert set(objs) == {"a", "b"}
            ```
        """
        if self._store is None:
            raise ValueError("Cannot get objects when track_objects=False")
        return [t.obj for t in self._store.items() if t.obj is not None]

    def get_all_items(self) -> list[ItemType]:
        """
        Return all Item wrappers in the tree.

         Example:
            ```python
            _ = qt.insert((1.0, 1.0), obj=None)
            items = qt.get_all_items()
            assert hasattr(items[0], "id_") and hasattr(items[0], "geom")
            ```
        """
        if self._store is None:
            raise ValueError("Cannot get items when track_objects=False")
        return list(self._store.items())

    def get_all_node_boundaries(self) -> list[Bounds]:
        """
        Return all node boundaries in the tree. Useful for visualization.

        Example:
            ```python
            bounds = qt.get_all_node_boundaries()
            assert isinstance(bounds, list)
            ```
        """
        return self._native.get_all_node_boundaries()

    def get(self, id_: int) -> Any | None:
        """
        Return the object associated with id, if tracking is enabled.

        Example:
            ```python
            i = qt.insert((1.0, 2.0), obj={"k": "v"})
            obj = qt.get(i)
            assert obj == {"k": "v"}
            ```
        """
        if self._store is None:
            raise ValueError("Cannot get objects when track_objects=False")
        item = self._store.by_id(id_)
        return None if item is None else item.obj

    def count_items(self) -> int:
        """
        Return the number of items currently in the tree (native count).

        Example:
            ```python
            before = qt.count_items()
            _ = qt.insert((2.0, 2.0))
            assert qt.count_items() == before + 1
            ```
        """
        return self._native.count_items()

    def __len__(self) -> int:
        return self._count
