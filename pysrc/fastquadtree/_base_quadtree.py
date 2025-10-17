# _abc_quadtree.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Iterable,
    Sequence,
    Tuple,
    TypeVar,
    overload,
)

from ._item import Item  # base class for PointItem and RectItem
from ._obj_store import ObjStore

if TYPE_CHECKING:
    from numpy.typing import NDArray

Bounds = Tuple[float, float, float, float]

# Generic parameters
G = TypeVar("G")  # geometry type, e.g. Point or Bounds
HitT = TypeVar("HitT")  # raw native tuple, e.g. (id,x,y) or (id,x0,y0,x1,y1)
ItemType = TypeVar("ItemType", bound=Item)  # e.g. PointItem or RectItem


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

    @abstractmethod
    def _make_item(self, id_: int, geom: G, obj: Any | None) -> ItemType:
        """Build an ItemType from id, geometry, and optional object."""

    # ---- ctor ----

    def __init__(
        self,
        bounds: Bounds,
        capacity: int,
        *,
        max_depth: int | None = None,
        track_objects: bool = False,
    ):
        self._bounds = bounds
        self._max_depth = max_depth
        self._capacity = capacity
        self._native = self._new_native(bounds, capacity, max_depth)

        self._track_objects = bool(track_objects)
        self._store: ObjStore[ItemType] | None = ObjStore() if track_objects else None

        # Auto ids when not using ObjStore.free slots
        self._next_id = 0
        self._count = 0

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
        Can accept either a Python sequence of geometries or a NumPy array of shape (N,2) or (N,4) with dtype float32.

        If tracking is enabled, the objects will be bulk stored internally.
        If no objects are provided, the items will have obj=None (if tracking).

        Args:
            geoms: List of geometries.
            objs: Optional list of Python objects aligned with geoms.

        Returns:
            Number of items inserted.

        Raises:
            ValueError: If any geometry is outside bounds.
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

            if geoms.dtype != _np.float32:
                raise TypeError("Numpy array must use dtype float32")

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

        Returns:
            True if the item was found and deleted.
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
        """
        self._native = self._new_native(self._bounds, self._capacity, self._max_depth)
        self._count = 0
        if self._store is not None:
            self._store.clear()
        self._next_id = 0

    def get_all_objects(self) -> list[Any]:
        """
        Return all tracked Python objects in the tree.
        """
        if self._store is None:
            raise ValueError("Cannot get objects when track_objects=False")
        return [t.obj for t in self._store.items() if t.obj is not None]

    def get_all_items(self) -> list[ItemType]:
        """
        Return all Item wrappers in the tree.
        """
        if self._store is None:
            raise ValueError("Cannot get items when track_objects=False")
        return list(self._store.items())

    def get_all_node_boundaries(self) -> list[Bounds]:
        """
        Return all node boundaries in the tree. Useful for visualization.
        """
        return self._native.get_all_node_boundaries()

    def get(self, id_: int) -> Any | None:
        """
        Return the object associated with id, if tracking is enabled.
        """
        if self._store is None:
            raise ValueError("Cannot get objects when track_objects=False")
        item = self._store.by_id(id_)
        return None if item is None else item.obj

    def count_items(self) -> int:
        """
        Return the number of items currently in the tree (native count).
        """
        return self._native.count_items()

    def __len__(self) -> int:
        return self._count
