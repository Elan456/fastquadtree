# _abc_quadtree.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Tuple, TypeVar

from ._bimap import BiMap
from ._item import Item  # base class for PointItem and RectItem

Bounds = Tuple[float, float, float, float]

# Generic parameters
G = TypeVar("G")  # geometry type, e.g. Point or Bounds
HitT = TypeVar("HitT")  # raw native tuple, e.g. (id,x,y) or (id,x0,y0,x1,y1)
ItemType = TypeVar(
    "ItemType", bound=Item
)  # Python Item subtype, e.g. PointItem or RectItem


class _BaseQuadTree(Generic[G, HitT, ItemType], ABC):
    """
    Shared logic for Python QuadTree wrappers over native Rust engines.

    Concrete subclasses must implement the few native hooks and item builders.
    """

    __slots__ = (
        "_bounds",
        "_capacity",
        "_count",
        "_items",
        "_max_depth",
        "_native",
        "_next_id",
        "_track_objects",
    )

    # ---- required native hooks ----

    @abstractmethod
    def _new_native(self, bounds: Bounds, capacity: int, max_depth: int | None) -> Any:
        """Create the native engine instance."""

    @abstractmethod
    def _native_insert(self, id_: int, geom: G) -> bool:
        """Call native insert for one geometry."""

    @abstractmethod
    def _native_delete(self, id_: int, geom: G) -> bool:
        """Call native delete for one geometry."""

    @abstractmethod
    def _native_query(self, rect: Bounds) -> list[HitT]:
        """Call native query. Returns raw hit tuples."""

    @abstractmethod
    def _native_count(self) -> int:
        """Call native count_items()."""

    @abstractmethod
    def _native_insert_many(self, start_id: int, geoms: list[G]) -> int:
        """
        Bulk insert. Default: loop calling single insert.
        Override to bind to a native bulk method returning last_id.
        """

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
        start_id: int = 1,
    ):
        self._bounds = bounds
        self._max_depth = max_depth
        self._capacity = capacity
        self._native = self._new_native(bounds, capacity, max_depth)
        # typed item map if tracking enabled
        self._track_objects = bool(track_objects)
        self._items: BiMap[ItemType] | None = BiMap() if track_objects else None
        self._next_id = int(start_id)
        self._count = 0

    # ---- shared helpers ----

    def _alloc_id(self, id_: int | None) -> int:
        if id_ is None:
            nid = self._next_id
            self._next_id += 1
            return nid
        if id_ >= self._next_id:
            self._next_id = id_ + 1
        return id_

    def _insert_common(self, geom: G, *, id_: int | None, obj: Any = None) -> int:
        use_id = self._alloc_id(id_)
        if not self._native_insert(use_id, geom):
            bx0, by0, bx1, by1 = self._bounds
            raise ValueError(
                f"Geometry {geom!r} is outside bounds ({bx0}, {by0}, {bx1}, {by1})"
            )

        if self._items is not None:
            self._items.add(self._make_item(use_id, geom, obj))

        self._count += 1
        return use_id

    def _insert_many_common(self, geoms: list[G]) -> int:
        start_id = self._next_id
        last_id = self._native_insert_many(start_id, geoms)
        num = last_id - start_id + 1
        if num < len(geoms):
            raise ValueError("One or more items are outside tree bounds")

        self._next_id = last_id + 1
        if self._items is not None:
            for i, id_ in enumerate(range(start_id, last_id + 1)):
                self._items.add(self._make_item(id_, geoms[i], None))
        self._count += num
        return num

    def _delete_exact(self, id_: int, geom: G) -> bool:
        deleted = self._native_delete(id_, geom)
        if deleted:
            self._count -= 1
            if self._items is not None:
                self._items.pop_id(id_)
        return deleted

    # ---- public shared API ----

    def attach(self, id_: int, obj: Any) -> None:
        if self._items is None:
            raise ValueError("Cannot attach objects when track_objects=False")
        it = self._items.by_id(id_)
        if it is None:
            raise KeyError(f"Id {id_} not found in quadtree")
        # Preserve geometry from existing item
        self._items.add(self._make_item(id_, it.geom, obj))  # type: ignore[attr-defined]

    def delete_by_object(self, obj: Any) -> bool:
        if self._items is None:
            raise ValueError("Cannot delete by object when track_objects=False")
        it = self._items.by_obj(obj)
        if it is None:
            return False
        # type of geom is determined by concrete Item subtype
        return self._delete_exact(it.id_, it.geom)  # type: ignore[arg-type]

    def clear(self, *, reset_ids: bool = False) -> None:
        self._native = self._new_native(self._bounds, self._capacity, self._max_depth)
        self._count = 0
        if self._items is not None:
            self._items.clear()
        if reset_ids:
            self._next_id = 1

    def get_all_objects(self) -> list[Any]:
        if self._items is None:
            raise ValueError("Cannot get objects when track_objects=False")
        return [t.obj for t in self._items.items() if t.obj is not None]

    def get_all_items(self) -> list[ItemType]:
        if self._items is None:
            raise ValueError("Cannot get items when track_objects=False")
        return list(self._items.items())

    def get_all_node_boundaries(self) -> list[Bounds]:
        return self._native.get_all_node_boundaries()

    def get(self, id_: int) -> Any | None:
        """
        Return the object associated with id, if tracking is enabled.
        """
        if self._items is None:
            raise ValueError("Cannot get objects when track_objects=False")
        item = self._items.by_id(id_)
        return None if item is None else item.obj

    def count_items(self) -> int:
        return self._native_count()

    def __len__(self) -> int:
        return self._count
