# _bimap.py
from __future__ import annotations

from operator import itemgetter
from typing import Any, Generic, Iterable, Iterator, Sequence, TypeVar

from ._item import Item  # base class for PointItem and RectItem

TItem = TypeVar("TItem", bound=Item)


class ObjStore(Generic[TItem]):
    """
    High-performance id <-> object store for dense, auto-assigned ids.

    Storage
      - _arr[id]  -> Item or None
      - _objs[id] -> Python object or None
      - _obj_to_ids: reverse identity map id(obj) -> set[id]
      - _free: LIFO free-list of reusable ids

    Assumptions
      - Ids are assigned by the shim and are dense [0..len) with possible holes
        created by deletes. New inserts reuse holes via the free-list.
      - If the same object is inserted multiple times, all IDs are tracked.
        by_obj() returns the item with the lowest ID (deterministic).
        delete_by_object() deletes all instances of the object.
    """

    __slots__ = ("_arr", "_free", "_len", "_obj_to_ids", "_objs")

    def __init__(self, items: Iterable[TItem] | None = None) -> None:
        self._arr: list[TItem | None] = []
        self._objs: list[Any | None] = []
        self._obj_to_ids: dict[int, set[int]] = {}
        self._free: list[int] = []  # LIFO
        self._len: int = 0  # live items

        if items:
            for it in items:
                self.add(it, handle_out_of_order=True)

            # Populate free-list from None holes created during initialization
            for i in range(len(self._arr)):
                if self._arr[i] is None:
                    self._free.append(i)

    # ---- Serialization ----
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to a dict suitable for JSON or other serialization.

        Returns:
            A dict with 'items' key containing list of serialized items.
        """
        items = [it.to_dict() for it in self._arr if it is not None]
        return {"items": items}

    @classmethod
    def from_dict(cls, data: dict[str, Any], item_factory: Any) -> ObjStore[TItem]:
        """
        Deserialize from a dict.

        Args:
            data: A dict with 'items' key containing list of serialized items.
            item_factory: A callable that takes (id, obj) and returns an Item.

        Returns:
            An ObjStore instance populated with the deserialized items.
        """
        items = []
        for item_data in data.get("items", []):
            item = Item.from_dict(item_data)
            items.append(item_factory(item.id_, item.geom, item.obj))
        return cls(items)

    # -------- core --------

    def add(self, item: TItem, handle_out_of_order: bool = False) -> None:
        """
        Insert or replace the mapping at item.id_. Reverse map updated so obj points to id.
        If the same object is inserted multiple times, all IDs are tracked.

        Args:
            item: The item to add.
            handle_out_of_order: If True, fill gaps with None instead of raising.
                Used during deserialization to preserve original IDs from serialized data.
                Gaps are NOT added to the free-list (permanent holes until post-processing).
        """
        id_ = item.id_
        obj = item.obj

        # ids must be dense and assigned by the caller
        if id_ > len(self._arr):
            if not handle_out_of_order:
                raise AssertionError(
                    "ObjStore.add received an out-of-order id, use alloc_id() to get the next available id"
                )
            # fill holes with None
            while len(self._arr) < id_:
                self._arr.append(None)
                self._objs.append(None)

        if id_ == len(self._arr):
            # append
            self._arr.append(item)
            self._objs.append(obj)
            self._len += 1
        else:
            # replace or fill a hole
            old = self._arr[id_]
            if old is None:
                self._len += 1
            elif old.obj is not None:
                # Remove old object mapping for this ID (fixes attach() bug)
                old_obj_id = id(old.obj)
                if old_obj_id in self._obj_to_ids:
                    self._obj_to_ids[old_obj_id].discard(id_)
                    if not self._obj_to_ids[old_obj_id]:
                        del self._obj_to_ids[old_obj_id]
            self._arr[id_] = item
            self._objs[id_] = obj

        # Add new object mapping
        if obj is not None:
            obj_id = id(obj)
            if obj_id not in self._obj_to_ids:
                self._obj_to_ids[obj_id] = set()
            self._obj_to_ids[obj_id].add(id_)

    def by_id(self, id_: int) -> TItem | None:
        return self._arr[id_] if 0 <= id_ < len(self._arr) else None

    def by_obj(self, obj: Any) -> TItem | None:
        """Return the item with the LOWEST id for this object (deterministic)."""
        obj_id = id(obj)
        ids = self._obj_to_ids.get(obj_id)
        if not ids:
            return None
        min_id = min(ids)
        return self.by_id(min_id)

    def by_obj_all(self, obj: Any) -> list[TItem]:
        """Return ALL items associated with this object, sorted by ID."""
        obj_id = id(obj)
        ids = self._obj_to_ids.get(obj_id, set())
        result = []
        for i in sorted(ids):
            item = self.by_id(i)
            if item is not None:
                result.append(item)
        return result

    def pop_id(self, id_: int) -> TItem | None:
        """Remove by id. Dense ids go to the free-list for reuse."""
        if not (0 <= id_ < len(self._arr)):
            return None
        it = self._arr[id_]
        if it is None:
            return None
        self._arr[id_] = None
        self._objs[id_] = None

        # Remove from reverse mapping
        if it.obj is not None:
            obj_id = id(it.obj)
            if obj_id in self._obj_to_ids:
                self._obj_to_ids[obj_id].discard(id_)
                if not self._obj_to_ids[obj_id]:
                    del self._obj_to_ids[obj_id]

        self._free.append(id_)
        self._len -= 1
        return it

    # -------- allocation --------

    def alloc_id(self) -> int:
        """
        Get a reusable dense id. Uses free-list else appends at the tail.
        Build your Item with this id then call add(item).
        """
        return self._free.pop() if self._free else len(self._arr)

    # -------- fast batch gathers --------

    def get_many_by_ids(self, ids: Sequence[int], *, chunk: int = 2048) -> list[TItem]:
        """
        Batch: return Items for ids, preserving order.
        Uses C-level itemgetter on the dense array in chunks.
        """
        out: list[TItem] = []
        extend = out.extend
        arr = self._arr
        for i in range(0, len(ids), chunk):
            block = ids[i : i + chunk]
            vals = itemgetter(*block)(arr)  # tuple or single item
            extend(vals if isinstance(vals, tuple) else (vals,))
        return out

    def get_many_objects(self, ids: Sequence[int], *, chunk: int = 2048) -> list[Any]:
        """
        Batch: return Python objects for ids, preserving order.
        Mirrors get_many_by_ids but reads from _objs.
        """

        out: list[Any] = []
        extend = out.extend
        objs = self._objs
        for i in range(0, len(ids), chunk):
            block = ids[i : i + chunk]
            vals = itemgetter(*block)(objs)  # tuple or single object
            extend(vals if isinstance(vals, tuple) else (vals,))
        return out

    # -------- convenience and iteration --------

    def __len__(self) -> int:
        return self._len

    def clear(self) -> None:
        self._arr.clear()
        self._objs.clear()
        self._obj_to_ids.clear()
        self._free.clear()
        self._len = 0

    def contains_id(self, id_: int) -> bool:
        return 0 <= id_ < len(self._arr) and self._arr[id_] is not None

    def contains_obj(self, obj: Any) -> bool:
        return id(obj) in self._obj_to_ids

    def items_by_id(self) -> Iterator[tuple[int, TItem]]:
        for i, it in enumerate(self._arr):
            if it is not None:
                yield i, it

    def items(self) -> Iterator[TItem]:
        for _, it in self.items_by_id():
            yield it
