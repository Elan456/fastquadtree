# _bimap.py
from __future__ import annotations

from operator import itemgetter
from typing import Any, Generic, Iterable, Iterator, TypeVar

from ._item import Item  # base class for PointItem and RectItem

TItem = TypeVar("TItem", bound=Item)


class ObjStore(Generic[TItem]):
    """
    High-performance id <-> object store for dense, auto-assigned ids.

    Storage
      - _arr[id]  -> Item or None
      - _objs[id] -> Python object or None
      - _obj_to_id: reverse identity map id(obj) -> id
      - _free: LIFO free-list of reusable ids

    Assumptions
      - Ids are assigned by the shim and are dense [0..len) with possible holes
        created by deletes. New inserts reuse holes via the free-list.
    """

    __slots__ = ("_arr", "_free", "_len", "_obj_to_id", "_objs")

    def __init__(self, items: Iterable[TItem] | None = None) -> None:
        self._arr: list[TItem | None] = []
        self._objs: list[Any | None] = []
        self._obj_to_id: dict[int, int] = {}
        self._free: list[int] = []  # LIFO
        self._len: int = 0  # live items

        if items:
            for it in items:
                self.add(it)

    # -------- core --------

    def add(self, item: TItem) -> None:
        """
        Insert or replace the mapping at item.id_. Reverse map updated so obj points to id.
        """
        id_ = item.id_
        obj = item.obj

        # ids must be dense and assigned by the caller
        if id_ > len(self._arr):
            raise AssertionError(
                "ObjStore.add received an out-of-order id, use alloc_id() to get the next available id"
            )

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
                self._obj_to_id.pop(id(old.obj), None)
            self._arr[id_] = item
            self._objs[id_] = obj

        if obj is not None:
            self._obj_to_id[id(obj)] = id_

    def by_id(self, id_: int) -> TItem | None:
        return self._arr[id_] if 0 <= id_ < len(self._arr) else None

    def by_obj(self, obj: Any) -> TItem | None:
        id_ = self._obj_to_id.get(id(obj))
        return self.by_id(id_) if id_ is not None else None

    def pop_id(self, id_: int) -> TItem | None:
        """Remove by id. Dense ids go to the free-list for reuse."""
        if not (0 <= id_ < len(self._arr)):
            return None
        it = self._arr[id_]
        if it is None:
            return None
        self._arr[id_] = None
        self._objs[id_] = None
        if it.obj is not None:
            self._obj_to_id.pop(id(it.obj), None)
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

    def get_many_by_ids(self, ids: Iterable[int], *, chunk: int = 2048) -> list[TItem]:
        """
        Batch: return Items for ids, preserving order.
        Uses C-level itemgetter on the dense array in chunks.
        """
        ids_list: list[int] = list(ids)
        if not ids_list:
            return []

        out: list[TItem] = []
        extend = out.extend
        arr = self._arr
        for i in range(0, len(ids_list), chunk):
            block = ids_list[i : i + chunk]
            vals = itemgetter(*block)(arr)  # tuple or single item
            extend(vals if isinstance(vals, tuple) else (vals,))
        return out

    def get_many_objects(self, ids: Iterable[int], *, chunk: int = 2048) -> list[Any]:
        """
        Batch: return Python objects for ids, preserving order.
        Mirrors get_many_by_ids but reads from _objs.
        """
        ids_list: list[int] = list(ids)
        if not ids_list:
            return []

        out: list[Any] = []
        extend = out.extend
        objs = self._objs
        for i in range(0, len(ids_list), chunk):
            block = ids_list[i : i + chunk]
            vals = itemgetter(*block)(objs)  # tuple or single object
            extend(vals if isinstance(vals, tuple) else (vals,))
        return out

    # -------- convenience and iteration --------

    def __len__(self) -> int:
        return self._len

    def clear(self) -> None:
        self._arr.clear()
        self._objs.clear()
        self._obj_to_id.clear()
        self._free.clear()
        self._len = 0

    def contains_id(self, id_: int) -> bool:
        return 0 <= id_ < len(self._arr) and self._arr[id_] is not None

    def contains_obj(self, obj: Any) -> bool:
        return id(obj) in self._obj_to_id

    def items_by_id(self) -> Iterator[tuple[int, TItem]]:
        for i, it in enumerate(self._arr):
            if it is not None:
                yield i, it

    def items(self) -> Iterator[TItem]:
        for _, it in self.items_by_id():
            yield it
