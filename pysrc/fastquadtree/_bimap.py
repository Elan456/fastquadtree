# _bimap.py
from __future__ import annotations

from typing import Any, Iterable, Iterator

from ._item import Item


class BiMap:
    """
    Bidirectional map to the same Item:
      id -> Item
      obj -> Item  (uses object identity)

    Rules:
      - One-to-one: an id maps to exactly one Item, and an object maps to exactly one Item.
      - add(item): inserts or replaces both sides so they point to 'item'.
      - If item.obj is None, only id -> Item is stored.
    """

    __slots__ = ("_id_to_item", "_objid_to_item")

    def __init__(self, items: Iterable[Item] | None = None) -> None:
        self._id_to_item: dict[int, Item] = {}
        self._objid_to_item: dict[int, Item] = {}
        if items:
            for it in items:
                self.add(it)

    # - core -

    def add(self, item: Item) -> None:
        """
        Insert or replace mapping for this Item.
        Handles conflicts so that both id and obj point to this exact Item.
        """
        id_ = item.id_
        obj = item.obj

        # Unlink any old item currently bound to this id
        old = self._id_to_item.get(id_)
        if old is not None and old is not item:
            old_obj = old.obj
            if old_obj is not None:
                self._objid_to_item.pop(id(old_obj), None)

        # Unlink any old item currently bound to this obj
        if obj is not None:
            prev = self._objid_to_item.get(id(obj))
            if prev is not None and prev is not item:
                self._id_to_item.pop(prev.id_, None)

        # Link new
        self._id_to_item[id_] = item
        if obj is not None:
            self._objid_to_item[id(obj)] = item

    def by_id(self, id_: int) -> Item | None:
        return self._id_to_item.get(id_)

    def by_obj(self, obj: Any) -> Item | None:
        return self._objid_to_item.get(id(obj))

    def pop_id(self, id_: int) -> Item | None:
        it = self._id_to_item.pop(id_, None)
        if it is not None:
            obj = it.obj
            if obj is not None:
                self._objid_to_item.pop(id(obj), None)
        return it

    def pop_obj(self, obj: Any) -> Item | None:
        it = self._objid_to_item.pop(id(obj), None)
        if it is not None:
            self._id_to_item.pop(it.id_, None)
        return it

    def pop_item(self, item: Item) -> Item | None:
        """
        Remove this exact Item if present on either side.
        """
        removed = None
        # Remove by id first
        removed = self._id_to_item.pop(item.id_)

        # Remove by obj side
        obj = item.obj
        if obj is not None:
            self._objid_to_item.pop(id(obj), None)
            removed = removed or item
        return removed

    # - convenience -

    def __len__(self) -> int:
        return len(self._id_to_item)

    def clear(self) -> None:
        self._id_to_item.clear()
        self._objid_to_item.clear()

    def contains_id(self, id_: int) -> bool:
        return id_ in self._id_to_item

    def contains_obj(self, obj: Any) -> bool:
        return id(obj) in self._objid_to_item

    def items_by_id(self) -> Iterator[tuple[int, Item]]:
        return iter(self._id_to_item.items())

    def items(self) -> Iterator[Item]:
        return iter(self._id_to_item.values())
