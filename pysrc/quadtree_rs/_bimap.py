# _bimap.py
from __future__ import annotations
from typing import Any, Iterable, Iterator, Tuple

class BiMap:
    """
    Strong, one-to-one bidirectional map:
      id (int) -> object
      object (by identity) -> id

    Notes:
      - Objects are held strongly, so they will not be GC'd while stored here.
      - Identity is used for object keys, so unhashable or mutable objects work.
      - Setting a pair removes any previous associations that would violate 1-1.
    """

    __slots__ = ("_id_to_obj", "_objid_to_id")

    def __init__(self, pairs: Iterable[Tuple[int, Any]] | None = None) -> None:
        self._id_to_obj: dict[int, Any] = {}
        self._objid_to_id: dict[int, int] = {}
        if pairs:
            for k, v in pairs:
                self.set(k, v)

    # ---- core API ----

    def set(self, id_: int, obj: Any) -> None:
        """Associate id_ <-> obj, replacing any conflicting mappings."""
        # If id_ already mapped, unlink old object
        old_obj = self._id_to_obj.get(id_)
        if old_obj is not None and old_obj is not obj:
            self._objid_to_id.pop(id(old_obj), None)

        # If obj already mapped to a different id, unlink that id
        oid = id(obj)
        old_id = self._objid_to_id.get(oid)
        if old_id is not None and old_id != id_:
            self._id_to_obj.pop(old_id, None)

        self._id_to_obj[id_] = obj
        self._objid_to_id[oid] = id_

    def by_id(self, id_: int) -> Any | None:
        return self._id_to_obj.get(id_)

    def by_obj(self, obj: Any) -> int | None:
        return self._objid_to_id.get(id(obj))

    def pop_id(self, id_: int) -> Any | None:
        obj = self._id_to_obj.pop(id_, None)
        if obj is not None:
            self._objid_to_id.pop(id(obj), None)
        return obj

    def pop_obj(self, obj: Any) -> int | None:
        oid = id(obj)
        id_ = self._objid_to_id.pop(oid, None)
        if id_ is not None:
            self._id_to_obj.pop(id_, None)
        return id_

    # ---- convenience ----

    def __len__(self) -> int:
        return len(self._id_to_obj)

    def clear(self) -> None:
        self._id_to_obj.clear()
        self._objid_to_id.clear()

    def contains_id(self, id_: int) -> bool:
        return id_ in self._id_to_obj

    def contains_obj(self, obj: Any) -> bool:
        return id(obj) in self._objid_to_id

    def keys(self) -> Iterator[int]:
        return iter(self._id_to_obj.keys())

    def values(self) -> Iterator[Any]:
        return iter(self._id_to_obj.values())

    def items(self) -> Iterator[Tuple[int, Any]]:
        return iter(self._id_to_obj.items())
