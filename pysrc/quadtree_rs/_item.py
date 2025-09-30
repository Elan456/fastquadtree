# item.py
from __future__ import annotations
from typing import Any, Callable, Optional


class Item:
    """
    Lightweight view of an index entry.

    Attributes:
        id: Integer identifier.
        x: X coordinate.
        y: Y coordinate.
        obj: The attached Python object if available, else None.

    Notes:
        - Holds a strong reference to the object when provided.
        - If you want lazy lookup from an external store, you can pass get_obj.
    """

    __slots__ = ("id", "x", "y", "_obj")

    def __init__(self, id: int, x: float, y: float, obj: Any | None = None):
        self.id = id
        self.x = x
        self.y = y
        self._obj = obj

    @property
    def obj(self) -> Any | None:
        if self._obj is not None:
            return self._obj
        return None
