# item.py
from __future__ import annotations
from typing import Any


class Item:
    """
    Lightweight view of an index entry.

    Attributes:
        id: Integer identifier.
        x: X coordinate.
        y: Y coordinate.
        obj: The attached Python object if available, else None.
    """

    __slots__ = ("id", "x", "y", "obj")

    def __init__(self, id: int, x: float, y: float, obj: Any | None = None):
        self.id = id
        self.x = x
        self.y = y
        self.obj = obj
