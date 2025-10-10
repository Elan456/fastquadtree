# item.py
from __future__ import annotations

from typing import Any, Tuple

Bounds = Tuple[float, float, float, float]
"""Axis-aligned rectangle as (min_x, min_y, max_x, max_y)."""

Point = Tuple[float, float]
"""2D point as (x, y)."""


class Item:
    """
    Lightweight view of an index entry.

    Attributes:
        id_: Integer identifier.
        geom: The geometry, either a Point or Rectangle Bounds.
        obj: The attached Python object if available, else None.
    """

    __slots__ = ("geom", "id_", "obj")

    def __init__(self, id_: int, geom: Point | Bounds, obj: Any | None = None):
        self.id_ = id_
        self.geom = geom
        self.obj = obj


class PointItem(Item):
    """
    Lightweight point item wrapper for tracking and as_items results.
    """

    __slots__ = ("geom", "id_", "obj", "x", "y")

    def __init__(self, id_: int, geom: Point, obj: Any | None = None):
        super().__init__(id_, geom, obj)
        self.x, self.y = geom


class RectItem(Item):
    """
    Lightweight rectangle item wrapper for tracking and as_items results.
    """

    __slots__ = ("geom", "id_", "max_x", "max_y", "min_x", "min_y", "obj")

    def __init__(self, id_: int, geom: Bounds, obj: Any | None = None):
        super().__init__(id_, geom, obj)
        self.min_x, self.min_y, self.max_x, self.max_y = geom
