"""Item wrappers for quadtree query results and object tracking."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from ._common import Bounds, Point

G = TypeVar("G", Point, Bounds)


class Item(Generic[G]):
    """
    Generic container for quadtree index entries.

    This class provides a lightweight wrapper around spatial index entries,
    containing an ID, geometry, and optional associated Python object.

    Attributes:
        id_: Integer identifier for this entry.
        geom: Geometry data (Point or Bounds depending on tree type).
        obj: Associated Python object, or None if not set.
    """

    __slots__ = ("geom", "id_", "obj")

    def __init__(self, id_: int, geom: G, obj: Any = None):
        self.id_: int = id_
        self.geom: G = geom
        self.obj: Any = obj

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the item to a dictionary.

        Returns:
            Dictionary with 'id', 'geom', and 'obj' keys.
        """
        return {
            "id": self.id_,
            "geom": self.geom,
            "obj": self.obj,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Item:
        """
        Deserialize an item from a dictionary.

        Args:
            data: Dictionary with 'id', 'geom', and 'obj' keys.

        Returns:
            Item instance populated with the deserialized data.
        """
        id_ = data["id"]
        geom = data["geom"]
        obj = data["obj"]
        return cls(id_, geom, obj)


class PointItem(Item[Point]):
    """
    Specialized item container for point geometries.

    This subclass of Item adds convenient x and y attributes for direct
    access to point coordinates.

    Attributes:
        id_: Integer identifier.
        geom: Point geometry as (x, y) tuple.
        obj: Associated Python object, or None if not set.
        x: X coordinate (convenience accessor).
        y: Y coordinate (convenience accessor).
    """

    __slots__ = ("x", "y")

    def __init__(self, id_: int, geom: Point, obj: Any = None):
        super().__init__(id_, geom, obj)
        self.x, self.y = geom


class RectItem(Item[Bounds]):
    """
    Specialized item container for rectangle geometries.

    This subclass of Item adds convenient min_x, min_y, max_x, and max_y
    attributes for direct access to rectangle bounds.

    Attributes:
        id_: Integer identifier.
        geom: Rectangle geometry as (min_x, min_y, max_x, max_y) tuple.
        obj: Associated Python object, or None if not set.
        min_x: Minimum X coordinate (convenience accessor).
        min_y: Minimum Y coordinate (convenience accessor).
        max_x: Maximum X coordinate (convenience accessor).
        max_y: Maximum Y coordinate (convenience accessor).
    """

    __slots__ = ("max_x", "max_y", "min_x", "min_y")

    def __init__(self, id_: int, geom: Bounds, obj: Any = None):
        super().__init__(id_, geom, obj)
        self.min_x, self.min_y, self.max_x, self.max_y = geom
