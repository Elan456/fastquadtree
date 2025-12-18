"""fastquadtree - High-performance spatial indexing for Python."""

from ._insert_result import InsertResult
from ._item import Item, PointItem, RectItem
from .point_quadtree import QuadTree
from .point_quadtree_objects import QuadTreeObjects
from .rect_quadtree import RectQuadTree
from .rect_quadtree_objects import RectQuadTreeObjects

__all__ = [
    "InsertResult",
    "Item",
    "PointItem",
    "QuadTree",
    "QuadTreeObjects",
    "RectItem",
    "RectQuadTree",
    "RectQuadTreeObjects",
]
