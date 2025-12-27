"""
fastquadtree - High-performance spatial indexing for Python.

This package provides fast quadtree-based spatial indexing with multiple variants
to suit different use cases:

- `QuadTree`: Point quadtree without object association
- `QuadTreeObjects`: Point quadtree with Python object association
- `RectQuadTree`: Rectangle quadtree without object association
- `RectQuadTreeObjects`: Rectangle quadtree with Python object association

All implementations are backed by a high-performance Rust core for optimal speed.
"""

from ._insert_result import InsertResult
from ._item import Item, PointItem, RectItem
from .point_quadtree import QuadTree
from .point_quadtree_objects import QuadTreeObjects
from .rect_quadtree import RectQuadTree
from .rect_quadtree_objects import RectQuadTreeObjects

# Allow lowercase version of quadtree for convenience
Quadtree = QuadTree
QuadtreeObjects = QuadTreeObjects
Rectquadtree = RectQuadTree
RectquadtreeObjects = RectQuadTreeObjects

__all__ = [
    "InsertResult",
    "Item",
    "PointItem",
    "QuadTree",
    "QuadTreeObjects",
    "Quadtree",
    "QuadtreeObjects",
    "RectItem",
    "RectQuadTree",
    "RectQuadTreeObjects",
    "Rectquadtree",
    "RectquadtreeObjects",
]
