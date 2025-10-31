"""
Python Shim to mimic the interface of pyqtree and allow for a
drop-in replacement to fastquadtree.
"""

from __future__ import annotations

from collections.abc import Iterable
from operator import itemgetter
from typing import Any, SupportsFloat, Tuple

from ._native import RectQuadTree

Point = Tuple[SupportsFloat, SupportsFloat]  # only for type hints in docstrings

# Default parameters from pyqtree
MAX_ITEMS = 10
MAX_DEPTH = 20


# Helper to gather objects by ids in chunks
# Performance improvement over list comprehension for large result sets
# 2.945 median query time --> 2.030 median query time (500k items, 500 queries)
def gather_objs(objs, ids, chunk=2048):
    out = []
    for i in range(0, len(ids), chunk):
        getter = itemgetter(*ids[i : i + chunk])
        vals = getter(objs)  # tuple or single object
        if isinstance(vals, tuple):
            out.extend(vals)
        else:
            out.append(vals)
    return out


class Index:
    """
    The interface of the class below is taken from the pyqtree package, but the implementation
    has been modified to use the fastquadtree package as a backend instead of
    the original pure-python implementation.
    Based on the benchmarks, this gives a overall performance boost of 6.514x.
    See the benchmark section of the docs for more details and the latest numbers.

    Index is  the top-level class for creating and using a quadtree spatial index
    with the original pyqtree interface. If you are not migrating from pyqtree,
    consider using the RectQuadTree class for detailed control and better performance.

    This class wraps a RectQuadTree instance and provides methods to insert items with bounding boxes,
    remove items, and query for items intersecting a given bounding box.

    Example usage:
    ```python
    from fastquadtree.pyqtree import Index


    spindex = Index(bbox=(0, 0, 100, 100))
    spindex.insert('duck', (50, 30, 53, 60))
    spindex.insert('cookie', (10, 20, 15, 25))
    spindex.insert('python', (40, 50, 95, 90))
    results = spindex.intersect((51, 51, 86, 86))
    sorted(results) # ['duck', 'python']
    ```
    """

    __slots__ = ("_free", "_item_to_id", "_objects", "_qt")

    def __init__(
        self,
        bbox: Iterable[SupportsFloat] | None = None,
        x: float | int | None = None,
        y: float | int | None = None,
        width: float | int | None = None,
        height: float | int | None = None,
        max_items: int = MAX_ITEMS,
        max_depth: int = MAX_DEPTH,
    ):
        """
        Initiate by specifying either 1) a bbox to keep track of, or 2) with an xy centerpoint and a width and height.

        Args:
          bbox: The coordinate system bounding box of the area that the quadtree should
            keep track of, as a 4-length sequence (xmin,ymin,xmax,ymax)
          x:
            The x center coordinate of the area that the quadtree should keep track of.
          y:
            The y center coordinate of the area that the quadtree should keep track of.
          width:
            How far from the xcenter that the quadtree should look when keeping track.
          height:
            How far from the ycenter that the quadtree should look when keeping track
          max_items (optional): The maximum number of items allowed per quad before splitting
              up into four new subquads. Default is 10.
          max_depth (optional): The maximum levels of nested subquads, after which no more splitting
            occurs and the bottommost quad nodes may grow indefinately. Default is 20.

        Note:
            Either the bbox argument must be set, or the x, y, width, and height
            arguments must be set.
        """
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            self._qt = RectQuadTree((x1, y1, x2, y2), max_items, max_depth=max_depth)

        elif (
            x is not None and y is not None and width is not None and height is not None
        ):
            self._qt = RectQuadTree(
                (x - width / 2, y - height / 2, x + width / 2, y + height / 2),
                max_items,
                max_depth=max_depth,
            )

        else:
            raise ValueError(
                "Either the bbox argument must be set, or the x, y, width, and height arguments must be set"
            )

        self._objects = []
        self._free = []
        self._item_to_id = {}

    def insert(self, item: Any, bbox: Iterable[SupportsFloat]):
        """
        Inserts an item into the quadtree along with its bounding box.

        Args:
          item: The item to insert into the index, which will be returned by the intersection method
          bbox: The spatial bounding box tuple of the item, with four members (xmin,ymin,xmax,ymax)
        """
        if type(bbox) is not tuple:  # Handle non-tuple input
            bbox = tuple(bbox)

        if self._free:
            rid = self._free.pop()
            self._objects[rid] = item
        else:
            rid = len(self._objects)
            self._objects.append(item)
        self._qt.insert(rid, bbox)
        self._item_to_id[id(item)] = rid

    def remove(self, item: Any, bbox: Iterable[SupportsFloat]):
        """
        Removes an item from the quadtree.

        Args:
          item: The item to remove from the index
          bbox: The spatial bounding box tuple of the item, with four members (xmin,ymin,xmax,ymax)

        Note:
            Both parameters need to exactly match the parameters provided to the insert method.
        """
        if type(bbox) is not tuple:  # Handle non-tuple input
            bbox = tuple(bbox)

        rid = self._item_to_id.pop(id(item))
        self._qt.delete(rid, bbox)
        self._objects[rid] = None
        self._free.append(rid)

    def intersect(self, bbox: Iterable[SupportsFloat]) -> list:
        """
        Intersects an input bounding box rectangle with all of the items
        contained in the quadtree.

        Args:
          bbox: A spatial bounding box tuple with four members (xmin,ymin,xmax,ymax)

        Returns:
          A list of inserted items whose bounding boxes intersect with the input bbox.
        """
        if type(bbox) is not tuple:  # Handle non-tuple input
            bbox = tuple(bbox)
        result = self._qt.query_ids(bbox)
        # result = [id1, id2, ...]
        return gather_objs(self._objects, result)
