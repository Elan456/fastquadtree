"""
Python Shim to mimic the interface of pyqtree and allow for a
drop-in replacement to fastquadtree.
"""

from __future__ import annotations

from operator import itemgetter
from typing import Any, Tuple

from ._native import RectQuadTree

Point = Tuple[float, float]  # only for type hints in docstrings

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
    The class below is taken from the pyqtree package, but the implementation
    has been modified to use the fastquadtree package as a backend instead of
    the original pure-python implementation.
    Based on the benchmarks, this gives a overall performance boost of 6.514x.
    See the benchmark section of the docs for more details and the latest numbers.


    The top spatial index to be created by the user. Once created it can be
    populated with geographically placed members that can later be tested for
    intersection with a user inputted geographic bounding box. Note that the
    index can be iterated through in a for-statement, which loops through all
    all the quad instances and lets you access their properties.

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
        bbox=None,
        x=None,
        y=None,
        width=None,
        height=None,
        max_items=MAX_ITEMS,
        max_depth=MAX_DEPTH,
    ):
        """
        Initiate by specifying either 1) a bbox to keep track of, or 2) with an xy centerpoint and a width and height.

        Parameters:
        - **bbox**: The coordinate system bounding box of the area that the quadtree should
            keep track of, as a 4-length sequence (xmin,ymin,xmax,ymax)
        - **x**:
            The x center coordinate of the area that the quadtree should keep track of.
        - **y**
            The y center coordinate of the area that the quadtree should keep track of.
        - **width**:
            How far from the xcenter that the quadtree should look when keeping track.
        - **height**:
            How far from the ycenter that the quadtree should look when keeping track
        - **max_items** (optional): The maximum number of items allowed per quad before splitting
            up into four new subquads. Default is 10.
        - **max_depth** (optional): The maximum levels of nested subquads, after which no more splitting
            occurs and the bottommost quad nodes may grow indefinately. Default is 20.
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

    def insert(self, item: Any, bbox):  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Inserts an item into the quadtree along with its bounding box.

        Parameters:
        - **item**: The item to insert into the index, which will be returned by the intersection method
        - **bbox**: The spatial bounding box tuple of the item, with four members (xmin,ymin,xmax,ymax)
        """
        if self._free:
            rid = self._free.pop()
            self._objects[rid] = item
        else:
            rid = len(self._objects)
            self._objects.append(item)
        self._qt.insert(rid, bbox)
        self._item_to_id[id(item)] = rid

    def remove(self, item, bbox):
        """
        Removes an item from the quadtree.

        Parameters:
        - **item**: The item to remove from the index
        - **bbox**: The spatial bounding box tuple of the item, with four members (xmin,ymin,xmax,ymax)

        Both parameters need to exactly match the parameters provided to the insert method.
        """
        rid = self._item_to_id.pop(id(item))
        self._qt.delete(rid, bbox)
        self._objects[rid] = None
        self._free.append(rid)

    def intersect(self, bbox):
        """
        Intersects an input boundingbox rectangle with all of the items
        contained in the quadtree.

        Parameters:
        - **bbox**: A spatial bounding box tuple with four members (xmin,ymin,xmax,ymax)

        Returns:
        - A list of inserted items whose bounding boxes intersect with the input bbox.
        """
        result = self._qt.query_ids(bbox)
        # result = [id1, id2, ...]
        return gather_objs(self._objects, result)
