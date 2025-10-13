"""
Python Shim to mimic the interface of pyqtree and allow for a
drop-in replacement to fastquadtree.
"""

from __future__ import annotations

from typing import Any, Tuple

from .rect_quadtree import RectQuadTree

Point = Tuple[float, float]  # only for type hints in docstrings

# Default parameters from pyqtree
MAX_ITEMS = 10
MAX_DEPTH = 20


class Index:
    """
    The class below is taken from the pyqtree package, but the implementation
    has been modified to use the fastquadtree package as a backend instead of
    the original pure-python implementation.


    The top spatial index to be created by the user. Once created it can be
    populated with geographically placed members that can later be tested for
    intersection with a user inputted geographic bounding box. Note that the
    index can be iterated through in a for-statement, which loops through all
    all the quad instances and lets you access their properties.

    Example usage:

    >>> spindex = Index(bbox=(0, 0, 100, 100))
    >>> spindex.insert('duck', (50, 30, 53, 60))
    >>> spindex.insert('cookie', (10, 20, 15, 25))
    >>> spindex.insert('python', (40, 50, 95, 90))
    >>> results = spindex.intersect((51, 51, 86, 86))
    >>> sorted(results)
    ['duck', 'python']
    """

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
            self.rect_quadtree = RectQuadTree(
                (x1, y1, x2, y2), max_items, max_depth=max_depth, track_objects=True
            )

        elif (
            x is not None and y is not None and width is not None and height is not None
        ):
            self.rect_quadtree = RectQuadTree(
                (x - width / 2, y - height / 2, x + width / 2, y + height / 2),
                max_items,
                max_depth=max_depth,
                track_objects=True,
            )

        else:
            raise ValueError(
                "Either the bbox argument must be set, or the x, y, width, and height arguments must be set"
            )

    def insert(self, item: Any, bbox):  # pyright: ignore[reportIncompatibleMethodOverride]
        """
        Inserts an item into the quadtree along with its bounding box.

        Parameters:
        - **item**: The item to insert into the index, which will be returned by the intersection method
        - **bbox**: The spatial bounding box tuple of the item, with four members (xmin,ymin,xmax,ymax)
        """
        self.rect_quadtree.insert(geom=bbox, obj=item)

    def remove(self, item, bbox):
        """
        Removes an item from the quadtree.

        Parameters:
        - **item**: The item to remove from the index
        - **bbox**: The spatial bounding box tuple of the item, with four members (xmin,ymin,xmax,ymax)

        Both parameters need to exactly match the parameters provided to the insert method.
        """
        self.rect_quadtree.delete_by_object(obj=item)

    def intersect(self, bbox):
        """
        Intersects an input boundingbox rectangle with all of the items
        contained in the quadtree.

        Parameters:
        - **bbox**: A spatial bounding box tuple with four members (xmin,ymin,xmax,ymax)

        Returns:
        - A list of inserted items whose bounding boxes intersect with the input bbox.
        """
        out = list(self.rect_quadtree.query(bbox, as_items=True))
        return [item.obj for item in out]
