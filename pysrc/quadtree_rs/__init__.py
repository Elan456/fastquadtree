from __future__ import annotations

"""
Python shim for the Rust-backed quadtree.

This module exposes a small, typed API that wraps the compiled Rust engine
for fast spatial indexing and queries in 2D.

Quickstart:
    >>> from quadtree_rs import QuadTree
    >>> qt = QuadTree(bounds=(0.0, 0.0, 1000.0, 1000.0), capacity=8, max_depth=12, track_objects=True)
    >>> qt.insert((10.0, 20.0), obj={"name": "A"})
    1
    >>> qt.insert_many_points([(100.0, 100.0), (250.0, 250.0)])
    2
    >>> qt.query((0.0, 0.0, 150.0, 150.0))
    [(1, 10.0, 20.0), (2, 100.0, 100.0)]
    >>> item = qt.nearest_neighbor((12.0, 18.0), as_item=True)
    >>> item.id, item.x, item.y, item.obj
    (1, 10.0, 20.0, {'name': 'A'})

Notes:
- Coordinates are floats
- IDs are integers that you may provide or let the wrapper auto-assign
- Object tracking is optional; enable it when you want id <-> object mapping
"""

from typing import Any, Iterable, List, Optional, Tuple, overload
from typing import Literal

# Compiled Rust module is provided by maturin (tool.maturin.module-name)
from ._native import QuadTree as _RustQuadTree
from ._bimap import BiMap  # type: ignore[attr-defined]

Bounds = Tuple[float, float, float, float]
"""Axis-aligned rectangle as (min_x, min_y, max_x, max_y)."""

Point = Tuple[float, float]
"""2D point as (x, y)."""

_IdCoord = Tuple[int, float, float]
"""Result tuple as (id, x, y)."""


class Item:
    """
    Lightweight view of a result row.

    Provides direct access to id, x, y and lazy lookup of the attached Python
    object if object tracking is enabled on the owning QuadTree.

    Attributes:
        id: Integer identifier for the item.
        x: X coordinate.
        y: Y coordinate.
        obj: The attached Python object if available, else None.

    Notes:
        Uses __slots__ and defers object lookup to keep overhead low.
        Access .obj only when needed.
    """

    __slots__ = ("id", "x", "y", "_map_get")

    def __init__(self, id: int, x: float, y: float, map_get):
        self.id = id
        self.x = x
        self.y = y
        self._map_get = map_get  # either BiMap.by_id or None

    @property
    def obj(self) -> Any | None:
        """Return the attached object for this id or None if not tracked."""
        get = self._map_get
        return None if get is None else get(self.id)


class QuadTree:
    """
    High-level Python wrapper over the Rust quadtree engine.

    The quadtree stores points with integer IDs. You may attach an arbitrary
    Python object per ID when object tracking is enabled.

    Performance characteristics:
        Inserts: average O(log n)
        Rect queries: average O(log n + k) where k is matches returned
        Nearest neighbor: average O(log n)

    Thread-safety:
        Instances are not thread-safe. Use external synchronization if you
        mutate the same tree from multiple threads.

    Args:
        bounds: World bounds as (min_x, min_y, max_x, max_y).
        capacity: Max number of points per node before splitting.
        max_depth: Optional max tree depth. If omitted, engine decides.
        track_objects: Enable id <-> object mapping inside Python.
        start_id: Starting auto-assigned id when you omit id on insert.

    Raises:
        ValueError: If parameters are invalid or inserts are out of bounds.
    """

    __slots__ = ("_native", "_objects", "_next_id", "_count", "_bounds")

    def __init__(
        self,
        bounds: Bounds,
        capacity: int,
        *,
        max_depth: Optional[int] = None,
        track_objects: bool = False,
        start_id: int = 1,
    ):
        if max_depth is None:
            self._native = _RustQuadTree(bounds, capacity)
        else:
            self._native = _RustQuadTree(bounds, capacity, max_depth=max_depth)
        self._objects: Optional[BiMap] = BiMap() if track_objects else None
        self._next_id: int = int(start_id)
        self._count: int = 0
        self._bounds = bounds

    # ---------- inserts ----------

    def insert(self, xy: Point, *, id: Optional[int] = None, obj: Any = None) -> int:
        """
        Insert a single point.

        Args:
            xy: Point (x, y).
            id: Optional integer id. If None, an auto id is assigned.
            obj: Optional Python object to associate with id. Stored only if
                object tracking is enabled.

        Returns:
            The id used for this insert.

        Raises:
            ValueError: If the point is outside tree bounds.
        """
        if id is None:
            id = self._next_id
            self._next_id += 1
        else:
            # ensure future auto-ids do not collide
            if id >= self._next_id:
                self._next_id = id + 1

        if not self._native.insert(id, xy):
            x, y = xy
            bx0, by0, bx1, by1 = self._bounds
            raise ValueError(f"Point ({x}, {y}) is outside bounds ({bx0}, {by0}, {bx1}, {by1})")

        if self._objects is not None and obj is not None:
            self._objects.set(id, obj)

        self._count += 1
        return id

    def insert_many_points(self, points: Iterable[Point]) -> int:
        """
        Bulk insert points with auto-assigned ids.

        Args:
            points: Iterable of (x, y) points.

        Returns:
            Number of points successfully inserted.

        Raises:
            ValueError: If any point is outside tree bounds.
        """
        ins = self._native.insert
        nid = self._next_id
        inserted = 0
        bx0, by0, bx1, by1 = self._bounds
        for xy in points:
            id_ = nid
            nid += 1
            if not ins(id_, xy):
                x, y = xy
                raise ValueError(f"Point ({x}, {y}) is outside bounds ({bx0}, {by0}, {bx1}, {by1})")
            inserted += 1
        self._next_id = nid
        self._count += inserted
        return inserted

    def attach(self, id: int, obj: Any) -> None:
        """
        Attach or replace the Python object for an existing id.

        If object tracking was disabled at construction time, a BiMap is
        created on first use.

        Args:
            id: Target id.
            obj: Object to associate with id.
        """
        if self._objects is None:
            self._objects = BiMap()
        self._objects.set(id, obj)

    def delete(self, id: int, xy: Point) -> bool:
        """
        Delete an item by id and exact coordinates.

        Args:
            id: Integer id to remove.
            xy: Coordinates (x, y) of the item.

        Returns:
            True if the item was found and deleted, else False.
        """
        deleted = self._native.delete(id, xy)
        if deleted:
            self._count -= 1
            if self._objects is not None and self._objects.contains_id(id):
                self._objects.pop_id(id)
        return deleted

    def delete_by_object(self, obj: Any, xy: Point) -> bool:
        """
        Delete an item by Python object and coordinates.

        Requires object tracking to be enabled. Performs an O(1) reverse
        lookup to get the id, then deletes that entry at the given location.

        Args:
            obj: The tracked Python object to remove.
            xy: Coordinates (x, y) of the item.

        Returns:
            True if the item was found and deleted, else False.

        Raises:
            ValueError: If object tracking is disabled.
        """
        if self._objects is None:
            raise ValueError("Cannot delete by object when track_objects=False. Use delete(id, xy) instead.")

        item_id = self._objects.by_obj(obj)
        if item_id is None:
            return False

        return self.delete(item_id, xy)

    # ---------- queries ----------

    @overload
    def query(self, rect: Bounds, *, as_items: Literal[False] = ...) -> List[_IdCoord]:
        ...

    @overload
    def query(self, rect: Bounds, *, as_items: Literal[True]) -> List[Item]:
        ...

    def query(self, rect: Bounds, *, as_items: bool = False) -> List[_IdCoord] | List[Item]:
        """
        Return all points inside an axis-aligned rectangle.

        Args:
            rect: Query rectangle as (min_x, min_y, max_x, max_y).
            as_items: If True, return Item wrappers. If False, return raw tuples.

        Returns:
            If as_items is False: list of (id, x, y) tuples.
            If as_items is True: list of Item objects.
        """
        raw = self._native.query(rect)
        if not as_items:
            return raw
        out: List[Item] = []
        ap = out.append
        map_get = self._objects.by_id if self._objects is not None else None
        Item_ = Item
        for id_, x, y in raw:
            ap(Item_(id_, x, y, map_get))
        return out

    @overload
    def nearest_neighbor(self, xy: Point, *, as_item: Literal[False] = ...) -> Optional[_IdCoord]:
        ...

    @overload
    def nearest_neighbor(self, xy: Point, *, as_item: Literal[True]) -> Optional[Item]:
        ...

    def nearest_neighbor(self, xy: Point, *, as_item: bool = False):
        """
        Return the single nearest neighbor to the query point.

        Args:
            xy: Query point (x, y).
            as_item: If True, return Item. If False, return (id, x, y).

        Returns:
            The nearest neighbor or None if the tree is empty.
        """
        t = self._native.nearest_neighbor(xy)
        if t is None or not as_item:
            return t
        id_, x, y = t
        map_get = self._objects.by_id if self._objects is not None else None
        return Item(id_, x, y, map_get)

    @overload
    def nearest_neighbors(self, xy: Point, k: int, *, as_items: Literal[False] = ...) -> List[_IdCoord]:
        ...

    @overload
    def nearest_neighbors(self, xy: Point, k: int, *, as_items: Literal[True]) -> List[Item]:
        ...

    def nearest_neighbors(self, xy: Point, k: int, *, as_items: bool = False):
        """
        Return the k nearest neighbors to the query point.

        Args:
            xy: Query point (x, y).
            k: Number of neighbors to return.
            as_items: If True, return Item wrappers. If False, return raw tuples.

        Returns:
            List of results in ascending distance order.
        """
        raw = self._native.nearest_neighbors(xy, k)
        if not as_items:
            return raw
        map_get = self._objects.by_id if self._objects is not None else None
        Item_ = Item
        return [Item_(id_, x, y, map_get) for (id_, x, y) in raw]

    # ---------- misc ----------

    def get(self, id: int) -> Any | None:
        """
        Return the object associated with id.

        Returns:
            The tracked object if present and tracking is enabled, else None.
        """
        return None if self._objects is None else self._objects.by_id(id)

    def get_all_rectangles(self) -> List[Bounds]:
        """
        Return all node rectangles in the current quadtree.

        Returns:
            List of (min_x, min_y, max_x, max_y) for each node in the tree.
        """
        return self._native.get_all_rectangles()

    def get_all_objects(self) -> List[Any]:
        """
        Return all tracked objects.

        Returns:
            List of objects if tracking is enabled, else an empty list.
        """
        if self._objects is None:
            return []
        return list(self._objects.values())

    def count_items(self) -> int:
        """
        Return the number of items stored in the native tree.

        Notes:
            This calls the native engine and may differ from len(self) if
            you create multiple wrappers around the same native structure.
        """
        return self._native.count_items()

    def __len__(self) -> int:
        """
        Return the number of successful inserts done via this wrapper.

        Notes:
            This is the Python-side counter that tracks calls that returned True.
            use count_items() to get the authoritative native-side count.
        """
        return self._count

    # Power users can access the raw class
    NativeQuadTree = _RustQuadTree


__all__ = ["QuadTree", "Item", "Bounds", "Point"]
