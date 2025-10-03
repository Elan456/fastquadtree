from __future__ import annotations

from typing import Any, Literal, Tuple, overload

from ._bimap import BiMap  # type: ignore[attr-defined]
from ._item import Item

# Compiled Rust module is provided by maturin (tool.maturin.module-name)
from ._native import QuadTree as _RustQuadTree

Bounds = Tuple[float, float, float, float]
"""Axis-aligned rectangle as (min_x, min_y, max_x, max_y)."""

Point = Tuple[float, float]
"""2D point as (x, y)."""

_IdCoord = Tuple[int, float, float]
"""Result tuple as (id, x, y)."""


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

    __slots__ = ("_bounds", "_count", "_items", "_native", "_next_id")

    def __init__(
        self,
        bounds: Bounds,
        capacity: int,
        *,
        max_depth: int | None = None,
        track_objects: bool = False,
        start_id: int = 1,
    ):
        if max_depth is None:
            self._native = _RustQuadTree(bounds, capacity)
        else:
            self._native = _RustQuadTree(bounds, capacity, max_depth=max_depth)
        self._items: BiMap | None = BiMap() if track_objects else None
        self._next_id: int = int(start_id)
        self._count: int = 0
        self._bounds = bounds

    # ---------- inserts ----------

    def insert(self, xy: Point, *, id_: int | None = None, obj: Any = None) -> int:
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
        if id_ is None:
            id_ = self._next_id
            self._next_id += 1
        # ensure future auto-ids do not collide
        elif id_ >= self._next_id:
            self._next_id = id_ + 1

        if not self._native.insert(id_, xy):
            x, y = xy
            bx0, by0, bx1, by1 = self._bounds
            raise ValueError(
                f"Point ({x}, {y}) is outside bounds ({bx0}, {by0}, {bx1}, {by1})"
            )

        if self._items is not None:
            self._items.add(Item(id_, xy[0], xy[1], obj))

        self._count += 1
        return id_

    def insert_many_points(self, points: list[Point]) -> int:
        """
        Bulk insert points with auto-assigned ids.

        Args:
            points: List of (x, y) points.

        Returns:
            The number of points inserted
        """
        start_id = self._next_id
        last_id = self._native.insert_many_points(start_id, points)

        num_inserted = last_id - start_id + 1

        if num_inserted < len(points):
            raise ValueError("One or more points are outside tree bounds")

        self._next_id = last_id + 1

        # Update the item tracker if needed
        if self._items is not None:
            for i, id_ in enumerate(range(start_id, last_id + 1)):
                x, y = points[i]
                self._items.add(Item(id_, x, y, None))

        return num_inserted

    def attach(self, id_: int, obj: Any) -> None:
        """
        Attach or replace the Python object for an existing id.
        Tracking must be enabled.

        Args:
            id_: Target id.
            obj: Object to associate with id.
        """
        if self._items is None:
            raise ValueError("Cannot attach objects when track_objects=False")

        item = self._items.by_id(id_)
        if item is None:
            raise KeyError(f"Id {id_} not found in quadtree")
        self._items.add(Item(id_, item.x, item.y, obj))

    def delete(self, id_: int, xy: Point) -> bool:
        """
        Delete an item by id and exact coordinates.

        Args:
            id_: Integer id to remove.
            xy: Coordinates (x, y) of the item.

        Returns:
            True if the item was found and deleted, else False.
        """
        deleted = self._native.delete(id_, xy)
        if deleted:
            self._count -= 1
            if self._items is not None:
                self._items.pop_id(id_)  # ignore result
        return deleted

    def delete_by_object(self, obj: Any) -> bool:
        """
        Delete an item by Python object.

        Requires object tracking to be enabled. Performs an O(1) reverse
        lookup to get the id, then deletes that entry at the given location.

        Args:
            obj: The tracked Python object to remove.

        Returns:
            True if the item was found and deleted, else False.

        Raises:
            ValueError: If object tracking is disabled.
        """
        if self._items is None:
            raise ValueError(
                "Cannot delete by object when track_objects=False. Use delete(id, xy) instead."
            )

        item = self._items.by_obj(obj)
        if item is None:
            return False

        return self.delete(item.id_, (item.x, item.y))

    # ---------- queries ----------

    @overload
    def query(
        self, rect: Bounds, *, as_items: Literal[False] = ...
    ) -> list[_IdCoord]: ...

    @overload
    def query(self, rect: Bounds, *, as_items: Literal[True]) -> list[Item]: ...

    def query(
        self, rect: Bounds, *, as_items: bool = False
    ) -> list[_IdCoord] | list[Item]:
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

        if self._items is None:
            raise ValueError("Cannot return results as items with track_objects=False")
        out: list[Item] = []
        for id_, _, _ in raw:
            item = self._items.by_id(id_)
            if item is None:
                raise RuntimeError(
                    f"Internal error: id {id_} found in native tree but missing from object tracker. "
                    f"Ensure all inserts/deletes are done via this wrapper."
                )
            out.append(item)
        return out

    @overload
    def nearest_neighbor(
        self, xy: Point, *, as_item: Literal[False] = ...
    ) -> _IdCoord | None: ...

    @overload
    def nearest_neighbor(self, xy: Point, *, as_item: Literal[True]) -> Item | None: ...

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

        if self._items is None:
            raise ValueError("Cannot return result as item with track_objects=False")
        id_, _x, _y = t
        item = self._items.by_id(id_)
        if item is None:
            raise RuntimeError(
                f"Internal error: id {id_} found in native tree but missing from object tracker. "
                f"Ensure all inserts/deletes are done via this wrapper."
            )
        return item

    @overload
    def nearest_neighbors(
        self, xy: Point, k: int, *, as_items: Literal[False] = ...
    ) -> list[_IdCoord]: ...

    @overload
    def nearest_neighbors(
        self, xy: Point, k: int, *, as_items: Literal[True]
    ) -> list[Item]: ...

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
        if self._items is None:
            raise ValueError("Cannot return results as items with track_objects=False")

        out: list[Item] = []
        for id_, _, _ in raw:
            item = self._items.by_id(id_)
            if item is None:
                raise RuntimeError(
                    f"Internal error: id {id_} found in native tree but missing from object tracker. "
                    f"Ensure all inserts/deletes are done via this wrapper."
                )
            out.append(item)
        return out

    # ---------- misc ----------

    def get(self, id_: int) -> Any | None:
        """
        Return the object associated with id.

        Returns:
            The tracked object if present and tracking is enabled, else None.
        """
        if self._items is None:
            raise ValueError("Cannot get objects when track_objects=False")
        item = self._items.by_id(id_)
        if item is None:
            return None
        return item.obj

    def get_all_rectangles(self) -> list[Bounds]:
        """
        Return all node rectangles in the current quadtree.

        Returns:
            List of (min_x, min_y, max_x, max_y) for each node in the tree.
        """
        return self._native.get_all_rectangles()

    def get_all_objects(self) -> list[Any]:
        """
        Return all tracked objects.

        Returns:
            List of objects if tracking is enabled, else an empty list.
        """
        if self._items is None:
            raise ValueError("Cannot get objects when track_objects=False")
        return [t.obj for t in self._items.items() if t.obj is not None]

    def get_all_items(self) -> list[Item]:
        """
        Return all tracked items.

        Returns:
            List of Item if tracking is enabled, else an empty list.
        """
        if self._items is None:
            raise ValueError("Cannot get items when track_objects=False")
        return list(self._items.items())

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


__all__ = ["Bounds", "Item", "Point", "QuadTree"]
