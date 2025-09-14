from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

# Compiled Rust module lives here thanks to tool.maturin.module-name
from ._native import QuadTree as _RustQuadTree  # type: ignore[attr-defined]

Bounds = Tuple[float, float, float, float]
Point = Tuple[float, float]


class Item:
    """
    Lightweight result wrapper.

    Uses __slots__ and lazy object lookup to keep overhead low.
    Access .obj only if you actually need the Python object.
    """
    __slots__ = ("id", "x", "y", "_map_get")

    def __init__(self, id: int, x: float, y: float, map_get):
        self.id = id
        self.x = x
        self.y = y
        self._map_get = map_get  # either dict.get or None

    @property
    def obj(self) -> Any | None:
        get = self._map_get
        return None if get is None else get(self.id)


class QuadTree:
    """
    High-level Python wrapper over the Rust quadtree.

    - Inserts are by integer id and point
    - Tracking objects is optional. Enable it if you want id -> obj mapping
    - Queries return raw tuples by default for speed
    - Opt into Item wrappers via as_items=True when you need them
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
        """
        Create a quadtree.

        bounds: (min_x, min_y, max_x, max_y)
        capacity: max number of points stored per node before splitting
        max_depth: optional depth cap
        track_objects: store an id -> obj map in Python
        start_id: starting id used when auto-assigning ids
        """
        if max_depth is None:
            self._native = _RustQuadTree(bounds, capacity)
        else:
            self._native = _RustQuadTree(bounds, capacity, max_depth=max_depth)
        self._objects: Optional[dict[int, Any]] = {} if track_objects else None
        self._next_id: int = int(start_id)
        self._count: int = 0
        self._bounds = bounds

    # ---------- inserts ----------

    def insert(self, xy: Point, *, id: Optional[int] = None, obj: Any = None) -> int:
        """
        Insert a point.

        xy: (x, y)
        id: optional integer id. If omitted, an auto id is assigned
        obj: optional Python object to associate with id. Only stored if track_objects=True
        returns: the id used
        """
        if id is None:
            id = self._next_id
            self._next_id += 1
        else:
            # make sure next_id is always ahead of any manually provided ids
            if id >= self._next_id:
                self._next_id = id + 1

        if not self._native.insert(id, xy):
            x, y = xy
            bx0, by0, bx1, by1 = self._bounds
            raise ValueError(f"Point ({x}, {y}) is outside bounds ({bx0}, {by0}, {bx1}, {by1})")

        # Store obj only if mapping is enabled and obj was provided
        if self._objects is not None and obj is not None:
            self._objects[id] = obj

        self._count += 1
        return id

    def insert_many_points(self, points: Iterable[Point]) -> int:
        """
        Bulk insert points with auto-assigned ids.

        points: iterable of (x, y)
        returns: number of points inserted
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
        Attach or replace an object for an existing id.

        If track_objects is False, a map will be created on first use.
        """
        if self._objects is None:
            self._objects = {}
        self._objects[id] = obj

    # ---------- queries ----------

    def query(self, rect: Bounds, *, as_items: bool = False) -> List[Tuple[int, float, float]] | List[Item]:
        """
        Query points inside an axis-aligned rectangle.

        rect: (min_x, min_y, max_x, max_y)
        as_items: return Item wrappers if True, else raw (id, x, y) tuples
        """
        raw = self._native.query(rect)
        if not as_items:
            return raw
        out: List[Item] = []
        ap = out.append
        map_get = self._objects.get if self._objects is not None else None
        Item_ = Item
        for id_, x, y in raw:
            ap(Item_(id_, x, y, map_get))
        return out

    def nearest_neighbor(self, xy: Point, *, as_item: bool = False):
        """
        Nearest neighbor to xy.

        as_item: return an Item if True, else a tuple (id, x, y). Returns None if empty.
        """
        t = self._native.nearest_neighbor(xy)
        if t is None or not as_item:
            return t
        id_, x, y = t
        map_get = self._objects.get if self._objects is not None else None
        return Item(id_, x, y, map_get)

    def nearest_neighbors(self, xy: Point, k: int, *, as_items: bool = False):
        """
        k nearest neighbors to xy.

        as_items: return Item wrappers if True, else raw tuples
        """
        raw = self._native.nearest_neighbors(xy, k)
        if not as_items:
            return raw
        map_get = self._objects.get if self._objects is not None else None
        Item_ = Item
        return [Item_(id_, x, y, map_get) for (id_, x, y) in raw]

    # ---------- misc ----------

    def get(self, id: int) -> Any | None:
        """
        Get the object associated with id. Returns None if not tracked or not present.
        """
        return None if self._objects is None else self._objects.get(id)

    def __len__(self) -> int:
        """
        Number of successful inserts through this wrapper.
        """
        return self._count

    # Power users can access the raw class
    NativeQuadTree = _RustQuadTree


__all__ = ["QuadTree", "Item", "Bounds", "Point"]
