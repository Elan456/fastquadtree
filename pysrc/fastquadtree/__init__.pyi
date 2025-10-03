from typing import (
    Any,
    Iterable,
    Literal as _Literal,  # avoid polluting public namespace
    overload,
)

Bounds = tuple[float, float, float, float]
Point = tuple[float, float]

class Item:
    id_: int
    x: float
    y: float
    obj: Any | None

class QuadTree:
    # Expose the raw native class for power users
    NativeQuadTree: type

    def __init__(
        self,
        bounds: Bounds,
        capacity: int,
        *,
        max_depth: int | None = None,
        track_objects: bool = False,
        start_id: int = 1,
    ) -> None: ...

    # Inserts
    def insert(self, xy: Point, *, id_: int | None = ..., obj: Any = ...) -> int: ...
    def insert_many_points(self, points: Iterable[Point]) -> int: ...
    def attach(self, id_: int, obj: Any) -> None: ...

    # Deletions
    def delete(self, id_: int, xy: Point) -> bool: ...
    def delete_by_object(self, obj: Any) -> bool: ...

    # Queries
    @overload
    def query(
        self, rect: Bounds, *, as_items: _Literal[False] = ...
    ) -> list[tuple[int, float, float]]: ...
    @overload
    def query(self, rect: Bounds, *, as_items: _Literal[True]) -> list[Item]: ...
    @overload
    def nearest_neighbor(
        self, xy: Point, *, as_item: _Literal[False] = ...
    ) -> tuple[int, float, float] | None: ...
    @overload
    def nearest_neighbor(
        self, xy: Point, *, as_item: _Literal[True]
    ) -> Item | None: ...
    @overload
    def nearest_neighbors(
        self, xy: Point, k: int, *, as_items: _Literal[False] = ...
    ) -> list[tuple[int, float, float]]: ...
    @overload
    def nearest_neighbors(
        self, xy: Point, k: int, *, as_items: _Literal[True]
    ) -> list[Item]: ...

    # Misc
    def get(self, id_: int) -> Any | None: ...
    def get_all_rectangles(self) -> list[Bounds]: ...
    def get_all_objects(self) -> list[Any]: ...
    def get_all_items(self) -> list[Item]: ...
    def count_items(self) -> int: ...
    def __len__(self) -> int: ...
