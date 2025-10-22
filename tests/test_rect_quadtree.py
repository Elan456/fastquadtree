# tests/test_rect_quadtree.py
from __future__ import annotations

from typing import Tuple

import pytest

from fastquadtree import rect_quadtree as rq

# ---- Test double for the native Rust RectQuadTree ----

Bounds = Tuple[float, float, float, float]
_IdRect = Tuple[int, float, float, float, float]


class FakeNative:
    """
    Minimal in-memory stand-in for RectQuadTreeF32:
      - insert / insert_many
      - delete
      - query / query_ids
      - count_items
      - get_all_node_boundaries
    """

    def __init__(self, bounds: Bounds, capacity: int, max_depth: int | None = None):
        self.bounds = bounds
        self.capacity = capacity
        self.max_depth = max_depth
        self.items: dict[int, Bounds] = {}

    # helper: inclusive "touch or intersect"
    def _intersects(self, a: Bounds, b: Bounds) -> bool:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0

    # helper: consider "insertable" if it intersects world bounds
    def _insertable(self, r: Bounds) -> bool:
        return self._intersects(self.bounds, r)

    def insert(self, id_: int, rect: Bounds) -> bool:
        if not self._insertable(rect):
            return False
        self.items[id_] = rect
        return True

    def insert_many(self, start_id: int, rects: list[Bounds]) -> int:
        next_id = start_id
        for r in rects:
            if self.insert(next_id, r):
                next_id += 1
        return next_id - 1

    def delete(self, id_: int, rect: Bounds) -> bool:
        cur = self.items.get(id_)
        if cur is None:
            return False
        if cur != rect:
            return False
        del self.items[id_]
        return True

    def query(self, rect: Bounds) -> list[_IdRect]:
        out: list[_IdRect] = []
        for id_, r in self.items.items():
            if self._intersects(r, rect):
                x0, y0, x1, y1 = r
                out.append((id_, x0, y0, x1, y1))
        return out

    def query_ids(self, rect: Bounds) -> list[int]:
        return [t[0] for t in self.query(rect)]

    def count_items(self) -> int:
        return len(self.items)

    def get_all_node_boundaries(self) -> list[Bounds]:
        # Single node world for testing
        return [self.bounds]

    # test-only escape hatch to create tracker-missing situations
    def _force_raw(self, id_: int, rect: Bounds) -> None:
        self.items[id_] = rect


# ---- Fixtures ----


@pytest.fixture(autouse=True)
def _patch_native(monkeypatch):
    """
    Replace rect_quadtree.RectQuadTreeF32 with FakeNative for all tests.
    """
    monkeypatch.setattr(rq, "RectQuadTreeF32", FakeNative, raising=True)


# ---- Helpers ----


def b_to_float(x0, y0, x1, y1) -> Bounds:
    return (float(x0), float(y0), float(x1), float(y1))


# ---- Tests ----


def test_insert_delete_and_count_and_len_no_tracking():
    qt = rq.RectQuadTree(b_to_float(0, 0, 100, 100), 8, track_objects=False)

    # Insert inside bounds
    id1 = qt.insert(b_to_float(10, 10, 20, 20))
    assert id1 == 0
    assert len(qt) == 1
    assert qt.count_items() == 1

    # Delete exact
    assert qt.delete(id1, b_to_float(10, 10, 20, 20)) is True
    assert len(qt) == 0
    assert qt.count_items() == 0

    # Insert outside bounds -> ValueError via _insert_common
    with pytest.raises(ValueError):
        qt.insert(b_to_float(200, 200, 210, 210))


def test_bulk_insert_success_and_error_paths(monkeypatch):
    qt = rq.RectQuadTree(b_to_float(0, 0, 100, 100), 8)

    # success path
    n = qt.insert_many([b_to_float(1, 1, 2, 2), b_to_float(3, 3, 4, 4)])
    assert n == 2
    assert qt.count_items() == 2
    assert len(qt) == 2

    # error path: one out of bounds so _insert_many_common raises ValueError
    with pytest.raises(ValueError):
        qt.insert_many([b_to_float(5, 5, 6, 6), b_to_float(1000, 1000, 1001, 1001)])


def test_query_paths_without_tracking_raw_and_as_items():
    qt = rq.RectQuadTree(b_to_float(0, 0, 10, 10), 8, track_objects=False)
    a = qt.insert(b_to_float(1, 1, 2, 2))
    b_id = qt.insert(b_to_float(5, 5, 6, 6))

    # raw tuples
    raw = qt.query(b_to_float(0, 0, 10, 10), as_items=False)
    ids = sorted(t[0] for t in raw)
    assert ids == [a, b_id]

    # as_items branch when _items is None -> build RectItem instances (obj=None)
    with pytest.raises(ValueError):
        qt.query(b_to_float(0, 0, 10, 10), as_items=True)


def test_node_boundaries_and_delete_miss():
    qt = rq.RectQuadTree(b_to_float(0, 0, 10, 10), 8)
    a = qt.insert(b_to_float(1, 1, 2, 2))
    qt.insert(b_to_float(5, 5, 6, 6))

    # get_all_node_boundaries delegates to native hook
    bounds = qt.get_all_node_boundaries()
    assert bounds == [b_to_float(0, 0, 10, 10)]

    # delete miss (wrong rect)
    assert qt.delete(a, b_to_float(1, 1, 3, 3)) is False


def test_accurate_obj_output_with_tracking():
    qt = rq.RectQuadTree(b_to_float(0, 0, 10, 10), 8, track_objects=True)
    id1 = qt.insert(b_to_float(1, 1, 2, 2), obj="first")
    qt.insert(b_to_float(5, 5, 6, 6), obj={"name": "second"})

    items = qt.query(b_to_float(0, 0, 10, 10), as_items=True)
    assert len(items) == 2
    assert "first" in [it.obj for it in items]
    assert {"name": "second"} in [it.obj for it in items]

    # Small query that only hits one
    items1 = qt.query(b_to_float(0, 0, 3, 3), as_items=True)
    assert len(items1) == 1
    assert items1[0].obj == "first"
    assert items1[0].id_ == id1
    assert items1[0].min_x == 1.0
    assert items1[0].min_y == 1.0
    assert items1[0].max_x == 2.0
    assert items1[0].max_y == 2.0

    # delete one and query again
    assert qt.delete(id1, b_to_float(1, 1, 2, 2)) is True
    items2 = qt.query(b_to_float(0, 0, 10, 10), as_items=True)
    assert len(items2) == 1
    assert items2[0].obj == {"name": "second"}


def test_unsupported_dtype():
    """Test that providing an unsupported dtype raises ValueError."""
    with pytest.raises(ValueError):
        rq.RectQuadTree(
            b_to_float(0, 0, 100, 100), capacity=4, track_objects=True, dtype="f128"
        )  # type: ignore

    # From bytes
    qt = rq.RectQuadTree(
        b_to_float(0, 0, 100, 100), capacity=4, track_objects=True, dtype="f32"
    )
    data = qt.to_bytes()
    with pytest.raises(ValueError):
        rq.RectQuadTree.from_bytes(data, dtype="f128")  # type: ignore


def test_all_other_dtypes():
    """Test that all supported dtypes can be used without error."""
    for dtype in ["f32", "f64", "i32", "i64"]:
        print("Testing dtype:", dtype)
        qt = rq.RectQuadTree(
            (0, 0, 100, 100), capacity=4, track_objects=True, dtype=dtype
        )
        id1 = qt.insert((10, 10, 20, 20), obj="test")
        assert id1 == 0
        data = qt.to_bytes()
        qt2 = rq.RectQuadTree.from_bytes(data, dtype=dtype)
        items = qt2.get_all_items()
        assert len(items) == 1
        assert items[0].obj == "test"

        # Run some queries
        results = qt2.query((5, 5, 15, 15), as_items=True)
        assert len(results) == 1
        assert results[0].obj == "test"
