import pytest

from fastquadtree._base_quadtree import _BaseQuadTree
from fastquadtree._item import Item

# ---------- test doubles ----------


class DummyNative:
    """
    Minimal native engine stub with the API expected by _BaseQuadTree.
    Works on 2D points (x, y). Bounds are inclusive.
    """

    def __init__(self, bounds, capacity, max_depth):
        self.bounds = bounds
        self.capacity = capacity
        self.max_depth = max_depth
        self._items = {}  # id -> geom

    def _inside(self, geom):
        x0, y0, x1, y1 = self.bounds
        x, y = geom
        return x0 <= x <= x1 and y0 <= y <= y1

    def insert(self, id_, geom):
        if not self._inside(geom):
            return False
        self._items[id_] = geom
        return True

    def insert_many(self, start_id, geoms):
        inserted = 0
        for g in geoms:
            if self._inside(g):
                self._items[start_id + inserted] = g
                inserted += 1
        return (start_id + inserted - 1) if inserted > 0 else (start_id - 1)

    def delete(self, id_, geom):
        if id_ in self._items and self._items[id_] == geom:
            del self._items[id_]
            return True
        return False

    def count_items(self):
        return len(self._items)

    def get_all_node_boundaries(self):
        return [self.bounds]


class SimpleItem(Item):
    """Item-like object with the attributes _BaseQuadTree expects."""

    __slots__ = ("geom", "id_", "obj")

    def __init__(self, id_, geom, obj):
        self.id_ = id_
        self.geom = geom
        self.obj = obj


class PointQT(_BaseQuadTree[tuple[float, float], tuple[int, float, float], SimpleItem]):
    """Concrete wrapper for tests over DummyNative (points only)."""

    def _new_native(self, bounds, capacity, max_depth):
        return DummyNative(bounds, capacity, max_depth)

    def _make_item(self, id_, geom, obj):
        return SimpleItem(id_, geom, obj)


# ---------- helpers ----------

B = (0.0, 0.0, 10.0, 10.0)
IN = (5.0, 5.0)
IN2 = (7.0, 3.0)
OUT = (50.0, 50.0)


# ---------- core coverage (from earlier suite) ----------


def test_ctor_and_len_and_count_items_no_tracking():
    qt = PointQT(bounds=B, capacity=4, track_objects=False)
    assert len(qt) == 0
    assert qt.count_items() == 0
    assert qt.get_all_node_boundaries() == [B]


def test_ids_to_objects_raises_when_not_tracking():
    qt = PointQT(bounds=B, capacity=4, track_objects=False)
    with pytest.raises(ValueError, match="track_objects=False"):
        qt._ids_to_objects([0, 1])


def test_insert_without_tracking_ok_and_outside_raises():
    qt = PointQT(bounds=B, capacity=4, track_objects=False)
    rid = qt.insert(IN)
    assert rid == 0
    assert len(qt) == 1
    assert qt.count_items() == 1

    rid2 = qt.insert(IN2)
    assert rid2 == 1
    assert len(qt) == 2
    assert qt.count_items() == 2

    with pytest.raises(ValueError, match="outside bounds"):
        qt.insert(OUT)
    assert len(qt) == 2


def test_insert_with_tracking_and_get_attach_and_delete_by_object_paths():
    qt = PointQT(bounds=B, capacity=8, track_objects=True)

    rid = qt.insert(IN, obj={"k": 1})
    assert rid == 0
    assert len(qt) == 1
    assert qt.get(rid) == {"k": 1}
    assert qt.get_all_objects() == [{"k": 1}]
    assert [it.obj for it in qt.get_all_items()] == [{"k": 1}]

    qt.attach(rid, obj="new")
    assert qt.get(rid) == "new"

    qt2 = PointQT(bounds=B, capacity=4, track_objects=False)
    with pytest.raises(ValueError, match="track_objects=False"):
        qt2.attach(0, obj="x")

    with pytest.raises(KeyError):
        qt.attach(999, obj="x")

    with pytest.raises(ValueError, match="track_objects=False"):
        qt2.delete_by_object("whatever")

    assert qt.delete_by_object("missing") is False
    assert qt.delete_by_object("new") is True
    assert len(qt) == 0
    assert qt.count_items() == 0


def test_get_errors_and_values():
    qt = PointQT(bounds=B, capacity=4, track_objects=True)
    rid = qt.insert(IN, obj="v")
    assert qt.get(rid) == "v"
    assert qt.get(999) is None

    qt2 = PointQT(bounds=B, capacity=4, track_objects=False)
    with pytest.raises(ValueError, match="track_objects=False"):
        qt2.get(0)


def test_delete_true_and_false_paths_with_tracking():
    qt = PointQT(bounds=B, capacity=4, track_objects=True)
    rid = qt.insert(IN, obj="x")

    assert qt.delete(rid, IN2) is False
    assert qt.delete(rid, IN) is True
    assert len(qt) == 0


def test_clear_resets_everything_and_next_id():
    qt = PointQT(bounds=B, capacity=4, track_objects=True)
    qt.insert(IN, obj=1)
    qt.insert(IN2, obj=2)
    assert len(qt) == 2

    qt.clear()
    assert len(qt) == 0
    assert qt.count_items() == 0

    rid = qt.insert(IN, obj="a")
    assert rid == 0
    assert qt.get(rid) == "a"


def test_insert_many_no_geoms_returns_zero():
    qt = PointQT(bounds=B, capacity=4, track_objects=False)
    assert qt.insert_many([]) == 0
    assert len(qt) == 0


def test_insert_many_one_is_oob_tracking():
    qt = PointQT(bounds=B, capacity=4, track_objects=True)
    with pytest.raises(ValueError):
        _ = qt.insert_many([(5.0, 5.0), (20.0, 20.0), (2.0, 2.0)]) == 0


def test_insert_many_without_tracking_success_and_partial_failure_raises():
    qt = PointQT(bounds=B, capacity=4, track_objects=False)

    num = qt.insert_many([IN, IN2, (1.0, 9.0)])
    assert num == 3
    assert len(qt) == 3

    with pytest.raises(ValueError, match="outside tree bounds"):
        qt.insert_many([IN, OUT, IN2])
    assert len(qt) == 3


def test_insert_many_with_tracking_free_list_branch_objs_none_and_objs_mismatch_and_success():
    qt = PointQT(bounds=B, capacity=8, track_objects=True)

    id0 = qt.insert(IN, obj="a")
    id1 = qt.insert(IN2, obj="b")
    assert {id0, id1} == {0, 1}
    assert qt.delete(id1, IN2) is True
    assert qt.delete(id0, IN) is True
    assert len(qt) == 0

    num = qt.insert_many([(2.0, 2.0), (3.0, 3.0)], objs=None)
    assert num == 2
    assert len(qt) == 2

    with pytest.raises(ValueError, match="objs length must match"):
        qt.insert_many([(4.0, 4.0), (5.0, 5.0)], objs=["only-one"])

    num2 = qt.insert_many([(6.0, 6.0)], objs=["v6"])
    assert num2 == 1
    has_v6 = any(qt.get(i) == "v6" for i in range(10))
    assert has_v6


def test_insert_many_with_tracking_no_free_bulk_path_success_and_mismatch():
    qt = PointQT(bounds=B, capacity=16, track_objects=True)

    geoms = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    objs = ["o1", "o2", "o3"]
    num = qt.insert_many(geoms, objs=objs)
    assert num == 3
    assert len(qt) == 3
    assert [qt.get(i) for i in range(3)] == objs

    with pytest.raises(ValueError, match="objs length must match"):
        qt.insert_many([(4.0, 4.0), (5.0, 5.0)], objs=["only-one"])


# ---------- extra tests to hit remaining uncovered lines ----------


def test_ids_to_objects_with_tracking_success():
    """Covers the non-error return in _ids_to_objects (line ~76)."""
    qt = PointQT(bounds=B, capacity=8, track_objects=True)
    a = qt.insert((1.0, 1.0), obj="A")
    b = qt.insert((2.0, 2.0), obj="B")
    c = qt.insert((3.0, 3.0), obj="C")
    out = qt._ids_to_objects([c, a, b])
    assert out == ["C", "A", "B"]


def test_insert_many_with_tracking_no_free_bulk_path_objs_none_branch():
    """
    Covers the objs is None branch after bulk insert when tracking and no free holes
    (lines ~154-158).
    """
    qt = PointQT(bounds=B, capacity=16, track_objects=True)
    # no frees yet, so bulk path without objs
    n = qt.insert_many([(1.0, 1.0), (2.0, 2.0)], objs=None)
    assert n == 2
    # None objects were stored
    assert qt.get(0) is None
    assert qt.get(1) is None


def test_next_id_monotonic_update_true_and_false():
    """
    Covers the condition that updates _next_id after bulk insert
    - true path: _next_id increases
    - false path: _next_id unchanged when already ahead
    (line ~166).
    """
    # true path
    qt_true = PointQT(bounds=B, capacity=16, track_objects=True)
    assert qt_true._next_id == 0
    qt_true.insert_many([(1.0, 1.0)], objs=None)  # no frees, bulk path
    assert qt_true._next_id == 1  # updated

    # false path
    qt_false = PointQT(bounds=B, capacity=16, track_objects=True)
    qt_false._next_id = 100  # ahead of last_id + 1
    qt_false.insert_many([(2.0, 2.0), (3.0, 3.0)], objs=None)
    assert qt_false._next_id == 100  # unchanged


def test_get_all_objects_and_items_raise_when_not_tracking():
    """
    Covers the error branches around lines ~181-184.
    """
    qt = PointQT(bounds=B, capacity=4, track_objects=False)
    with pytest.raises(ValueError, match="track_objects=False"):
        qt.get_all_objects()
    with pytest.raises(ValueError, match="track_objects=False"):
        qt.get_all_items()
