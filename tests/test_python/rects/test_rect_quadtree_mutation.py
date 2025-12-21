import pytest
from tests.test_python.conftest import get_bounds_for_dtype

from fastquadtree.rect_quadtree import RectQuadTree


def test_delete_by_coords_and_tuple(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTree(bounds_use, capacity=4, dtype=dtype)
    rect1 = (5, 5, 6, 6) if dtype.startswith("i") else (5.0, 5.0, 6.0, 6.0)
    rid = rqt.insert(rect1)
    assert rqt.delete(rid, *rect1) is True
    assert len(rqt) == 0

    rect2 = (7, 7, 8, 8) if dtype.startswith("i") else (7.0, 7.0, 8.0, 8.0)
    rid = rqt.insert(rect2)
    tup = (rid, *rect2)
    assert len(tup) == 5
    assert rqt.delete_tuple(tup) is True
    assert len(rqt) == 0
    assert rqt.delete(rid, *rect2) is False


def test_update_and_update_tuple(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTree(bounds_use, capacity=4, dtype=dtype)
    rect_a = (1, 1, 2, 2) if dtype.startswith("i") else (1.0, 1.0, 2.0, 2.0)
    rect_b = (3, 3, 4, 4) if dtype.startswith("i") else (3.0, 3.0, 4.0, 4.0)
    rect_c = (5, 5, 6, 6) if dtype.startswith("i") else (5.0, 5.0, 6.0, 6.0)
    rid = rqt.insert(rect_a)
    assert rqt.update(rid, *rect_a, *rect_b) is True
    assert rect_b in [tuple(r[1:]) for r in rqt.query(bounds_use)]

    assert rqt.update_tuple(rid, rect_b, rect_c) is True
    assert rect_c in [tuple(r[1:]) for r in rqt.query(bounds_use)]


def test_update_rollback_out_of_bounds(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTree(bounds_use, capacity=4, dtype=dtype)
    rect = (1, 1, 2, 2) if dtype.startswith("i") else (1.0, 1.0, 2.0, 2.0)
    rid = rqt.insert(rect)
    with pytest.raises(ValueError):
        rqt.update(
            rid,
            *rect,
            200 if dtype.startswith("i") else 200.0,
            200 if dtype.startswith("i") else 200.0,
            210 if dtype.startswith("i") else 210.0,
            210 if dtype.startswith("i") else 210.0,
        )
    assert rect in [tuple(r[1:]) for r in rqt.query(bounds_use)]


def test_clear_resets_state(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTree(bounds_use, capacity=4, dtype=dtype)
    rqt.insert((1, 1, 2, 2) if dtype.startswith("i") else (1.0, 1.0, 2.0, 2.0))
    rqt.insert((3, 3, 4, 4) if dtype.startswith("i") else (3.0, 3.0, 4.0, 4.0))
    assert rqt._next_id == 2
    rqt.clear()
    assert len(rqt) == 0
    assert rqt._next_id == 0
