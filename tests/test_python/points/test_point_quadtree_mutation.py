import pytest
from tests.test_python.conftest import get_bounds_for_dtype

from fastquadtree.point_quadtree import QuadTree


def test_delete_by_coords_and_tuple(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTree(bounds_use, capacity=4, dtype=dtype)
    pt = (5, 5) if dtype.startswith("i") else (5.0, 5.0)
    rid = qt.insert(pt)
    del_val = pt[0]
    assert qt.delete(rid, del_val, del_val) is True
    assert len(qt) == 0

    pt2 = (6, 6) if dtype.startswith("i") else (6.0, 6.0)
    rid = qt.insert(pt2)
    tup = (rid, pt2[0], pt2[1])
    assert qt.delete_tuple(tup) is True
    assert len(qt) == 0
    assert qt.delete(rid, pt2[0], pt2[1]) is False


def test_update_and_update_tuple_success(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTree(bounds_use, capacity=4, dtype=dtype)
    start = (1, 1) if dtype.startswith("i") else (1.0, 1.0)
    rid = qt.insert(start)
    new1 = (2, 2) if dtype.startswith("i") else (2.0, 2.0)
    new2 = (3, 3) if dtype.startswith("i") else (3.0, 3.0)
    assert qt.update(rid, start[0], start[1], new1[0], new1[1]) is True
    assert new1 in [t[1:] for t in qt.query(bounds_use)]

    assert qt.update_tuple(rid, new1, new2) is True
    if not dtype.startswith("i"):
        _ = new2 in qt
    assert new2 in [t[1:] for t in qt.query(bounds_use)]


def test_update_rollback_on_out_of_bounds(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTree(bounds_use, capacity=4, dtype=dtype)
    pt_old = (10, 10) if dtype.startswith("i") else (10.0, 10.0)
    rid = qt.insert(pt_old)
    with pytest.raises(ValueError):
        qt.update(
            rid,
            pt_old[0],
            pt_old[1],
            200 if dtype.startswith("i") else 200.0,
            200 if dtype.startswith("i") else 200.0,
        )
    # Item remains at old location
    assert pt_old in [t[1:] for t in qt.query(bounds_use)]


def test_clear_resets_count_and_next_id(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTree(bounds_use, capacity=4, dtype=dtype)
    qt.insert((1, 1) if dtype.startswith("i") else (1.0, 1.0))
    qt.insert((2, 2) if dtype.startswith("i") else (2.0, 2.0))
    assert qt._next_id == 2  # internal counter advanced by two inserts
    qt.clear()
    assert len(qt) == 0
    assert qt._next_id == 0
