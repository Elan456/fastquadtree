import numpy as np
import pytest
from tests.test_python.conftest import (
    assert_query_matches_np,
    get_bounds_for_dtype,
    make_np_coords,
)

from fastquadtree.point_quadtree_objects import QuadTreeObjects


def test_insert_many_np_with_and_without_objects(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTreeObjects(bounds_use, capacity=8, dtype=dtype)
    coords = make_np_coords(dtype, [(1.0, 1.0), (2.0, 2.0)])
    res = qt.insert_many_np(coords)
    assert res.count == 2
    assert list(res.ids) == [0, 1]
    rect = (0, 0, 5, 5) if dtype.startswith("i") else (0.0, 0.0, 5.0, 5.0)
    assert_query_matches_np(qt, rect)

    objs = ["a", "b"]
    more = make_np_coords(dtype, [(3.0, 3.0), (4.0, 4.0)])
    res2 = qt.insert_many_np(more, objs=objs)
    assert list(res2.ids) == [2, 3]
    assert [it.obj for it in qt.query(rect) if it.obj] == objs


def test_insert_many_np_errors(bounds):
    qt = QuadTreeObjects(bounds, capacity=4, dtype="f64")
    wrong = np.array([[1, 2]], dtype=np.float32)
    with pytest.raises(TypeError):
        qt.insert_many_np(wrong)

    not_np = [(1.0, 1.0)]
    with pytest.raises(TypeError):
        qt.insert_many_np(not_np)  # type: ignore[arg-type]

    coords = make_np_coords("f64", [(1.0, 1.0)])
    with pytest.raises(ValueError):
        qt.insert_many_np(coords, objs=["x", "y"])


def test_insert_many_np_empty(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTreeObjects(bounds_use, capacity=4, dtype=dtype)
    empty = np.empty((0, 2), dtype=make_np_coords(dtype, [(0.0, 0.0)]).dtype)
    res = qt.insert_many_np(empty)
    assert res.count == 0
    assert res.start_id == 0
    assert res.end_id == -1
