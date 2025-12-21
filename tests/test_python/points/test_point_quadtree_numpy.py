import numpy as np
import pytest
from tests.test_python.conftest import (
    assert_query_matches_np,
    get_bounds_for_dtype,
    make_np_coords,
)

from fastquadtree.point_quadtree import QuadTree


def test_insert_many_np_round_trip(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTree(bounds_use, capacity=8, dtype=dtype)
    coords = make_np_coords(dtype, [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])
    result = qt.insert_many_np(coords)
    assert result.count == 3
    assert list(result.ids) == [0, 1, 2]

    rect = (0, 0, 5, 5) if dtype.startswith("i") else (0.0, 0.0, 5.0, 5.0)
    assert_query_matches_np(qt, rect)


def test_insert_many_np_dtype_mismatch_raises(bounds):
    qt = QuadTree(bounds, capacity=4, dtype="f32")
    wrong = np.array([[1, 2], [3, 4]], dtype=np.float64)
    with pytest.raises(TypeError):
        qt.insert_many_np(wrong)


def test_numpy_passed_to_non_np_methods_rejected(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTree(bounds_use, capacity=4, dtype=dtype)
    coords = make_np_coords(dtype, [(1.0, 1.0), (2.0, 2.0)])
    with pytest.raises(TypeError):
        qt.insert_many(coords)  # type: ignore[arg-type]


def test_insert_many_np_empty_array(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTree(bounds_use, capacity=4, dtype=dtype)
    empty = np.empty((0, 2), dtype=make_np_coords(dtype, [(0.0, 0.0)]).dtype)
    res = qt.insert_many_np(empty)
    assert res.count == 0
    assert res.start_id == 0
    assert res.end_id == -1
