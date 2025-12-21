import numpy as np
import pytest
from tests.test_python.conftest import (
    assert_query_matches_np,
    get_bounds_for_dtype,
    make_np_coords,
)

from fastquadtree.rect_quadtree import RectQuadTree


def rect_for_dtype(dtype: str, coords: tuple[float, float, float, float]) -> tuple:
    return tuple(map(int, coords)) if dtype.startswith("i") else coords


def test_insert_many_np_round_trip(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTree(bounds_use, capacity=8, dtype=dtype)
    rects = make_np_coords(
        dtype,
        [
            rect_for_dtype(dtype, (1.0, 1.0, 2.0, 2.0)),
            rect_for_dtype(dtype, (3.0, 3.0, 4.0, 4.0)),
        ],
    )
    res = rqt.insert_many_np(rects)
    assert res.count == 2
    assert list(res.ids) == [0, 1]
    assert_query_matches_np(rqt, rect_for_dtype(dtype, (0.0, 0.0, 10.0, 10.0)))


def test_insert_many_np_dtype_mismatch(bounds):
    bounds = get_bounds_for_dtype(bounds, "i32")
    rqt = RectQuadTree(bounds, capacity=4, dtype="i32")
    wrong = np.array([[1.0, 1.0, 2.0, 2.0]], dtype=np.float64)
    with pytest.raises(TypeError):
        rqt.insert_many_np(wrong)


def test_numpy_into_non_np_methods_raises(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTree(bounds_use, capacity=4, dtype=dtype)
    rects = make_np_coords(dtype, [rect_for_dtype(dtype, (1.0, 1.0, 2.0, 2.0))])
    with pytest.raises(TypeError):
        rqt.insert_many(rects)  # type: ignore[arg-type]


def test_insert_many_np_empty(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTree(bounds_use, capacity=4, dtype=dtype)
    empty = np.empty(
        (0, 4),
        dtype=make_np_coords(
            dtype, [rect_for_dtype(dtype, (0.0, 0.0, 1.0, 1.0))]
        ).dtype,
    )
    res = rqt.insert_many_np(empty)
    assert res.count == 0
    assert res.start_id == 0
    assert res.end_id == -1
