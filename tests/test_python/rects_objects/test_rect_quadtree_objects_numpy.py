import numpy as np
import pytest
from tests.test_python.conftest import (
    assert_query_matches_np,
    get_bounds_for_dtype,
    make_np_coords,
)

from fastquadtree.rect_quadtree_objects import RectQuadTreeObjects


def rect_for_dtype(dtype: str, coords: tuple[float, float, float, float]) -> tuple:
    return tuple(map(int, coords)) if dtype.startswith("i") else coords


def test_insert_many_np_with_objects(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTreeObjects(bounds_use, capacity=8, dtype=dtype)
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
    assert_query_matches_np(rqt, rect_for_dtype(dtype, (0.0, 0.0, 5.0, 5.0)))

    objs = ["a", "b"]
    more = make_np_coords(
        dtype,
        [
            rect_for_dtype(dtype, (5.0, 5.0, 6.0, 6.0)),
            rect_for_dtype(dtype, (7.0, 7.0, 8.0, 8.0)),
        ],
    )
    res2 = rqt.insert_many_np(more, objs=objs)
    assert list(res2.ids) == [2, 3]
    assert [
        it.obj
        for it in rqt.query(rect_for_dtype(dtype, (0.0, 0.0, 10.0, 10.0)))
        if it.obj
    ] == objs


def test_insert_many_np_errors(bounds):
    rqt = RectQuadTreeObjects(bounds, capacity=4, dtype="f64")
    wrong = np.array([[1, 1, 2, 2]], dtype=np.float32)
    with pytest.raises(TypeError):
        rqt.insert_many_np(wrong)

    with pytest.raises(TypeError):
        rqt.insert_many_np([(1.0, 1.0, 2.0, 2.0)])  # type: ignore[arg-type]

    coords = make_np_coords("f64", [(1.0, 1.0, 2.0, 2.0)])
    with pytest.raises(ValueError):
        rqt.insert_many_np(coords, objs=["too", "long"])


def test_insert_many_np_empty(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTreeObjects(bounds_use, capacity=4, dtype=dtype)
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
