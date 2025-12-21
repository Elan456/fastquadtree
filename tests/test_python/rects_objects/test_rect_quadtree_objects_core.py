import pytest
from tests.test_python.conftest import get_bounds_for_dtype

from fastquadtree.rect_quadtree_objects import RectQuadTreeObjects


def rect_for_dtype(dtype: str, coords: tuple[float, float, float, float]) -> tuple:
    return tuple(map(int, coords)) if dtype.startswith("i") else coords


def test_insert_and_query_rects(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTreeObjects(bounds_use, capacity=4, dtype=dtype)
    rect1 = rect_for_dtype(dtype, (1.0, 1.0, 2.0, 2.0))
    rect2 = rect_for_dtype(dtype, (3.0, 3.0, 4.0, 4.0))
    rid1 = rqt.insert(rect1, obj="a")
    rid2 = rqt.insert(rect2)

    assert len(rqt) == 2
    res = rqt.query(rect_for_dtype(dtype, (0.0, 0.0, 5.0, 5.0)))
    assert {it.id_ for it in res} == {rid1, rid2}
    assert res[0].obj in {"a", None}

    ids = rqt.query_ids(rect_for_dtype(dtype, (0.0, 0.0, 5.0, 5.0)))
    assert set(ids) == {rid1, rid2}

    assert rect1 in rqt
    assert rect_for_dtype(dtype, (10.0, 10.0, 11.0, 11.0)) not in rqt

    iter_ids = [it.id_ for it in rqt]
    assert set(iter_ids) == {rid1, rid2}


def test_insert_many_and_mismatch_errors(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTreeObjects(bounds_use, capacity=10, dtype=dtype)
    rects = [
        rect_for_dtype(dtype, (1.0, 1.0, 2.0, 2.0)),
        rect_for_dtype(dtype, (3.0, 3.0, 4.0, 4.0)),
    ]
    res = rqt.insert_many(rects)
    assert list(res.ids) == [0, 1]

    objs = ["x", "y"]
    res2 = rqt.insert_many(rects, objs=objs)
    assert list(res2.ids) == [2, 3]
    assert [
        it.obj
        for it in rqt.query(rect_for_dtype(dtype, (0.0, 0.0, 5.0, 5.0)))
        if it.obj
    ] == objs

    with pytest.raises(ValueError):
        rqt.insert_many(rects, objs=["only one"])
