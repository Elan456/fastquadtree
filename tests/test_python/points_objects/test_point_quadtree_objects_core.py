import pytest

from fastquadtree.point_quadtree_objects import QuadTreeObjects


def test_insert_and_query_points_with_objects(bounds, dtype):
    bounds_use = tuple(map(int, bounds)) if dtype.startswith("i") else bounds
    qt = QuadTreeObjects(bounds_use, capacity=4, dtype=dtype)
    p1 = (1, 1) if dtype.startswith("i") else (1.0, 1.0)
    p2 = (2, 2) if dtype.startswith("i") else (2.0, 2.0)
    rid1 = qt.insert(p1, obj="a")
    rid2 = qt.insert(p2)

    assert len(qt) == 2
    rect = (0, 0, 5, 5) if dtype.startswith("i") else (0.0, 0.0, 5.0, 5.0)
    results = qt.query(rect)
    assert {it.id_ for it in results} == {rid1, rid2}
    assert any(it.obj == "a" for it in results)

    ids = qt.query_ids(rect)
    assert set(ids) == {rid1, rid2}

    if not dtype.startswith("i"):
        _ = p1 in qt

    # Iteration yields PointItem objects
    iter_ids = [it.id_ for it in qt]
    assert set(iter_ids) == {rid1, rid2}


def test_insert_many_with_optional_objects(bounds, dtype):
    bounds_use = tuple(map(int, bounds)) if dtype.startswith("i") else bounds
    qt = QuadTreeObjects(bounds_use, capacity=10, dtype=dtype)
    geoms = (
        [(1, 1), (2, 2), (3, 3)]
        if dtype.startswith("i")
        else [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    )
    res = qt.insert_many(geoms)
    assert res.start_id == 0
    assert res.end_id == 2

    objs = ["a", "b", "c"]
    res2 = qt.insert_many(geoms, objs=objs)
    assert res2.start_id == 3
    assert res2.end_id == 5
    rect = (0, 0, 5, 5) if dtype.startswith("i") else (0.0, 0.0, 5.0, 5.0)
    assert [it.obj for it in qt.query(rect)][-3:] == objs

    with pytest.raises(ValueError):
        qt.insert_many(geoms, objs=["only two"])
