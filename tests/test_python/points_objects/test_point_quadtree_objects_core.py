from typing import List

import pytest
from tests.test_python.conftest import get_bounds_for_dtype

from fastquadtree._common import Point
from fastquadtree.point_quadtree_objects import QuadTreeObjects


def test_insert_and_query_points_with_objects(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
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
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTreeObjects(bounds_use, capacity=10, dtype=dtype)
    geoms: List[Point] = (
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


def test_contains_robust(bounds, dtype):
    """Comprehensive test for __contains__ with various dtypes and edge cases."""
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTreeObjects(bounds_use, capacity=4, dtype=dtype)

    # Test empty tree
    test_pt = (10, 10) if dtype.startswith("i") else (10.0, 10.0)
    assert test_pt not in qt, "Empty tree should not contain any point"

    # Insert points and test exact matches
    if dtype.startswith("i"):
        pts = [(10, 20), (30, 40), (50, 60)]
    else:
        pts = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]

    for pt in pts:
        qt.insert(pt)

    # Test exact matches
    for pt in pts:
        assert pt in qt, f"Point {pt} should be found in tree"

    # Test non-existent points
    if dtype.startswith("i"):
        non_existent = [(11, 20), (10, 21), (99, 99)]
    else:
        non_existent = [(10.1, 20.0), (10.0, 20.1), (99.0, 99.0)]

    for pt in non_existent:
        assert pt not in qt, f"Point {pt} should not be found in tree"

    # Test with duplicates at same location
    dup_pt = (70, 80) if dtype.startswith("i") else (70.0, 80.0)
    qt.insert(dup_pt)
    qt.insert(dup_pt)  # Insert same point twice
    assert dup_pt in qt, "Duplicate point should be found"

    # Test integer dtype with exact integer coordinates
    if dtype.startswith("i"):
        exact_int_pt = (25, 35)
        qt.insert(exact_int_pt)
        assert exact_int_pt in qt, "Integer point should be found with exact match"
        assert (25, 36) not in qt, "Off-by-one integer should not match"

    # Test float dtype with floating point precision
    if dtype.startswith("f"):
        float_pt = (12.5, 22.5)
        qt.insert(float_pt)
        assert float_pt in qt, "Float point should be found with exact match"
        # These should NOT match due to floating point difference
        assert (12.50001, 22.5) not in qt, "Slightly different float should not match"
        assert (12.5, 22.50001) not in qt, "Slightly different float should not match"

    # Test boundaries (points at tree edges)
    min_x, min_y, max_x, max_y = bounds_use
    if dtype.startswith("i"):
        edge_pts = [
            (min_x, min_y),  # Bottom-left corner
            (max_x - 1, max_y - 1),  # Near top-right (inside bounds)
        ]
    else:
        edge_pts = [
            (float(min_x), float(min_y)),  # Bottom-left corner
            (
                float(max_x) - 1.0,
                float(max_y) - 1.0,
            ),  # Near top-right (use exact value)
        ]

    for pt in edge_pts:
        qt.insert(pt)
        assert pt in qt, f"Edge point {pt} should be found in tree"
