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


def test_contains_robust(bounds, dtype):
    """Comprehensive test for __contains__ with various dtypes and edge cases."""
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTreeObjects(bounds_use, capacity=4, dtype=dtype)

    # Test empty tree
    test_rect = (10, 10, 20, 20) if dtype.startswith("i") else (10.0, 10.0, 20.0, 20.0)
    assert test_rect not in rqt, "Empty tree should not contain any rectangle"

    # Insert rectangles and test exact matches
    if dtype.startswith("i"):
        rects = [
            (10, 20, 30, 40),
            (50, 60, 70, 80),
            (15, 25, 35, 45),
        ]
    else:
        rects = [
            (10.0, 20.0, 30.0, 40.0),
            (50.0, 60.0, 70.0, 80.0),
            (15.0, 25.0, 35.0, 45.0),
        ]

    for rect in rects:
        rqt.insert(rect)

    # Test exact matches
    for rect in rects:
        assert rect in rqt, f"Rectangle {rect} should be found in tree"

    # Test non-existent rectangles (slightly different coordinates)
    if dtype.startswith("i"):
        non_existent = [
            (10, 20, 30, 41),  # Different max_y
            (11, 20, 30, 40),  # Different min_x
            (10, 21, 30, 40),  # Different min_y
            (10, 20, 31, 40),  # Different max_x
            (99, 99, 100, 100),  # Completely different
        ]
    else:
        non_existent = [
            (10.0, 20.0, 30.0, 40.1),  # Slightly different max_y
            (10.1, 20.0, 30.0, 40.0),  # Slightly different min_x
            (10.0, 20.1, 30.0, 40.0),  # Slightly different min_y
            (10.0, 20.0, 30.1, 40.0),  # Slightly different max_x
            (99.0, 99.0, 100.0, 100.0),  # Completely different
        ]

    for rect in non_existent:
        assert rect not in rqt, f"Rectangle {rect} should not be found in tree"

    # Test with duplicates at same location
    dup_rect = (5, 5, 8, 8) if dtype.startswith("i") else (5.0, 5.0, 8.0, 8.0)
    rqt.insert(dup_rect)
    rqt.insert(dup_rect)  # Insert same rectangle twice
    assert dup_rect in rqt, "Duplicate rectangle should be found"

    # Test integer dtype with exact integer coordinates
    if dtype.startswith("i"):
        exact_int_rect = (25, 35, 45, 55)
        rqt.insert(exact_int_rect)
        assert (
            exact_int_rect in rqt
        ), "Integer rectangle should be found with exact match"
        # Off-by-one should not match
        assert (25, 35, 45, 56) not in rqt, "Off-by-one rectangle should not match"
        assert (26, 35, 45, 55) not in rqt, "Off-by-one rectangle should not match"

    # Test float dtype with floating point precision
    if dtype.startswith("f"):
        float_rect = (12.5, 22.5, 32.5, 42.5)
        rqt.insert(float_rect)
        assert float_rect in rqt, "Float rectangle should be found with exact match"
        # These should NOT match due to floating point difference
        assert (
            12.50001,
            22.5,
            32.5,
            42.5,
        ) not in rqt, "Slightly different float should not match"
        assert (
            12.5,
            22.5,
            32.5,
            42.50001,
        ) not in rqt, "Slightly different float should not match"

    # Test boundaries (rectangles at tree edges)
    tree_min_x, tree_min_y, tree_max_x, tree_max_y = bounds_use
    if dtype.startswith("i"):
        edge_rects = [
            (tree_min_x, tree_min_y, tree_min_x + 5, tree_min_y + 5),  # Bottom-left
            (
                tree_max_x - 10,
                tree_max_y - 10,
                tree_max_x - 1,
                tree_max_y - 1,
            ),  # Near top-right
        ]
    else:
        edge_rects = [
            (
                float(tree_min_x),
                float(tree_min_y),
                float(tree_min_x) + 5.0,
                float(tree_min_y) + 5.0,
            ),  # Bottom-left
            (
                float(tree_max_x) - 10.0,
                float(tree_max_y) - 10.0,
                float(tree_max_x) - 1.0,
                float(tree_max_y) - 1.0,
            ),  # Near top-right (use exact values)
        ]

    for rect in edge_rects:
        rqt.insert(rect)
        assert rect in rqt, f"Edge rectangle {rect} should be found in tree"

    # Test that overlapping rectangles are distinct in __contains__
    if dtype.startswith("i"):
        overlap1 = (40, 40, 60, 60)
        overlap2 = (45, 45, 65, 65)  # Overlaps with overlap1 but different coords
    else:
        overlap1 = (40.0, 40.0, 60.0, 60.0)
        overlap2 = (45.0, 45.0, 65.0, 65.0)

    rqt.insert(overlap1)
    rqt.insert(overlap2)
    assert overlap1 in rqt, "First overlapping rectangle should be found"
    assert overlap2 in rqt, "Second overlapping rectangle should be found"
    # A rectangle that overlaps both but matches neither exactly
    if dtype.startswith("i"):
        assert (
            42,
            42,
            62,
            62,
        ) not in rqt, "Non-matching overlapping rectangle should not be found"
    else:
        assert (
            42.0,
            42.0,
            62.0,
            62.0,
        ) not in rqt, "Non-matching overlapping rectangle should not be found"
