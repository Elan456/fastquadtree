import numpy as np
import pytest

from fastquadtree import Item, QuadTree, RectQuadTree

BOUNDS = (0, 0, 1000, 1000)


def ids(hits):
    """Return sorted list of ids from [(id, x, y), ...]."""
    return sorted(h[0] for h in hits)


def test_insert_many_seeds_items_and_query_as_items_round_trip():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    n = qt.insert_many([(10, 10), (20, 20), (30, 30)])
    assert n == 3

    qt_np = QuadTree(BOUNDS, capacity=8, track_objects=True)

    points = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
    n = qt_np.insert_many(points)
    assert n == 3

    raw = qt.query((0, 0, 40, 40), as_items=False)
    its = qt.query((0, 0, 40, 40), as_items=True)

    raw_np = qt_np.query((0, 0, 40, 40), as_items=False)
    its_np = qt_np.query((0, 0, 40, 40), as_items=True)

    assert len(raw) == len(its) == 3
    assert len(raw_np) == len(its_np) == 3
    # ids and positions match
    m_raw = {t[0]: (t[1], t[2]) for t in raw}
    for it in its:
        assert isinstance(it, Item)
        assert (it.x, it.y) == m_raw[it.id_]

    m_raw_np = {t[0]: (t[1], t[2]) for t in raw_np}
    for it in its_np:
        assert isinstance(it, Item)
        assert (it.x, it.y) == m_raw_np[it.id_]
    # ids match between raw and raw_np
    assert ids(raw) == ids(raw_np)


def test_type_error_on_wrong_dtype():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    points = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float64)  # Wrong dtype
    with pytest.raises(TypeError):
        qt.insert_many(points)
    assert len(qt) == 0


def test_insert_empty_numpy_array():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    points = np.empty((0, 2), dtype=np.float32)
    n = qt.insert_many(points)
    assert n == 0
    assert len(qt) == 0


def test_insert_many_numpy_out_of_bounds():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    points = np.array([[10, 10], (2000, 2000), [30, 30]], dtype=np.float32)
    with pytest.raises(ValueError):
        qt.insert_many(points)
    assert len(qt) == 0


def test_insert_many_without_tracking_numpy():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=False)
    points = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
    n = qt.insert_many(points)
    assert n == 3
    assert len(qt) == 3

    raw = qt.query((0, 0, 40, 40), as_items=False)

    assert len(raw) == 3
    # ids and positions match
    m_raw = {t[0]: (t[1], t[2]) for t in raw}
    for t in raw:
        assert (t[1], t[2]) == m_raw[t[0]]


def test_insert_many_rect_quadtree_numpy():
    qt = RectQuadTree(BOUNDS, capacity=8, track_objects=True)
    rects = np.array(
        [[10, 10, 15, 15], [20, 20, 25, 25], [30, 30, 35, 35]], dtype=np.float32
    )
    n = qt.insert_many(rects)
    assert n == 3

    raw = qt.query((0, 0, 40, 40), as_items=False)
    its = qt.query((0, 0, 40, 40), as_items=True)
    assert len(raw) == len(its) == 3
    # ids and positions match
    m_raw = {t[0]: (t[1], t[2], t[3], t[4]) for t in raw}
    for it in its:
        assert isinstance(it, Item)
        assert (it.min_x, it.min_y, it.max_x, it.max_y) == m_raw[it.id_]

    # Query that will only hit one rect
    raw2 = qt.query((12, 12, 13, 13), as_items=False)

    assert len(raw2) == 1
    assert raw2[0][0] == 0  # id of the first rect


def test_point_query_accuracy_robust_numpy():
    qt = QuadTree(BOUNDS, capacity=4, track_objects=True)
    num_points = 10000
    np.random.seed(42)
    points = np.random.uniform(0, 999, size=(num_points, 2)).astype(np.float32)
    qt.insert_many(points)

    # Query a random rectangle and verify all returned points are within it
    query_rect = (250, 250, 750, 750)
    results = qt.query(query_rect, as_items=False)

    for _, x, y in results:
        assert query_rect[0] <= x < query_rect[2]
        assert query_rect[1] <= y < query_rect[3]

    # Verify that no points within the rectangle are missed
    all_points_set = {(int(x), int(y)) for x, y in points}
    queried_points_set = {(int(x), int(y)) for _, x, y in results}

    for x in range(int(query_rect[0]), int(query_rect[2])):
        for y in range(int(query_rect[1]), int(query_rect[3])):
            if (x, y) in all_points_set:
                assert (x, y) in queried_points_set


def test_rect_query_accuracy_robust_numpy():
    qt = RectQuadTree(BOUNDS, capacity=4, track_objects=True)
    num_rects = 10000
    np.random.seed(42)
    rects = np.random.uniform(0, 950, size=(num_rects, 2)).astype(np.float32)
    sizes = np.random.uniform(5, 50, size=(num_rects, 2)).astype(np.float32)
    rects = np.hstack((rects, rects + sizes))
    qt.insert_many(rects)

    # Query a random rectangle and verify all returned rects intersect it
    query_rect = (250, 250, 750, 750)
    results = qt.query(query_rect, as_items=False)

    def intersects(r1, r2):
        return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[3] < r2[1] or r1[1] > r2[3])

    for _, min_x, min_y, max_x, max_y in results:
        assert intersects((min_x, min_y, max_x, max_y), query_rect)

    # Verify that no rects intersecting the query rectangle are missed
    all_rects = list(rects)
    queried_rects_set = {
        (int(min_x), int(min_y), int(max_x), int(max_y))
        for _, min_x, min_y, max_x, max_y in results
    }

    for rect in all_rects:
        if intersects(rect, query_rect):
            assert (
                int(rect[0]),
                int(rect[1]),
                int(rect[2]),
                int(rect[3]),
            ) in queried_rects_set


def test_insert_objects_numpy():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    qt.insert_many(
        np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32),
        objs=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
    )

    items = qt.query((0, 0, 40, 40), as_items=True)
    assert len(items) == 3

    names = {item.obj["name"] for item in items if item.obj is not None}
    assert names == {"A", "B", "C"}
