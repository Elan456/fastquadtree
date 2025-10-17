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
    points = np.array([[10, 10], [20, 20], [30, 30]])
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
