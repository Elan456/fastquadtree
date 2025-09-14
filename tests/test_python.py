import math
import pytest
from quadtree_rs import QuadTree


def ids(hits):
    """Return sorted list of ids from [(id, x, y), ...]."""
    return sorted(h[0] for h in hits)


def test_import_and_ctor():
    qt = QuadTree((0.0, 0.0, 100.0, 100.0), 2)
    assert isinstance(qt, QuadTree)


def test_insert_and_query_leaf():
    qt = QuadTree((0.0, 0.0, 100.0, 100.0), 10)
    assert qt.insert((10.0, 10.0))
    assert qt.insert((30.0, 30.0))
    assert qt.insert((70.0, 70.0))

    hits = qt.query((0.0, 0.0, 40.0, 40.0))
    assert ids(hits) == [1, 2]

    # half-open: max edges excluded, so (10,10) not in [0,0,10,10]
    hits2 = qt.query((0.0, 0.0, 10.0, 10.0))
    assert hits2 == []


def test_query_outside_returns_empty():
    qt = QuadTree((0.0, 0.0, 100.0, 100.0), 2)
    qt.insert((50.0, 50.0))
    hits = qt.query((200.0, 200.0, 300.0, 300.0))
    assert hits == []


def test_split_and_quadrant_queries():
    # capacity 1 to force early splits
    qt = QuadTree((0.0, 0.0, 100.0, 100.0), 1)

    # one point per quadrant plus the exact center
    assert qt.insert((10.0, 10.0))   # Q0
    assert qt.insert((75.0, 10.0))   # Q1
    assert qt.insert((10.0, 75.0))   # Q2
    assert qt.insert((75.0, 75.0))   # Q3
    assert qt.insert((50.0, 50.0))   # center -> right-top with >= rule

    lb = qt.query((0.0, 0.0, 50.0, 50.0))
    assert ids(lb) == [1]

    rb = qt.query((50.0, 0.0, 100.0, 50.0))
    assert ids(rb) == [2]

    lt = qt.query((0.0, 50.0, 50.0, 100.0))
    assert ids(lt) == [3]

    rt = qt.query((50.0, 50.0, 100.0, 100.0))
    assert ids(rt) == [4, 5]


def test_half_open_edges_on_insert():
    qt = QuadTree((0.0, 0.0, 100.0, 100.0), 4)
    # min edges included
    assert qt.insert((0.0, 0.0))
    # max edges excluded
    with pytest.raises(ValueError):
        qt.insert((100.0, 0.0))

    with pytest.raises(ValueError): 
        qt.insert((0.0, 100.0))

    with pytest.raises(ValueError):
        qt.insert((100.0, 100.0))

    hits = qt.query((0.0, 0.0, 1.0, 1.0))
    assert ids(hits) == [1]


def test_nearest_neighbor_basic():
    qt = QuadTree((0.0, 0.0, 100.0, 100.0), 2)
    qt.insert((10.0, 10.0))
    qt.insert((60.0, 60.0))
    nn = qt.nearest_neighbor((55.0, 55.0))
    assert nn is not None and nn[0] == 2  # id 2 should be closer


def test_nearest_neighbors_k():
    qt = QuadTree((0.0, 0.0, 100.0, 100.0), 2)
    pts = [
        (1, (10.0, 10.0)),
        (2, (20.0, 20.0)),
        (3, (30.0, 30.0)),
        (4, (80.0, 80.0)),
    ]
    for i, xy in pts:
        assert qt.insert(xy)

    res = qt.nearest_neighbors((25.0, 25.0), 3, as_items=True)
    assert sorted([item.id for item in res]) == [1, 2, 3]
    # check the closest is indeed (20,20) or (30,30) depending on your distance tie rules
    dists = [(item.id, math.hypot(item.x - 25.0, item.y - 25.0)) for item in res]
    assert len(dists) == 3
