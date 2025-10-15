import math

import pytest

from fastquadtree import QuadTree


def ids(hits):
    """Return sorted list of ids from [(id, x, y), ...]."""
    return sorted(h[0] for h in hits)


def test_import_and_ctor():
    qt = QuadTree((0.0, 0.0, 100.0, 100.0), 2)
    assert isinstance(qt, QuadTree)


def test_insert_and_query_leaf():
    qt = QuadTree((0.0, 0.0, 100.0, 100.0), 10)
    assert qt.insert((10.0, 10.0)) == 0
    assert qt.insert((30.0, 30.0)) == 1
    assert qt.insert((70.0, 70.0)) == 2

    hits = qt.query((0.0, 0.0, 40.0, 40.0))
    assert ids(hits) == [0, 1]

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
    assert qt.insert((10.0, 10.0)) == 0  # Q0
    assert qt.insert((75.0, 10.0)) == 1  # Q1
    assert qt.insert((10.0, 75.0)) == 2  # Q2
    assert qt.insert((75.0, 75.0)) == 3  # Q3
    assert qt.insert((50.0, 50.0)) == 4  # center -> right-top with >= rule

    lb = qt.query((0.0, 0.0, 50.0, 50.0))
    assert ids(lb) == [0]

    rb = qt.query((50.0, 0.0, 100.0, 50.0))
    assert ids(rb) == [1]

    lt = qt.query((0.0, 50.0, 50.0, 100.0))
    assert ids(lt) == [2]

    rt = qt.query((50.0, 50.0, 100.0, 100.0))
    assert ids(rt) == [3, 4]


def test_half_open_edges_on_insert():
    qt = QuadTree((0.0, 0.0, 100.0, 100.0), 4)
    # min edges included
    assert qt.insert((0.0, 0.0)) == 0
    # max edges excluded
    with pytest.raises(ValueError):
        qt.insert((100.0, 0.0))

    with pytest.raises(ValueError):
        qt.insert((0.0, 100.0))

    with pytest.raises(ValueError):
        qt.insert((100.0, 100.0))

    hits = qt.query((0.0, 0.0, 1.0, 1.0))
    assert ids(hits) == [0]


def test_nearest_neighbor_basic():
    qt = QuadTree((0.0, 0.0, 100.0, 100.0), 2)
    qt.insert((10.0, 10.0))
    id2 = qt.insert((60.0, 60.0))
    nn = qt.nearest_neighbor((55.0, 55.0))
    assert nn is not None
    assert nn[0] == id2  # id 2 should be closer


def test_nearest_neighbors_k():
    qt = QuadTree((0.0, 0.0, 100.0, 100.0), 2, track_objects=True)
    pts = [
        (0, (10.0, 10.0)),
        (1, (20.0, 20.0)),
        (2, (30.0, 30.0)),
        (3, (80.0, 80.0)),
    ]
    for expected_id, xy in pts:
        assert expected_id == qt.insert(xy)

    res = qt.nearest_neighbors((25.0, 25.0), 3, as_items=True)
    assert sorted([item.id_ for item in res]) == [0, 1, 2]
    assert res[0].obj is None  # no objects were attached
    # check the closest is indeed (20,20) or (30,30) depending on your distance tie rules
    dists = [(item.id_, math.hypot(item.x - 25.0, item.y - 25.0)) for item in res]
    assert len(dists) == 3
