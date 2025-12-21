from tests.test_python.conftest import get_bounds_for_dtype

from fastquadtree._common import Bounds
from fastquadtree.rect_quadtree import RectQuadTree


def test_insert_query_contains_and_iter(bounds: Bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTree(bounds_use, capacity=4, dtype=dtype)
    rects = (
        [(1, 1, 2, 2), (3, 3, 4, 4)]
        if dtype.startswith("i")
        else [
            (1.0, 1.0, 2.0, 2.0),
            (3.0, 3.0, 4.0, 4.0),
        ]
    )
    ids = [rqt.insert(rect) for rect in rects]

    assert len(rqt) == 2
    assert rects[0] in [tuple(r[1:]) for r in rqt.query(bounds_use)]

    results = rqt.query((bounds_use[0], bounds_use[1], 5, 5))
    assert {r[0] for r in results} == set(ids)
    assert sorted(rqt) == sorted(results)


def test_custom_id_insertion(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTree(bounds_use, capacity=4, dtype=dtype)
    rect = (5, 5, 6, 6) if dtype.startswith("i") else (5.0, 5.0, 6.0, 6.0)
    rid = rqt.insert(rect, id_=7)
    assert rid == 7
    assert len(rqt) == 1
    assert rect in [tuple(r[1:]) for r in rqt.query(bounds_use)]


def test_overlapping_rectangles_query(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTree(bounds_use, capacity=4, dtype=dtype)
    r1 = (10, 10, 30, 30) if dtype.startswith("i") else (10.0, 10.0, 30.0, 30.0)
    r2 = (20, 20, 40, 40) if dtype.startswith("i") else (20.0, 20.0, 40.0, 40.0)
    rqt.insert(r1)
    rqt.insert(r2)
    results = rqt.query(
        (15, 15, 35, 35) if dtype.startswith("i") else (15.0, 15.0, 35.0, 35.0)
    )
    assert len(results) == 2


def test_nearest_neighbor_variants(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTree(bounds_use, capacity=4, dtype=dtype)
    rects = [
        (10, 10, 12, 12),
        (20, 20, 22, 22),
        (40, 40, 42, 42),
    ]
    for rect in rects:
        rqt.insert(rect)

    query_pt = (21, 21) if dtype.startswith("i") else (21.0, 21.0)
    nn = rqt.nearest_neighbor(query_pt)
    assert nn is not None
    assert nn[1:] == rects[1]

    knn = rqt.nearest_neighbors(query_pt, k=2)
    assert [tuple(map(float, t[1:])) for t in knn] == [
        tuple(map(float, rects[1])),
        tuple(map(float, rects[0])),
    ]

    nn_np = rqt.nearest_neighbor_np(query_pt)
    assert nn_np is not None
    assert tuple(nn_np[1].tolist()) == tuple(map(float, rects[1]))

    ids_np, coords_np = rqt.nearest_neighbors_np(query_pt, k=2)
    assert list(ids_np) == [1, 0]
    assert [tuple(row) for row in coords_np.tolist()] == [
        tuple(map(float, rects[1])),
        tuple(map(float, rects[0])),
    ]


def test_node_boundaries_and_max_depth(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTree(bounds_use, capacity=2, max_depth=4, dtype=dtype)
    r1 = (10, 10, 15, 15) if dtype.startswith("i") else (10.0, 10.0, 15.0, 15.0)
    r2 = (80, 80, 90, 90) if dtype.startswith("i") else (80.0, 80.0, 90.0, 90.0)
    rqt.insert(r1)
    rqt.insert(r2)
    boundaries = rqt.get_all_node_boundaries()
    assert boundaries
    assert all(len(b) == 4 for b in boundaries)
    assert rqt.get_inner_max_depth() == 4
