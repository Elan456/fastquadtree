from fastquadtree.point_quadtree import QuadTree


def test_insert_query_len_contains_and_iter(bounds, dtype):
    # Use integer bounds/coords when dtype is integral to satisfy native expectations
    if dtype.startswith("i"):
        bounds_use = tuple(map(int, bounds))
        coords = [(1, 1), (2, 2), (3, 3)]
    else:
        bounds_use = bounds
        coords = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]

    qt = QuadTree(bounds_use, capacity=4, dtype=dtype)
    ids = [qt.insert(pt) for pt in coords]

    assert len(qt) == 3
    # __contains__ is executed for coverage; presence verified via query
    if not dtype.startswith("i"):
        _ = coords[0] in qt
    assert coords[0] in [t[1:] for t in qt.query(bounds_use)]

    got = qt.query((bounds_use[0], bounds_use[1], 5, 5))
    assert {p[0] for p in got} == set(ids)

    # __iter__ returns all items; order may match query of bounds
    iter_items = list(qt)
    assert sorted(iter_items) == sorted(got)


def test_query_empty_region_and_outside_bounds(bounds, dtype):
    bounds_use = tuple(map(int, bounds)) if dtype.startswith("i") else bounds
    qt = QuadTree(bounds_use, capacity=4, dtype=dtype)
    assert qt.query(bounds_use) == []
    outside = (
        bounds_use[2] + 1,
        bounds_use[3] + 1,
        bounds_use[2] + 2,
        bounds_use[3] + 2,
    )
    assert qt.query(outside) == []


def test_custom_id_insertion(bounds, dtype):
    bounds_use = tuple(map(int, bounds)) if dtype.startswith("i") else bounds
    pt = (5, 5) if dtype.startswith("i") else (5.0, 5.0)
    qt = QuadTree(bounds_use, capacity=4, dtype=dtype)
    rid = qt.insert(pt, id_=42)
    assert rid == 42
    assert len(qt) == 1
    assert pt in [t[1:] for t in qt.query(bounds_use)]


def test_nearest_neighbor_variants(bounds, dtype):
    if dtype.startswith("i"):
        bounds_use = tuple(map(int, bounds))
        pts = [(10, 10), (20, 20), (35, 35)]
    else:
        bounds_use = bounds
        pts = [(10.0, 10.0), (20.0, 20.0), (35.0, 35.0)]

    qt = QuadTree(bounds_use, capacity=4, dtype=dtype)
    for pt in pts:
        qt.insert(pt)

    query_pt = (22, 22) if dtype.startswith("i") else (22.0, 22.0)
    nearest = qt.nearest_neighbor(query_pt)
    assert nearest is not None
    assert nearest[1:] == (20.0, 20.0)

    k_nearest = qt.nearest_neighbors(query_pt, k=2)
    assert [tuple(map(float, coords[1:])) for coords in k_nearest] == [
        (20.0, 20.0),
        (10.0, 10.0),
    ]

    nn_np = qt.nearest_neighbor_np(query_pt)
    assert nn_np is not None
    assert tuple(nn_np[1].tolist()) == (20.0, 20.0)

    ids_np, coords_np = qt.nearest_neighbors_np(query_pt, k=2)
    assert list(ids_np) == [1, 0]
    assert [tuple(map(float, row)) for row in coords_np.tolist()] == [
        (20.0, 20.0),
        (10.0, 10.0),
    ]


def test_nearest_neighbor_empty_and_k_exceeds_count(bounds, dtype):
    bounds_use = tuple(map(int, bounds)) if dtype.startswith("i") else bounds
    qt = QuadTree(bounds_use, capacity=4, dtype=dtype)
    query_pt = (5, 5) if dtype.startswith("i") else (5.0, 5.0)
    assert qt.nearest_neighbor(query_pt) is None
    assert qt.nearest_neighbors(query_pt, k=3) == []

    pt = (1, 1) if dtype.startswith("i") else (1.0, 1.0)
    qt.insert(pt)
    knn = qt.nearest_neighbors(query_pt, k=5)
    assert len(knn) == 1
    assert knn[0][1:] == tuple(map(float, pt))


def test_get_all_node_boundaries_and_max_depth(bounds, dtype):
    bounds_use = tuple(map(int, bounds)) if dtype.startswith("i") else bounds
    qt = QuadTree(bounds_use, capacity=2, max_depth=6, dtype=dtype)
    pt1 = (10, 10) if dtype.startswith("i") else (10.0, 10.0)
    pt2 = (90, 90) if dtype.startswith("i") else (90.0, 90.0)
    pt3 = (25, 25) if dtype.startswith("i") else (25.0, 25.0)
    qt.insert(pt1)
    qt.insert(pt2)
    qt.insert(pt3)

    boundaries = qt.get_all_node_boundaries()
    assert boundaries, "expected at least one boundary"
    assert all(len(b) == 4 for b in boundaries)

    assert qt.get_inner_max_depth() == 6
