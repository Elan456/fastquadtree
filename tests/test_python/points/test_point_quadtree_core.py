from tests.test_python.conftest import get_bounds_for_dtype

from fastquadtree.point_quadtree import QuadTree


def test_insert_query_len_contains_and_iter(bounds, dtype):
    # Use integer bounds/coords when dtype is integral to satisfy native expectations
    if dtype.startswith("i"):
        bounds_use = get_bounds_for_dtype(bounds, dtype)
        coords = [(1, 1), (2, 2), (3, 3)]
    else:
        bounds_use = bounds
        coords = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]

    qt = QuadTree(bounds_use, capacity=4, dtype=dtype)
    ids = [qt.insert(pt) for pt in coords]

    assert len(qt) == 3
    # __contains__ is executed for coverage; presence verified via query
    assert coords[0] in qt
    assert coords[0] in [t[1:] for t in qt.query(bounds_use)]

    got = qt.query((bounds_use[0], bounds_use[1], 5, 5))
    assert {p[0] for p in got} == set(ids)

    # __iter__ returns all items; order may match query of bounds
    iter_items = list(qt)
    assert sorted(iter_items) == sorted(got)


def test_query_empty_region_and_outside_bounds(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
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
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    pt = (5, 5) if dtype.startswith("i") else (5.0, 5.0)
    qt = QuadTree(bounds_use, capacity=4, dtype=dtype)
    rid = qt.insert(pt, id_=42)
    assert rid == 42
    assert len(qt) == 1
    assert pt in [t[1:] for t in qt.query(bounds_use)]


def test_nearest_neighbor_variants(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    if dtype.startswith("i"):
        pts = [(10, 10), (20, 20), (35, 35)]
    else:
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
    bounds_use = get_bounds_for_dtype(bounds, dtype)
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
    bounds_use = get_bounds_for_dtype(bounds, dtype)
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


def test_contains_robust(bounds, dtype):
    """Comprehensive test for __contains__ with various dtypes and edge cases."""
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTree(bounds_use, capacity=4, dtype=dtype)

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
