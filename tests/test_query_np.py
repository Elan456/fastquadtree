import numpy as np
import pytest

# Adjust import names if your package exposes different class names
from fastquadtree import QuadTree

BOUNDS = (0, 0, 100, 100)


def _mk_tree_f32(points, capacity=4):
    qt = QuadTree(BOUNDS, capacity=capacity)
    # If your API requires start_id use: qt.insert_many_with_start(0, points)
    qt.insert_many(points)
    return qt


def _ids_coords_to_sets(ids_np, coords_np):
    ids = ids_np.tolist()
    coords = [tuple(map(float, c)) for c in coords_np.tolist()]
    return set(ids), set(coords)


def _sorted_by_id(ids_np, coords_np):
    order = np.argsort(ids_np)
    return ids_np[order], coords_np[order]


def test_query_np_empty_tree_returns_empty_arrays():
    qt = QuadTree(BOUNDS, capacity=4)
    ids_np, coords_np = qt.query_np((10, 10, 20, 20))
    assert ids_np.dtype == np.uint64
    assert coords_np.ndim == 2
    assert coords_np.shape[1] == 2
    assert ids_np.size == 0
    assert coords_np.shape == (0, 2)


def test_query_np_single_hit():
    pts = np.array([[10, 10], [50, 50], [90, 90]], dtype=np.float32)
    qt = _mk_tree_f32(pts)
    ids_np, coords_np = qt.query_np((49.9, 49.9, 50.1, 50.1))
    assert ids_np.size == 1
    assert coords_np.shape == (1, 2)
    # IDs are auto-assigned in insertion order: 0,1,2
    assert ids_np[0] == 1
    assert np.allclose(coords_np[0], [50, 50])


def test_query_np_no_hits():
    pts = np.array([[10, 10], [20, 20]], dtype=np.float32)
    qt = _mk_tree_f32(pts)
    ids_np, coords_np = qt.query_np((30, 30, 40, 40))
    assert ids_np.size == 0
    assert coords_np.shape == (0, 2)


def test_query_np_many_hits_accuracy_and_dtype():
    pts = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.float32)
    qt = _mk_tree_f32(pts)
    ids_np, coords_np = qt.query_np((15, 15, 35, 35))

    assert ids_np.dtype == np.uint64
    assert coords_np.dtype == np.float32
    assert coords_np.shape[1] == 2

    ids_set, coords_set = _ids_coords_to_sets(ids_np, coords_np)
    assert ids_set == {1, 2}
    assert coords_set == {(20.0, 20.0), (30.0, 30.0)}


def test_query_np_matches_python_list_api_order_agnostic():
    pts = np.array([[5, 5], [15, 15], [25, 25], [35, 35]], dtype=np.float32)
    qt = _mk_tree_f32(pts)
    rect = (10, 10, 30, 30)
    # Reference via list-of-tuples API
    ref = qt.query(rect)  # [(id, x, y), ...]
    ref_ids = {r[0] for r in ref}
    ref_xy = {(float(r[1]), float(r[2])) for r in ref}

    ids_np, coords_np = qt.query_np(rect)
    got_ids = set(ids_np.tolist())
    got_xy = {tuple(map(float, xy)) for xy in coords_np.tolist()}

    assert got_ids == ref_ids
    assert got_xy == ref_xy


def test_query_np_shapes_and_contiguity():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 100, size=(1000, 2)).astype(np.float32)
    qt = _mk_tree_f32(pts)
    ids_np, coords_np = qt.query_np((25, 25, 75, 75))

    assert coords_np.ndim == 2
    assert coords_np.shape[1] == 2
    assert ids_np.ndim == 1
    assert ids_np.flags.c_contiguous
    assert coords_np.flags.c_contiguous


def test_query_np_alignment_of_ids_and_coords():
    pts = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.float32)
    qt = _mk_tree_f32(pts)
    rect = (0, 0, 100, 100)
    ids_np, coords_np = qt.query_np(rect)

    # Sort both by id and verify coordinates match the inserted points
    ids_np, coords_np = _sorted_by_id(ids_np, coords_np)
    assert np.all(ids_np == np.arange(pts.shape[0], dtype=np.uint64))
    assert np.allclose(coords_np, pts)


def test_query_np_mutation_does_not_affect_tree_state():
    pts = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
    qt = _mk_tree_f32(pts)
    rect = (0, 0, 100, 100)
    ids_np, coords_np = qt.query_np(rect)
    orig = coords_np.copy()

    # Mutate the returned arrays
    coords_np[:] = -123.45
    ids_np[:] = 999

    # Re-query - the tree should be unchanged
    ids2, coords2 = qt.query_np(rect)
    ids2, coords2 = _sorted_by_id(ids2, coords2)
    assert np.allclose(coords2, pts)
    assert np.all(ids2 == np.arange(pts.shape[0], dtype=np.uint64))

    # And our original `orig` still matches expected
    assert np.allclose(orig[_sorted_by_id(ids_np.copy(), orig)[0].argsort()], orig)


def test_query_np_boundary_inclusion_float_eps():
    # Use a tight rectangle with epsilon to avoid ambiguity if max is exclusive
    eps = 1e-6
    pts = np.array([[10, 10]], dtype=np.float32)
    qt = _mk_tree_f32(pts)
    ids_np, coords_np = qt.query_np((10 - eps, 10 - eps, 10 + eps, 10 + eps))
    assert ids_np.size == 1
    assert np.allclose(coords_np[0], [10, 10])


def test_query_np_large_random_matches_naive_filter():
    rng = np.random.default_rng(42)
    n = 20000
    pts = rng.uniform(0, 100, size=(n, 2)).astype(np.float32)
    qt = _mk_tree_f32(pts, capacity=8)

    rect = (12.5, 37.5, 76.25, 88.0)
    ids_np, coords_np = qt.query_np(rect)

    # Naive filter on the same points - ids are insertion order [0..n-1]
    mask = (
        (pts[:, 0] >= rect[0])
        & (pts[:, 0] <= rect[2])
        & (pts[:, 1] >= rect[1])
        & (pts[:, 1] <= rect[3])
    )
    exp_ids = np.flatnonzero(mask).astype(np.uint64)
    exp_coords = pts[mask]

    # Order agnostic compare
    got_ids, got_xy = _ids_coords_to_sets(ids_np, coords_np)
    exp_ids_set = set(exp_ids.tolist())
    exp_xy_set = {tuple(map(float, xy)) for xy in exp_coords.tolist()}

    assert got_ids == exp_ids_set
    assert got_xy == exp_xy_set


@pytest.mark.parametrize(
    "rects",
    [
        ((0, 0, 50, 100), (50, 0, 100, 100)),  # vertical split, disjoint
        ((0, 0, 100, 50), (0, 50, 100, 100)),  # horizontal split, disjoint
    ],
)
def test_query_np_additivity_on_disjoint_rects(rects):
    rng = np.random.default_rng(7)
    pts = rng.uniform(0, 100, size=(5000, 2)).astype(np.float32)
    qt = _mk_tree_f32(pts)

    ids_all, _ = qt.query_np((0, 0, 100, 100))
    ids_a, _ = qt.query_np(rects[0])
    ids_b, _ = qt.query_np(rects[1])

    # For disjoint rectangles, counts add up
    assert ids_a.size + ids_b.size == ids_all.size

    # And sets union to the global set
    set_all = set(ids_all.tolist())
    assert set(ids_a.tolist()).union(set(ids_b.tolist())) == set_all


def test_query_np_f64_dtype_and_accuracy():
    pts = np.array([[1.25, 2.5], [3.75, 6.25]], dtype=np.float64)
    qt = QuadTree(BOUNDS, capacity=4, dtype="f64")
    qt.insert_many(pts)
    ids_np, coords_np = qt.query_np((0, 0, 10, 10))
    assert coords_np.dtype == np.float64
    ids_np, coords_np = _sorted_by_id(ids_np, coords_np)
    assert np.allclose(coords_np, pts)


def test_query_np_i32_dtype_and_accuracy():
    pts = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    qt = QuadTree(BOUNDS, capacity=4, dtype="i32")
    qt.insert_many(pts)
    ids_np, coords_np = qt.query_np((0, 0, 10, 10))
    assert coords_np.dtype == np.int32
    ids_np, coords_np = _sorted_by_id(ids_np, coords_np)
    assert np.array_equal(coords_np, pts)


def test_query_np_i64_dtype_and_accuracy():
    pts = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)
    qt = QuadTree(BOUNDS, capacity=4, dtype="i64")
    qt.insert_many(pts)
    ids_np, coords_np = qt.query_np((0, 0, 10, 10))
    assert coords_np.dtype == np.int64
    ids_np, coords_np = _sorted_by_id(ids_np, coords_np)
    assert np.array_equal(coords_np, pts)
