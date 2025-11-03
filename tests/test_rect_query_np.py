from __future__ import annotations

import numpy as np
import pytest

from fastquadtree import RectQuadTree

BOUNDS = (0, 0, 100, 100)


def _mk_rects_f32(rects, capacity=4):
    rqt = RectQuadTree(BOUNDS, capacity=capacity)
    # If your API requires start_id, adapt accordingly
    rqt.insert_many(rects)
    return rqt


def _ids_rects_to_sets(ids_np, rects_np):
    ids = ids_np.tolist()
    rects = [tuple(map(float, r)) for r in rects_np.tolist()]
    return set(ids), set(rects)


def _sorted_by_id(ids_np, rects_np):
    order = np.argsort(ids_np)
    return ids_np[order], rects_np[order]


def _intersects(a, b) -> bool:
    # a and b are (x0, y0, x1, y1)
    # Inclusive edges are considered intersection
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def test_query_np_empty_tree_returns_empty_arrays():
    rqt = RectQuadTree(BOUNDS, capacity=4)
    ids_np, rects_np = rqt.query_np((10, 10, 20, 20))
    assert ids_np.dtype == np.uint64
    assert rects_np.ndim == 2
    assert rects_np.shape[1] == 4
    assert ids_np.size == 0
    assert rects_np.shape == (0, 4)


def test_query_np_single_hit():
    rects = np.array(
        [
            [10, 10, 20, 20],
            [30, 30, 40, 40],
            [60, 60, 80, 90],
        ],
        dtype=np.float32,
    )
    rqt = _mk_rects_f32(rects)
    ids_np, rects_np = rqt.query_np((15, 15, 18, 18))
    assert ids_np.size == 1
    assert rects_np.shape == (1, 4)
    # Auto IDs are insertion order 0,1,2
    assert ids_np[0] == 0
    assert np.allclose(rects_np[0], [10, 10, 20, 20])


def test_query_np_no_hits():
    rects = np.array(
        [
            [0, 0, 5, 5],
            [10, 10, 15, 15],
        ],
        dtype=np.float32,
    )
    rqt = _mk_rects_f32(rects)
    ids_np, rects_np = rqt.query_np((20, 20, 25, 25))
    assert ids_np.size == 0
    assert rects_np.shape == (0, 4)


def test_query_np_many_hits_and_dtype():
    rects = np.array(
        [
            [10, 10, 20, 20],
            [18, 18, 22, 22],  # overlaps both
            [30, 30, 35, 35],
            [40, 40, 41, 41],
        ],
        dtype=np.float32,
    )
    rqt = _mk_rects_f32(rects)
    q = (19, 19, 30, 30)
    ids_np, rects_np = rqt.query_np(q)
    assert ids_np.dtype == np.uint64
    assert rects_np.dtype == np.float32
    assert rects_np.shape[1] == 4

    # Reference via list API
    ref = rqt.query(q)
    ref_ids = {r[0] for r in ref}
    ref_rects = {(float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in ref}

    got_ids, got_rects = _ids_rects_to_sets(ids_np, rects_np)
    assert got_ids == ref_ids
    assert got_rects == ref_rects


def test_query_np_matches_python_list_api_order_agnostic():
    rects = np.array(
        [
            [0, 0, 10, 10],
            [5, 5, 15, 15],
            [20, 20, 30, 30],
            [25, 25, 35, 35],
        ],
        dtype=np.float32,
    )
    rqt = _mk_rects_f32(rects)
    q = (9, 9, 22, 22)

    ref = rqt.query(q)  # [(id, x0, y0, x1, y1), ...]
    ref_ids = {r[0] for r in ref}
    ref_rects = {(float(r[1]), float(r[2]), float(r[3]), float(r[4])) for r in ref}

    ids_np, rects_np = rqt.query_np(q)
    got_ids, got_rects = _ids_rects_to_sets(ids_np, rects_np)

    assert got_ids == ref_ids
    assert got_rects == ref_rects


def test_query_np_shapes_and_contiguity():
    rng = np.random.default_rng(0)
    # Make small rects to keep intersections simple
    centers = rng.uniform(5, 95, size=(2000, 2)).astype(np.float32)
    wh = rng.uniform(0.1, 1.0, size=(2000, 2)).astype(np.float32)
    rects = np.column_stack(
        [
            centers[:, 0] - wh[:, 0],
            centers[:, 1] - wh[:, 1],
            centers[:, 0] + wh[:, 0],
            centers[:, 1] + wh[:, 1],
        ]
    )
    rqt = _mk_rects_f32(rects)
    ids_np, rects_np = rqt.query_np((25, 25, 75, 75))

    assert rects_np.ndim == 2
    assert rects_np.shape[1] == 4
    assert ids_np.ndim == 1
    assert ids_np.flags.c_contiguous
    assert rects_np.flags.c_contiguous


def test_query_np_alignment_of_ids_and_rects():
    rects = np.array(
        [
            [10, 10, 20, 20],
            [30, 30, 40, 40],
            [50, 50, 60, 60],
            [70, 70, 80, 80],
        ],
        dtype=np.float32,
    )
    rqt = _mk_rects_f32(rects)
    q = (0, 0, 100, 100)
    ids_np, rects_np = rqt.query_np(q)

    ids_np, rects_np = _sorted_by_id(ids_np, rects_np)
    assert np.all(ids_np == np.arange(rects.shape[0], dtype=np.uint64))
    assert np.allclose(rects_np, rects)


def test_query_np_mutation_does_not_affect_tree_state():
    rects = np.array(
        [
            [10, 10, 20, 20],
            [30, 30, 40, 40],
            [50, 50, 60, 60],
        ],
        dtype=np.float32,
    )
    rqt = _mk_rects_f32(rects)
    q = (0, 0, 100, 100)
    ids_np, rects_np = rqt.query_np(q)
    orig_ids = ids_np.copy()
    orig_rects = rects_np.copy()

    # Mutate the returned arrays
    rects_np[:] = -123.45
    ids_np[:] = 999

    # Re-query, tree should be unchanged
    ids2, rects2 = rqt.query_np(q)
    ids2, rects2 = _sorted_by_id(ids2, rects2)
    assert np.allclose(rects2, rects)
    assert np.all(ids2 == np.arange(rects.shape[0], dtype=np.uint64))

    assert rects2.shape == (3, 4)

    # The original snapshots still hold their old values
    assert np.allclose(
        orig_rects[_sorted_by_id(ids_np.copy(), orig_rects)[0].argsort()], orig_rects
    )
    assert np.all(
        orig_ids[_sorted_by_id(ids_np.copy(), orig_rects)[0].argsort()] == orig_ids
    )


def test_query_np_boundary_inclusion_float_eps():
    eps = 1e-6
    rects = np.array(
        [
            [10, 10, 20, 20],
        ],
        dtype=np.float32,
    )
    rqt = _mk_rects_f32(rects)
    # Tight query box that clearly overlaps
    ids_np, rects_np = rqt.query_np((20 - eps, 20 - eps, 30, 30))
    assert ids_np.size == 1
    assert np.allclose(rects_np[0], [10, 10, 20, 20])


def test_query_np_large_random_matches_naive_intersection():
    rng = np.random.default_rng(42)
    n = 15000
    centers = rng.uniform(0, 100, size=(n, 2)).astype(np.float32)
    # Keep small rects to limit cross partition overlap
    wh = rng.uniform(0.05, 0.5, size=(n, 2)).astype(np.float32)
    rects = np.column_stack(
        [
            centers[:, 0] - wh[:, 0],
            centers[:, 1] - wh[:, 1],
            centers[:, 0] + wh[:, 0],
            centers[:, 1] + wh[:, 1],
        ]
    )
    rqt = _mk_rects_f32(rects, capacity=8)

    q = (12.5, 37.5, 76.25, 88.0)
    ids_np, rects_np = rqt.query_np(q)

    # Naive filter: any rectangle that intersects q
    mask = np.array([_intersects(r, q) for r in rects.tolist()], dtype=bool)
    exp_ids = np.flatnonzero(mask).astype(np.uint64)
    exp_rects = rects[mask]

    got_ids, got_rects = _ids_rects_to_sets(ids_np, rects_np)
    exp_ids_set = set(exp_ids.tolist())
    exp_rects_set = {tuple(map(float, r)) for r in exp_rects.tolist()}

    assert got_ids == exp_ids_set
    assert got_rects == exp_rects_set


@pytest.mark.parametrize(
    "parts",
    [
        # Two disjoint halves that cover the space
        ((0, 0, 50, 100), (50, 0, 100, 100)),
        ((0, 0, 100, 50), (0, 50, 100, 100)),
    ],
)
def test_query_np_additivity_on_disjoint_queries(parts):
    rng = np.random.default_rng(7)
    # Generate rects that are narrow and placed fully inside one half
    left_centers = rng.uniform(5, 45, size=(2000, 2)).astype(np.float32)
    right_centers = rng.uniform(55, 95, size=(2000, 2)).astype(np.float32)
    wh = rng.uniform(0.05, 0.5, size=(4000, 2)).astype(np.float32)

    left_rects = np.column_stack(
        [
            left_centers[:, 0] - wh[:2000, 0],
            left_centers[:, 1] - wh[:2000, 1],
            left_centers[:, 0] + wh[:2000, 0],
            left_centers[:, 1] + wh[:2000, 1],
        ]
    )
    right_rects = np.column_stack(
        [
            right_centers[:, 0] - wh[2000:, 0],
            right_centers[:, 1] - wh[2000:, 1],
            right_centers[:, 0] + wh[2000:, 0],
            right_centers[:, 1] + wh[2000:, 1],
        ]
    )
    rects = np.vstack([left_rects, right_rects])

    rqt = _mk_rects_f32(rects)

    ids_all, _ = rqt.query_np((0, 0, 100, 100))
    ids_a, _ = rqt.query_np(parts[0])
    ids_b, _ = rqt.query_np(parts[1])

    # Since we constructed rects to sit entirely in one half, counts add up
    assert ids_a.size + ids_b.size == ids_all.size

    # Sets should union to the global set
    set_all = set(ids_all.tolist())
    assert set(ids_a.tolist()).union(set(ids_b.tolist())) == set_all


def test_query_np_dtype_variants_f64():
    rects = np.array([[1.25, 2.5, 3.75, 6.0], [3.0, 4.0, 6.25, 6.5]], dtype=np.float64)
    rqt = RectQuadTree(BOUNDS, capacity=4, dtype="f64")
    rqt.insert_many(rects)
    ids_np, rects_np = rqt.query_np((0, 0, 10, 10))
    assert rects_np.dtype == np.float64
    ids_np, rects_np = _sorted_by_id(ids_np, rects_np)
    assert np.allclose(rects_np, rects)


def test_query_np_dtype_variants_i32():
    rects = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int32)
    rqt = RectQuadTree(BOUNDS, capacity=4, dtype="i32")
    rqt.insert_many(rects)
    ids_np, rects_np = rqt.query_np((0, 0, 100, 100))
    assert rects_np.dtype == np.int32
    ids_np, rects_np = _sorted_by_id(ids_np, rects_np)
    assert np.array_equal(rects_np, rects)


def test_query_np_dtype_variants_i64():
    rects = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int64)
    rqt = RectQuadTree(BOUNDS, capacity=4, dtype="i64")
    rqt.insert_many(rects)
    ids_np, rects_np = rqt.query_np((0, 0, 100, 100))
    assert rects_np.dtype == np.int64
    ids_np, rects_np = _sorted_by_id(ids_np, rects_np)
    assert np.array_equal(rects_np, rects)
