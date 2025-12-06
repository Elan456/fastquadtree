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


def test_non_default_dtype_insert_many():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True, dtype="f64")
    points = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float64)
    n = qt.insert_many(points)
    assert n == 3
    assert len(qt) == 3

    raw = qt.query((0, 0, 40, 40), as_items=False)

    assert len(raw) == 3
    # ids and positions match
    m_raw = {t[0]: (t[1], t[2]) for t in raw}
    for t in raw:
        assert (t[1], t[2]) == m_raw[t[0]]


def test_non_default_quadtree_dtype_with_default_numpy_dtype_raises():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True, dtype="f64")
    points = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)  # Wrong dtype
    with pytest.raises(TypeError):
        qt.insert_many(points)
    assert len(qt) == 0


def test_unspported_quadtree_dtype_insert_many_raises():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True, dtype="i32")
    points = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)  # Wrong dtype
    with pytest.raises(TypeError):
        qt.insert_many(points)
    assert len(qt) == 0

    points = np.array(
        [[10, 10], [20, 20], [30, 30]], dtype=np.uint32
    )  # unsupported dtype
    with pytest.raises(TypeError):
        qt.insert_many(points)

    # QT is also unsupported
    with pytest.raises(TypeError):
        qt = QuadTree(BOUNDS, capacity=8, track_objects=True, dtype="u32")


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


def test_query_as_np_accuracy():
    qt = QuadTree(BOUNDS, capacity=4, track_objects=True)
    points = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.float32)
    qt.insert_many(points)

    ids_np, coords_np = qt.query_np((15, 15, 35, 35))

    ids_list = ids_np.tolist()
    coords_list = coords_np.tolist()

    assert len(ids_list) == 2
    assert set(ids_list) == {1, 2}  # IDs of points (20,20) and (30,30)

    expected_coords = [(20.0, 20.0), (30.0, 30.0)]
    for coord in coords_list:
        assert tuple(coord) in expected_coords

    # Check ids
    assert set(ids_list) == {1, 2}


# -------------------------
# Additional thorough tests
# -------------------------


class _Tag:
    """Equality-by-value but distinct identity."""

    def __init__(self, v: int):
        self.v = v

    def __eq__(self, other):
        return isinstance(other, _Tag) and self.v == other.v

    def __hash__(self):
        return hash(self.v)


def test_insert_many_get_start_id_non_tracking_numpy_returns_contiguous_range():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=False)
    pts = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)

    n, start = qt.insert_many(pts, get_start_id=True)
    assert (n, start) == (3, 0)

    raw = qt.query((0, 0, 40, 40), as_items=False)
    got = sorted(t[0] for t in raw)
    assert got == list(range(start, start + n))


def test_insert_many_get_start_id_non_tracking_sequence_returns_contiguous_range():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=False)
    n, start = qt.insert_many([(10, 10), (20, 20), (30, 30)], get_start_id=True)
    assert (n, start) == (3, 0)

    raw = qt.query((0, 0, 40, 40), as_items=False)
    got = sorted(t[0] for t in raw)
    assert got == list(range(start, start + n))


def test_insert_many_get_start_id_non_tracking_start_after_prior_single_inserts():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=False)

    id0 = qt.insert((5, 5))
    id1 = qt.insert((6, 6))
    assert (id0, id1) == (0, 1)

    n, start = qt.insert_many(
        np.array([[10, 10], [20, 20]], dtype=np.float32), get_start_id=True
    )
    assert (n, start) == (2, 2)

    raw = qt.query((0, 0, 40, 40), as_items=False)
    got = sorted(t[0] for t in raw)
    assert got == [0, 1, 2, 3]


def test_insert_many_get_start_id_tracking_appends_tail_does_not_fill_holes():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)

    a = qt.insert((10, 10), obj="a")
    b = qt.insert((20, 20), obj="b")
    c = qt.insert((30, 30), obj="c")
    assert (a, b, c) == (0, 1, 2)

    # Delete id 1, leaving a hole in ObjStore
    assert qt.delete(1, (20, 20)) is True
    assert len(qt) == 2

    # Bulk insert should append at tail in your implementation (start_id == len(_arr) == 3)
    n, start = qt.insert_many([(40, 40), (50, 50)], get_start_id=True)
    assert (n, start) == (2, 3)

    raw = qt.query((0, 0, 60, 60), as_items=False)
    got_ids = sorted(t[0] for t in raw)
    assert got_ids == [0, 2, 3, 4]

    # A single insert should reuse the free-list hole (LIFO), which is id 1 here
    rid = qt.insert((60, 60), obj="d")
    assert rid == 1


def test_insert_many_get_start_id_empty_inputs_returns_tuple_and_does_not_mutate():
    # non-tracking
    qt = QuadTree(BOUNDS, capacity=8, track_objects=False)
    qt.insert((1, 1))
    assert len(qt) == 1

    # empty list
    n, start = qt.insert_many([], get_start_id=True)
    assert n == 0
    assert start == 1
    assert len(qt) == 1

    # empty numpy
    empty = np.empty((0, 2), dtype=np.float32)
    n, start = qt.insert_many(empty, get_start_id=True)
    assert (n, start) == (0, 1)
    assert len(qt) == 1

    # tracking
    qt2 = QuadTree(BOUNDS, capacity=8, track_objects=True)
    qt2.insert((1, 1), obj="x")
    assert len(qt2) == 1

    n, start = qt2.insert_many([], get_start_id=True)
    assert (n, start) == (0, 1)
    assert len(qt2) == 1

    n, start = qt2.insert_many(np.empty((0, 2), dtype=np.float32), get_start_id=True)
    assert (n, start) == (0, 1)
    assert len(qt2) == 1


def test_insert_many_get_start_id_positional_backcompat_third_arg():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=False)
    pts = np.array([[10, 10], [20, 20]], dtype=np.float32)

    # positional: (geoms, objs, get_start_id)
    n, start = qt.insert_many(pts, None, True)
    assert (n, start) == (2, 0)


def test_insert_many_objs_len_mismatch_raises_tracking():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    pts = np.array([[10, 10], [20, 20]], dtype=np.float32)

    with pytest.raises(ValueError):
        qt.insert_many(pts, objs=[{"only": "one"}])


def test_delete_by_object_is_identity_not_equality():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)

    a = _Tag(7)
    b = _Tag(7)  # equal to a, but not the same object
    assert a == b
    assert a is not b

    qt.insert((10, 10), obj=a)

    # Should not delete using a different equal object if identity semantics are used
    assert qt.delete_by_object(b) is False
    assert len(qt) == 1

    # But should delete with the original object
    assert qt.delete_by_object(a) is True
    assert len(qt) == 0


def test_attach_then_get_and_query_items_and_get_all_objects():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)

    i = qt.insert((10, 10), obj=None)
    qt.attach(i, {"k": "v"})
    assert qt.get(i) == {"k": "v"}

    items = qt.query((0, 0, 20, 20), as_items=True)
    assert len(items) == 1
    assert items[0].id_ == i
    assert items[0].obj == {"k": "v"}

    objs = qt.get_all_objects()
    assert objs == [{"k": "v"}]

    all_items = qt.get_all_items()
    assert len(all_items) == 1
    assert all_items[0].id_ == i


def test_clear_resets_ids_in_both_modes():
    # non-tracking
    qt = QuadTree(BOUNDS, capacity=8, track_objects=False)
    assert qt.insert((1, 1)) == 0
    assert qt.insert((2, 2)) == 1
    qt.clear()
    assert len(qt) == 0
    assert qt.insert((3, 3)) == 0

    # tracking
    qt2 = QuadTree(BOUNDS, capacity=8, track_objects=True)
    assert qt2.insert((1, 1), obj="a") == 0
    assert qt2.insert((2, 2), obj="b") == 1
    qt2.clear()
    assert len(qt2) == 0
    assert qt2.insert((3, 3), obj="c") == 0


def test_to_bytes_from_bytes_round_trip_preserves_objects_and_queries():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)

    pts = np.array([[10, 10], [20, 20], [30, 30]], dtype=np.float32)
    qt.insert_many(pts, objs=["a", "b", "c"])
    assert len(qt) == 3

    blob = qt.to_bytes()
    qt2 = type(qt).from_bytes(blob, dtype="f32")

    assert len(qt2) == 3
    res = qt2.query((0, 0, 40, 40), as_items=True)
    assert {it.obj for it in res} == {"a", "b", "c"}


def test_query_np_shapes_and_types_sanity():
    qt = QuadTree(BOUNDS, capacity=4, track_objects=False)
    pts = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.float32)
    qt.insert_many(pts)

    ids_np, coords_np = qt.query_np((0, 0, 100, 100))

    assert ids_np.ndim == 1
    assert coords_np.ndim == 2
    assert coords_np.shape[1] == 2
    assert ids_np.shape[0] == coords_np.shape[0]
