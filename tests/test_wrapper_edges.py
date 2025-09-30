import re
import pytest

from quadtree_rs import QuadTree, Item

BOUNDS = (0.0, 0.0, 1000.0, 1000.0)


def test_insert_many_seeds_items_and_query_as_items_round_trip():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    n = qt.insert_many_points([(10, 10), (20, 20), (30, 30)])
    assert n == 3

    raw = qt.query((0, 0, 40, 40), as_items=False)
    its = qt.query((0, 0, 40, 40), as_items=True)

    assert len(raw) == len(its) == 3
    # ids and positions match
    m_raw = {t[0]: (t[1], t[2]) for t in raw}
    for it in its:
        assert isinstance(it, Item)
        assert (it.x, it.y) == m_raw[it.id]


def test_delete_returns_native_result_even_if_bimap_missing():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    id_ = qt.insert((50, 50))
    # remove bimap entry to simulate drift
    qt._items.pop_id(id_)  # type: ignore[attr-defined]

    assert qt.delete(id_, (50, 50)) is True
    assert qt.count_items() == 0
    assert len(qt) == 0  # wrapper counter


def test_delete_by_object_uses_cached_coords_and_updates_counts():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    obj = {"name": "slime"}
    id_ = qt.insert((123, 456), obj=obj)

    assert qt.count_items() == 1
    assert len(qt) == 1

    assert qt.delete_by_object(obj) is True
    assert qt.count_items() == 0
    assert len(qt) == 0


def test_bounds_error_message_includes_point_and_bounds():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=False)
    with pytest.raises(
        ValueError, match=r"Point \([^)]*\) is outside bounds \([^)]*\)"
    ):
        qt.insert((1500, -10))


def test_nearest_neighbors_as_items_work_when_items_are_seeded():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    # Use wrapper inserts so BiMap is populated
    ids = [qt.insert((x, x)) for x in (100, 200, 300)]
    raw = qt.nearest_neighbors((190, 190), 2, as_items=False)
    its = qt.nearest_neighbors((190, 190), 2, as_items=True)

    assert len(raw) == len(its) == 2
    raw_ids = [t[0] for t in raw]
    item_ids = [it.id for it in its]
    assert raw_ids == item_ids


def test_query_as_items_does_not_mutate_bimap_when_inserts_are_wrapped():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    ids = [qt.insert((10, 10)), qt.insert((20, 20)), qt.insert((30, 30))]
    # Snapshot of Item object identities in the BiMap
    before = {i: qt._items.by_id(i) for i in ids}  # type: ignore[attr-defined]
    its = qt.query((0, 0, 40, 40), as_items=True)
    after = {i: qt._items.by_id(i) for i in ids}  # type: ignore[attr-defined]
    # Items are the same objects. Query did not create new Items.
    assert [it.id for it in its] == ids
    assert before == after
    for i in ids:
        assert before[i] is after[i]


def test_nearest_neighbor_as_item_requires_seeded_items():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    qt.insert((100, 100))
    got = qt.nearest_neighbor((101, 101), as_item=False)
    it = qt.nearest_neighbor((101, 101), as_item=True)
    assert it is not None
    assert (it.id, it.x, it.y) == got


def test_invariant_violation_raises_on_query_as_items_when_native_bypassed():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    # Bypass wrapper to simulate a bug or misuse
    assert qt._native.insert(42, (500, 500))  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        qt.query((400, 400, 600, 600), as_items=True)


def test_invariant_violation_raises_on_nearest_as_item_when_native_bypassed():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    assert qt._native.insert(77, (200, 200))  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        qt.nearest_neighbor((201, 201), as_item=True)
