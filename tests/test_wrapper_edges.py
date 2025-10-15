import pytest

from fastquadtree import Item, QuadTree

BOUNDS = (0.0, 0.0, 1000.0, 1000.0)


def test_insert_many_seeds_items_and_query_as_items_round_trip():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    n = qt.insert_many([(10, 10), (20, 20), (30, 30)])
    assert n == 3

    raw = qt.query((0, 0, 40, 40), as_items=False)
    its = qt.query((0, 0, 40, 40), as_items=True)

    assert len(raw) == len(its) == 3
    # ids and positions match
    m_raw = {t[0]: (t[1], t[2]) for t in raw}
    for it in its:
        assert isinstance(it, Item)
        assert (it.x, it.y) == m_raw[it.id_]


def test_delete_returns_native_result_even_if_bimap_missing():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    id_ = qt.insert((50, 50))
    # remove bimap entry to simulate drift
    qt._store.pop_id(id_)  # type: ignore[attr-defined]

    assert qt.delete(id_, (50, 50)) is True
    assert qt.count_items() == 0
    assert len(qt) == 0  # wrapper counter


def test_delete_by_object_uses_cached_coords_and_updates_counts():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    obj = {"name": "slime"}
    qt.insert((123, 456), obj=obj)

    assert qt.count_items() == 1
    assert len(qt) == 1

    assert qt.delete_by_object(obj) is True
    assert qt.count_items() == 0
    assert len(qt) == 0


def test_bounds_error_message_includes_point_and_bounds():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=False)
    with pytest.raises(
        ValueError, match=r"Geometry \([^)]*\) is outside bounds \([^)]*\)"
    ):
        qt.insert((1500, -10))


def test_nearest_neighbors_as_items_work_when_items_are_seeded():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    # Use wrapper inserts so BiMap is populated
    [qt.insert((x, x)) for x in (100, 200, 300)]
    raw = qt.nearest_neighbors((190, 190), 2, as_items=False)
    its = qt.nearest_neighbors((190, 190), 2, as_items=True)

    assert len(raw) == len(its) == 2
    raw_ids = [t[0] for t in raw]
    item_ids = [it.id_ for it in its]
    assert raw_ids == item_ids


def test_query_as_items_does_not_mutate_bimap_when_inserts_are_wrapped():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    ids = [qt.insert((10, 10)), qt.insert((20, 20)), qt.insert((30, 30))]
    # Snapshot of Item object identities in the BiMap
    before = {i: qt._store.by_id(i) for i in ids}  # type: ignore[attr-defined]
    its = qt.query((0, 0, 40, 40), as_items=True)
    after = {i: qt._store.by_id(i) for i in ids}  # type: ignore[attr-defined]
    # Items are the same objects. Query did not create new Items.
    assert [it.id_ for it in its] == ids
    assert before == after
    for i in ids:
        assert before[i] is after[i]


def test_nearest_neighbor_as_item_requires_seeded_items():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    qt.insert((100, 100))
    got = qt.nearest_neighbor((101, 101), as_item=False)
    it = qt.nearest_neighbor((101, 101), as_item=True)
    assert it is not None
    assert (it.id_, it.x, it.y) == got


def test_gets_query_items_without_tracking():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=False, max_depth=4)
    id1 = qt.insert((10, 10))
    id2 = qt.insert((20, 20))

    with pytest.raises(ValueError):
        qt.get(id1)

    with pytest.raises(ValueError):
        qt.get(id2)

    # Get all items
    with pytest.raises(ValueError):
        qt.get_all_items()

    with pytest.raises(ValueError):
        qt.query((0, 0, 30, 30), as_items=True)

    with pytest.raises(ValueError):
        qt.nearest_neighbor((15, 15), as_item=True)

    with pytest.raises(ValueError):
        qt.nearest_neighbors((15, 15), 2, as_items=True)

    with pytest.raises(ValueError):
        # Attach
        qt.attach(id1, {"name": "point1"})

    with pytest.raises(ValueError):
        qt.get_all_objects()


def test_out_of_bounds_insert():
    qt = QuadTree((0, 0, 100, 100), capacity=4)
    with pytest.raises(ValueError):
        qt.insert((150, 50))  # x is out of bounds
    with pytest.raises(ValueError):
        qt.insert((50, -10))  # y is out of bounds
    with pytest.raises(ValueError):
        qt.insert((100, 100))  # on max edge, should be excluded

    assert len(qt.get_all_node_boundaries()) == 1

    with pytest.raises(ValueError):
        qt.get_all_objects()


def test_get_all_items_returns_tracked_items():
    qt = QuadTree(BOUNDS, capacity=8, track_objects=True)
    obj1 = {"name": "point1"}
    obj2 = {"name": "point2"}
    id1 = qt.insert((10, 10), obj=obj1)
    id2 = qt.insert((20, 20), obj=obj2)

    items = qt.get_all_items()
    assert len(items) == 2
    ids = {item.id_ for item in items}
    assert ids == {id1, id2}

    for item in items:
        if item.id_ == id1:
            assert item.obj is obj1
        elif item.id_ == id2:
            assert item.obj is obj2
        else:
            pytest.fail(f"Unexpected item ID {item.id_}")


def test_insert_many_exception_for_out_of_bounds():
    qt = QuadTree((0, 0, 100, 100), capacity=4)
    points = [(10.0, 10.0), (20.0, 20.0), (150.0, 150.0)]  # Last point is out of bounds

    with pytest.raises(ValueError):
        qt.insert_many(points)
