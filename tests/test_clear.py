#!/usr/bin/env python3
import pytest

from fastquadtree import QuadTree


def make_tree(track=True, with_max_depth=False):
    kwargs = {"track_objects": track}
    if with_max_depth:
        kwargs["max_depth"] = 8  # pyright: ignore[reportArgumentType]
    return QuadTree((0, 0, 100, 100), capacity=4, **kwargs)


def test_clear_preserves_wrapper_identity_and_swaps_native():
    qt = make_tree(track=True)
    # Insert a couple to change internal state
    a = qt.insert((10, 10), obj={"a": 1})
    b = qt.insert((20, 20), obj={"b": 2})
    assert qt.count_items() == 2
    assert len(qt) == 2

    # Capture Python object identity and native handle
    wrapper_id_before = id(qt)
    native_before = getattr(qt, "_native", None)
    assert native_before is not None  # we expect the private attr to exist

    qt.clear()  # default reset_ids=False

    # Same Python wrapper object
    assert id(qt) == wrapper_id_before

    # New native instance was created
    native_after = getattr(qt, "_native", None)
    assert native_after is not None
    assert native_after is not native_before

    # Emptied state
    assert qt.count_items() == 0
    assert len(qt) == 0

    # Tracker emptied
    assert qt.get(a) is None
    assert qt.get(b) is None


def test_clear_reset_ids_behavior_false_then_true():
    qt = make_tree(track=True)

    # Insert 3 items, auto ids should be 1,2,3
    id1 = qt.insert((1, 1), obj="x")
    id2 = qt.insert((2, 2), obj="y")
    id3 = qt.insert((3, 3), obj="z")
    assert [id1, id2, id3] == [0, 1, 2]

    qt.clear()
    id4 = qt.insert((4, 4), obj="w")
    assert id4 == 0
    assert qt.count_items() == 1
    assert len(qt) == 1

    # Now clear and reset ids back to 0 again
    qt.clear()
    id1_again = qt.insert((10, 10), obj="again")
    assert id1_again == 0
    assert qt.count_items() == 1
    assert len(qt) == 1


@pytest.mark.parametrize("with_max_depth", [False, True])
@pytest.mark.parametrize("track_objects", [False, True])
def test_clear_keeps_config_and_bounds(with_max_depth, track_objects):
    qt = make_tree(track=track_objects, with_max_depth=with_max_depth)

    # Before clear, an out-of-bounds insert should fail the same way as after clear
    with pytest.raises(ValueError, match="outside bounds"):
        qt.insert((200, 200))

    # Insert an in-bounds point, then clear
    qt.insert((50, 50), obj="ok") if track_objects else qt.insert((50, 50))
    assert qt.count_items() == 1

    qt.clear()

    # After clear, bounds and capacity are preserved, so the same checks apply
    with pytest.raises(ValueError, match="outside bounds"):
        qt.insert((200, 200))
    ok2 = qt.insert((51, 51))  # still in-bounds
    assert ok2 >= 0
    assert qt.count_items() == 1


def test_clear_with_multiple_references_observe_empty_state():
    qt = make_tree(track=True)
    alias = qt  # another reference to the same wrapper

    qt.insert((10, 10), obj="a")
    qt.insert((20, 20), obj="b")
    assert alias.count_items() == 2

    # Clear through one reference
    alias.clear()

    # The other reference sees the emptied state because it is the same object
    assert qt.count_items() == 0
    assert len(qt) == 0

    # Can insert again after clear
    nid = qt.insert((30, 30), obj="c")
    assert nid >= 0
    assert alias.count_items() == 1


def test_clear_syncs_tracker_and_queries():
    qt = make_tree(track=True)

    ids = [qt.insert((x, x), obj={"i": x}) for x in (5, 15, 25)]
    assert qt.count_items() == 3

    # Sanity check query returns all three
    hits = qt.query((0, 0, 100, 100))
    assert sorted(h[0] for h in hits) == sorted(ids)

    qt.clear()

    # Queries are now empty and tracker has no objects
    hits_after = qt.query((0, 0, 100, 100))
    assert hits_after == []
    for i in ids:
        assert qt.get(i) is None
