import random

import pyqtree
import pytest

# Import your shim
from fastquadtree.pyqtree import Index as FQTIndex

WORLD = (0.0, 0.0, 100.0, 100.0)

EPS = 1e-6  # Floating point tolerance for edge cases


def rand_rect(rng, world=WORLD, min_size=1.0, max_size=20.0):
    x1 = rng.uniform(world[0], world[2] - min_size)
    y1 = rng.uniform(world[1], world[3] - min_size)
    w = rng.uniform(min_size, max_size)
    h = rng.uniform(min_size, max_size)
    x2 = min(world[2], x1 + w)
    y2 = min(world[3], y1 + h)
    return (x1, y1, x2, y2)


def build_indices(items, ctor="bbox"):
    """
    items: list[tuple[item_obj, (xmin, ymin, xmax, ymax)]]
    ctor: "bbox" or "xywh"
    """
    if ctor == "bbox":
        fqt = FQTIndex(bbox=WORLD)
        pyq = pyqtree.Index(bbox=WORLD)
    elif ctor == "xywh":
        x = (WORLD[0] + WORLD[2]) / 2.0
        y = (WORLD[1] + WORLD[3]) / 2.0
        w = WORLD[2] - WORLD[0]
        h = WORLD[3] - WORLD[1]
        fqt = FQTIndex(x=x, y=y, width=w, height=h)
        pyq = pyqtree.Index(x=x, y=y, width=w, height=h)
    else:
        raise ValueError("bad ctor")

    for obj, box in items:
        # Both APIs are item, bbox
        ret1 = fqt.insert(obj, box)
        ret2 = pyq.insert(obj, box)
        # pyqtree returns None, so enforce parity
        assert ret1 is None
        assert ret2 is None

    return fqt, pyq


def results_match_exact(fqt, pyq, query):
    """Compare lists exactly, not just as sets."""
    got_fqt = sorted(fqt.intersect(query))
    got_pyq = sorted(pyq.intersect(query))
    assert (
        got_fqt == got_pyq
    ), f"\nquery={query}\nfastquadtree={got_fqt}\npyqtree={got_pyq}"


def test_ctor_error_branch():
    # Exercise the constructor error path for 100% coverage
    with pytest.raises(ValueError):
        FQTIndex()  # neither bbox nor x,y,width,height


@pytest.mark.parametrize("ctor", ["bbox", "xywh"])
def test_basic_insert_intersect_remove_matches_pyqtree(ctor):
    rng = random.Random(123)
    # Make a small deterministic dataset with some overlaps and some isolated
    items = [(name, rand_rect(rng)) for name in ["a", "b", "c", "d", "e", "f", "g"]]

    fqt, pyq = build_indices(items, ctor=ctor)

    # Queries that hit various cases
    queries = [
        (0, 0, 1, 1),  # miss everything
        (10, 10, 90, 90),  # broad overlap
        items[0][1],  # exactly the first item's bbox
        items[-1][1],  # exactly the last item's bbox
        (25, 25, 26, 26),  # tiny box
        (0, 0, 100, 100),  # world box
    ]

    for q in queries:
        results_match_exact(fqt, pyq, q)

    # Remove two items and recheck
    to_remove = [items[1], items[4]]  # remove b and e
    for obj, box in to_remove:
        fqt.remove(obj, box)
        pyq.remove(obj, box)

    # After removal, both should match on the same queries
    for q in queries:
        results_match_exact(fqt, pyq, q)

    # Also check that removed objects are truly gone
    for obj, box in to_remove:
        assert obj not in fqt.intersect(box)
        assert obj not in pyq.intersect(box)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_randomized_equivalence_many_queries(seed):
    rng = random.Random(seed)
    # More items to stress traversal order
    items = [(f"obj_{i}", rand_rect(rng)) for i in range(50)]
    fqt, pyq = build_indices(items, ctor="bbox")

    # 30 random queries
    queries = [rand_rect(rng, max_size=40.0) for _ in range(30)]
    for q in queries:
        results_match_exact(fqt, pyq, q)


def test_order_is_identical_to_pyqtree_for_same_insert_order():
    """
    pyqtree does not document result ordering, but many users
    implicitly depend on the current behavior. This test locks
    your shim to whatever pyqtree returns.
    """
    # Crafted rectangles that overlap in a chainy way
    items = [
        ("one", (10, 10, 30, 30)),
        ("two", (20, 20, 40, 40)),
        ("three", (15, 15, 35, 35)),
        ("four", (12, 12, 18, 18)),
        ("five", (28, 28, 45, 45)),
    ]
    fqt, pyq = build_indices(items, ctor="bbox")
    q = (14, 14, 29, 29)
    # Compare lists directly for strict equality
    assert fqt.intersect(q) == pyq.intersect(q)


def test_insert_and_remove_return_none_and_accepts_any_object():
    # Mixed object types as items
    items = [
        ({"id": 1}, (5, 5, 10, 10)),
        (("tuple", 2), (8, 8, 12, 12)),
        (42, (0, 0, 3, 3)),
        ("str", (1, 1, 2, 2)),
    ]
    fqt, pyq = build_indices(items, ctor="bbox")

    # Both insert already asserted to return None inside build_indices
    # Now remove and assert None as well
    for obj, box in items:
        assert fqt.remove(obj, box) is None
        assert pyq.remove(obj, box) is None

    # All gone now
    assert fqt.intersect((0, 0, 100, 100)) == []
    assert pyq.intersect((0, 0, 100, 100)) == []


def _rect(x1, y1, x2, y2):
    return (x1, y1, x2, y2)


def test_free_slot_reuse_single_and_lifo():
    idx = FQTIndex(bbox=WORLD)

    # Insert three distinct items at non-overlapping places
    a, b, c = "a", "b", "c"
    ra = _rect(0, 0, 10, 10)
    rb = _rect(20, 20, 30, 30)
    rc = _rect(40, 40, 50, 50)

    assert idx.insert(a, ra) is None
    assert idx.insert(b, rb) is None
    assert idx.insert(c, rc) is None

    # RIDs are dense: 0, 1, 2
    rid_a = idx._item_to_id[id(a)]
    rid_b = idx._item_to_id[id(b)]
    rid_c = idx._item_to_id[id(c)]
    assert (rid_a, rid_b, rid_c) == (0, 1, 2)

    # Remove c then a to create two free slots; free should be [2, 0]
    assert idx.remove(c, rc) is None
    assert idx.remove(a, ra) is None
    assert idx._objects[rid_c] is None
    assert idx._objects[rid_a] is None
    assert idx._free == [rid_c, rid_a]

    before_len = len(idx._objects)

    # Insert x. It should reuse last freed slot (LIFO): rid_a
    x, rx = "x", _rect(60, 60, 70, 70)
    assert idx.insert(x, rx) is None
    rid_x = idx._item_to_id[id(x)]
    assert rid_x == rid_a
    assert len(idx._objects) == before_len

    # Insert y. It should reuse the next free slot: rid_c
    y, ry = "y", _rect(80, 80, 90, 90)
    assert idx.insert(y, ry) is None
    rid_y = idx._item_to_id[id(y)]
    assert rid_y == rid_c
    assert len(idx._objects) == before_len

    # Removed items do not appear; new items do
    assert a not in idx.intersect(ra)
    assert c not in idx.intersect(rc)
    assert x in idx.intersect(rx)
    assert y in idx.intersect(ry)

    # Free list consumed
    assert idx._free == []


def test_free_slot_reuse_no_growth_under_churn():
    rng = random.Random(123)
    idx = FQTIndex(bbox=WORLD)

    # Insert N items
    n = 200
    items = [f"obj_{i}" for i in range(n)]
    boxes = [
        (_x := rng.uniform(0, 90), _y := rng.uniform(0, 90), _x + 5, _y + 5)
        for _ in range(n)
    ]
    for obj, box in zip(items, boxes):
        idx.insert(obj, box)

    base_len = len(idx._objects)

    # Remove half of them
    removed = []
    for obj, box in zip(items[::2], boxes[::2]):
        idx.remove(obj, box)
        removed.append((obj, box))

    # Reinsert the same count of new items; length should not grow
    for k in range(len(removed)):
        obj = f"new_{k}"
        # use different boxes to ensure spatial removal did not leave stale entries
        x = rng.uniform(0, 90)
        y = rng.uniform(0, 90)
        box = (x, y, x + 3, y + 3)
        idx.insert(obj, box)

    assert len(idx._objects) == base_len

    # None of the removed items should be found
    for obj, box in removed:
        assert obj not in idx.intersect(box)


def _boxes_touching_edges_and_corners():
    # World is (0,0,100,100). Partition lines around 50 are common split lines.
    return [
        ("left_edge", (0.0, 10.0, 5.0, 20.0)),  # touches world min-x
        ("right_edge", (95.0, 10.0, 100.0, 20.0)),  # touches world max-x
        ("bottom_edge", (10.0, 0.0, 20.0, 5.0)),  # touches world min-y
        ("top_edge", (10.0, 95.0, 20.0, 100.0)),  # touches world max-y
        ("bottom_left_pt", (0.0, 0.0, 5.0, 5.0)),  # corner touch
        ("top_right_pt", (95.0, 95.0, 100.0, 100.0)),
        # Straddle the vertical split line x=50 with tiny thickness
        ("straddle_x50", (50.0 - EPS, 40.0, 50.0 + EPS, 60.0)),
        # Straddle the horizontal split line y=50 with tiny thickness
        ("straddle_y50", (40.0, 50.0 - EPS, 60.0, 50.0 + EPS)),
        # Very thin but > 0 width and height
        ("thin_horizontal", (20.0, 33.333, 80.0, 33.333 + 1e-5)),
        ("thin_vertical", (33.333, 20.0, 33.333 + 1e-5, 80.0)),
        # Boxes that just touch each other at an edge or corner
        ("touch_A", (30.0, 30.0, 40.0, 40.0)),
        ("touch_B", (40.0, 30.0, 50.0, 40.0)),  # shares an edge with A
        ("touch_C", (40.0, 40.0, 50.0, 50.0)),  # touches B at one corner
    ]


def _queries_covering_touch_cases():
    return [
        (0.0, 0.0, 100.0, 100.0),  # world
        (0.0, 10.0, 5.0, 20.0),  # exact edge box
        (95.0, 10.0, 100.0, 20.0),
        (10.0, 0.0, 20.0, 5.0),
        (10.0, 95.0, 20.0, 100.0),
        (30.0, 30.0, 50.0, 40.0),  # spans touch_A and touch_B shared edge
        (39.9999, 39.9999, 40.0001, 40.0001),  # tiny around touching corner
        (50.0 - 2 * EPS, 49.0, 50.0 + 2 * EPS, 51.0),  # around x=50 straddle
        (49.0, 50.0 - 2 * EPS, 51.0, 50.0 + 2 * EPS),  # around y=50 straddle
        (20.0, 33.333 - 1e-4, 80.0, 33.333 + 1e-4),  # thin horizontal
        (33.333 - 1e-4, 20.0, 33.333 + 1e-4, 80.0),  # thin vertical
    ]


@pytest.mark.parametrize("ctor", ["bbox", "xywh"])
def test_edge_and_boundary_semantics_match_pyqtree(ctor):
    items = _boxes_touching_edges_and_corners()
    fqt, pyq = build_indices(items, ctor=ctor)

    # Check that every crafted query matches exactly
    for q in _queries_covering_touch_cases():
        results_match_exact(fqt, pyq, q)

    # Add a few more probes around the partition lines to stress boundary math
    for dx in (-EPS, 0.0, EPS):
        for dy in (-EPS, 0.0, EPS):
            q = (50.0 + dx - 1.0, 50.0 + dy - 1.0, 50.0 + dx + 1.0, 50.0 + dy + 1.0)
            results_match_exact(fqt, pyq, q)


def test_dense_targets_along_partition_lines_match_pyqtree():
    """
    Insert many small rectangles centered along x=50 and y=50 to stress
    splitting and equality on boundary math.
    """
    objs = []
    boxes = []

    # 40 tiny boxes along x=50 at different y
    for i in range(40):
        y = 2.0 + i * 2.4  # spread across the world
        boxes.append((50.0 - 0.25, y - 0.25, 50.0 + 0.25, y + 0.25))
        objs.append(f"vx_{i}")

    # 40 tiny boxes along y=50 at different x
    for i in range(40):
        x = 2.0 + i * 2.4
        boxes.append((x - 0.25, 50.0 - 0.25, x + 0.25, 50.0 + 0.25))
        objs.append(f"hy_{i}")

    items = list(zip(objs, boxes))
    fqt, pyq = build_indices(items, ctor="bbox")

    # Probe a grid of queries around the center to catch any off by epsilon
    for cx in (49.5, 50.0, 50.5):
        for cy in (49.5, 50.0, 50.5):
            q = (cx - 1.0, cy - 1.0, cx + 1.0, cy + 1.0)
            results_match_exact(fqt, pyq, q)

    # Also check a sweep of thin queries that align to the lines
    thin_queries = [
        (49.9, 0.0, 50.1, 100.0),
        (0.0, 49.9, 100.0, 50.1),
        (49.999, 10.0, 50.001, 90.0),
        (10.0, 49.999, 90.0, 50.001),
    ]
    for q in thin_queries:
        results_match_exact(fqt, pyq, q)
