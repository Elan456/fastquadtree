#!/usr/bin/env python3
"""Test unconventional quadtree bounding boxes including negative regions."""

from fastquadtree import QuadTree, RectQuadTree


def test_serialization_point_quadtree():
    """Test serialization and deserialization of PointQuadTree."""
    qt = QuadTree((0, 0, 100, 100), capacity=4)
    points = [(10, 10), (20, 20), (30, 30), (40, 40)]
    for pt in points:
        qt.insert(pt)

    data = qt.to_bytes()
    qt2 = QuadTree.from_bytes(data)

    assert qt.count_items() == qt2.count_items()
    for rect in [(0, 0, 50, 50), (15, 15, 35, 35)]:
        res1 = qt.query(rect)
        res2 = qt2.query(rect)
        assert sorted(res1) == sorted(res2)


def test_serialization_rect_quadtree():
    """Test serialization and deserialization of RectQuadTree."""
    rqt = RectQuadTree((0, 0, 100, 100), capacity=4)
    rects = [(5, 5, 15, 15), (20, 20, 30, 30), (35, 35, 45, 45), (50, 50, 60, 60)]
    for rect in rects:
        rqt.insert(rect)

    data = rqt.to_bytes()
    rqt2 = RectQuadTree.from_bytes(data)

    assert rqt.count_items() == rqt2.count_items()
    for query_rect in [(0, 0, 25, 25), (30, 30, 55, 55)]:
        res1 = rqt.query(query_rect)
        res2 = rqt2.query(query_rect)
        assert sorted(res1) == sorted(res2)


def test_serialization_with_objects_point():
    """Test serialization of quadtree with associated objects."""
    qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=True)
    items = [((10, 10), "A"), ((20, 20), "B"), ((30, 30), "C")]
    for pt, obj in items:
        qt.insert(pt, obj=obj)

    data = qt.to_bytes()
    qt2 = QuadTree.from_bytes(data)

    assert qt.count_items() == qt2.count_items()

    # Check that the objects in the object store are preserved
    all_qt2_items = [item.to_dict() for item in qt2.get_all_items()]
    all_qt_items = [item.to_dict() for item in qt.get_all_items()]

    for item in all_qt_items:
        assert item in all_qt2_items

    assert type(qt2.get_all_items()[0]) is type(qt.get_all_items()[0])
    assert type(qt2.get_all_items()[0].obj) is type(qt.get_all_items()[0].obj)


def test_serialization_with_objects_rect():
    """Test serialization of rect quadtree with associated objects."""
    rqt = RectQuadTree((0, 0, 100, 100), capacity=4, track_objects=True)
    items = [
        ((5, 5, 15, 15), "RectA"),
        ((20, 20, 30, 30), "RectB"),
        ((35, 35, 45, 45), "RectC"),
    ]
    for rect, obj in items:
        rqt.insert(rect, obj=obj)

    data = rqt.to_bytes()
    rqt2 = RectQuadTree.from_bytes(data)

    assert rqt.count_items() == rqt2.count_items()

    # Check that the objects in the object store are preserved
    all_rqt2_items = [item.to_dict() for item in rqt2.get_all_items()]
    all_rqt_items = [item.to_dict() for item in rqt.get_all_items()]

    for item in all_rqt_items:
        assert item in all_rqt2_items

    assert type(rqt2.get_all_items()[0]) is type(rqt.get_all_items()[0])
    assert type(rqt2.get_all_items()[0].obj) is type(rqt.get_all_items()[0].obj)


def test_serialization_perserves_ids():
    """Test that serialization and deserialization preserves item ids."""
    qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=True)
    items = [((10, 10), "A"), ((20, 20), "B"), ((30, 30), "C")]
    for pt, obj in items:
        qt.insert(pt, obj=obj)

    original_ids = [item.id_ for item in qt.get_all_items()]

    data = qt.to_bytes()
    qt2 = QuadTree.from_bytes(data)

    deserialized_ids = [item.id_ for item in qt2.get_all_items()]

    assert sorted(original_ids) == sorted(deserialized_ids)

    # Delete id 1
    qt.delete(1, (20, 20))
    ids_after_delete = [item.id_ for item in qt.get_all_items()]

    qt3 = QuadTree.from_bytes(qt.to_bytes())
    ids_after_delete_deserialized = [item.id_ for item in qt3.get_all_items()]
    assert sorted(ids_after_delete) == [0, 2]
    assert sorted(ids_after_delete_deserialized) == [0, 2]


def test_serialization_reconstructs_freelist():
    """Test that deserialization properly reconstructs the free-list from holes.

    This ensures that deserialized trees reuse deleted IDs just like the original tree.
    """
    qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=True)

    # Insert three items with IDs 0, 1, 2
    qt.insert((10, 10), obj="A")
    qt.insert((20, 20), obj="B")
    qt.insert((30, 30), obj="C")

    # Delete ID 1, which should add it to the free-list
    qt.delete(1, (20, 20))

    # Verify state: items [0, 2], ID 1 is free
    ids_after_delete = sorted([item.id_ for item in qt.get_all_items()])
    assert ids_after_delete == [0, 2]

    # Insert new item - should reuse ID 1
    new_id_before_serialization = qt.insert((40, 40), obj="D")
    assert new_id_before_serialization == 1, "Original tree should reuse deleted ID 1"

    # Now delete ID 1 again for serialization test
    qt.delete(1, (40, 40))

    # Serialize and deserialize
    data = qt.to_bytes()
    qt_deserialized = QuadTree.from_bytes(data)

    # Verify IDs are preserved
    ids_deserialized = sorted([item.id_ for item in qt_deserialized.get_all_items()])
    assert ids_deserialized == [0, 2]

    # CRITICAL TEST: Insert into deserialized tree should reuse ID 1
    new_id_after_deserialization = qt_deserialized.insert((50, 50), obj="E")
    assert new_id_after_deserialization == 1, (
        "Deserialized tree should reuse deleted ID 1, not append at ID 3. "
        "Free-list should be reconstructed from None holes during deserialization."
    )

    # Verify final state
    final_ids = sorted([item.id_ for item in qt_deserialized.get_all_items()])
    assert final_ids == [0, 1, 2], "Should have items at IDs 0, 1 (reused), and 2"
