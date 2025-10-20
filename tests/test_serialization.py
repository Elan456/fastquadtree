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
