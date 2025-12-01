#!/usr/bin/env python3
"""Test delete_by_object functionality with bimap implementation."""

import pytest

from fastquadtree import QuadTree


def test_delete_by_object_basic():
    """Test basic delete_by_object functionality."""
    qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=True)

    # Create some test objects
    obj1 = {"name": "point1", "data": "test1"}
    obj2 = {"name": "point2", "data": "test2"}
    obj3 = {"name": "point3", "data": "test3"}

    # Insert objects
    id1 = qt.insert((10, 10), obj=obj1)
    id2 = qt.insert((20, 20), obj=obj2)
    id3 = qt.insert((30, 30), obj=obj3)

    assert qt.count_items() == 3
    assert qt.get(id1) is obj1
    assert qt.get(id2) is obj2
    assert qt.get(id3) is obj3

    # Delete by object
    result = qt.delete_by_object(obj2)
    assert result is True
    assert qt.count_items() == 2
    assert qt.get(id2) is None

    # Verify other objects still exist
    assert qt.get(id1) is obj1
    assert qt.get(id3) is obj3


def test_delete_by_object_not_found():
    """Test delete_by_object when object is not tracked."""
    qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=True)

    obj1 = {"name": "point1"}
    obj2 = {"name": "point2"}  # Not inserted

    qt.insert((10, 10), obj=obj1)

    # Try to delete object that was never inserted
    result = qt.delete_by_object(obj2)
    assert result is False
    assert qt.count_items() == 1


def test_delete_by_object_without_tracking():
    """Test delete_by_object raises error when track_objects=False."""
    qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=False)

    obj1 = {"name": "point1"}

    with pytest.raises(
        ValueError, match="Cannot delete by object when track_objects=False"
    ):
        qt.delete_by_object(obj1)


def test_delete_by_object_multiple_same_location():
    """Test delete_by_object when multiple objects are at same location."""
    qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=True)

    obj1 = {"name": "point1"}
    obj2 = {"name": "point2"}
    obj3 = {"name": "point3"}

    # Insert multiple objects at same location
    id1 = qt.insert((10, 10), obj=obj1)
    id2 = qt.insert((10, 10), obj=obj2)
    id3 = qt.insert((10, 10), obj=obj3)

    assert qt.count_items() == 3

    # Delete specific object by reference
    result = qt.delete_by_object(obj2)
    assert result is True
    assert qt.count_items() == 2
    assert qt.get(id2) is None

    # Other objects should still exist
    assert qt.get(id1) is obj1
    assert qt.get(id3) is obj3


def test_delete_by_object_performance():
    """Test that delete_by_object is fast (O(1) lookup)."""
    qt = QuadTree((0, 0, 1000, 1000), capacity=4, track_objects=True)

    # Insert many objects
    objects = []
    for i in range(1000):
        obj = {"id": i, "data": f"object_{i}"}
        objects.append(obj)
        qt.insert((i % 100, i // 100), obj=obj)

    # Delete objects - should be fast
    target_obj = objects[500]
    result = qt.delete_by_object(target_obj)
    assert result is True
    assert qt.count_items() == 999


def test_delete_by_object_with_attach():
    """Test delete_by_object works with objects added via attach."""
    qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=True)

    obj1 = {"name": "attached_object"}

    # Insert without object, then attach
    id1 = qt.insert((10, 10))
    qt.attach(id1, obj1)

    assert qt.get(id1) is obj1

    # Delete by object
    result = qt.delete_by_object(obj1)
    assert result is True
    assert qt.count_items() == 0
    assert qt.get(id1) is None


def test_delete_by_object_replace_then_delete():
    """Test delete_by_object after replacing an object."""
    qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=True)

    obj1 = {"name": "original"}
    obj2 = {"name": "replacement"}

    # Insert with original object
    id1 = qt.insert((10, 10), obj=obj1)

    # Replace object for same ID
    qt.attach(id1, obj2)

    # Original object should no longer be tracked
    result1 = qt.delete_by_object(obj1)
    assert result1 is False

    # New object should be deletable
    result2 = qt.delete_by_object(obj2)
    assert result2 is True
    assert qt.count_items() == 0


def test_delete_by_object_edge_cases():
    """Test edge cases like None objects and empty quadtree."""
    qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=True)

    # Test with empty quadtree
    obj1 = {"name": "test"}
    result = qt.delete_by_object(obj1)
    assert result is False

    # Test inserting with None object (should not be tracked)
    id1 = qt.insert((10, 10), obj=None)
    assert qt.get(id1) is None

    # Test object identity (different dicts with same content)
    obj2 = {"name": "test"}
    obj3 = {"name": "test"}  # Same content, different object

    qt.insert((20, 20), obj=obj2)
    result = qt.delete_by_object(obj3)
    assert result is False  # Different object identity
