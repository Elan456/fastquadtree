#!/usr/bin/env python3
"""Test unconventional quadtree bounding boxes including negative regions."""

import pytest

from fastquadtree import QuadTree


def test_negative_region_basic():
    """Test basic operations in completely negative coordinate space."""
    qt = QuadTree((-500, -500, -100, -100), capacity=4)

    # Insert points in negative space
    qt.insert((-200, -200))
    qt.insert((-300, -400))
    qt.insert((-150, -450))
    qt.insert((-450, -150))

    assert qt.count_items() == 4

    # Query entire region
    all_items = qt.query((-500, -500, -100, -100))
    assert len(all_items) == 4

    # Query subregion - should find the point at (-200, -200)
    subset = qt.query((-250, -250, -150, -150))
    assert len(subset) == 1
    assert subset[0][1] == -200  # x coordinate
    assert subset[0][2] == -200  # y coordinate


def test_negative_region_with_object_tracking():
    """Test negative coordinates with object tracking enabled."""
    qt = QuadTree((-1000, -1000, 0, 0), capacity=4, track_objects=True)

    # Create test objects
    obj1 = {"name": "neg_point1", "value": -100}
    obj2 = {"name": "neg_point2", "value": -200}
    obj3 = {"name": "neg_point3", "value": -300}

    # Insert with objects in negative space
    id1 = qt.insert((-100, -100), obj=obj1)
    id2 = qt.insert((-500, -500), obj=obj2)
    id3 = qt.insert((-900, -200), obj=obj3)

    assert qt.count_items() == 3
    assert qt.get(id1) is obj1
    assert qt.get(id2) is obj2
    assert qt.get(id3) is obj3

    # Test delete_by_object in negative space
    deleted = qt.delete_by_object(obj2)
    assert deleted is True
    assert qt.count_items() == 2
    assert qt.get(id2) is None


def test_cross_origin_region():
    """Test region that spans across origin (0,0)."""
    qt = QuadTree((-100, -100, 100, 100), capacity=4)

    qt.insert((-50, -50))  # Bottom-left
    qt.insert((50, -50))  # Bottom-right
    qt.insert((-50, 50))  # Top-left
    qt.insert((50, 50))  # Top-right
    qt.insert((0, 0))  # Origin

    assert qt.count_items() == 5

    # Query each quadrant
    bl = qt.query((-100, -100, 0, 0))
    br = qt.query((0, -100, 100, 0))
    tl = qt.query((-100, 0, 0, 100))
    tr = qt.query((0, 0, 100, 100))

    # Origin point should appear in multiple quadrants due to boundary behavior
    assert len(bl) >= 1
    assert len(br) >= 1
    assert len(tl) >= 1
    assert len(tr) >= 1


def test_very_large_negative_coordinates():
    """Test with very large negative coordinates."""
    qt = QuadTree((-1e6, -1e6, -1e5, -1e5), capacity=4)

    qt.insert((-500000, -500000))
    qt.insert((-200000, -800000))
    qt.insert((-900000, -200000))

    assert qt.count_items() == 3

    # Test query
    all_items = qt.query((-1e6, -1e6, -1e5, -1e5))
    assert len(all_items) == 3

    # Test nearest neighbor
    nearest = qt.nearest_neighbor((-500001, -500001))
    assert nearest is not None
    assert nearest[1] == -500000  # x coordinate
    assert nearest[2] == -500000  # y coordinate


def test_fractional_negative_coordinates():
    """Test with fractional negative coordinates."""
    qt = QuadTree((-1.0, -1.0, -0.1, -0.1), capacity=4)

    qt.insert((-0.5, -0.5))
    qt.insert((-0.2, -0.8))
    qt.insert((-0.9, -0.3))

    assert qt.count_items() == 3

    # Query small region
    items = qt.query((-0.6, -0.6, -0.4, -0.4))
    assert len(items) == 1
    assert items[0][1] == -0.5  # x coordinate

    # Test k-nearest neighbors
    neighbors = qt.nearest_neighbors((-0.5, -0.5), 2)
    assert len(neighbors) == 2
    # First should be exact match
    assert neighbors[0][1] == -0.5
    assert neighbors[0][2] == -0.5


def test_boundary_edge_cases():
    """Test boundary conditions with negative coordinates."""
    qt = QuadTree((-10, -10, -1, -1), capacity=4)

    # Insert points at various boundaries
    qt.insert((-10, -10))  # Min corner
    qt.insert((-5.5, -5.5))  # Center
    qt.insert((-10, -5.5))  # Left edge
    qt.insert((-5.5, -10))  # Bottom edge

    # Test insertion at max boundary (should be rejected)
    with pytest.raises(ValueError):
        qt.insert((-1, -1))  # Exactly at max boundary

    with pytest.raises(ValueError):
        qt.insert((0, 0))  # Outside bounds


def test_delete_operations_negative_space():
    """Test deletion operations in negative coordinate space."""
    qt = QuadTree((-100, -100, -10, -10), capacity=4)

    id1 = qt.insert((-50, -50))
    id2 = qt.insert((-30, -80))
    qt.insert((-80, -30))

    assert qt.count_items() == 3

    # Delete a point
    deleted = qt.delete(id2, (-30, -80))
    assert deleted is True
    assert qt.count_items() == 2

    # Verify remaining points
    remaining = qt.query((-100, -100, -10, -10))
    assert len(remaining) == 2

    # Test delete with wrong coordinates
    deleted = qt.delete(id1, (-51, -50))  # Wrong x coordinate
    assert deleted is False
    assert qt.count_items() == 2


def test_mixed_positive_negative_operations():
    """Test operations across positive and negative coordinate space."""
    qt = QuadTree((-1000, -1000, 1000, 1000), capacity=4)

    # Insert in all quadrants
    qt.insert((-500, -500))  # Negative both
    qt.insert((500, -500))  # Positive X, negative Y
    qt.insert((-500, 500))  # Negative X, positive Y
    qt.insert((500, 500))  # Positive both
    qt.insert((0, 0))  # Origin

    assert qt.count_items() == 5

    # Test queries across regions
    negative_quad = qt.query((-1000, -1000, 0, 0))
    positive_quad = qt.query((0, 0, 1000, 1000))

    assert len(negative_quad) >= 1
    assert len(positive_quad) >= 1

    # Test nearest neighbor across origin
    nearest_to_origin = qt.nearest_neighbor((1, 1))
    assert nearest_to_origin is not None


def test_stress_negative_coordinates():
    """Stress test with many points in negative space."""
    qt = QuadTree((-500, -500, 0, 0), capacity=4)

    # Insert grid of points
    inserted_ids = []
    for i in range(20):
        for j in range(20):
            x = -25 * i - 10  # Range from -10 to -485
            y = -25 * j - 10  # Range from -10 to -485
            if -500 <= x < 0 and -500 <= y < 0:
                id_val = qt.insert((x, y))
                inserted_ids.append(id_val)

    assert qt.count_items() == len(inserted_ids)
    assert qt.count_items() > 100  # Should have many points

    # Test querying subregions
    corner_query = qt.query((-100, -100, -50, -50))
    center_query = qt.query((-300, -300, -200, -200))

    assert len(corner_query) > 0
    assert len(center_query) > 0


def test_nearest_neighbors_negative_space():
    """Test k-nearest neighbors in negative coordinate space."""
    qt = QuadTree((-200, -200, -50, -50), capacity=4)

    qt.insert((-100, -100))
    qt.insert((-150, -150))
    qt.insert((-75, -125))
    qt.insert((-125, -75))

    # Find k nearest neighbors
    neighbors = qt.nearest_neighbors((-80, -110), 3)
    assert len(neighbors) == 3

    # Verify they're sorted by distance
    # Calculate distances and verify ordering
    query_point = (-80, -110)
    distances = []
    for _, x, y in neighbors:
        dist = ((x - query_point[0]) ** 2 + (y - query_point[1]) ** 2) ** 0.5
        distances.append(dist)

    # Should be sorted in ascending order
    assert distances == sorted(distances)


def test_insert_many_negative_coordinates():
    """Test bulk insertion in negative coordinate space."""
    qt = QuadTree((-1000, -1000, 0, 0), capacity=4)

    # Create list of negative coordinate points
    points = [(-100.0 - i * 10, -100.0 - j * 10) for i in range(10) for j in range(10)]

    # Filter to ensure all points are within bounds
    valid_points = [(x, y) for x, y in points if -1000 <= x < 0 and -1000 <= y < 0]

    count = qt.insert_many(valid_points)
    assert count == len(valid_points)
    assert qt.count_items() == len(valid_points)

    # Verify we can query all points back
    all_items = qt.query((-1000, -1000, 0, 0))
    assert len(all_items) == len(valid_points)


def test_item_wrappers_negative_coordinates():
    """Test Item wrappers work correctly with negative coordinates."""
    qt = QuadTree((-100, -100, 0, 0), capacity=4, track_objects=True)

    obj1 = {"name": "negative_item1"}
    obj2 = {"name": "negative_item2"}

    qt.insert((-50, -75), obj=obj1)
    qt.insert((-80, -20), obj=obj2)

    # Query with as_items=True
    items = qt.query((-100, -100, 0, 0), as_items=True)
    assert len(items) == 2

    for item in items:
        assert hasattr(item, "id_")
        assert hasattr(item, "x")
        assert hasattr(item, "y")
        assert hasattr(item, "obj")
        assert item.x < 0  # Should be negative
        assert item.y < 0  # Should be negative
        assert item.obj is not None
