"""Tests for RectQuadTree nearest neighbor functionality."""

from __future__ import annotations

import pytest

from fastquadtree import RectQuadTree


def test_nearest_neighbor_single_rect_no_tracking():
    """Test finding nearest neighbor with a single rectangle, no object tracking."""
    qt = RectQuadTree(bounds=(0.0, 0.0, 100.0, 100.0), capacity=4, track_objects=False)
    id1 = qt.insert((10.0, 10.0, 20.0, 20.0))

    # Query point inside the rectangle (distance = 0)
    result = qt.nearest_neighbor((15.0, 15.0), as_item=False)
    assert result is not None
    id_, x0, y0, x1, y1 = result
    assert id_ == id1
    assert (x0, y0, x1, y1) == (10.0, 10.0, 20.0, 20.0)

    # Query point outside the rectangle
    result2 = qt.nearest_neighbor((25.0, 25.0), as_item=False)
    assert result2 is not None
    assert result2[0] == id1


def test_nearest_neighbor_multiple_rects_no_tracking():
    """Test finding nearest among multiple rectangles."""
    qt = RectQuadTree(bounds=(0.0, 0.0, 100.0, 100.0), capacity=4, track_objects=False)

    # Insert rectangles at different distances from query point (50, 50)
    _id1 = qt.insert((10.0, 10.0, 20.0, 20.0))  # far
    _id2 = qt.insert((30.0, 30.0, 40.0, 40.0))  # closer
    id3 = qt.insert((45.0, 45.0, 55.0, 55.0))  # closest (contains query point)
    _id4 = qt.insert((70.0, 70.0, 80.0, 80.0))  # far

    result = qt.nearest_neighbor((50.0, 50.0), as_item=False)
    assert result is not None
    # Should find rect 3 since the point is inside it (distance = 0)
    assert result[0] == id3


def test_nearest_neighbor_empty_tree():
    """Test that nearest neighbor on empty tree returns None."""
    qt = RectQuadTree(bounds=(0.0, 0.0, 100.0, 100.0), capacity=4, track_objects=False)
    result = qt.nearest_neighbor((50.0, 50.0), as_item=False)
    assert result is None


def test_nearest_neighbor_with_tracking():
    """Test nearest neighbor with object tracking enabled."""
    qt = RectQuadTree(bounds=(0.0, 0.0, 100.0, 100.0), capacity=4, track_objects=True)
    id1 = qt.insert((10.0, 10.0, 20.0, 20.0), obj="rect1")
    _id2 = qt.insert((30.0, 30.0, 40.0, 40.0), obj="rect2")

    # Test as_item=False
    result_raw = qt.nearest_neighbor((15.0, 15.0), as_item=False)
    assert result_raw is not None
    assert result_raw[0] == id1

    # Test as_item=True
    result_item = qt.nearest_neighbor((15.0, 15.0), as_item=True)
    assert result_item is not None
    assert result_item.id_ == id1
    assert result_item.obj == "rect1"
    assert result_item.geom == (10.0, 10.0, 20.0, 20.0)


def test_nearest_neighbor_as_item_without_tracking_raises():
    """Test that as_item=True without tracking raises ValueError."""
    qt = RectQuadTree(bounds=(0.0, 0.0, 100.0, 100.0), capacity=4, track_objects=False)
    qt.insert((10.0, 10.0, 20.0, 20.0))

    with pytest.raises(ValueError, match="Cannot return result as item"):
        qt.nearest_neighbor((15.0, 15.0), as_item=True)


def test_nearest_neighbors_k_multiple_no_tracking():
    """Test K-NN with multiple rectangles, no tracking."""
    qt = RectQuadTree(bounds=(0.0, 0.0, 100.0, 100.0), capacity=4, track_objects=False)

    id1 = qt.insert((10.0, 10.0, 20.0, 20.0))
    id2 = qt.insert((30.0, 30.0, 40.0, 40.0))
    _id3 = qt.insert((50.0, 50.0, 60.0, 60.0))
    _id4 = qt.insert((70.0, 70.0, 80.0, 80.0))

    # Query from point (25, 25) and get 2 nearest
    results = qt.nearest_neighbors((25.0, 25.0), 2, as_items=False)
    assert len(results) == 2

    # Collect IDs to check which ones were returned
    result_ids = sorted([r[0] for r in results])
    # Should be IDs 1 and 2 (the two closest to point 25,25)
    assert result_ids == sorted([id1, id2])


def test_nearest_neighbors_k_exceeds_count():
    """Test K-NN when K exceeds number of rectangles."""
    qt = RectQuadTree(bounds=(0.0, 0.0, 100.0, 100.0), capacity=4, track_objects=False)

    qt.insert((10.0, 10.0, 20.0, 20.0))
    qt.insert((30.0, 30.0, 40.0, 40.0))

    # Request more neighbors than exist
    results = qt.nearest_neighbors((25.0, 25.0), 10, as_items=False)
    # Should return only the 2 that exist
    assert len(results) == 2


def test_nearest_neighbors_k_zero():
    """Test K-NN with k=0 returns empty list."""
    qt = RectQuadTree(bounds=(0.0, 0.0, 100.0, 100.0), capacity=4, track_objects=False)
    qt.insert((10.0, 10.0, 20.0, 20.0))

    results = qt.nearest_neighbors((25.0, 25.0), 0, as_items=False)
    assert len(results) == 0


def test_nearest_neighbors_with_tracking():
    """Test K-NN with object tracking enabled."""
    qt = RectQuadTree(bounds=(0.0, 0.0, 100.0, 100.0), capacity=4, track_objects=True)

    id1 = qt.insert((10.0, 10.0, 20.0, 20.0), obj="rect1")
    id2 = qt.insert((30.0, 30.0, 40.0, 40.0), obj="rect2")
    _id3 = qt.insert((50.0, 50.0, 60.0, 60.0), obj="rect3")

    # Test as_items=False
    results_raw = qt.nearest_neighbors((25.0, 25.0), 2, as_items=False)
    assert len(results_raw) == 2
    result_ids = sorted([r[0] for r in results_raw])
    assert result_ids == sorted([id1, id2])

    # Test as_items=True
    results_items = qt.nearest_neighbors((25.0, 25.0), 2, as_items=True)
    assert len(results_items) == 2
    result_objs = sorted([item.obj for item in results_items])
    assert result_objs == ["rect1", "rect2"]


def test_nearest_neighbors_as_items_without_tracking_raises():
    """Test that as_items=True without tracking raises ValueError."""
    qt = RectQuadTree(bounds=(0.0, 0.0, 100.0, 100.0), capacity=4, track_objects=False)
    qt.insert((10.0, 10.0, 20.0, 20.0))
    qt.insert((30.0, 30.0, 40.0, 40.0))

    with pytest.raises(ValueError, match="Cannot return results as items"):
        qt.nearest_neighbors((25.0, 25.0), 2, as_items=True)


def test_nearest_neighbors_with_deep_tree():
    """Test K-NN with a tree that has multiple levels."""
    qt = RectQuadTree(bounds=(0.0, 0.0, 100.0, 100.0), capacity=2, track_objects=False)

    # Insert many rectangles to force splitting
    for i in range(20):
        offset = float(i) * 4.0
        qt.insert((offset, offset, offset + 3.0, offset + 3.0))

    # Query for 5 nearest
    results = qt.nearest_neighbors((10.0, 10.0), 5, as_items=False)
    assert len(results) == 5

    # All results should be valid
    for r in results:
        assert r[0] < 20


def test_nearest_neighbor_all_dtypes():
    """Test that nearest neighbor works with all supported dtypes."""
    for dtype in ["f32", "f64", "i32", "i64"]:
        qt = RectQuadTree(
            bounds=(0, 0, 100, 100), capacity=4, track_objects=False, dtype=dtype
        )
        id1 = qt.insert((10, 10, 20, 20))
        id2 = qt.insert((30, 30, 40, 40))

        # Test nearest_neighbor
        result = qt.nearest_neighbor((15, 15), as_item=False)
        assert result is not None
        assert result[0] == id1

        # Test nearest_neighbors
        results = qt.nearest_neighbors((25, 25), 2, as_items=False)
        assert len(results) == 2
        result_ids = sorted([r[0] for r in results])
        assert result_ids == sorted([id1, id2])


def test_nearest_neighbors_ordering():
    """Test that nearest neighbors are returned in order of increasing distance."""
    qt = RectQuadTree(bounds=(0.0, 0.0, 100.0, 100.0), capacity=4, track_objects=True)

    # Insert rectangles at known distances from origin (0, 0)
    qt.insert((5.0, 5.0, 10.0, 10.0), obj="dist5")  # closest corner at (5,5)
    qt.insert((15.0, 15.0, 20.0, 20.0), obj="dist15")  # closest corner at (15,15)
    qt.insert((25.0, 25.0, 30.0, 30.0), obj="dist25")  # closest corner at (25,25)

    # Query from origin and get all 3
    results = qt.nearest_neighbors((0.0, 0.0), 3, as_items=True)
    assert len(results) == 3

    # The objects should be in distance order
    objs = [item.obj for item in results]
    assert objs == ["dist5", "dist15", "dist25"]


def test_nearest_neighbor_empty_with_tracking():
    """Test nearest neighbor on empty tree with tracking returns None."""
    qt = RectQuadTree(bounds=(0.0, 0.0, 100.0, 100.0), capacity=4, track_objects=True)

    result_raw = qt.nearest_neighbor((50.0, 50.0), as_item=False)
    assert result_raw is None

    result_item = qt.nearest_neighbor((50.0, 50.0), as_item=True)
    assert result_item is None


def test_nearest_neighbors_empty_with_tracking():
    """Test K-NN on empty tree with tracking returns empty list."""
    qt = RectQuadTree(bounds=(0.0, 0.0, 100.0, 100.0), capacity=4, track_objects=True)

    results_raw = qt.nearest_neighbors((50.0, 50.0), 5, as_items=False)
    assert results_raw == []

    results_items = qt.nearest_neighbors((50.0, 50.0), 5, as_items=True)
    assert results_items == []


def test_missing_tracked_item_runtim_error():
    """Test that missing tracked item raises RuntimeError."""
    qt = RectQuadTree(bounds=(0.0, 0.0, 100.0, 100.0), capacity=4, track_objects=True)

    # Insert direct with native
    qt._native.insert(1, (10.0, 10.0, 20.0, 20.0))

    with pytest.raises(RuntimeError, match="Internal error: missing tracked item"):
        qt.nearest_neighbor((15.0, 15.0), as_item=True)

    with pytest.raises(RuntimeError, match="Internal error: missing tracked item"):
        qt.nearest_neighbors((15.0, 15.0), 1, as_items=True)
