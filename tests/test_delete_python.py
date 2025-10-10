from fastquadtree._native import QuadTree


def test_delete_simple():
    """Test basic delete functionality"""
    tree = QuadTree((0, 0, 100, 100), capacity=4)

    # Insert some points
    tree.insert(1, (10, 10))
    tree.insert(2, (20, 20))
    tree.insert(3, (30, 30))

    assert tree.count_items() == 3

    # Delete existing item
    assert tree.delete(2, (20, 20))
    assert tree.count_items() == 2

    # Try to delete the same item again
    assert not tree.delete(2, (20, 20))
    assert tree.count_items() == 2

    # Delete another item
    assert tree.delete(1, (10, 10))
    assert tree.count_items() == 1

    # Delete last item
    assert tree.delete(3, (30, 30))
    assert tree.count_items() == 0


def test_delete_non_existent():
    """Test deleting non-existent items"""
    tree = QuadTree((0, 0, 100, 100), capacity=4)

    # Insert some points
    tree.insert(1, (10, 10))
    tree.insert(2, (20, 20))

    # Try to delete non-existent item (wrong ID)
    assert not tree.delete(99, (10, 10))
    assert tree.count_items() == 2

    # Try to delete non-existent point (wrong location)
    assert not tree.delete(1, (30, 30))
    assert tree.count_items() == 2

    # Try to delete point outside boundary
    assert not tree.delete(1, (200, 200))
    assert tree.count_items() == 2


def test_delete_with_split_and_merge():
    """Test delete with tree splitting and merging"""
    tree = QuadTree((0, 0, 100, 100), capacity=2)

    # Insert points that will cause splits
    tree.insert(1, (10, 10))
    tree.insert(2, (20, 20))
    tree.insert(3, (30, 30))  # This should cause a split
    tree.insert(4, (40, 40))
    tree.insert(5, (60, 60))  # Different quadrant

    initial_rectangles = len(tree.get_all_node_boundaries())
    assert initial_rectangles > 1  # Should have split
    assert tree.count_items() == 5

    # Delete points to trigger merging
    assert tree.delete(3, (30, 30))
    assert tree.delete(4, (40, 40))
    assert tree.delete(5, (60, 60))

    assert tree.count_items() == 2

    # Tree should have merged back to fewer rectangles
    final_rectangles = len(tree.get_all_node_boundaries())
    assert final_rectangles <= initial_rectangles


def test_delete_preserves_other_operations():
    """Test that delete doesn't break other operations"""
    tree = QuadTree((0, 0, 100, 100), capacity=4)

    # Insert points
    tree.insert(1, (10, 10))
    tree.insert(2, (20, 20))
    tree.insert(3, (80, 80))
    tree.insert(4, (90, 90))

    # Delete one point
    assert tree.delete(2, (20, 20))

    # Test that queries still work correctly
    results = tree.query((5, 5, 25, 25))
    assert len(results) == 1  # Should only find point (10,10)
    assert results[0][0] == 1  # ID should be 1

    # Test nearest neighbor
    nearest = tree.nearest_neighbor((15, 15))
    assert nearest is not None
    assert nearest[0] == 1  # Should be point (10,10)

    # Test that we can still insert
    tree.insert(5, (50, 50))
    assert tree.count_items() == 4


def test_delete_all_points():
    """Test deleting all points from tree"""
    tree = QuadTree((0, 0, 100, 100), capacity=3)

    points = [(10, 10), (20, 20), (30, 30), (80, 80), (90, 10)]

    # Insert all points
    for i, point in enumerate(points):
        tree.insert(i, point)

    assert tree.count_items() == 5

    # Delete all points
    for i, point in enumerate(points):
        assert tree.delete(i, point)

    assert tree.count_items() == 0

    # Tree should be back to just the root rectangle
    final_rectangles = len(tree.get_all_node_boundaries())
    assert final_rectangles == 1


def testdelete_point_matching():
    """Test that delete requires exact ID and point matching"""
    tree = QuadTree((0, 0, 100, 100), capacity=4)

    # Insert points that are very close but not identical, and same point with different IDs
    tree.insert(1, (10.0, 10.0))
    tree.insert(2, (10.000001, 10.0))
    tree.insert(3, (10.0, 10.000001))
    tree.insert(4, (10.0, 10.0))  # Same location, different ID

    assert tree.count_items() == 4

    # Delete by exact ID and point
    assert tree.delete(1, (10.0, 10.0))
    assert tree.count_items() == 3

    # The item with ID 4 at the same location should still be there
    assert tree.delete(4, (10.0, 10.0))
    assert tree.count_items() == 2

    # Try to delete with wrong ID
    assert not tree.delete(1, (10.0, 10.0))  # Already deleted
    assert tree.count_items() == 2

    # Delete the close but not identical points
    assert tree.delete(2, (10.000001, 10.0))
    assert tree.delete(3, (10.0, 10.000001))
    assert tree.count_items() == 0


def test_delete_multiple_items_same_location():
    """Test deleting specific items when multiple items exist at the same location"""
    tree = QuadTree((0, 0, 100, 100), capacity=4)

    # Insert multiple items at the exact same location
    location = (50.0, 50.0)
    tree.insert(10, location)
    tree.insert(20, location)
    tree.insert(30, location)

    assert tree.count_items() == 3

    # Delete by specific ID - should only delete that item
    assert tree.delete(20, location)
    assert tree.count_items() == 2

    # Verify the other items are still there by querying
    results = tree.query((49, 49, 51, 51))
    assert len(results) == 2

    # Verify we can find both remaining items
    ids = [item[0] for item in results]
    assert 10 in ids
    assert 30 in ids
    assert 20 not in ids  # Should be deleted

    # Delete the remaining items
    assert tree.delete(10, location)
    assert tree.delete(30, location)
    assert tree.count_items() == 0

    # Try to delete again - should fail
    assert not tree.delete(10, location)
    assert not tree.delete(20, location)
    assert not tree.delete(30, location)
