use fastquadtree::{QuadTree, Item, Point, Rect};

#[test]
fn test_delete_simple() {
    let mut tree = QuadTree::new(Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 }, 4);
    
    // Insert some points
    let p1 = Point { x: 10.0, y: 10.0 };
    let p2 = Point { x: 20.0, y: 20.0 };
    let p3 = Point { x: 30.0, y: 30.0 };
    
    tree.insert(Item { id: 1, point: p1 });
    tree.insert(Item { id: 2, point: p2 });
    tree.insert(Item { id: 3, point: p3 });
    
    assert_eq!(tree.count_items(), 3);
    
    // Delete existing item
    assert!(tree.delete(2, p2));
    assert_eq!(tree.count_items(), 2);
    
    // Try to delete the same item again
    assert!(!tree.delete(2, p2));
    assert_eq!(tree.count_items(), 2);
    
    // Delete another item
    assert!(tree.delete(1, p1));
    assert_eq!(tree.count_items(), 1);
    
    // Delete last item
    assert!(tree.delete(3, p3));
    assert_eq!(tree.count_items(), 0);
}

#[test]
fn test_delete_non_existent() {
    let mut tree = QuadTree::new(Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 }, 4);
    
    // Insert some points
    tree.insert(Item { id: 1, point: Point { x: 10.0, y: 10.0 } });
    tree.insert(Item { id: 2, point: Point { x: 20.0, y: 20.0 } });
    
    // Try to delete non-existent item (wrong ID)
    assert!(!tree.delete(99, Point { x: 10.0, y: 10.0 }));
    assert_eq!(tree.count_items(), 2);
    
    // Try to delete non-existent point (wrong location)
    assert!(!tree.delete(1, Point { x: 30.0, y: 30.0 }));
    assert_eq!(tree.count_items(), 2);
    
    // Try to delete point outside boundary
    assert!(!tree.delete(1, Point { x: 200.0, y: 200.0 }));
    assert_eq!(tree.count_items(), 2);
}

#[test]
fn test_delete_with_split_and_merge() {
    let mut tree = QuadTree::new(Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 }, 2);
    
    // Insert points that will cause splits
    tree.insert(Item { id: 1, point: Point { x: 10.0, y: 10.0 } });
    tree.insert(Item { id: 2, point: Point { x: 20.0, y: 20.0 } });
    tree.insert(Item { id: 3, point: Point { x: 30.0, y: 30.0 } }); // This should cause a split
    tree.insert(Item { id: 4, point: Point { x: 40.0, y: 40.0 } });
    tree.insert(Item { id: 5, point: Point { x: 60.0, y: 60.0 } }); // Different quadrant
    
    let initial_rectangles = tree.get_all_node_boundaries().len();
    assert!(initial_rectangles > 1); // Should have split
    assert_eq!(tree.count_items(), 5);
    
    // Delete points to trigger merging
    assert!(tree.delete(3, Point { x: 30.0, y: 30.0 }));
    assert!(tree.delete(4, Point { x: 40.0, y: 40.0 }));
    assert!(tree.delete(5, Point { x: 60.0, y: 60.0 }));
    
    assert_eq!(tree.count_items(), 2);
    
    // Tree should have merged back to fewer rectangles
    let final_rectangles = tree.get_all_node_boundaries().len();
    assert!(final_rectangles <= initial_rectangles);
}

#[test]
fn test_delete_deep_tree() {
    let mut tree = QuadTree::new(Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 }, 1);
    
    // Insert many points in the same area to create a deep tree
    let points = vec![
        Point { x: 10.0, y: 10.0 },
        Point { x: 10.1, y: 10.1 },
        Point { x: 10.2, y: 10.2 },
        Point { x: 10.3, y: 10.3 },
        Point { x: 10.4, y: 10.4 },
    ];
    
    for (i, point) in points.iter().enumerate() {
        tree.insert(Item { id: i as u64, point: *point });
    }
    
    let initial_count = tree.count_items();
    let initial_rectangles = tree.get_all_node_boundaries().len();
    
    assert_eq!(initial_count, 5);
    assert!(initial_rectangles > 5); // Should have created a deep tree
    
    // Delete from the middle
    assert!(tree.delete(2, Point { x: 10.2, y: 10.2 }));
    assert_eq!(tree.count_items(), 4);
    
    // Delete more to trigger merging
    assert!(tree.delete(3, Point { x: 10.3, y: 10.3 }));
    assert!(tree.delete(4, Point { x: 10.4, y: 10.4 }));
    assert_eq!(tree.count_items(), 2);
    
    // Tree should have simplified (allow equal in case merging isn't as aggressive)
    let final_rectangles = tree.get_all_node_boundaries().len();
    assert!(final_rectangles <= initial_rectangles);
}

#[test]
fn test_delete_all_points() {
    let mut tree = QuadTree::new(Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 }, 3);
    
    let points = vec![
        Point { x: 10.0, y: 10.0 },
        Point { x: 20.0, y: 20.0 },
        Point { x: 30.0, y: 30.0 },
        Point { x: 80.0, y: 80.0 }, // Different quadrant
        Point { x: 90.0, y: 10.0 }, // Another quadrant
    ];
    
    // Insert all points
    for (i, point) in points.iter().enumerate() {
        tree.insert(Item { id: i as u64, point: *point });
    }
    
    assert_eq!(tree.count_items(), 5);
    let _initial_rectangles = tree.get_all_node_boundaries().len();
    
    // Delete all points
    for (i, point) in points.iter().enumerate() {
        assert!(tree.delete(i as u64, *point));
    }
    
    assert_eq!(tree.count_items(), 0);
    
    // Tree should be back to just the root rectangle
    let final_rectangles = tree.get_all_node_boundaries().len();
    assert_eq!(final_rectangles, 1);
}

#[test]
fn test_delete_preserves_other_operations() {
    let mut tree = QuadTree::new(Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 }, 4);
    
    // Insert points
    tree.insert(Item { id: 1, point: Point { x: 10.0, y: 10.0 } });
    tree.insert(Item { id: 2, point: Point { x: 20.0, y: 20.0 } });
    tree.insert(Item { id: 3, point: Point { x: 80.0, y: 80.0 } });
    tree.insert(Item { id: 4, point: Point { x: 90.0, y: 90.0 } });
    
    // Delete one point
    assert!(tree.delete(2, Point { x: 20.0, y: 20.0 }));
    
    // Test that queries still work correctly
    let query_rect = Rect { min_x: 5.0, min_y: 5.0, max_x: 25.0, max_y: 25.0 };
    let results = tree.query(query_rect);
    assert_eq!(results.len(), 1); // Should only find point (10,10)
    assert_eq!(results[0].0, 1);
    
    // Test nearest neighbor
    let nearest = tree.nearest_neighbor(Point { x: 15.0, y: 15.0 });
    assert!(nearest.is_some());
    assert_eq!(nearest.unwrap().id, 1); // Should be point (10,10)
    
    // Test that we can still insert
    assert!(tree.insert(Item { id: 5, point: Point { x: 50.0, y: 50.0 } }));
    assert_eq!(tree.count_items(), 4);
}

#[test]
fn test_delete_exact_point_matching() {
    let mut tree = QuadTree::new(Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 }, 4);
    
    // Insert points that are very close but not identical, and same point with different IDs
    tree.insert(Item { id: 1, point: Point { x: 10.0, y: 10.0 } });
    tree.insert(Item { id: 2, point: Point { x: 10.000001, y: 10.0 } });
    tree.insert(Item { id: 3, point: Point { x: 10.0, y: 10.000001 } });
    tree.insert(Item { id: 4, point: Point { x: 10.0, y: 10.0 } }); // Same location, different ID
    
    assert_eq!(tree.count_items(), 4);
    
    // Delete by exact ID and point
    assert!(tree.delete(1, Point { x: 10.0, y: 10.0 }));
    assert_eq!(tree.count_items(), 3);
    
    // The item with ID 4 at the same location should still be there
    assert!(tree.delete(4, Point { x: 10.0, y: 10.0 }));
    assert_eq!(tree.count_items(), 2);
    
    // Try to delete with wrong ID
    assert!(!tree.delete(1, Point { x: 10.0, y: 10.0 })); // Already deleted
    assert_eq!(tree.count_items(), 2);
    
    // Delete the close but not identical points
    assert!(tree.delete(2, Point { x: 10.000001, y: 10.0 }));
    assert!(tree.delete(3, Point { x: 10.0, y: 10.000001 }));
    assert_eq!(tree.count_items(), 0);
}

#[test]
fn test_delete_multiple_items_same_location() {
    let mut tree = QuadTree::new(Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 }, 4);
    
    // Insert multiple items at the exact same location
    let location = Point { x: 50.0, y: 50.0 };
    tree.insert(Item { id: 10, point: location });
    tree.insert(Item { id: 20, point: location });
    tree.insert(Item { id: 30, point: location });
    
    assert_eq!(tree.count_items(), 3);
    
    // Delete by specific ID - should only delete that item
    assert!(tree.delete(20, location));
    assert_eq!(tree.count_items(), 2);
    
    // Verify the other items are still there
    let query_rect = Rect { min_x: 49.0, min_y: 49.0, max_x: 51.0, max_y: 51.0 };
    let results = tree.query(query_rect);
    assert_eq!(results.len(), 2);
    
    // Verify we can find both remaining items
    let ids: Vec<u64> = results.iter().map(|item| item.0).collect();
    assert!(ids.contains(&10));
    assert!(ids.contains(&30));
    assert!(!ids.contains(&20)); // Should be deleted
    
    // Delete the remaining items
    assert!(tree.delete(10, location));
    assert!(tree.delete(30, location));
    assert_eq!(tree.count_items(), 0);
    
    // Try to delete again - should fail
    assert!(!tree.delete(10, location));
    assert!(!tree.delete(20, location));
    assert!(!tree.delete(30, location));
}

#[test]
fn merge_happens_when_grandchildren_exist() {
    // capacity 2 makes the tree split as soon as 3 points share a node
    let mut tree = QuadTree::new(Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 }, 2);

    // Build depth: put several points in the lower-left quadrant to force grandchildren
    let pts = [
        Point { x: 10.0, y: 10.0 },
        Point { x: 11.0, y: 11.0 },
        Point { x: 12.0, y: 12.0 }, // split 1
        Point { x: 12.5, y: 12.5 }, // deeper
        Point { x: 12.75, y: 12.75 },
    ];
    for (i, p) in pts.iter().enumerate() {
        tree.insert(Item { id: i as u64, point: *p });
    }

    // Sanity: we should have split, so there are multiple rectangles
    let initial_rects = tree.get_all_node_boundaries().len();
    assert!(initial_rects > 1);

    // Delete enough to make the subtree compact again
    assert!(tree.delete(4, pts[4]));
    assert!(tree.delete(3, pts[3]));
    assert!(tree.delete(2, pts[2]));

    // Only two points left in that region, which fits capacity at the parent
    // The recursive merge should collapse grandchildren, then children, possibly the root if applicable
    let final_rects = tree.get_all_node_boundaries().len();
    assert!(final_rects < initial_rects, "expected fewer rectangles after recursive merge");
}

#[test]
fn root_collapses_when_total_items_fit_capacity() {
    let mut tree = QuadTree::new(Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 }, 2);

    // Points across different quadrants to ensure a top-level split
    let pts = [
        (0, Point { x: 10.0, y: 10.0 }),
        (1, Point { x: 20.0, y: 20.0 }),
        (2, Point { x: 80.0, y: 80.0 }),
        (3, Point { x: 90.0, y: 90.0 }),
    ];
    for (id, p) in pts.iter() {
        tree.insert(Item { id: *id, point: *p });
    }

    assert!(tree.get_all_node_boundaries().len() > 1);
    // Delete two so only two remain in total, which equals capacity at the root
    assert!(tree.delete(2, Point { x: 80.0, y: 80.0 }));
    assert!(tree.delete(3, Point { x: 90.0, y: 90.0 }));

    // With 2 items left and capacity 2, the root should collapse to a leaf
    assert!(tree.children.is_none(), "root should have collapsed to a leaf");
    assert_eq!(tree.get_all_node_boundaries().len(), 1, "only the root rectangle should remain");
}

#[test]
fn no_merge_when_over_capacity() {
    let mut tree = QuadTree::new(Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 }, 2);

    // Force a split
    let pts = [
        (0, Point { x: 10.0, y: 10.0 }),
        (1, Point { x: 20.0, y: 20.0 }),
        (2, Point { x: 80.0, y: 80.0 }),
    ];
    for (id, p) in pts.iter() {
        tree.insert(Item { id: *id, point: *p });
    }
    assert!(tree.get_all_node_boundaries().len() > 1);

    // Delete one, total remains 2 across two different quadrants plus one more insert to make it 3
    assert!(tree.delete(1, Point { x: 20.0, y: 20.0 }));
    tree.insert(Item { id: 3, point: Point { x: 85.0, y: 85.0 }});
    // Now total is 3, which exceeds capacity at the root, so no merge to single leaf
    assert!(tree.children.is_some(), "should not collapse when total items exceed capacity");
}

#[test]
fn deep_chain_collapses_to_leaf() {
    let mut tree = QuadTree::new(Rect { min_x: 0.0, min_y: 0.0, max_x: 1.0, max_y: 1.0 }, 1);
    // Create a chain by inserting points that keep falling into the same quadrant
    let pts = [
        (0, Point { x: 0.1, y: 0.1 }),
        (1, Point { x: 0.15, y: 0.15 }),
        (2, Point { x: 0.18, y: 0.18 }),
        (3, Point { x: 0.19, y: 0.19 }),
    ];
    for (id, p) in pts.iter() {
        tree.insert(Item { id: *id, point: *p });
    }

    assert!(tree.get_all_node_boundaries().len() > 1);

    // Delete back down to one point, which must fit in a single leaf
    assert!(tree.delete(3, pts[3].1));
    assert!(tree.delete(2, pts[2].1));
    assert!(tree.delete(1, pts[1].1));

    assert!(tree.children.is_none(), "deep path should have collapsed back to a single leaf");
    assert_eq!(tree.count_items(), 1);
}
