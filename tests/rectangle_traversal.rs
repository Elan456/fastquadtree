use fastquadtree::{QuadTree, Item, Point, Rect};

#[test]
fn test_get_all_node_boundaries_single_node() {
    let boundary = Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 };
    let qt = QuadTree::new(boundary, 4);
    
    let rectangles = qt.get_all_node_boundaries();
    assert_eq!(rectangles.len(), 1);
    assert_eq!(rectangles[0], boundary);
}

#[test]
fn test_get_all_node_boundaries_after_split() {
    let boundary = Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 };
    let mut qt = QuadTree::new(boundary, 2);
    
    // Insert items in different quadrants to avoid deep splits
    qt.insert(Item { id: 1, point: Point { x: 10.0, y: 10.0 } }); // Bottom-left
    qt.insert(Item { id: 2, point: Point { x: 80.0, y: 20.0 } }); // Bottom-right
    qt.insert(Item { id: 3, point: Point { x: 30.0, y: 80.0 } }); // Top-left - This should trigger split
    
    let rectangles = qt.get_all_node_boundaries();
    
    // Should have root + 4 children = 5 rectangles (since items are in different quadrants)
    assert_eq!(rectangles.len(), 5);
    
    // Root rectangle should be in the list
    assert!(rectangles.contains(&boundary));
    
    // Verify we have the expected child rectangles
    let expected_children = [
        Rect { min_x: 0.0, min_y: 0.0, max_x: 50.0, max_y: 50.0 },   // Bottom-left
        Rect { min_x: 50.0, min_y: 0.0, max_x: 100.0, max_y: 50.0 }, // Bottom-right
        Rect { min_x: 0.0, min_y: 50.0, max_x: 50.0, max_y: 100.0 }, // Top-left
        Rect { min_x: 50.0, min_y: 50.0, max_x: 100.0, max_y: 100.0 }, // Top-right
    ];
    
    for expected_child in &expected_children {
        assert!(rectangles.contains(expected_child), 
                "Missing expected child rectangle: {:?}", expected_child);
    }
}

#[test]
fn test_get_all_node_boundaries_deep_tree() {
    let boundary = Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 };
    let mut qt = QuadTree::new(boundary, 1); // Small capacity to force deep splits
    
    // Insert items to create a deeper tree structure
    qt.insert(Item { id: 1, point: Point { x: 10.0, y: 10.0 } });
    qt.insert(Item { id: 2, point: Point { x: 12.0, y: 12.0 } }); // Close to first, will split further
    qt.insert(Item { id: 3, point: Point { x: 14.0, y: 14.0 } }); // Even deeper split
    
    let rectangles = qt.get_all_node_boundaries();
    
    // Should have more than just root + 4 children due to deeper splits
    assert!(rectangles.len() > 5, "Expected more rectangles due to deep splits, got {}", rectangles.len());
    
    // Root should still be there
    assert!(rectangles.contains(&boundary));
}

#[test]
fn test_get_all_node_boundaries_same_quadrant_deep_split() {
    let boundary = Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 };
    let mut qt = QuadTree::new(boundary, 2);
    
    // Insert items that all fall in the same quadrant to force deep splits
    qt.insert(Item { id: 1, point: Point { x: 10.0, y: 10.0 } });
    qt.insert(Item { id: 2, point: Point { x: 20.0, y: 20.0 } });
    qt.insert(Item { id: 3, point: Point { x: 30.0, y: 30.0 } }); // All in bottom-left, forces deep split
    
    let rectangles = qt.get_all_node_boundaries();
    
    // Should have root + first level children + deeper subdivisions = 9 rectangles
    assert_eq!(rectangles.len(), 9);
    
    // Root rectangle should be in the list
    assert!(rectangles.contains(&boundary));
    
    // The bottom-left quadrant should be further subdivided
    let bottom_left_child = Rect { min_x: 0.0, min_y: 0.0, max_x: 50.0, max_y: 50.0 };
    assert!(rectangles.contains(&bottom_left_child));
    
    // The bottom-left child should have its own 4 children
    let bottom_left_grandchildren = [
        Rect { min_x: 0.0, min_y: 0.0, max_x: 25.0, max_y: 25.0 },
        Rect { min_x: 25.0, min_y: 0.0, max_x: 50.0, max_y: 25.0 },
        Rect { min_x: 0.0, min_y: 25.0, max_x: 25.0, max_y: 50.0 },
        Rect { min_x: 25.0, min_y: 25.0, max_x: 50.0, max_y: 50.0 },
    ];
    
    for grandchild in &bottom_left_grandchildren {
        assert!(rectangles.contains(grandchild), 
                "Missing expected grandchild rectangle: {:?}", grandchild);
    }
}