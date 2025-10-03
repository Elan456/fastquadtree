use fastquadtree::{Point, Rect, Item, QuadTree};

fn r(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
    Rect { min_x: x0, min_y: y0, max_x: x1, max_y: y1 }
}

fn pt(x: f32, y: f32) -> Point { 
    Point { x, y } 
}

fn item(id: u64, x: f32, y: f32) -> Item {
    Item { id, point: pt(x, y) }
}

fn ids(v: &Vec<(u64, f32, f32)>) -> Vec<u64> {
    let mut out: Vec<u64> = v.iter().map(|it| it.0).collect();
    out.sort_unstable();
    out
}

#[test]
fn negative_region_basic_operations() {
    // Test completely negative region
    let mut qt = QuadTree::new(r(-500.0, -500.0, -100.0, -100.0), 4);
    
    // Insert points in negative space
    assert!(qt.insert(item(1, -200.0, -200.0)));
    assert!(qt.insert(item(2, -300.0, -400.0)));
    assert!(qt.insert(item(3, -150.0, -450.0)));
    assert!(qt.insert(item(4, -450.0, -150.0)));
    
    // Query entire region
    let all_items = qt.query(r(-500.0, -500.0, -100.0, -100.0));
    assert_eq!(ids(&all_items), vec![1, 2, 3, 4]);
    
    // Query subregion
    let subset = qt.query(r(-250.0, -250.0, -150.0, -150.0));
    assert_eq!(ids(&subset), vec![1]);
}

#[test]
fn negative_region_with_splits() {
    // Force splits in negative space
    let mut qt = QuadTree::new(r(-1000.0, -1000.0, 0.0, 0.0), 2);
    
    // Insert enough points to force multiple splits
    assert!(qt.insert(item(1, -100.0, -100.0)));   // NE quadrant
    assert!(qt.insert(item(2, -900.0, -100.0)));   // NW quadrant
    assert!(qt.insert(item(3, -100.0, -900.0)));   // SE quadrant
    assert!(qt.insert(item(4, -900.0, -900.0)));   // SW quadrant
    assert!(qt.insert(item(5, -200.0, -200.0)));   // Force split in NE
    assert!(qt.insert(item(6, -150.0, -150.0)));   // Another in NE
    
    // Verify all items can be found
    let all_items = qt.query(r(-1000.0, -1000.0, 0.0, 0.0));
    assert_eq!(ids(&all_items), vec![1, 2, 3, 4, 5, 6]);
    
    // Query specific quadrants
    let ne_items = qt.query(r(-500.0, -500.0, 0.0, 0.0));
    assert_eq!(ids(&ne_items), vec![1, 5, 6]);
    
    let sw_items = qt.query(r(-1000.0, -1000.0, -500.0, -500.0));
    assert_eq!(ids(&sw_items), vec![4]);
}

#[test]
fn cross_origin_region() {
    // Region that spans across origin (0,0)
    let mut qt = QuadTree::new(r(-100.0, -100.0, 100.0, 100.0), 4);
    
    assert!(qt.insert(item(1, -50.0, -50.0)));  // Bottom-left quadrant
    assert!(qt.insert(item(2, 50.0, -50.0)));   // Bottom-right quadrant
    assert!(qt.insert(item(3, -50.0, 50.0)));   // Top-left quadrant
    assert!(qt.insert(item(4, 50.0, 50.0)));    // Top-right quadrant
    assert!(qt.insert(item(5, 0.0, 0.0)));      // Exactly at origin


    // min edge is included max edge is excluded
    
    // Query each quadrant
    let bl = qt.query(r(-100.0, -100.0, 0.0, 0.0));
    assert_eq!(ids(&bl), vec![1]);
    
    let br = qt.query(r(0.0, -100.0, 100.0, 0.0));
    assert_eq!(ids(&br), vec![2]);
    
    let tl = qt.query(r(-100.0, 0.0, 0.0, 100.0));
    assert_eq!(ids(&tl), vec![3]);
    
    let tr = qt.query(r(0.0, 0.0, 100.0, 100.0));
    assert_eq!(ids(&tr), vec![4, 5]);
}

#[test]
fn very_large_negative_coordinates() {
    // Test with very large negative coordinates
    let mut qt = QuadTree::new(r(-1e6, -1e6, -1e5, -1e5), 4);
    
    assert!(qt.insert(item(1, -500000.0, -500000.0)));
    assert!(qt.insert(item(2, -200000.0, -800000.0)));
    assert!(qt.insert(item(3, -900000.0, -200000.0)));
    
    let all_items = qt.query(r(-1e6, -1e6, -1e5, -1e5));
    assert_eq!(ids(&all_items), vec![1, 2, 3]);
    
    // Test nearest neighbor in large negative space
    let nearest = qt.nearest_neighbor(pt(-500001.0, -500001.0));
    assert!(nearest.is_some());
    assert_eq!(nearest.unwrap().id, 1);
}

#[test]
fn fractional_negative_coordinates() {
    // Test with fractional negative coordinates
    let mut qt = QuadTree::new(r(-1.0, -1.0, -0.1, -0.1), 4);
    
    assert!(qt.insert(item(1, -0.5, -0.5)));
    assert!(qt.insert(item(2, -0.2, -0.8)));
    assert!(qt.insert(item(3, -0.9, -0.3)));
    
    let items = qt.query(r(-0.6, -0.6, -0.4, -0.4));
    assert_eq!(ids(&items), vec![1]);
    
    // Test k-nearest neighbors
    let neighbors = qt.nearest_neighbors(pt(-0.5, -0.5), 2);
    assert_eq!(neighbors.len(), 2);
    assert_eq!(neighbors[0].id, 1); // Closest should be the exact match
}

#[test]
fn mixed_positive_negative_large_region() {
    // Large region spanning both positive and negative space
    let mut qt = QuadTree::new(r(-1000.0, -1000.0, 1000.0, 1000.0), 4);
    
    // Insert points in all quadrants
    assert!(qt.insert(item(1, -500.0, -500.0)));  // Negative both
    assert!(qt.insert(item(2, 500.0, -500.0)));   // Positive X, negative Y
    assert!(qt.insert(item(3, -500.0, 500.0)));   // Negative X, positive Y
    assert!(qt.insert(item(4, 500.0, 500.0)));    // Positive both
    assert!(qt.insert(item(5, 0.0, 0.0)));        // Origin
    
    // Test queries across different regions
    let negative_quadrant = qt.query(r(-1000.0, -1000.0, 0.0, 0.0));
    assert_eq!(ids(&negative_quadrant), vec![1]);
    
    let positive_quadrant = qt.query(r(0.0, 0.0, 1000.0, 1000.0));
    assert_eq!(ids(&positive_quadrant), vec![4, 5]);
    
    let all_items = qt.query(r(-1000.0, -1000.0, 1000.0, 1000.0));
    assert_eq!(ids(&all_items), vec![1, 2, 3, 4, 5]);
}

#[test]
fn edge_boundary_handling() {
    // Test boundary conditions with negative coordinates
    let mut qt = QuadTree::new(r(-10.0, -10.0, 0.0, 0.0), 4);
    
    // Insert points exactly on boundaries
    assert!(qt.insert(item(1, -10.0, -10.0)));  // Min corner
    assert!(qt.insert(item(2, -1.0, -1.0)));    // Max corner (should be excluded)
    assert!(qt.insert(item(3, -5.5, -5.5)));    // Center
    assert!(qt.insert(item(4, -10.0, -5.5)));   // Left edge
    assert!(qt.insert(item(5, -5.5, -10.0)));   // Bottom edge
    
    // Query entire region - max corner point should not be included
    let all_items = qt.query(r(-10.0, -10.0, -1.0, -1.0));
    // Point at (-1.0, -1.0) should be excluded due to half-open interval
    assert!(all_items.len() <= 4);
    
    // Query that should include the boundary point
    let edge_query = qt.query(r(-10.1, -10.1, -0.9, -0.9));
    assert!(ids(&edge_query).contains(&2)); // Now includes the max corner
}

#[test]
fn delete_operations_negative_space() {
    // Test deletion in negative coordinate space
    let mut qt = QuadTree::new(r(-100.0, -100.0, -10.0, -10.0), 4);
    
    assert!(qt.insert(item(1, -50.0, -50.0)));
    assert!(qt.insert(item(2, -30.0, -80.0)));
    assert!(qt.insert(item(3, -80.0, -30.0)));
    
    // Verify initial state
    assert_eq!(qt.count_items(), 3);
    
    // Delete a point
    assert!(qt.delete(2, pt(-30.0, -80.0)));
    assert_eq!(qt.count_items(), 2);
    
    // Verify remaining points
    let remaining = qt.query(r(-100.0, -100.0, -10.0, -10.0));
    assert_eq!(ids(&remaining), vec![1, 3]);
    
    // Test delete with wrong coordinates
    assert!(!qt.delete(1, pt(-51.0, -50.0))); // Wrong x coordinate
    assert_eq!(qt.count_items(), 2); // Should still be 2
}

#[test]
fn nearest_neighbor_negative_space() {
    let mut qt = QuadTree::new(r(-200.0, -200.0, -50.0, -50.0), 4);
    
    assert!(qt.insert(item(1, -100.0, -100.0)));
    assert!(qt.insert(item(2, -150.0, -150.0)));
    assert!(qt.insert(item(3, -75.0, -125.0)));
    
    // Find nearest to a query point in negative space
    let nearest = qt.nearest_neighbor(pt(-80.0, -110.0));
    assert!(nearest.is_some());
    // Should be item 3 as it's closest to (-80, -110)
    assert_eq!(nearest.unwrap().id, 3);
    
    // Test k-nearest neighbors
    let k_nearest = qt.nearest_neighbors(pt(-100.0, -100.0), 2);
    assert_eq!(k_nearest.len(), 2);
    assert_eq!(k_nearest[0].id, 1); // Exact match should be first
}

#[test]
fn stress_test_negative_coordinates() {
    // Stress test with many points in negative space
    let mut qt = QuadTree::new(r(-1000.0, -1000.0, 0.0, 0.0), 4);
    
    // Insert a grid of points
    let mut inserted_count = 0;
    for i in 0..50 {
        for j in 0..50 {
            let x = -20.0 * (i as f32) - 10.0; // Range from -10 to -990
            let y = -20.0 * (j as f32) - 10.0; // Range from -10 to -990
            let id = (i * 50 + j) as u64 + 1;
            if x >= -1000.0 && y >= -1000.0 && x < 0.0 && y < 0.0 {
                assert!(qt.insert(item(id, x, y)));
                inserted_count += 1;
            }
        }
    }
    
    assert_eq!(qt.count_items(), inserted_count);
    
    // Test querying various subregions
    let corner_query = qt.query(r(-100.0, -100.0, -50.0, -50.0));
    assert!(!corner_query.is_empty());
    
    let center_query = qt.query(r(-600.0, -600.0, -400.0, -400.0));
    assert!(!center_query.is_empty());
}

#[test]
fn out_of_bounds_insertion_negative() {
    let mut qt = QuadTree::new(r(-100.0, -100.0, -10.0, -10.0), 4);
    
    // These should succeed (within bounds)
    assert!(qt.insert(item(1, -50.0, -50.0)));
    assert!(qt.insert(item(2, -99.0, -99.0)));
    
    // These should fail (out of bounds)
    assert!(!qt.insert(item(3, -5.0, -50.0)));   // X too high
    assert!(!qt.insert(item(4, -50.0, -5.0)));   // Y too high
    assert!(!qt.insert(item(5, -101.0, -50.0))); // X too low
    assert!(!qt.insert(item(6, -50.0, -101.0))); // Y too low
    assert!(!qt.insert(item(7, 0.0, 0.0)));      // Both positive
    
    // Verify only the valid insertions succeeded
    assert_eq!(qt.count_items(), 2);
}