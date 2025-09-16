use quadtree_rs::{QuadTree, Item, Point, Rect};

fn main() {
    println!("=== QuadTree Delete by ID+Location Demo ===\n");
    
    // Create a quadtree
    let mut tree = QuadTree::new(Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 }, 4);
    
    // Insert multiple items at the same location
    let shared_location = Point { x: 50.0, y: 50.0 };
    tree.insert(Item { id: 1, point: shared_location });
    tree.insert(Item { id: 2, point: shared_location });
    tree.insert(Item { id: 3, point: shared_location });
    
    // Insert items at different locations
    tree.insert(Item { id: 4, point: Point { x: 25.0, y: 25.0 } });
    tree.insert(Item { id: 5, point: Point { x: 75.0, y: 75.0 } });
    
    println!("Initial tree with {} items", tree.count_items());
    
    // Query the area around the shared location
    let query_rect = Rect { min_x: 45.0, min_y: 45.0, max_x: 55.0, max_y: 55.0 };
    let items_at_location = tree.query(query_rect);
    println!("Items at (50,50): {:?}", items_at_location.iter().map(|i| i.id).collect::<Vec<_>>());
    
    // Delete specific item by ID+location - only removes that specific item
    println!("\nDeleting item with ID=2 at (50,50)...");
    let deleted = tree.delete(2, shared_location);
    println!("Delete successful: {}", deleted);
    println!("Tree now has {} items", tree.count_items());
    
    // Verify the other items at the same location are still there
    let remaining_items = tree.query(query_rect);
    println!("Remaining items at (50,50): {:?}", remaining_items.iter().map(|i| i.id).collect::<Vec<_>>());
    
    // Try to delete the same item again - should fail
    println!("\nTrying to delete ID=2 again...");
    let deleted_again = tree.delete(2, shared_location);
    println!("Delete successful: {}", deleted_again);
    
    // Try to delete with wrong ID - should fail  
    println!("\nTrying to delete ID=999 at (50,50)...");
    let wrong_id = tree.delete(999, shared_location);
    println!("Delete successful: {}", wrong_id);
    
    // Try to delete with wrong location - should fail
    println!("\nTrying to delete ID=1 at wrong location (60,60)...");
    let wrong_location = tree.delete(1, Point { x: 60.0, y: 60.0 });
    println!("Delete successful: {}", wrong_location);
    
    // Delete the remaining items at the shared location
    println!("\nDeleting remaining items at (50,50)...");
    tree.delete(1, shared_location);
    tree.delete(3, shared_location);
    
    println!("Final tree has {} items", tree.count_items());
    
    // Verify the items at other locations are still there
    let all_items = tree.query(Rect { min_x: 0.0, min_y: 0.0, max_x: 100.0, max_y: 100.0 });
    println!("All remaining items: {:?}", all_items.iter().map(|i| (i.id, i.point.x, i.point.y)).collect::<Vec<_>>());
    
    println!("\n=== Demo Complete ===");
}