use fastquadtree::{RectQuadTree, QuadTree, Item, RectItem, Point, Rect};

#[test]
fn quadtree_roundtrip_bytes() {
    // Build a small tree
    let mut qt = QuadTree::new(
        Rect { min_x: 0.0, min_y: 0.0, max_x: 10.0, max_y: 10.0 },
        4,
        8
    );
    for (i, (x, y)) in [(1.0, 1.0), (2.0, 3.0), (7.5, 8.5), (9.0, 0.5)].into_iter().enumerate() {
        qt.insert(Item { id: i as u64 + 1, point: Point { x, y } });
    }

    // Serialize
    let bytes = qt.to_bytes().expect("serialize quadtree");

    // Deserialize
    let qt2 = QuadTree::from_bytes(&bytes).expect("deserialize quadtree");

    // Basic invariants
    assert_eq!(qt.count_items(), qt2.count_items());

    // Query equality for a region
    let rect = Rect { min_x: 0.0, min_y: 0.0, max_x: 5.0, max_y: 5.0 };
    let a: Vec<_> = qt.query(rect).into_iter().map(|(id, _, _)| id).collect();
    let b: Vec<_> = qt2.query(rect).into_iter().map(|(id, _, _)| id).collect();
    assert_eq!(a, b);

    // Nearest neighbor equality
    let nn1 = qt.nearest_neighbor(Point { x: 1.2, y: 1.1 }).map(|it| it.id);
    let nn2 = qt2.nearest_neighbor(Point { x: 1.2, y: 1.1 }).map(|it| it.id);
    assert_eq!(nn1, nn2);
}


#[test]
fn rectquadtree_roundtrip_bytes() {
    // Build a small tree
    let mut qt = RectQuadTree::new(
        Rect { min_x: 0.0, min_y: 0.0, max_x: 10.0, max_y: 10.0 },
        4,
        8
    );
    for (i, (x, y)) in [(1.0, 1.0), (2.0, 3.0), (7.5, 8.5), (9.0, 0.5)].into_iter().enumerate() {
        qt.insert(RectItem { id: i as u64 + 1, rect: Rect { min_x: x, min_y: y, max_x: x + 1.0, max_y: y + 1.0 } });
    }

    // Serialize
    let bytes = qt.to_bytes().expect("serialize quadtree");

    // Deserialize
    let qt2 = RectQuadTree::from_bytes(&bytes).expect("deserialize quadtree");

    // Basic invariants
    assert_eq!(qt.count_items(), qt2.count_items());

    // Query equality for a region
    let rect = Rect { min_x: 0.0, min_y: 0.0, max_x: 5.0, max_y: 5.0 };
    let a: Vec<_> = qt.query(rect).into_iter().map(|it| it.0).collect();
    let b: Vec<_> = qt2.query(rect).into_iter().map(|it| it.0).collect();
    assert_eq!(a, b);
}
