use fastquadtree::{Point, Rect, Item, QuadTree};

fn r(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
    Rect { min_x: x0, min_y: y0, max_x: x1, max_y: y1 }
}
fn pt(x: f32, y: f32) -> Point { Point { x, y } }
fn ids(v: &Vec<(u64, f32, f32)>) -> Vec<u64> {
    let mut out: Vec<u64> = v.iter().map(|it| it.0).collect();
    out.sort_unstable();
    out
}

#[test]
fn query_on_empty_tree_is_empty() {
    let qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 4);
    let hits = qt.query(r(10.0, 10.0, 20.0, 20.0));
    assert!(hits.is_empty());
}

#[test]
fn leaf_query_without_split() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 10);
    let a = Item { id: 1, point: pt(10.0, 10.0) };
    let b = Item { id: 2, point: pt(30.0, 30.0) };
    let c = Item { id: 3, point: pt(70.0, 70.0) };
    assert!(qt.insert(a));
    assert!(qt.insert(b));
    assert!(qt.insert(c));

    // Query a small box that should only include a and b
    let hits = qt.query(r(0.0, 0.0, 40.0, 40.0));
    assert_eq!(ids(&hits), vec![1, 2]);

    // Query a box that excludes max edges
    let hits2 = qt.query(r(0.0, 0.0, 10.0, 10.0)); // half-open excludes point at (10,10)
    assert!(hits2.is_empty());
}

#[test]
fn query_outside_root_returns_empty() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 2);
    assert!(qt.insert(Item { id: 1, point: pt(50.0, 50.0) }));
    let hits = qt.query(r(200.0, 200.0, 300.0, 300.0));
    assert!(hits.is_empty());
}

#[test]
fn query_after_split_picks_from_correct_children() {
    // capacity 1 to force early splits
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 1);

    // One point in each quadrant plus the exact center
    assert!(qt.insert(Item { id: 1, point: pt(10.0, 10.0) }));  // Q0
    assert!(qt.insert(Item { id: 2, point: pt(75.0, 10.0) }));  // Q1
    assert!(qt.insert(Item { id: 3, point: pt(10.0, 75.0) }));  // Q2
    assert!(qt.insert(Item { id: 4, point: pt(75.0, 75.0) }));  // Q3
    assert!(qt.insert(Item { id: 5, point: pt(50.0, 50.0) }));  // center -> right-top with >= rule

    // Query left-bottom child bounds [0,50) x [0,50)
    let lb = qt.query(r(0.0, 0.0, 50.0, 50.0));
    assert_eq!(ids(&lb), vec![1]);

    // Query right-bottom child [50,100) x [0,50)
    let rb = qt.query(r(50.0, 0.0, 100.0, 50.0));
    assert_eq!(ids(&rb), vec![2]);

    // Query left-top child [0,50) x [50,100)
    let lt = qt.query(r(0.0, 50.0, 50.0, 100.0));
    assert_eq!(ids(&lt), vec![3]);

    // Query right-top child [50,100) x [50,100) includes 4 and center 5
    let rt = qt.query(r(50.0, 50.0, 100.0, 100.0));
    assert_eq!(ids(&rt), vec![4, 5]);
}

#[test]
fn query_range_covering_multiple_children_returns_union() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 1);
    let pts = [
        (1, pt(10.0, 10.0)),  // Q0
        (2, pt(75.0, 10.0)),  // Q1
        (3, pt(10.0, 75.0)),  // Q2
        (4, pt(75.0, 75.0)),  // Q3
        (5, pt(60.0, 40.0)),  // crosses into right-bottom
    ];
    for (id, p) in pts {
        assert!(qt.insert(Item { id, point: p }));
    }

    // Query right half of the root
    let hits = qt.query(r(50.0, 0.0, 100.0, 100.0));
    assert_eq!(ids(&hits), vec![2, 4, 5]);

    // Query a band across the middle two children
    let hits2 = qt.query(r(0.0, 40.0, 100.0, 60.0));
    // Items with y in [40,60), which includes id 5 only in this setup
    assert_eq!(ids(&hits2), vec![5]);
}

#[test]
fn full_range_query_returns_all_items() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 3);
    for i in 0..9 {
        let x = 10.0 + 10.0 * (i as f32);
        let y = 20.0 + 5.0 * (i as f32);
        assert!(qt.insert(Item { id: i + 1, point: pt(x, y) }));
    }
    let hits = qt.query(r(0.0, 0.0, 100.0, 100.0));
    assert_eq!(hits.len(), 9);
    assert_eq!(ids(&hits), (1..=9).collect::<Vec<_>>());
}

#[test]
fn half_open_edges_behavior() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 4);
    // min edges are included, max edges are excluded
    assert!(qt.insert(Item { id: 1, point: pt(0.0, 0.0) }));
    assert!(!qt.insert(Item { id: 2, point: pt(100.0, 0.0) }));
    assert!(!qt.insert(Item { id: 3, point: pt(0.0, 100.0) }));
    assert!(!qt.insert(Item { id: 4, point: pt(100.0, 100.0) }));

    let hits = qt.query(r(0.0, 0.0, 1.0, 1.0));
    assert_eq!(ids(&hits), vec![1]);
}

#[test]
fn triple_insert_query() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 4);
    // min edges are included, max edges are excluded
    assert!(qt.insert(Item { id: 1, point: pt(10.0, 10.0) }));
    assert!(qt.insert(Item { id: 2, point: pt(5.0, 5.0) }));
    assert!(qt.insert(Item { id: 3, point: pt(83.0, 83.0) }));

    let hits = qt.query(r(0.0, 0.0, 84.0, 84.0));
    assert_eq!(ids(&hits), vec![1,2,3]);
}
