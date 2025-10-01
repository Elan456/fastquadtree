use fastquadtree::{Point, Rect, Item, QuadTree};

fn r(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
    Rect { min_x: x0, min_y: y0, max_x: x1, max_y: y1 }
}

fn pt(x: f32, y: f32) -> Point { Point { x, y } }

#[test]
fn rect_contains_half_open() {
    let a = r(0.0, 0.0, 10.0, 10.0);

    // inside
    assert!(a.contains(&pt(0.0, 0.0)));
    assert!(a.contains(&pt(5.0, 5.0)));
    assert!(a.contains(&pt(9.999, 9.999)));

    // on max edges should be excluded
    assert!(!a.contains(&pt(10.0, 5.0)));
    assert!(!a.contains(&pt(5.0, 10.0)));
    assert!(!a.contains(&pt(10.0, 10.0)));

    // outside
    assert!(!a.contains(&pt(-0.1, 0.0)));
    assert!(!a.contains(&pt(0.0, -0.1)));
}

#[test]
fn insert_inside_leaf_until_capacity() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 2);

    let ok1 = qt.insert(Item { id: 1, point: pt(10.0, 10.0) });
    let ok2 = qt.insert(Item { id: 2, point: pt(20.0, 20.0) });

    assert!(ok1 && ok2, "inserts up to capacity should succeed");
}

#[test]
fn insert_outside_returns_false() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 2);

    assert!(!qt.insert(Item { id: 1, point: pt(-1.0, 50.0) }));
    assert!(!qt.insert(Item { id: 2, point: pt(50.0, 101.0) }));
}

#[test]
fn split_then_midline_inserts_succeed() {
    // capacity 1 forces a split on the second insert
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 1);

    // first insert goes to parent as a leaf
    assert!(qt.insert(Item { id: 1, point: pt(25.0, 25.0) }));

    // second insert triggers split; midlines are at x=50, y=50
    // This point lies on the vertical midline x = 50, above center
    // With half-open rectangles and child routing using >= for right/top,
    // this must insert into the right-top child and succeed.
    assert!(qt.insert(Item { id: 2, point: pt(50.0, 75.0) }));

    // Another midline case on horizontal midline y = 50, left side
    assert!(qt.insert(Item { id: 3, point: pt(25.0, 50.0) }));

    // Exactly at center should also succeed and be routed to right-top
    assert!(qt.insert(Item { id: 4, point: pt(50.0, 50.0) }));
}

#[test]
fn many_inserts_all_succeed_when_inside() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 1000.0, 1000.0), 4);

    // generate a grid of interior points
    let mut ok = 0usize;
    let mut id = 1u64;
    for x in (10..1000).step_by(100) {
        for y in (10..1000).step_by(100) {
            if qt.insert(Item { id, point: pt(x as f32, y as f32) }) {
                ok += 1;
            }
            id += 1;
        }
    }
    assert_eq!(ok, 10 * 10, "all interior inserts should succeed");
}

#[test]
fn boundary_points_respect_half_open_rule() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 2);

    // Allowed: on min edges
    assert!(qt.insert(Item { id: 1, point: pt(0.0, 0.0) }));
    assert!(qt.insert(Item { id: 2, point: pt(0.0, 50.0) }));

    // Not allowed: on max edges
    assert!(!qt.insert(Item { id: 3, point: pt(100.0, 0.0) }));
    assert!(!qt.insert(Item { id: 4, point: pt(0.0, 100.0) }));
    assert!(!qt.insert(Item { id: 5, point: pt(100.0, 100.0) }));
}
