// tests/rect_quadtree_tests.rs (or inline with #[cfg(test)] mod tests { ... })

use fastquadtree::{Rect, RectQuadTree, RectItem};

fn r(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
    Rect { min_x: x0, min_y: y0, max_x: x1, max_y: y1 }
}

fn item(id: u64, x0: f32, y0: f32, x1: f32, y1: f32) -> RectItem {
    RectItem { id, rect: r(x0, y0, x1, y1) }
}

fn ids(v: &[(u64, Rect)]) -> Vec<u64> {
    let mut out: Vec<u64> = v.iter().map(|it| it.0).collect();
    out.sort_unstable();
    out
}

#[test]
fn query_on_empty_tree_is_empty() {
    let qt = RectQuadTree::new(r(0.0, 0.0, 100.0, 100.0), 4);
    let hits = qt.query(r(10.0, 10.0, 20.0, 20.0));
    assert!(hits.is_empty());
}

#[test]
fn insert_outside_boundary_returns_false() {
    let mut qt = RectQuadTree::new(r(0.0, 0.0, 100.0, 100.0), 4);
    // Well outside with a gap
    let ok = qt.insert(item(1, 200.0, 200.0, 210.0, 210.0));
    assert!(!ok);
    assert!(qt.query(r(0.0, 0.0, 100.0, 100.0)).is_empty());
}

#[test]
fn leaf_insert_no_split_and_inclusive_edge_touch_query() {
    let mut qt = RectQuadTree::new(r(0.0, 0.0, 100.0, 100.0), 8);
    let a = item(1, 10.0, 10.0, 20.0, 20.0);
    assert!(qt.insert(a));

    // Query that only touches the right edge of `a` at x=20
    let hits_touch = qt.query(r(20.0, 0.0, 21.0, 100.0));
    assert_eq!(ids(&hits_touch), vec![1]);

    // Query that clearly intersects
    let hits_inside = qt.query(r(15.0, 15.0, 25.0, 25.0));
    assert_eq!(ids(&hits_inside), vec![1]);
}

#[test]
fn new_with_max_depth_prevents_split() {
    // max_depth = 0 at root, force everything to stay in the root even beyond capacity
    let mut qt = RectQuadTree::new_with_max_depth(r(0.0, 0.0, 100.0, 100.0), 1, 0);
    assert!(qt.insert(item(1, 10.0, 10.0, 20.0, 20.0)));
    assert!(qt.insert(item(2, 30.0, 30.0, 35.0, 35.0)));
    assert!(qt.insert(item(3, 60.0, 60.0, 70.0, 70.0)));
    // Depth limit should keep it as a leaf
    assert!(qt.children.is_none());
    assert_eq!(qt.count_items(), 3);

    // Full-cover query hits all (report-all branch)
    let hits = qt.query(r(0.0, 0.0, 100.0, 100.0));
    assert_eq!(ids(&hits), vec![1, 2, 3]);
}

#[test]
fn split_child_routing_and_straddler_stays_at_parent() {
    // capacity 1 to force split as soon as we try to insert the second item
    let mut qt = RectQuadTree::new(r(0.0, 0.0, 100.0, 100.0), 1);

    // Rectangles that fully fit each quadrant
    let q0 = item(1,  5.0,  5.0,  10.0, 10.0); // left-bottom
    let q1 = item(2, 60.0,  5.0,  70.0, 10.0); // right-bottom
    let q2 = item(3,  5.0, 60.0,  10.0, 70.0); // left-top
    let q3 = item(4, 60.0, 60.0,  70.0, 70.0); // right-top

    // Straddler right at center that should stay at parent
    let s  = item(5, 49.0, 49.0, 51.0, 51.0);

    assert!(qt.insert(q0)); // no split yet
    assert!(qt.insert(q1)); // triggers split
    assert!(qt.insert(q2));
    assert!(qt.insert(q3));
    assert!(qt.insert(s));  // does not fit a child, stays in parent

    // Parent should have the straddler, children should exist
    assert!(qt.children.is_some());
    assert!(qt.items.iter().any(|it| it.id == 5));

    // Query tight boxes inside each child so they do not intersect the straddler
    let lb = qt.query(r(0.0, 0.0, 45.0, 45.0));
    assert_eq!(ids(&lb), vec![1]);

    let rb = qt.query(r(55.0, 0.0, 100.0, 45.0));
    assert_eq!(ids(&rb), vec![2]);

    let lt = qt.query(r(0.0, 55.0, 45.0, 100.0));
    assert_eq!(ids(&lt), vec![3]);

    let rt = qt.query(r(55.0, 55.0, 100.0, 100.0));
    assert_eq!(ids(&rt), vec![4]);

    // Query exactly the root bounds - report-all path
    let all = qt.query(r(0.0, 0.0, 100.0, 100.0));
    assert_eq!(ids(&all), vec![1, 2, 3, 4, 5]);
}

#[test]
fn query_ids_convenience() {
    let mut qt = RectQuadTree::new(r(0.0, 0.0, 100.0, 100.0), 2);
    assert!(qt.insert(item(1, 0.0, 0.0, 10.0, 10.0)));
    assert!(qt.insert(item(2, 20.0, 20.0, 30.0, 30.0)));

    let mut got = qt.query_ids(r(0.0, 0.0, 100.0, 100.0));
    got.sort_unstable();
    assert_eq!(got, vec![1, 2]);
}

#[test]
fn delete_from_child_triggers_merge_when_possible() {
    // Setup to create children, then delete and allow merge
    let mut qt = RectQuadTree::new(r(0.0, 0.0, 100.0, 100.0), 1);
    let a = item(1,  5.0,  5.0, 10.0, 10.0); // will go to left-bottom
    let b = item(2, 60.0,  5.0, 70.0, 10.0); // will go to right-bottom

    assert!(qt.insert(a)); // no split yet
    assert!(qt.insert(b)); // split happens, a and b in children

    assert!(qt.children.is_some());
    assert!(qt.delete(2, b.rect)); // remove from child, triggers try_merge
    // After deletion, only one item remains in children and capacity=1, so merge up
    assert!(qt.children.is_none());
    assert_eq!(qt.count_items(), 1);
    // Remaining item should be `a` in the parent items
    assert!(qt.items.iter().any(|it| it.id == 1));
}

#[test]
fn delete_no_merge_when_a_child_has_children() {
    // Build a tree where one child will split (becoming non-leaf)
    let mut qt = RectQuadTree::new(r(0.0, 0.0, 100.0, 100.0), 1);

    // Two rectangles in the left-bottom quadrant to force that child to split
    let lb1 = item(1, 5.0,  5.0, 10.0, 10.0); // goes to Q0
    let lb2 = item(2, 12.0, 5.0, 18.0,  9.0); // also Q0 -> that child will split
    let other = item(3, 60.0,  5.0, 70.0, 10.0); // different quadrant

    assert!(qt.insert(lb1));   // no split yet at root
    assert!(qt.insert(other)); // split root, lb1 and other in children
    assert!(qt.insert(lb2));   // insert second into left-bottom child -> that child splits

    // Confirm left-bottom child has its own children
    let children = qt.children.as_ref().unwrap();
    let lb_child = &children[0];
    assert!(lb_child.children.is_some());

    // Delete `other` from its child - parent will call try_merge but should not merge
    assert!(qt.delete(3, other.rect));
    assert!(qt.children.is_some(), "should not merge because a child is non-leaf");
}

#[test]
fn delete_no_merge_when_capacity_would_be_exceeded() {
    // Parent capacity=1, keep one straddler at parent plus at least one child item
    let mut qt = RectQuadTree::new(r(0.0, 0.0, 100.0, 100.0), 1);

    // Straddler at center stays in parent
    let s  = item(100, 49.0, 49.0, 51.0, 51.0);
    // Two child items in different quadrants
    let a = item(1,  5.0,  5.0, 10.0, 10.0);
    let b = item(2, 60.0,  5.0, 70.0, 10.0);

    assert!(qt.insert(s)); // parent holds this
    assert!(qt.insert(a)); // triggers split, s stays at parent, a goes to child
    assert!(qt.insert(b)); // goes to another child

    // Delete one child item
    assert!(qt.delete(2, b.rect));

    // Now children total = 1 and parent items = 1 -> 1 + 1 > capacity, so no merge
    assert!(qt.children.is_some(), "should not merge because capacity would be exceeded");
}

#[test]
fn delete_parent_straddler_and_delete_outside() {
    let mut qt = RectQuadTree::new(r(0.0, 0.0, 100.0, 100.0), 1);

    // Keep a straddler at parent and one child item
    let s  = item(7, 49.0, 49.0, 51.0, 51.0);
    let a  = item(8,  5.0,  5.0, 10.0, 10.0);

    assert!(qt.insert(s));
    assert!(qt.insert(a)); // split occurs

    // Delete the parent-held straddler
    assert!(qt.delete(7, s.rect));
    assert!(!qt.items.iter().any(|it| it.id == 7));

    // Try to delete something outside the root - should return false
    assert!(!qt.delete(999, r(200.0, 200.0, 210.0, 210.0)));
}

#[test]
fn count_items_and_get_all_node_boundaries() {
    let mut qt = RectQuadTree::new(r(0.0, 0.0, 100.0, 100.0), 1);

    // Build a small hierarchy
    assert!(qt.insert(item(1,  5.0,  5.0, 10.0, 10.0))); // cause split with next
    assert!(qt.insert(item(2, 60.0,  5.0, 70.0, 10.0)));
    assert!(qt.insert(item(3,  5.0, 60.0, 10.0, 70.0))); // fill another child

    assert_eq!(qt.count_items(), 3);

    // Boundaries should include at least the root and its four children after the split
    let nodes = qt.get_all_node_boundaries();
    assert!(!nodes.is_empty());
    // Root must be present as first collected by implementation
    assert_eq!(nodes[0], r(0.0, 0.0, 100.0, 100.0));
    // There should be more than 1 node since we split
    assert!(nodes.len() >= 5);
}
