use fastquadtree::{Point, Rect, Item, QuadTree};

fn r(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
    Rect { min_x: x0, min_y: y0, max_x: x1, max_y: y1 }
}
fn pt(x: f32, y: f32) -> Point { Point { x, y } }
fn ids(v: &[Item]) -> Vec<u64> {
    let mut out: Vec<u64> = v.iter().map(|it| it.id).collect();
    out.sort_unstable();
    out
}
fn dist2(a: Point, b: Point) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}

#[test]
fn nearest_neighbor_zero_poins() {
    // keep user's original test name
    let qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 4);
    assert!(qt.nearest_neighbor(pt(10.0, 10.0)).is_none());
}

#[test]
fn nearest_neighbor_one_point() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 4);
    qt.insert(Item { id: 1, point: pt(50.0, 50.0) });
    assert_eq!(qt.nearest_neighbor(pt(10.0, 10.0)).unwrap().id, 1);
}

#[test]
fn nearest_neighbor_two_points() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 4);
    qt.insert(Item { id: 1, point: pt(50.0, 50.0) });
    qt.insert(Item { id: 2, point: pt(60.0, 60.0) });
    assert_eq!(qt.nearest_neighbor(pt(10.0, 10.0)).unwrap().id, 1);
}

#[test]
fn nn_exact_hit_distance_zero() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 20.0, 20.0), 8);
    qt.insert(Item { id: 7, point: pt(10.0, 10.0) });
    qt.insert(Item { id: 8, point: pt(3.0, 4.0) });
    let q = pt(10.0, 10.0);
    assert_eq!(qt.nearest_neighbor(q).unwrap().id, 7);

    // ask for more neighbors than exist
    let res = qt.nearest_neighbors(q, 5);
    let uniq = ids(&res);
    assert_eq!(res.len(), uniq.len());
    assert!(uniq.contains(&7));
    assert!(uniq.contains(&8));
}

#[test]
fn knn_basic_ordering_no_split() {
    // capacity high so we do not split
    let mut qt = QuadTree::new(r(0.0, 0.0, 40.0, 40.0), 16);
    // unique distances from query (6, 6)
    qt.insert(Item { id: 1, point: pt(5.0, 5.0) });     // d2 = 2
    qt.insert(Item { id: 2, point: pt(6.5, 6.0) });     // d2 = 0.25
    qt.insert(Item { id: 3, point: pt(10.0, 10.0) });   // d2 = 32
    qt.insert(Item { id: 4, point: pt(7.8, 7.8) });     // d2 = 6.48
    qt.insert(Item { id: 5, point: pt(20.0, 20.0) });   // d2 = 392
    let q = pt(6.0, 6.0);
    let res = qt.nearest_neighbors(q, 5);
    let order: Vec<u64> = res.iter().map(|it| it.id).collect();
    assert_eq!(order, vec![2, 1, 4, 3, 5]);

    // distances should be nondecreasing
    let d: Vec<f32> = res.iter().map(|it| dist2(q, it.point)).collect();
    for w in d.windows(2) {
        assert!(w[0] <= w[1] + 1e-12);
    }
}

#[test]
fn knn_respects_capacity_and_split() {
    // capacity small so we force splits across all quadrants
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 1);
    qt.insert(Item { id: 1, point: pt(10.0, 10.0) });
    qt.insert(Item { id: 2, point: pt(90.0, 10.0) });
    qt.insert(Item { id: 3, point: pt(10.0, 90.0) });
    qt.insert(Item { id: 4, point: pt(90.0, 90.0) });
    qt.insert(Item { id: 5, point: pt(12.0, 12.0) });

    let q = pt(11.0, 11.0);
    let nn = qt.nearest_neighbor(q).unwrap();
    assert_eq!(nn.id, 5);

    let res = qt.nearest_neighbors(q, 3);
    let order: Vec<u64> = res.iter().map(|it| it.id).collect();
    // next nearest after id 5 should be id 1
    assert_eq!(order[0], 5);
    assert_eq!(order[1], 1);
}

#[test]
fn knn_k_greater_than_len_no_duplicates() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 10.0, 10.0), 2);
    qt.insert(Item { id: 1, point: pt(1.0, 1.0) });
    qt.insert(Item { id: 2, point: pt(2.0, 2.0) });
    qt.insert(Item { id: 3, point: pt(9.0, 9.0) });

    let res = qt.nearest_neighbors(pt(0.0, 0.0), 10);
    let idset = ids(&res);
    assert_eq!(res.len(), 3);
    assert_eq!(res.len(), idset.len());
    assert!(idset.contains(&1));
    assert!(idset.contains(&2));
    assert!(idset.contains(&3));
}

#[test]
fn within_strictly_less_than_max_distance() {
    // current implementation uses d2 < max_d2, not <=
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 4);
    let p = pt(10.0, 10.0);
    qt.insert(Item { id: 1, point: pt(13.0, 14.0) }); // distance = 5.0
    qt.insert(Item { id: 2, point: pt(30.0, 30.0) });

    // exactly at max distance should return empty
    let res_eq = qt.nearest_neighbors_within(p, 1, 5.0);
    assert!(res_eq.is_empty());

    // slightly larger should include the point
    let res_gt = qt.nearest_neighbors_within(p, 1, 5.0001);
    assert_eq!(res_gt.len(), 1);
    assert_eq!(res_gt[0].id, 1);
}

#[test]
fn equidistant_tie_returns_one_of_the_candidates() {
    // two points equidistant from query
    let mut qt = QuadTree::new(r(0.0, 0.0, 50.0, 50.0), 4);
    qt.insert(Item { id: 10, point: pt(10.0, 10.0) });
    qt.insert(Item { id: 20, point: pt(10.0, 12.0) });
    let q = pt(10.0, 11.0);
    let nn = qt.nearest_neighbor(q).unwrap().id;
    assert!(nn == 10 || nn == 20);
}

#[test]
fn identical_locations_two_items_pick_one_due_to_strict_lt() {
    // both items at the same coordinates
    // with current algorithm and strict <, k=2 will only return one of them
    let mut qt = QuadTree::new(r(0.0, 0.0, 10.0, 10.0), 4);
    qt.insert(Item { id: 1, point: pt(5.0, 5.0) });
    qt.insert(Item { id: 2, point: pt(5.0, 5.0) });

    let q = pt(4.5, 5.0);
    let res = qt.nearest_neighbors(q, 2);
    assert_eq!(res.len(), 2, "an item shouldnt be ignored because it is in the same position as another");
    assert!(res[0].id == 1 || res[0].id == 2);
}

#[test]
fn query_far_outside_root_works() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 4);
    qt.insert(Item { id: 1, point: pt(0.0, 0.0) });
    qt.insert(Item { id: 2, point: pt(100.0, 100.0) });
    qt.insert(Item { id: 3, point: pt(100.0, 0.0) });
    qt.insert(Item { id: 4, point: pt(0.0, 100.0) });

    // query well outside the root bounds
    let q = pt(-100.0, -100.0);
    let nn = qt.nearest_neighbor(q).unwrap();
    assert_eq!(nn.id, 1);
}

#[test]
fn ordering_is_by_distance_even_after_splits() {
    let mut qt = QuadTree::new(r(0.0, 0.0, 64.0, 64.0), 1); // force deep subdivisions
    // place a small grid of points
    let mut id = 1u64;
    for y in (4..=60).step_by(8) {
        for x in (4..=60).step_by(8) {
            qt.insert(Item { id, point: pt(x as f32, y as f32) });
            id += 1;
        }
    }
    let q = pt(7.0, 7.0);
    let res = qt.nearest_neighbors(q, 10);
    assert!(!res.is_empty());

    // distances should be nondecreasing
    let d: Vec<f32> = res.iter().map(|it| dist2(q, it.point)).collect();
    for w in d.windows(2) {
        assert!(w[0] <= w[1] + 1e-12);
    }

    // the very first neighbor should be close to (4,4) or (12,4) or (4,12)
    let first = res[0].point;
    let candidates = [pt(4.0, 4.0), pt(12.0, 4.0), pt(4.0, 12.0), pt(12.0, 12.0)];
    let ok = candidates.iter().any(|&c| (first.x - c.x).abs() < 1e-9 && (first.y - c.y).abs() < 1e-9);
    assert!(ok, "unexpected first NN {:?}", first);
}

#[test]
fn midline_and_center_points_are_handled() {
    // insert points that lie on child midlines and the exact center
    let mut qt = QuadTree::new(r(0.0, 0.0, 100.0, 100.0), 2);
    let center = pt(50.0, 50.0);
    qt.insert(Item { id: 1, point: center });
    qt.insert(Item { id: 2, point: pt(50.0, 10.0) }); // vertical midline
    qt.insert(Item { id: 3, point: pt(10.0, 50.0) }); // horizontal midline
    qt.insert(Item { id: 4, point: pt(75.0, 50.0) });

    // query at center should return the center item
    assert_eq!(qt.nearest_neighbor(center).unwrap().id, 1);

    // query near vertical midline should find the midline point or center
    let nn = qt.nearest_neighbor(pt(50.0, 12.0)).unwrap().id;
    assert!(nn == 2 || nn == 1);
}
