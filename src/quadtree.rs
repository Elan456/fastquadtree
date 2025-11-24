use crate::geom::{Point, Rect, dist_sq_point_to_rect, dist_sq_points, Coord, mid};
use smallvec::SmallVec;
use serde::{Serialize, Deserialize};
use bincode::config::standard;
use bincode::serde::{encode_to_vec, decode_from_slice};

#[derive(Copy, Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct Item<T: Coord> {
    pub id: u64,
    pub point: Point<T>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct QuadTree<T: Coord> {
    pub boundary: Rect<T>,
    pub items: Vec<Item<T>>,
    pub capacity: usize,
    pub children: Option<Box<[QuadTree<T>; 4]>>,
    depth: usize,
    max_depth: usize,
}

// Child index mapping (y increases upward or downward, both fine):
// 0: (x < cx, y < cy)
// 1: (x >= cx, y < cy)
// 2: (x < cx, y >= cy)
// 3: (x >= cx, y >= cy)
#[inline(always)]
fn child_index_for_point<T: Coord>(b: &Rect<T>, p: &Point<T>) -> usize {
    let cx = mid(b.min_x, b.max_x);
    let cy = mid(b.min_y, b.max_y);
    let x_ge = (p.x >= cx) as usize; // right half-bit
    let y_ge = (p.y >= cy) as usize; // upper or lower half-bit
    (y_ge << 1) | x_ge
}

impl<T: Coord> QuadTree<T> {
    pub fn new(boundary: Rect<T>, capacity: usize, max_depth: usize) -> Self {
        QuadTree {
            boundary,
            items: Vec::with_capacity(capacity),
            capacity,
            children: None,
            depth: 0,
            max_depth: max_depth,
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::error::EncodeError> {
        encode_to_vec(self, standard())
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::error::DecodeError> {
        let (qt, _len): (Self, usize) = decode_from_slice(bytes, standard())?;
        Ok(qt)
    }

    pub fn new_child(boundary: Rect<T>, capacity: usize, depth: usize, max_depth: usize) -> Self {
        QuadTree {
            boundary,
            items: Vec::with_capacity(capacity),
            capacity,
            children: None,
            depth: depth,
            max_depth: max_depth,
        }
    }

    // Returns True if the item is inserted successfully
    pub fn insert(&mut self, item: Item<T>) -> bool {
        if !self.boundary.contains(&item.point) {
            return false;
        }

        // If children is None, we are a leaf node
        if self.children.is_none() {
            // If we have room or we are at the max depth, store it here
            if self.items.len() < self.capacity || self.depth >= self.max_depth {
                // We have room to store it here
                self.items.push(item);
                return true;
            }
            self.split();
        }

        // Need to insert this item into the right child
        // Internal node: delegate to a child
        let idx = child_index_for_point(&self.boundary, &item.point);
        if let Some(children) = self.children.as_mut() {
            return children[idx].insert(item);
        }

        return true;
    }

    pub fn split(&mut self){
        // Create child rectangles
        let cx = mid(self.boundary.min_x, self.boundary.max_x);
        let cy = mid(self.boundary.min_y, self.boundary.max_y);

        let quads = [
            Rect { min_x: self.boundary.min_x, min_y: self.boundary.min_y, max_x: cx,               max_y: cy               }, // 0
            Rect { min_x: cx,                    min_y: self.boundary.min_y, max_x: self.boundary.max_x, max_y: cy               }, // 1
            Rect { min_x: self.boundary.min_x,   min_y: cy,                  max_x: cx,               max_y: self.boundary.max_y }, // 2
            Rect { min_x: cx,                    min_y: cy,                  max_x: self.boundary.max_x, max_y: self.boundary.max_y }, // 3
        ];

        // Allocate children
        let d = self.depth + 1;
        let mut kids: [QuadTree<T>; 4] = [
            QuadTree::new_child(quads[0], self.capacity, d, self.max_depth),
            QuadTree::new_child(quads[1], self.capacity, d, self.max_depth),
            QuadTree::new_child(quads[2], self.capacity, d, self.max_depth),
            QuadTree::new_child(quads[3], self.capacity, d, self.max_depth),
        ];
        // Move existing items down
        for it in self.items.drain(..) {
            let idx = child_index_for_point(&self.boundary, &it.point);
            kids[idx].insert(it);
        }
        self.children = Some(Box::new(kids));
    }

    #[inline(always)]
    fn rect_contains_rect(a: &Rect<T>, b: &Rect<T>) -> bool {
        a.min_x <= b.min_x && a.min_y <= b.min_y &&
        a.max_x >= b.max_x && a.max_y >= b.max_y
    }

    pub fn query(&self, range: Rect<T>) -> Vec<(u64, T, T)> {
        #[derive(Copy, Clone)]
        enum Mode { Filter, ReportAll }

        // Hoist bounds for tight leaf checks
        let rx0 = range.min_x;
        let ry0 = range.min_y;
        let rx1 = range.max_x;
        let ry1 = range.max_y;

        let mut out: Vec<(u64, T, T)> = Vec::with_capacity(128);
        let mut stack: SmallVec<[(&QuadTree<T>, Mode); 64]> = SmallVec::new();
        stack.push((self, Mode::Filter));

        while let Some((node, mode)) = stack.pop() {
            match mode {
                Mode::ReportAll => {
                    if let Some(children) = node.children.as_ref() {
                        // Entire subtree is inside the query.
                        // No filtering, just recurse in ReportAll.
                        stack.push((&children[0], Mode::ReportAll));
                        stack.push((&children[1], Mode::ReportAll));
                        stack.push((&children[2], Mode::ReportAll));
                        stack.push((&children[3], Mode::ReportAll));
                    } else {
                        // Leaf: append all items, no per-point test
                        let items = &node.items;
                        out.reserve(items.len());
                        out.extend(items.iter().map(|it| (it.id, it.point.x, it.point.y)));
                    }
                }

                Mode::Filter => {
                    // Node cull
                    if !range.intersects(&node.boundary) {
                        continue;
                    }

                    // Full cover: switch to ReportAll
                    if Self::rect_contains_rect(&range, &node.boundary) {
                        stack.push((node, Mode::ReportAll));
                        continue;
                    }

                    // Partial overlap
                    if let Some(children) = node.children.as_ref() {
                        // Only push intersecting children
                        let c0 = &children[0];
                        if range.intersects(&c0.boundary) { stack.push((c0, Mode::Filter)); }
                        let c1 = &children[1];
                        if range.intersects(&c1.boundary) { stack.push((c1, Mode::Filter)); }
                        let c2 = &children[2];
                        if range.intersects(&c2.boundary) { stack.push((c2, Mode::Filter)); }
                        let c3 = &children[3];
                        if range.intersects(&c3.boundary) { stack.push((c3, Mode::Filter)); }
                    } else {
                        // Leaf scan with tight predicate
                        let items = &node.items;
                        // Reserve a little to reduce reallocs if many will pass
                        out.reserve(items.len().min(64));
                        for it in items {
                            let p = &it.point;
                            if p.x >= rx0 && p.x < rx1 && p.y >= ry0 && p.y < ry1 {
                                out.push((it.id, p.x, p.y));
                            }
                        }
                    }
                }
            }
        }

        out
    }

     // Default: unbounded search when max_distance == 0
    pub fn nearest_neighbor(&self, point: Point<T>) -> Option<Item<T>> {
        self.nearest_neighbors_within(point, 1, T::zero())
            .into_iter()
            .next()
    }

    // Default: unbounded search when max_distance == 0
    pub fn nearest_neighbors(&self, point: Point<T>, k: usize) -> Vec<Item<T>> {
        self.nearest_neighbors_within(point, k, T::zero())
    }

    // If max_distance == 0, treat as "no max"
    pub fn nearest_neighbors_within(
        &self,
        point: Point<T>,
        k: usize,
        max_distance: T,
    ) -> Vec<Item<T>> {
        if k == 0 {
            return Vec::new();
        }

        let mut picked = std::collections::HashSet::<u64>::new();
        let mut out = Vec::with_capacity(k);

        // Fixed cap for all iterations
        let cap2: Option<T> = if max_distance == T::zero() {
            None
        } else {
            Some(max_distance * max_distance)
        };

        for _ in 0..k {
            // stack holds (node_ref, bbox_distance_sq)
            let mut stack: Vec<(&QuadTree<T>, T)> = Vec::new();
            stack.push((self, dist_sq_point_to_rect(&point, &self.boundary)));

            let mut best: Option<Item<T>> = None;
            // Best for this iteration, starts at the cap (or None for unbounded)
            let mut best_d2: Option<T> = cap2;

            while let Some((node, node_d2)) = stack.pop() {
                // prune if strictly farther than current best
                if let Some(b) = best_d2 {
                    if node_d2 > b {
                        continue;
                    }
                }

                if let Some(children) = node.children.as_ref() {
                    // Push children farthest-first so nearest is popped first
                    let mut kids: Vec<(&QuadTree<T>, T)> = children
                        .iter()
                        .map(|c| (c, dist_sq_point_to_rect(&point, &c.boundary)))
                        .filter(|&(_, d2)| best_d2.map(|b| d2 < b).unwrap_or(true))
                        .collect();

                    kids.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    for e in kids {
                        stack.push(e);
                    }
                } else {
                    // leaf scan
                    for it in &node.items {
                        if picked.contains(&it.id) {
                            continue;
                        }
                        let d2 = dist_sq_points(&point, &it.point);
                        // strict < keeps cap exclusive and picks exactly one per iteration
                        if best_d2.map(|b| d2 < b).unwrap_or(true) {
                            best_d2 = Some(d2);
                            best = Some(*it);
                        }
                    }
                }
            }

            if let Some(it) = best {
                picked.insert(it.id);
                out.push(it);
                // Note: do NOT tighten cap here
            } else {
                break;
            }
        }

        out
    }

    // Traverses the entire quadtree and returns a list of all rectangle boundaries.
    pub fn get_all_node_boundaries(&self) -> Vec<Rect<T>> {
        let mut rectangles = Vec::new();
        self.collect_rectangles(&mut rectangles);
        rectangles
    }

    // Helper method to recursively collect all rectangle boundaries
    fn collect_rectangles(&self, rectangles: &mut Vec<Rect<T>>) {
        // Add this node's boundary
        rectangles.push(self.boundary);
        
        // Recursively collect from children if they exist
        if let Some(children) = self.children.as_ref() {
            for child in children.iter() {
                child.collect_rectangles(rectangles);
            }
        }
    }

    // Deletes an item by ID and location. Returns true if removed.
    pub fn delete(&mut self, id: u64, point: Point<T>) -> bool {
        if !self.boundary.contains(&point) {
            return false;
        }
        // Path-local merge is handled during recursion; avoid a second full walk.
        self.delete_internal(id, point)
    }

    fn delete_internal(&mut self, id: u64, point: Point<T>) -> bool {
        // Leaf: remove in-place
        if self.children.is_none() {
            if let Some(pos) = self.items.iter().position(|it|
                it.id == id && it.point.x == point.x && it.point.y == point.y
            ) {
                self.items.swap_remove(pos);
                return true;
            }
            return false;
        }

        // Internal: route to the child that contains the point
        let idx = child_index_for_point(&self.boundary, &point);
        if let Some(children) = self.children.as_mut() {
            let removed = children[idx].delete_internal(id, point);
            if removed {
                // Try to merge only at this node on the way back up.
                self.try_merge();
            }
            return removed;
        }
        false
    }

    // Attempts to merge this node's children back into this node if possible.
    // Local check only: no recursion. O(1) per call except for moving items.
    fn try_merge(&mut self) {
        let Some(children) = self.children.as_mut() else { return; };

        // Only merge if all children are leaves
        if !children.iter().all(|c| c.children.is_none()) {
            return;
        }

        let total: usize = children.iter().map(|c| c.items.len()).sum();
        if total <= self.capacity {
            // Move items up without cloning
            let mut merged = Vec::with_capacity(total);
            for c in children.iter_mut() {
                merged.append(&mut c.items);
            }
            self.items = merged;
            self.children = None;
        }
    }



    // Returns the total number of items in this subtree
    pub fn count_items(&self) -> usize {
        let mut count = self.items.len();
        if let Some(children) = self.children.as_ref() {
            for child in children.iter() {
                count += child.count_items();
            }
        }
        count
    }

    pub fn get_max_depth(&self) -> usize {
        self.max_depth
    }


}
