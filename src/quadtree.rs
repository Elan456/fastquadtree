use std::collections::HashSet;
use crate::geom::{Point, Rect, dist_sq_point_to_rect, dist_sq_points};

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Item {
    pub id: u64,
    pub point: Point, 
}


pub struct QuadTree {
    pub boundary: Rect,
    pub items: Vec<Item>,
    pub capacity: usize,
    pub children: Option<Box<[QuadTree; 4]>>,
    depth: usize,
    max_depth: usize,
}

// Child index mapping (y increases upward or downward, both fine):
// 0: (x < cx, y < cy)
// 1: (x >= cx, y < cy)
// 2: (x < cx, y >= cy)
// 3: (x >= cx, y >= cy)
fn child_index_for_point(b: &Rect, p: &Point) -> usize {
    let cx = 0.5 * (b.min_x + b.max_x);
    let cy = 0.5 * (b.min_y + b.max_y);
    let x_ge = (p.x >= cx) as usize; // right half-bit
    let y_ge = (p.y >= cy) as usize; // upper or lower half-bit
    (y_ge << 1) | x_ge
}

impl QuadTree {
    pub fn new(boundary: Rect, capacity: usize) -> Self {
        QuadTree {
            boundary,
            items: Vec::new(),
            capacity,
            children: None,
            depth: 0,
            max_depth: usize::MAX,
        }
    }

    pub fn new_with_max_depth(boundary: Rect, capacity: usize, max_depth: usize) -> Self {
        QuadTree {
            boundary,
            items: Vec::new(),
            capacity,
            children: None,
            depth: 0,
            max_depth: max_depth,
        }
    }

    pub fn new_child(boundary: Rect, capacity: usize, depth: usize, max_depth: usize) -> Self {
        QuadTree {
            boundary,
            items: Vec::new(),
            capacity,
            children: None,
            depth: depth,
            max_depth: max_depth,
        }
    }

    // Returns True if the item is inserted successfully
    pub fn insert(&mut self, item: Item) -> bool {
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
        let cx = 0.5 * (self.boundary.min_x + self.boundary.max_x);
        let cy = 0.5 * (self.boundary.min_y + self.boundary.max_y);

        let quads = [
            Rect { min_x: self.boundary.min_x, min_y: self.boundary.min_y, max_x: cx,               max_y: cy               }, // 0
            Rect { min_x: cx,                    min_y: self.boundary.min_y, max_x: self.boundary.max_x, max_y: cy               }, // 1
            Rect { min_x: self.boundary.min_x,   min_y: cy,                  max_x: cx,               max_y: self.boundary.max_y }, // 2
            Rect { min_x: cx,                    min_y: cy,                  max_x: self.boundary.max_x, max_y: self.boundary.max_y }, // 3
        ];

        // Allocate children
        let d = self.depth + 1;
        let mut kids: [QuadTree; 4] = [
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

    pub fn query(&self, range: Rect) -> Vec<Item> {
        let mut out = Vec::new();
        self.query_into(&range, &mut out);
        out
    }

    fn query_into(&self, range: &Rect, out: &mut Vec<Item>) {
        // prune if this node does not intersect the query
        if !self.boundary.intersects(range) {
            return;
        }

        // check items stored at this node
        for it in &self.items {
            if range.contains(&it.point) {
                out.push(*it); // Item is Copy
            }
        }

        // recurse to children
        if let Some(children) = self.children.as_ref() {
            for child in children.iter() {
                child.query_into(range, out);
            }
        }
    }

    pub fn nearest_neighbor(&self, point: Point) -> Option<Item> {
        self.nearest_neighbors_within(point, 1, f64::INFINITY)
            .into_iter()
            .next()
    }

    pub fn nearest_neighbors(&self, point: Point, k: usize) -> Vec<Item> {
        self.nearest_neighbors_within(point, k, f64::INFINITY)
    }

    pub fn nearest_neighbors_within(&self, point: Point, k: usize, max_distance: f64) -> Vec<Item> {
        if k == 0 {
            return Vec::new();
        }

        let mut picked = HashSet::<u64>::new();
        let mut out = Vec::with_capacity(k);
        let max_d2 = max_distance * max_distance;

        for _ in 0..k {
            // stack holds (node_ref, bbox_distance_sq)
            let mut stack: Vec<(&QuadTree, f64)> = Vec::new();
            stack.push((self, dist_sq_point_to_rect(&point, &self.boundary)));

            let mut best: Option<Item> = None;
            let mut best_d2 = max_d2;

            while let Some((node, node_d2)) = stack.pop() {
                // prune by bbox distance vs current best
                if node_d2 >= best_d2 {
                    continue;
                }

                if let Some(children) = node.children.as_ref() {
                    // compute and sort children by bbox distance, push farthest first
                    let mut kids: Vec<(&QuadTree, f64)> = children
                        .iter()
                        .map(|c| (c, dist_sq_point_to_rect(&point, &c.boundary)))
                        .filter(|(_, d2)| *d2 < best_d2)
                        .collect();

                    kids.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
                    for entry in kids {
                        stack.push(entry);
                    }
                } else {
                    // leaf scan
                    for it in &node.items {
                        if picked.contains(&it.id) {
                            continue;
                        }
                        let d2 = dist_sq_points(&point, &it.point);
                        if d2 < best_d2 {
                            best_d2 = d2;
                            best = Some(*it);
                        }
                    }
                }
            }

            if let Some(it) = best {
                picked.insert(it.id);
                out.push(it);
            } else {
                break; // no more neighbors that beat the cap
            }
        }

        out
    }

    // Traverses the entire quadtree and returns a list of all rectangle boundaries.
    pub fn get_all_rectangles(&self) -> Vec<Rect> {
        let mut rectangles = Vec::new();
        self.collect_rectangles(&mut rectangles);
        rectangles
    }

    // Helper method to recursively collect all rectangle boundaries
    fn collect_rectangles(&self, rectangles: &mut Vec<Rect>) {
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
    pub fn delete(&mut self, id: u64, point: Point) -> bool {
        if !self.boundary.contains(&point) {
            return false;
        }
        // Path-local merge is handled during recursion; avoid a second full walk.
        self.delete_internal(id, point)
    }

    fn delete_internal(&mut self, id: u64, point: Point) -> bool {
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


}
