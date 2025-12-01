use smallvec::SmallVec;
use crate::geom::{Point, Rect, Coord, mid, dist_sq_point_to_rect};
use serde::{Serialize, Deserialize};
use bincode::config::standard;
use bincode::serde::{encode_to_vec, decode_from_slice};

#[derive(Copy, Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct RectItem<T: Coord> {
    pub id: u64,
    pub rect: Rect<T>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct RectQuadTree<T: Coord> {
    pub boundary: Rect<T>,
    pub items: Vec<RectItem<T>>,
    pub capacity: usize,
    pub children: Option<Box<[RectQuadTree<T>; 4]>>,
    depth: usize,
    max_depth: usize,
}

// Child index mapping:
// 0: left,  bottom
// 1: right, bottom
// 2: left,  top
// 3: right, top
#[inline(always)]
fn child_index_for_rect<T: Coord>(boundary: &Rect<T>, r: &Rect<T>) -> Option<usize> {
    let cx = mid(boundary.min_x, boundary.max_x);
    let cy = mid(boundary.min_y, boundary.max_y);

    // fits entirely on one side per axis?
    let fits_x = r.max_x <= cx || r.min_x >= cx;
    let fits_y = r.max_y <= cy || r.min_y >= cy;
    if !(fits_x && fits_y) {
        return None;
    }
    // right=1 if min_x >= cx, top=1 if min_y >= cy
    let ix = (r.min_x >= cx) as usize;
    let iy = (r.min_y >= cy) as usize;
    Some(ix | (iy << 1)) // 0..3
}

#[inline(always)]
fn rects_touch_or_intersect<T: Coord>(a: &Rect<T>, b: &Rect<T>) -> bool {
    // Inclusive version so touching edges count
    a.min_x <= b.max_x && a.max_x >= b.min_x &&
    a.min_y <= b.max_y && a.max_y >= b.min_y
}

impl<T: Coord> RectQuadTree<T> {
    pub fn new(boundary: Rect<T>, capacity: usize, max_depth: usize) -> Self {
        RectQuadTree {
            boundary,
            items: Vec::with_capacity(capacity),
            capacity,
            children: None,
            depth: 0,
            max_depth,
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, bincode::error::EncodeError> {
        encode_to_vec(self, standard())
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, bincode::error::DecodeError> {
        let (qt, _len): (Self, usize) = decode_from_slice(bytes, standard())?;
        Ok(qt)
    }

    fn new_child(boundary: Rect<T>, capacity: usize, depth: usize, max_depth: usize) -> Self {
        RectQuadTree {
            boundary,
            items: Vec::with_capacity(capacity),
            capacity,
            children: None,
            depth,
            max_depth,
        }
    }

    /// Insert a rectangle. Returns true if inserted into the tree.
    pub fn insert(&mut self, item: RectItem<T>) -> bool {
        // Discard if completely outside this subtree
        if !rects_touch_or_intersect(&self.boundary, &item.rect) {
            return false;
        }

        // Leaf path
        if self.children.is_none() {
            if self.items.len() < self.capacity || self.depth >= self.max_depth {
                self.items.push(item);
                return true;
            }
            self.split();
        }

        // If it fits a child, delegate
        if let Some(children) = self.children.as_mut() {
            if let Some(idx) = child_index_for_rect(&self.boundary, &item.rect) {
                return children[idx].insert(item);
            }
        }

        // Otherwise keep it here
        self.items.push(item);
        true
    }

    fn split(&mut self) {
    let cx = mid(self.boundary.min_x, self.boundary.max_x);
    let cy = mid(self.boundary.min_y, self.boundary.max_y);

        let quads = [
            Rect { min_x: self.boundary.min_x, min_y: self.boundary.min_y, max_x: cx,                    max_y: cy },
            Rect { min_x: cx,                    min_y: self.boundary.min_y, max_x: self.boundary.max_x, max_y: cy },
            Rect { min_x: self.boundary.min_x,   min_y: cy,                  max_x: cx,                    max_y: self.boundary.max_y },
            Rect { min_x: cx,                    min_y: cy,                  max_x: self.boundary.max_x,  max_y: self.boundary.max_y },
        ];

        let d = self.depth + 1;
        let mut kids: [RectQuadTree<T>; 4] = [
            RectQuadTree::new_child(quads[0], self.capacity, d, self.max_depth),
            RectQuadTree::new_child(quads[1], self.capacity, d, self.max_depth),
            RectQuadTree::new_child(quads[2], self.capacity, d, self.max_depth),
            RectQuadTree::new_child(quads[3], self.capacity, d, self.max_depth),
        ];

        // Move any items that fully fit a child, in place
        let mut i = 0;
        while i < self.items.len() {
            let rect = self.items[i].rect;
            if let Some(idx) = child_index_for_rect(&self.boundary, &rect) {
                let it = self.items.swap_remove(i);
                kids[idx].insert(it);
                // do not advance i, we swapped in a new element at i
            } else {
                i += 1;
            }
        }

        self.children = Some(Box::new(kids));
    }

    #[inline(always)]
    fn rect_contains_rect_inclusive(a: &Rect<T>, b: &Rect<T>) -> bool {
        a.min_x <= b.min_x && a.min_y <= b.min_y &&
        a.max_x >= b.max_x && a.max_y >= b.max_y
    }

    /// Query for all rectangles that touch or intersect the given range.
    /// Returns a Vec of (id, Rect).
    pub fn query(&self, range: Rect<T>) -> Vec<(u64, Rect<T>)> {
        #[derive(Copy, Clone)]
        enum Mode { Filter, ReportAll }

        let mut out: SmallVec<[(u64, Rect<T>); 128]> = SmallVec::new();
        let mut stack: SmallVec<[(&RectQuadTree<T>, Mode); 64]> = SmallVec::new();
        stack.push((self, Mode::Filter));

        while let Some((node, mode)) = stack.pop() {
            match mode {
                Mode::ReportAll => {
                    // All descendants can be appended without per item checks
                    out.reserve(node.items.len());
                    for it in &node.items {
                        out.push((it.id, it.rect));
                    }
                    if let Some(children) = node.children.as_ref() {
                        stack.push((&children[0], Mode::ReportAll));
                        stack.push((&children[1], Mode::ReportAll));
                        stack.push((&children[2], Mode::ReportAll));
                        stack.push((&children[3], Mode::ReportAll));
                    }
                }
                Mode::Filter => {
                    // Node cull
                    if !rects_touch_or_intersect(&range, &node.boundary) {
                        continue;
                    }

                    // Full cover, switch to fast path
                    if Self::rect_contains_rect_inclusive(&range, &node.boundary) {
                        stack.push((node, Mode::ReportAll));
                        continue;
                    }

                    // Partial overlap
                    // Check items at this node
                    for it in &node.items {
                        if rects_touch_or_intersect(&range, &it.rect) {
                            out.push((it.id, it.rect));
                        }
                    }

                    // Recurse into intersecting children
                    if let Some(children) = node.children.as_ref() {
                        let c0 = &children[0];
                        if rects_touch_or_intersect(&range, &c0.boundary) { stack.push((c0, Mode::Filter)); }
                        let c1 = &children[1];
                        if rects_touch_or_intersect(&range, &c1.boundary) { stack.push((c1, Mode::Filter)); }
                        let c2 = &children[2];
                        if rects_touch_or_intersect(&range, &c2.boundary) { stack.push((c2, Mode::Filter)); }
                        let c3 = &children[3];
                        if rects_touch_or_intersect(&range, &c3.boundary) { stack.push((c3, Mode::Filter)); }
                    }
                }
            }
        }

        out.into_vec()
    }

    /// Convenience if you only want ids.
    pub fn query_ids(&self, range: Rect<T>) -> Vec<u64> {
        self.query(range).into_iter().map(|(id, _)| id).collect()
    }

    // Default: unbounded search when max_distance == 0
    pub fn nearest_neighbor(&self, point: Point<T>) -> Option<RectItem<T>> {
        self.nearest_neighbors_within(point, 1, T::zero())
            .into_iter()
            .next()
    }

    // Default: unbounded search when max_distance == 0
    pub fn nearest_neighbors(&self, point: Point<T>, k: usize) -> Vec<RectItem<T>> {
        self.nearest_neighbors_within(point, k, T::zero())
    }

    // If max_distance == 0, treat as "no max"
    // Distance is measured from the query point to the nearest edge of each rectangle
    pub fn nearest_neighbors_within(
        &self,
        point: Point<T>,
        k: usize,
        max_distance: T,
    ) -> Vec<RectItem<T>> {
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
            let mut stack: Vec<(&RectQuadTree<T>, T)> = Vec::new();
            stack.push((self, dist_sq_point_to_rect(&point, &self.boundary)));

            let mut best: Option<RectItem<T>> = None;
            // Best for this iteration, starts at the cap (or None for unbounded)
            let mut best_d2: Option<T> = cap2;

            while let Some((node, node_d2)) = stack.pop() {
                // prune if strictly farther than current best
                if let Some(b) = best_d2 {
                    if node_d2 > b {
                        continue;
                    }
                }

                // Check items at this node (internal nodes can have items that span children)
                for it in &node.items {
                    if picked.contains(&it.id) {
                        continue;
                    }
                    let d2 = dist_sq_point_to_rect(&point, &it.rect);
                    // strict < keeps cap exclusive and picks exactly one per iteration
                    if best_d2.map(|b| d2 < b).unwrap_or(true) {
                        best_d2 = Some(d2);
                        best = Some(*it);
                    }
                }

                if let Some(children) = node.children.as_ref() {
                    // Push children farthest-first so nearest is popped first
                    let mut kids: Vec<(&RectQuadTree<T>, T)> = children
                        .iter()
                        .map(|c| (c, dist_sq_point_to_rect(&point, &c.boundary)))
                        .filter(|&(_, d2)| best_d2.map(|b| d2 < b).unwrap_or(true))
                        .collect();

                    kids.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    for e in kids {
                        stack.push(e);
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

    /// Delete an item by id and rect. Returns true if removed.
    pub fn delete(&mut self, id: u64, rect: Rect<T>) -> bool {
        if !rects_touch_or_intersect(&self.boundary, &rect) {
            return false;
        }
        self.delete_internal(id, rect)
    }

    fn delete_internal(&mut self, id: u64, rect: Rect<T>) -> bool {
        // Try children if the rect fully fits one
        if let Some(children) = self.children.as_mut() {
            if let Some(idx) = child_index_for_rect(&self.boundary, &rect) {
                let removed = children[idx].delete_internal(id, rect);
                if removed {
                    self.try_merge();
                }
                return removed;
            }
        }

        // Remove from this node
        if let Some(pos) = self.items.iter().position(|it| it.id == id && it.rect == rect) {
            self.items.swap_remove(pos);
            return true;
        }
        false
    }

    /// Merge children back into this node if all are leaves and total items fit capacity.
    fn try_merge(&mut self) {
        let Some(children) = self.children.as_mut() else { return; };
        if !children.iter().all(|c| c.children.is_none()) {
            return;
        }

        let total: usize = children.iter().map(|c| c.items.len()).sum();
        if total + self.items.len() <= self.capacity {
            // Pull up all child items
            for c in children.iter_mut() {
                self.items.append(&mut c.items);
            }
            self.children = None;
        }
    }

    /// Count all items in the subtree.
    pub fn count_items(&self) -> usize {
        let mut count = self.items.len();
        if let Some(children) = self.children.as_ref() {
            for child in children.iter() {
                count += child.count_items();
            }
        }
        count
    }

    /// Debug helper: collect all node boundaries in this subtree.
    pub fn get_all_node_boundaries(&self) -> Vec<Rect<T>> {
        let mut rects = Vec::new();
        self.collect_boundaries(&mut rects);
        rects
    }

    fn collect_boundaries(&self, out: &mut Vec<Rect<T>>) {
        out.push(self.boundary);
        if let Some(children) = self.children.as_ref() {
            for child in children.iter() {
                child.collect_boundaries(out);
            }
        }
    }

    pub fn get_max_depth(&self) -> usize {
        self.max_depth
    }
}