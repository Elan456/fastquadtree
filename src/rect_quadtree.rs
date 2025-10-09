use smallvec::SmallVec;
use crate::geom::Rect;

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct RectItem {
    pub id: u64,
    pub rect: Rect,
}

pub struct RectQuadTree {
    pub boundary: Rect,
    pub items: Vec<RectItem>,
    pub capacity: usize,
    pub children: Option<Box<[RectQuadTree; 4]>>,
    depth: usize,
    max_depth: usize,
}

// Child index mapping:
// 0: left,  bottom
// 1: right, bottom
// 2: left,  top
// 3: right, top
#[inline(always)]
fn child_index_for_rect(boundary: &Rect, r: &Rect) -> Option<usize> {
    let cx = 0.5 * (boundary.min_x + boundary.max_x);
    let cy = 0.5 * (boundary.min_y + boundary.max_y);

    // Does r fit entirely on one side of the vertical split?
    let in_left  = r.max_x <= cx;
    let in_right = r.min_x >= cx;

    // Does r fit entirely on one side of the horizontal split?
    let in_bottom = r.max_y <= cy;
    let in_top    = r.min_y >= cy;

    let x_side = if in_left { Some(0) } else if in_right { Some(1) } else { None };
    let y_side = if in_bottom { Some(0) } else if in_top   { Some(1) } else { None };

    match (x_side, y_side) {
        (Some(0), Some(0)) => Some(0),
        (Some(1), Some(0)) => Some(1),
        (Some(0), Some(1)) => Some(2),
        (Some(1), Some(1)) => Some(3),
        _ => None, // Spans split line(s) so it must live at this node
    }
}

#[inline(always)]
fn rects_touch_or_intersect(a: &Rect, b: &Rect) -> bool {
    // Inclusive version so touching edges count
    a.min_x <= b.max_x && a.max_x >= b.min_x &&
    a.min_y <= b.max_y && a.max_y >= b.min_y
}

impl RectQuadTree {
    pub fn new(boundary: Rect, capacity: usize) -> Self {
        RectQuadTree {
            boundary,
            items: Vec::with_capacity(capacity),
            capacity,
            children: None,
            depth: 0,
            max_depth: usize::MAX,
        }
    }

    pub fn new_with_max_depth(boundary: Rect, capacity: usize, max_depth: usize) -> Self {
        RectQuadTree {
            boundary,
            items: Vec::with_capacity(capacity),
            capacity,
            children: None,
            depth: 0,
            max_depth,
        }
    }

    fn new_child(boundary: Rect, capacity: usize, depth: usize, max_depth: usize) -> Self {
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
    pub fn insert(&mut self, item: RectItem) -> bool {
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
        let cx = 0.5 * (self.boundary.min_x + self.boundary.max_x);
        let cy = 0.5 * (self.boundary.min_y + self.boundary.max_y);

        let quads = [
            Rect { min_x: self.boundary.min_x, min_y: self.boundary.min_y, max_x: cx,                    max_y: cy                    }, // 0
            Rect { min_x: cx,                    min_y: self.boundary.min_y, max_x: self.boundary.max_x, max_y: cy                    }, // 1
            Rect { min_x: self.boundary.min_x,   min_y: cy,                  max_x: cx,                    max_y: self.boundary.max_y }, // 2
            Rect { min_x: cx,                    min_y: cy,                  max_x: self.boundary.max_x,  max_y: self.boundary.max_y }, // 3
        ];

        let d = self.depth + 1;
        let mut kids: [RectQuadTree; 4] = [
            RectQuadTree::new_child(quads[0], self.capacity, d, self.max_depth),
            RectQuadTree::new_child(quads[1], self.capacity, d, self.max_depth),
            RectQuadTree::new_child(quads[2], self.capacity, d, self.max_depth),
            RectQuadTree::new_child(quads[3], self.capacity, d, self.max_depth),
        ];

        // Move any items that fully fit a child
        let mut stay: Vec<RectItem> = Vec::new();
        for it in self.items.drain(..) {
            if let Some(idx) = child_index_for_rect(&self.boundary, &it.rect) {
                kids[idx].insert(it);
            } else {
                stay.push(it);
            }
        }
        self.items = stay;
        self.children = Some(Box::new(kids));
    }

    #[inline(always)]
    fn rect_contains_rect_inclusive(a: &Rect, b: &Rect) -> bool {
        a.min_x <= b.min_x && a.min_y <= b.min_y &&
        a.max_x >= b.max_x && a.max_y >= b.max_y
    }

    /// Query for all rectangles that touch or intersect the given range.
    /// Returns a Vec of (id, Rect).
    pub fn query(&self, range: Rect) -> Vec<(u64, Rect)> {
        #[derive(Copy, Clone)]
        enum Mode { Filter, ReportAll }

        let mut out: Vec<(u64, Rect)> = Vec::with_capacity(128);
        let mut stack: SmallVec<[(&RectQuadTree, Mode); 64]> = SmallVec::new();
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

        out
    }

    /// Convenience if you only want ids.
    pub fn query_ids(&self, range: Rect) -> Vec<u64> {
        self.query(range).into_iter().map(|(id, _)| id).collect()
    }

    /// Delete an item by id and rect. Returns true if removed.
    pub fn delete(&mut self, id: u64, rect: Rect) -> bool {
        if !rects_touch_or_intersect(&self.boundary, &rect) {
            return false;
        }
        self.delete_internal(id, rect)
    }

    fn delete_internal(&mut self, id: u64, rect: Rect) -> bool {
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
    pub fn get_all_node_boundaries(&self) -> Vec<Rect> {
        let mut rects = Vec::new();
        self.collect_boundaries(&mut rects);
        rects
    }

    fn collect_boundaries(&self, out: &mut Vec<Rect>) {
        out.push(self.boundary);
        if let Some(children) = self.children.as_ref() {
            for child in children.iter() {
                child.collect_boundaries(out);
            }
        }
    }
}
