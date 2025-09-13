
mod geom;
pub use geom::{Point, Rect};

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
        }
    }

    // Returns True if the item is inserted successfully
    pub fn insert(&mut self, item: Item) -> bool {
        if !self.boundary.contains(&item.point) {
            return false;
        }

        // If children is None, we are a leaf node
        if self.children.is_none() {
            // Check if we need to subdivide
            if self.items.len() < self.capacity {
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
        let mut kids: [QuadTree; 4] = [
            QuadTree::new(quads[0], self.capacity),
            QuadTree::new(quads[1], self.capacity),
            QuadTree::new(quads[2], self.capacity),
            QuadTree::new(quads[3], self.capacity),
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


}
