pub mod geom;
pub mod quadtree;
pub mod rect_quadtree;

pub use crate::geom::{Point, Rect, dist_sq_point_to_rect, dist_sq_points};
pub use crate::quadtree::{Item, QuadTree};
pub use crate::rect_quadtree::{RectItem, RectQuadTree};

use pyo3::prelude::*;
use pyo3::types::PyList;

fn item_to_tuple(it: Item) -> (u64, f32, f32) {
    (it.id, it.point.x, it.point.y)
}

fn rect_to_tuple(r: Rect) -> (f32, f32, f32, f32) {
    (r.min_x, r.min_y, r.max_x, r.max_y)
}

#[pyclass(name = "QuadTree")]
pub struct PyQuadTree {
    inner: QuadTree,
}

#[pymethods]
impl PyQuadTree {
    #[new]
    #[pyo3(signature = (bounds, capacity, max_depth=None))]
    pub fn new(bounds: (f32, f32, f32, f32), capacity: usize, max_depth: Option<usize>) -> Self {
        let (min_x, min_y, max_x, max_y) = bounds;
        let rect = Rect { min_x, min_y, max_x, max_y };
        let inner = match max_depth {
            Some(d) => QuadTree::new_with_max_depth(rect, capacity, d),
            None => QuadTree::new(rect, capacity),
        };
        Self { inner }
    }

    pub fn insert(&mut self, id: u64, xy: (f32, f32)) -> bool {
        let (x, y) = xy;
        self.inner.insert(Item { id, point: Point { x, y } })
    }

    // Insert many points with auto ids starting at start_id: [(x, y), ...]
    // Returns the last id used
    pub fn insert_many(&mut self, start_id: u64, points: Vec<(f32, f32)>) -> u64 {
        let mut id = start_id;
        for (x, y) in points {
            if self.inner.insert(Item { id, point: Point { x, y } }) {
                id += 1;
            }
        }
        id.saturating_sub(1)
    }

    pub fn delete(&mut self, id: u64, xy: (f32, f32)) -> bool {
        let (x, y) = xy;
        self.inner.delete(id, Point { x, y })
    }

    // Returns list[(id, x, y)]
    pub fn query<'py>(&self, py: Python<'py>, rect: (f32, f32, f32, f32)) -> Bound<'py, PyList> {
        let (min_x, min_y, max_x, max_y) = rect;
        let tuples = self.inner.query(Rect { min_x, min_y, max_x, max_y });
        PyList::new(py, &tuples).expect("Failed to create Python list")
    }

    pub fn nearest_neighbor(&self, xy: (f32, f32)) -> Option<(u64, f32, f32)> {
        let (x, y) = xy;
        self.inner.nearest_neighbor(Point { x, y }).map(item_to_tuple)
    }

    pub fn nearest_neighbors(&self, xy: (f32, f32), k: usize) -> Vec<(u64, f32, f32)> {
        let (x, y) = xy;
        self.inner
            .nearest_neighbors(Point { x, y }, k)
            .into_iter()
            .map(item_to_tuple)
            .collect()
    }

    /// Returns all rectangle boundaries in the quadtree for visualization
    pub fn get_all_node_boundaries(&self) -> Vec<(f32, f32, f32, f32)> {
        self.inner
            .get_all_node_boundaries()
            .into_iter()
            .map(rect_to_tuple)
            .collect()
    }

    /// Returns the total number of items in the quadtree
    pub fn count_items(&self) -> usize {
        self.inner.count_items()
    }
}

#[pyclass(name = "RectQuadTree")]
pub struct PyRectQuadTree {
    inner: RectQuadTree,
}

#[pymethods]
impl PyRectQuadTree {
    #[new]
    #[pyo3(signature = (bounds, capacity, max_depth=None))]
    pub fn new(bounds: (f32, f32, f32, f32), capacity: usize, max_depth: Option<usize>) -> Self {
        let (min_x, min_y, max_x, max_y) = bounds;
        let rect = Rect { min_x, min_y, max_x, max_y };
        let inner = match max_depth {
            Some(d) => RectQuadTree::new_with_max_depth(rect, capacity, d),
            None => RectQuadTree::new(rect, capacity),
        };
        Self { inner }
    }

    /// Insert one AABB by id.
    pub fn insert(&mut self, id: u64, rect: (f32, f32, f32, f32)) -> bool {
        let (min_x, min_y, max_x, max_y) = rect;
        self.inner.insert(RectItem { id, rect: Rect { min_x, min_y, max_x, max_y } })
    }

    /// Insert many AABBs with auto ids starting at start_id. Returns the last id used.
    pub fn insert_many(&mut self, start_id: u64, rects: Vec<(f32, f32, f32, f32)>) -> u64 {
        let mut id = start_id;
        for (min_x, min_y, max_x, max_y) in rects {
            if self.inner.insert(RectItem { id, rect: Rect { min_x, min_y, max_x, max_y } }) {
                id += 1;
            }
        }
        id.saturating_sub(1)
    }

    /// Delete by id and exact rect.
    pub fn delete(&mut self, id: u64, rect: (f32, f32, f32, f32)) -> bool {
        let (min_x, min_y, max_x, max_y) = rect;
        self.inner.delete(id, Rect { min_x, min_y, max_x, max_y })
    }

    /// Query rectangles that touch or intersect the given range.
    /// Returns list[(id, min_x, min_y, max_x, max_y)]
    pub fn query<'py>(&self, py: Python<'py>, rect: (f32, f32, f32, f32)) -> Bound<'py, PyList> {
        let (min_x, min_y, max_x, max_y) = rect;
        let tuples: Vec<(u64, f32, f32, f32, f32)> = self.inner
            .query(Rect { min_x, min_y, max_x, max_y })
            .into_iter()
            .map(|(id, r)| (id, r.min_x, r.min_y, r.max_x, r.max_y))
            .collect();
        PyList::new(py, &tuples).expect("Failed to create Python list")
    }

    /// Collect all node boundaries for visualization or debugging.
    pub fn get_all_node_boundaries(&self) -> Vec<(f32, f32, f32, f32)> {
        self.inner
            .get_all_node_boundaries()
            .into_iter()
            .map(rect_to_tuple)
            .collect()
    }

    /// Total number of stored rectangles.
    pub fn count_items(&self) -> usize {
        self.inner.count_items()
    }
}

#[pymodule]
fn _native(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyQuadTree>()?;
    m.add_class::<PyRectQuadTree>()?;
    Ok(())
}
