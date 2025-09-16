
pub mod geom;
pub mod quadtree;

// Optional re-exports so users of the crate can do `use quadtree_rs::QuadTree;`
pub use crate::geom::{Point, Rect, dist_sq_point_to_rect, dist_sq_points};
pub use crate::quadtree::{Item, QuadTree};

use pyo3::prelude::*;

fn item_to_tuple(it: Item) -> (u64, f64, f64) {
    (it.id, it.point.x, it.point.y)
}

#[pyclass(name = "QuadTree")]
pub struct PyQuadTree {
    inner: QuadTree,
}

#[pymethods]
impl PyQuadTree {
    #[new]
    pub fn new(bounds: (f64, f64, f64, f64), capacity: usize, max_depth: Option<usize>) -> Self {
        let (min_x, min_y, max_x, max_y) = bounds;
        let rect = Rect { min_x, min_y, max_x, max_y };
        let inner = match max_depth {
            Some(d) => QuadTree::new_with_max_depth(rect, capacity, d),
            None => QuadTree::new(rect, capacity),
        };
        Self { inner }
    }

    pub fn insert(&mut self, id: u64, xy: (f64, f64)) -> bool {
        let (x, y) = xy;
        self.inner.insert(Item { id, point: Point { x, y } })
    }

    pub fn delete(&mut self, id: u64, xy: (f64, f64)) -> bool {
        let (x, y) = xy;
        self.inner.delete(id, Point { x, y })
    }

    pub fn query(&self, rect: (f64, f64, f64, f64)) -> Vec<(u64, f64, f64)> {
        let (min_x, min_y, max_x, max_y) = rect;
        self.inner
            .query(Rect { min_x, min_y, max_x, max_y })
            .into_iter()
            .map(item_to_tuple)
            .collect()
    }

    pub fn nearest_neighbor(&self, xy: (f64, f64)) -> Option<(u64, f64, f64)> {
        let (x, y) = xy;
        self.inner.nearest_neighbor(Point { x, y }).map(item_to_tuple)
    }

    pub fn nearest_neighbors(&self, xy: (f64, f64), k: usize) -> Vec<(u64, f64, f64)> {
        let (x, y) = xy;
        self.inner
            .nearest_neighbors(Point { x, y }, k)
            .into_iter()
            .map(item_to_tuple)
            .collect()
    }

    /// Returns all rectangle boundaries in the quadtree for visualization
    pub fn get_all_rectangles(&self) -> Vec<(f64, f64, f64, f64)> {
        self.inner
            .get_all_rectangles()
            .into_iter()
            .map(|rect| (rect.min_x, rect.min_y, rect.max_x, rect.max_y))
            .collect()
    }

    /// Returns the total number of items in the quadtree
    pub fn count_items(&self) -> usize {
        self.inner.count_items()
    }
}

#[pymodule]
fn _native(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<PyQuadTree>()?;
    Ok(())
}