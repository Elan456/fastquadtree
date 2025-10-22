pub mod geom;
pub mod quadtree;
pub mod rect_quadtree;

pub use crate::geom::{dist_sq_point_to_rect, dist_sq_points, mid, Coord, Point, Rect};
pub use crate::quadtree::{Item, QuadTree};
pub use crate::rect_quadtree::{RectItem, RectQuadTree};

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList};

fn item_to_tuple<T: Coord + Copy>(it: Item<T>) -> (u64, T, T) {
    (it.id, it.point.x, it.point.y)
}

fn rect_to_tuple<T: Coord + Copy>(r: Rect<T>) -> (T, T, T, T) {
    (r.min_x, r.min_y, r.max_x, r.max_y)
}

// Reusable core for point QuadTrees
macro_rules! define_point_quadtree_pyclass {
    ($t:ty, $rs_name:ident, $py_name:literal) => {
        #[pyclass(name = $py_name)]
        pub struct $rs_name {
            inner: QuadTree<$t>,
        }

        #[pymethods]
        impl $rs_name {
            #[new]
            #[pyo3(signature = (bounds, capacity, max_depth=None))]
            pub fn new(bounds: ($t, $t, $t, $t), capacity: usize, max_depth: Option<usize>) -> Self {
                let (min_x, min_y, max_x, max_y) = bounds;
                let rect = Rect { min_x, min_y, max_x, max_y };
                let inner = match max_depth {
                    Some(d) => QuadTree::new_with_max_depth(rect, capacity, d),
                    None => QuadTree::new(rect, capacity),
                };
                Self { inner }
            }

            pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
                let buf = self.inner.to_bytes().map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!("serialize failed: {e}"))
                })?;
                Ok(PyBytes::new(py, &buf))
            }

            #[staticmethod]
            pub fn from_bytes(bytes: &Bound<PyBytes>) -> PyResult<Self> {
                let inner = QuadTree::from_bytes(bytes.as_bytes()).map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!("deserialize failed: {e}"))
                })?;
                Ok(Self { inner })
            }

            pub fn insert(&mut self, id: u64, xy: ($t, $t)) -> bool {
                let (x, y) = xy;
                self.inner.insert(Item { id, point: Point { x, y } })
            }

            /// Insert many points with auto ids starting at start_id. Returns the last id used.
            pub fn insert_many(&mut self, start_id: u64, points: Vec<($t, $t)>) -> u64 {
                let mut id = start_id;
                for (x, y) in points {
                    if self.inner.insert(Item { id, point: Point { x, y } }) {
                        id += 1;
                    }
                }
                id.saturating_sub(1)
            }

            /// Assume (N x 2) numpy array of points with dtype matching this class.
            pub fn insert_many_np<'py>(
                &mut self,
                py: Python<'py>,
                start_id: u64,
                points: PyReadonlyArray2<'py, $t>,
            ) -> PyResult<u64> {
                let view = points.as_array();
                if view.ncols() != 2 {
                    return Err(PyValueError::new_err("points must have shape (N, 2)"));
                }
                let mut id = start_id;
                py.detach(|| {
                    if let Some(slice) = view.as_slice() {
                        for ch in slice.chunks_exact(2) {
                            let (x, y) = (ch[0], ch[1]);
                            if self.inner.insert(Item { id, point: Point { x, y } }) {
                                id += 1;
                            }
                        }
                    } else {
                        for row in view.outer_iter() {
                            let (x, y) = (row[0], row[1]);
                            if self.inner.insert(Item { id, point: Point { x, y } }) {
                                id += 1;
                            }
                        }
                    }
                });
                Ok(id.saturating_sub(1))
            }

            pub fn delete(&mut self, id: u64, xy: ($t, $t)) -> bool {
                let (x, y) = xy;
                self.inner.delete(id, Point { x, y })
            }

            /// Returns list[(id, x, y)]
            pub fn query<'py>(
                &self,
                py: Python<'py>,
                rect: ($t, $t, $t, $t),
            ) -> Bound<'py, PyList> {
                let (min_x, min_y, max_x, max_y) = rect;
                let tuples = self.inner.query(Rect { min_x, min_y, max_x, max_y });
                PyList::new(py, &tuples).expect("Failed to create Python list")
            }

            /// Returns list[id, ...]
            pub fn query_ids<'py>(
                &self,
                py: Python<'py>,
                rect: ($t, $t, $t, $t),
            ) -> Bound<'py, PyList> {
                let (min_x, min_y, max_x, max_y) = rect;
                let ids: Vec<u64> = self
                    .inner
                    .query(Rect { min_x, min_y, max_x, max_y })
                    .into_iter()
                    .map(|it| it.0)
                    .collect();
                PyList::new(py, &ids).expect("Failed to create Python list")
            }

            pub fn nearest_neighbor(&self, xy: ($t, $t)) -> Option<(u64, $t, $t)> {
                let (x, y) = xy;
                self.inner.nearest_neighbor(Point { x, y }).map(item_to_tuple)
            }

            pub fn nearest_neighbors(&self, xy: ($t, $t), k: usize) -> Vec<(u64, $t, $t)> {
                let (x, y) = xy;
                self.inner
                    .nearest_neighbors(Point { x, y }, k)
                    .into_iter()
                    .map(item_to_tuple)
                    .collect()
            }

            pub fn get_all_node_boundaries(&self) -> Vec<($t, $t, $t, $t)> {
                self.inner
                    .get_all_node_boundaries()
                    .into_iter()
                    .map(rect_to_tuple)
                    .collect()
            }

            pub fn count_items(&self) -> usize {
                self.inner.count_items()
            }
        }
    };
}

// Reusable core for rectangle QuadTrees
macro_rules! define_rect_quadtree_pyclass {
    ($t:ty, $rs_name:ident, $py_name:literal) => {
        #[pyclass(name = $py_name)]
        pub struct $rs_name {
            inner: RectQuadTree<$t>,
        }

        #[pymethods]
        impl $rs_name {
            #[new]
            #[pyo3(signature = (bounds, capacity, max_depth=None))]
            pub fn new(bounds: ($t, $t, $t, $t), capacity: usize, max_depth: Option<usize>) -> Self {
                let (min_x, min_y, max_x, max_y) = bounds;
                let rect = Rect { min_x, min_y, max_x, max_y };
                let inner = match max_depth {
                    Some(d) => RectQuadTree::new_with_max_depth(rect, capacity, d),
                    None => RectQuadTree::new(rect, capacity),
                };
                Self { inner }
            }

            pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
                let buf = self.inner.to_bytes().map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!("serialize failed: {e}"))
                })?;
                Ok(PyBytes::new(py, &buf))
            }

            #[staticmethod]
            pub fn from_bytes(bytes: &Bound<PyBytes>) -> PyResult<Self> {
                let inner = RectQuadTree::from_bytes(bytes.as_bytes()).map_err(|e| {
                    PyErr::new::<PyValueError, _>(format!("deserialize failed: {e}"))
                })?;
                Ok(Self { inner })
            }

            pub fn insert(&mut self, id: u64, rect: ($t, $t, $t, $t)) -> bool {
                let (min_x, min_y, max_x, max_y) = rect;
                self.inner.insert(RectItem {
                    id,
                    rect: Rect { min_x, min_y, max_x, max_y },
                })
            }

            /// Insert many rects with auto ids starting at start_id. Returns the last id used.
            pub fn insert_many(&mut self, start_id: u64, rects: Vec<($t, $t, $t, $t)>) -> u64 {
                let mut id = start_id;
                for (min_x, min_y, max_x, max_y) in rects {
                    if self.inner.insert(RectItem {
                        id,
                        rect: Rect { min_x, min_y, max_x, max_y },
                    }) {
                        id += 1;
                    }
                }
                id.saturating_sub(1)
            }

            /// Assume (N x 4) numpy array of rects with dtype matching this class.
            pub fn insert_many_np<'py>(
                &mut self,
                py: Python<'py>,
                start_id: u64,
                rects: PyReadonlyArray2<'py, $t>,
            ) -> PyResult<u64> {
                let view = rects.as_array();
                if view.ncols() != 4 {
                    return Err(PyValueError::new_err("rects must have shape (N, 4)"));
                }
                let mut id = start_id;
                py.detach(|| {
                    if let Some(slice) = view.as_slice() {
                        for ch in slice.chunks_exact(4) {
                            let r = Rect { min_x: ch[0], min_y: ch[1], max_x: ch[2], max_y: ch[3] };
                            if self.inner.insert(RectItem { id, rect: r }) {
                                id += 1;
                            }
                        }
                    } else {
                        for row in view.outer_iter() {
                            let r = Rect { min_x: row[0], min_y: row[1], max_x: row[2], max_y: row[3] };
                            if self.inner.insert(RectItem { id, rect: r }) {
                                id += 1;
                            }
                        }
                    }
                });
                Ok(id.saturating_sub(1))
            }

            pub fn delete(&mut self, id: u64, rect: ($t, $t, $t, $t)) -> bool {
                let (min_x, min_y, max_x, max_y) = rect;
                self.inner.delete(id, Rect { min_x, min_y, max_x, max_y })
            }

            /// Returns list[(id, min_x, min_y, max_x, max_y)]
            pub fn query<'py>(
                &self,
                py: Python<'py>,
                rect: ($t, $t, $t, $t),
            ) -> Bound<'py, PyList> {
                let (min_x, min_y, max_x, max_y) = rect;
                let tuples: Vec<(u64, $t, $t, $t, $t)> = self
                    .inner
                    .query(Rect { min_x, min_y, max_x, max_y })
                    .into_iter()
                    .map(|(id, r)| (id, r.min_x, r.min_y, r.max_x, r.max_y))
                    .collect();
                PyList::new(py, &tuples).expect("Failed to create Python list")
            }

            /// Returns list[id, ...]
            pub fn query_ids<'py>(
                &self,
                py: Python<'py>,
                rect: ($t, $t, $t, $t),
            ) -> Bound<'py, PyList> {
                let (min_x, min_y, max_x, max_y) = rect;
                let ids: Vec<u64> = self
                    .inner
                    .query(Rect { min_x, min_y, max_x, max_y })
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect();
                PyList::new(py, &ids).expect("Failed to create Python list")
            }

            pub fn get_all_node_boundaries(&self) -> Vec<($t, $t, $t, $t)> {
                self.inner
                    .get_all_node_boundaries()
                    .into_iter()
                    .map(rect_to_tuple)
                    .collect()
            }

            pub fn count_items(&self) -> usize {
                self.inner.count_items()
            }
        }
    };
}

// f32 default names for backward compat
define_point_quadtree_pyclass!(f32, PyQuadTreeF32, "QuadTree");
define_rect_quadtree_pyclass!(f32, PyRectQuadTreeF32, "RectQuadTree");

// f64
define_point_quadtree_pyclass!(f64, PyQuadTreeF64, "QuadTreeF64");
define_rect_quadtree_pyclass!(f64, PyRectQuadTreeF64, "RectQuadTreeF64");

// i32
define_point_quadtree_pyclass!(i32, PyQuadTreeI32, "QuadTreeI32");
define_rect_quadtree_pyclass!(i32, PyRectQuadTreeI32, "RectQuadTreeI32");

// i64
define_point_quadtree_pyclass!(i64, PyQuadTreeI64, "QuadTreeI64");
define_rect_quadtree_pyclass!(i64, PyRectQuadTreeI64, "RectQuadTreeI64");

#[pymodule]
fn _native(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    // f32 defaults
    m.add_class::<PyQuadTreeF32>()?;
    m.add_class::<PyRectQuadTreeF32>()?;

    // f64
    m.add_class::<PyQuadTreeF64>()?;
    m.add_class::<PyRectQuadTreeF64>()?;

    // i32
    m.add_class::<PyQuadTreeI32>()?;
    m.add_class::<PyRectQuadTreeI32>()?;

    // i64
    m.add_class::<PyQuadTreeI64>()?;
    m.add_class::<PyRectQuadTreeI64>()?;
    Ok(())
}
