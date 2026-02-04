pub mod geom;
pub mod quadtree;
pub mod rect_quadtree;
pub mod obj_store;

pub use crate::geom::{dist_sq_point_to_rect, dist_sq_points, mid, Coord, Point, Rect};
pub use crate::quadtree::{Item, QuadTree};
pub use crate::rect_quadtree::{RectItem, RectQuadTree};
pub use crate::obj_store::{ObjStore};

use numpy::PyReadonlyArray2;
use numpy::PyArray1;
use numpy::PyArray2;
use numpy::PyArrayMethods;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyModule, PyTuple};
use pyo3::PyResult;
use pyo3::ffi;
use std::any::TypeId;

fn item_to_tuple<T: Coord + Copy>(it: Item<T>) -> (u64, T, T) {
    (it.id, it.point.x, it.point.y)
}

fn rect_to_tuple<T: Coord + Copy>(r: Rect<T>) -> (T, T, T, T) {
    (r.min_x, r.min_y, r.max_x, r.max_y)
}

fn default_max_depth_for<T: 'static>() -> usize {
    // Caps aligned with meaningful resolution per dtype.
    // f32: 24 mantissa bits -> deeper splits stop helping.
    // f64: 53 mantissa bits.
    // ints: cap at bit width, beyond that subdivision degenerates anyway.
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        24
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        53
    } else if TypeId::of::<T>() == TypeId::of::<i32>() {
        32
    } else if TypeId::of::<T>() == TypeId::of::<i64>() {
        64
    } else {
        32
    }
}


// Reusable core for point QuadTrees
macro_rules! define_point_quadtree_pyclass {
    ($t:ty, $rs_name:ident, $py_name:literal) => {
        #[pyclass(name = $py_name)]
        pub struct $rs_name {
            inner: QuadTree<$t>,
            obj_store: ObjStore<Point<$t>>,
        }

        #[pymethods]
        impl $rs_name {
            #[new]
            #[pyo3(signature = (bounds, capacity, max_depth=None))]
            pub fn new(bounds: ($t, $t, $t, $t), capacity: usize, max_depth: Option<usize>) -> Self {
                let (min_x, min_y, max_x, max_y) = bounds;
                let rect = Rect { min_x, min_y, max_x, max_y };
                let inner = match max_depth {
                    Some(d) => QuadTree::new(rect, capacity, d),
                    None => QuadTree::new(
                        rect,
                        capacity,
                        default_max_depth_for::<$t>(),
                    ),
                };
                let obj_store = ObjStore::new();
                Self { inner, obj_store }
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
                Ok(Self { inner, obj_store: ObjStore::new() })
            }

            pub fn insert(&mut self, id: u64, xy: ($t, $t)) -> bool {
                let (x, y) = xy;
                self.inner.insert(Item { id, point: Point { x, y } })
            }

            pub fn insert_object<'py>(
                &mut self,
                obj: Bound<'py, PyAny>,
                point: ($t, $t),
            ) -> PyResult<u64> {
                let (x, y) = point;
                let p = Point { x, y };

                // Insert into store first to get an id
                let id = self.obj_store.insert(p, Some(obj.unbind()));

                // Insert into quadtree using the same id
                if self.inner.insert(Item { id, point: p }) {
                    Ok(id)
                } else {
                    // Roll back store so ids stay consistent
                    self.obj_store.pop_id(id);
                    Err(PyValueError::new_err("quadtree insert failed"))
                }
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

            pub fn query<'py>(
                &self,
                py: Python<'py>,
                rect: ($t, $t, $t, $t),
            ) -> Bound<'py, PyList> {
                let (min_x, min_y, max_x, max_y) = rect;

                // Release the GIL during the Rust search
                let tuples = py.detach(|| self.inner.query(Rect { min_x, min_y, max_x, max_y }));

                unsafe {
                    let len = tuples.len() as isize;
                    let list_ptr = ffi::PyList_New(len);
                    if list_ptr.is_null() {
                        ffi::PyErr_NoMemory();
                    }

                    // preserve ints for integer trees, floats for float trees
                    let is_float = TypeId::of::<$t>() == TypeId::of::<f32>()
                        || TypeId::of::<$t>() == TypeId::of::<f64>();

                    for (i, (id, x, y)) in tuples.into_iter().enumerate() {
                        let tup_ptr = ffi::PyTuple_New(3);
                        if tup_ptr.is_null() {
                            ffi::PyErr_NoMemory();
                        }

                        let py_id = ffi::PyLong_FromUnsignedLongLong(id);
                        let (py_x, py_y) = if is_float {
                            (ffi::PyFloat_FromDouble(x as f64),
                            ffi::PyFloat_FromDouble(y as f64))
                        } else {
                            (ffi::PyLong_FromLongLong(x as i64),
                            ffi::PyLong_FromLongLong(y as i64))
                        };

                        // SetItem functions steal references
                        ffi::PyTuple_SetItem(tup_ptr, 0, py_id);
                        ffi::PyTuple_SetItem(tup_ptr, 1, py_x);
                        ffi::PyTuple_SetItem(tup_ptr, 2, py_y);
                        ffi::PyList_SetItem(list_ptr, i as isize, tup_ptr);
                    }

                    // Create Bound<PyAny> then downcast to Bound<PyList>
                    Bound::from_owned_ptr(py, list_ptr).downcast_into_unchecked::<PyList>()
                }
            }

            /// Returns (ids: np.ndarray[u64], points: np.ndarray[Nx2])
            pub fn query_np<'py>(
                &self,
                py: Python<'py>,
                rect: ($t, $t, $t, $t),
            ) -> PyResult<Bound<'py, PyTuple>> {
                let (min_x, min_y, max_x, max_y) = rect;
                let (ids_vec, xs_vec, ys_vec) = py.detach(|| {
                    let tuples = self.inner.query(Rect { min_x, min_y, max_x, max_y });
                    let n = tuples.len();
                    let mut ids = Vec::with_capacity(n);
                    let mut xs  = Vec::with_capacity(n);
                    let mut ys  = Vec::with_capacity(n);
                    for (id, x, y) in tuples {
                        ids.push(id);
                        xs.push(x);
                        ys.push(y);
                    }
                    (ids, xs, ys)
                });

                let n = ids_vec.len();
                // Create NumPy arrays
                let ids_arr = PyArray1::<u64>::from_vec(py, ids_vec);      // Bound<'py, PyArray1<u64>>

                // Fill xy with no per element Python objects
                unsafe {
                    let xy_arr = PyArray2::<$t>::new(py, [n, 2], false);      // Bound<'py, PyArray2<$t>>
                    let mut a = xy_arr.as_array_mut();
                    for i in 0..n {
                        a[[i, 0]] = xs_vec[i];
                        a[[i, 1]] = ys_vec[i];
                    }
                    // Return a Python tuple (ids, xy)
                    let out = PyTuple::new(py, &[ids_arr.as_any(), xy_arr.as_any()])?;
                    Ok(out)
                }
            }

            /// Returns list[id, ...]
            pub fn query_ids<'py>(
                &self,
                py: Python<'py>,
                rect: ($t, $t, $t, $t),
            ) -> Bound<'py, PyList> {
                let (min_x, min_y, max_x, max_y) = rect;
                let ids: Vec<u64> = py.detach(|| {
                    self.inner
                        .query(Rect { min_x, min_y, max_x, max_y })
                        .into_iter()
                        .map(|it| it.0) // (id, x, y) -> id
                        .collect()
                });
                PyList::new(py, &ids).expect("Failed to create Python list")
            }

            /// Returns np.ndarray[u64] of ids only
            pub fn query_ids_np<'py>(
                &self,
                py: Python<'py>,
                rect: ($t, $t, $t, $t),
            ) -> PyResult<Bound<'py, PyArray1<u64>>> {
                let (min_x, min_y, max_x, max_y) = rect;

                // Run the search without the GIL and collect ids
                let ids: Vec<u64> = py.detach(|| {
                    self.inner
                        .query(Rect { min_x, min_y, max_x, max_y })
                        .into_iter()
                        .map(|it| it.0) // (id, x, y) -> id
                        .collect()
                });

                // Materialize as a NumPy array
                Ok(PyArray1::<u64>::from_vec(py, ids))
            }
            
            /// Returns list[Item, ...] by indexing into ObjStore._arr
            #[pyo3(signature = (rect, arr_list))]
            pub fn query_items<'py>(
                &self,
                py: Python<'py>,
                rect: ($t, $t, $t, $t),
                arr_list: &Bound<'py, PyList>,
            ) -> PyResult<Bound<'py, PyList>> {
                let (min_x, min_y, max_x, max_y) = rect;

                // 1) Run the quadtree query without the GIL and collect ids
                let ids: Vec<u64> = py.detach(|| {
                    self.inner
                        .query(Rect { min_x, min_y, max_x, max_y })
                        .into_iter()
                        .map(|it| it.0) // (id, x, y) -> id
                        .collect()
                });
                
                // 3) Build output list by indexing arr_list in C (no ids.tolist(), no itemgetter)
                unsafe {
                    let n = ids.len();
                    let out_ptr = ffi::PyList_New(n as isize);
                    if out_ptr.is_null() {
                        ffi::PyErr_NoMemory();
                        return Err(PyErr::fetch(py));
                    }

                    let storage_len = ffi::PyList_Size(arr_list.as_ptr()) as usize;

                    for (i, &id_u64) in ids.iter().enumerate() {
                        if id_u64 > (usize::MAX as u64) {
                            return Err(PyValueError::new_err("id does not fit usize"));
                        }
                        let id = id_u64 as usize;
                        if id >= storage_len {
                            return Err(PyValueError::new_err(format!(
                                "id {id} out of bounds for ObjStore._arr len {storage_len}"
                            )));
                        }

                        // Borrowed ref from _arr[id]
                        let item_ptr = ffi::PyList_GetItem(arr_list.as_ptr(), id as isize);
                        if item_ptr.is_null() {
                            return Err(PyErr::fetch(py));
                        }

                        // Optional: reject holes early (keeps behavior strict and avoids returning None)
                        // If you *want* to allow None results, delete this block.
                        if item_ptr == ffi::Py_None() {
                            return Err(PyValueError::new_err(format!(
                                "ObjStore has no item at id {id} (hole/None)"
                            )));
                        }

                        // PyList_SetItem steals a reference, so INCREF first
                        ffi::Py_INCREF(item_ptr);
                        let rc = ffi::PyList_SetItem(out_ptr, i as isize, item_ptr);
                        if rc != 0 {
                            return Err(PyErr::fetch(py));
                        }
                    }

                    Ok(Bound::from_owned_ptr(py, out_ptr).downcast_into_unchecked::<PyList>())
                }
            }

            pub fn query_items_using_rust_obj_store<'py>(
                &self,
                py: Python<'py>,
                rect: ($t, $t, $t, $t),
            ) -> PyResult<Bound<'py, PyList>> {
                let (min_x, min_y, max_x, max_y) = rect;

                // 1) Query ids without the GIL
                let ids: Vec<u64> = py.detach(|| {
                    self.inner
                        .query(Rect { min_x, min_y, max_x, max_y })
                        .into_iter()
                        .map(|it| it.0)
                        .collect()
                });

                // 2) Bulk lookup borrowed PyObject* pointers from the obj_store (no GIL needed)
                let ptrs: Vec<*mut pyo3::ffi::PyObject> = self
                    .obj_store
                    .bulk_obj_ptrs(&ids, /*strict_no_holes=*/ true)
                    .map_err(|e| PyValueError::new_err(format!("ObjStore lookup failed: {e:?}")))?;

                // 3) Build Python list in one allocation, fill via SET_ITEM
                unsafe {
                    let n = ptrs.len() as pyo3::ffi::Py_ssize_t;
                    let list_ptr = pyo3::ffi::PyList_New(n);
                    if list_ptr.is_null() {
                        return Err(PyErr::fetch(py));
                    }

                    for (i, obj_ptr) in ptrs.into_iter().enumerate() {
                        // list steals a reference; give it a new owned ref
                        pyo3::ffi::Py_INCREF(obj_ptr);
                        pyo3::ffi::PyList_SetItem(list_ptr, i as pyo3::ffi::Py_ssize_t, obj_ptr);
                    }

                    // from_owned_ptr yields Bound<PyAny>, so downcast
                    let any = Bound::from_owned_ptr(py, list_ptr);
                    any.downcast_into::<PyList>().map_err(|e| {
                        PyValueError::new_err(format!("Failed to downcast to PyList: {e}"))
                    })
                }
            }


            pub fn nearest_neighbor(&self, xy: ($t, $t)) -> Option<(u64, $t, $t)> {
                let (x, y) = xy;
                self.inner.nearest_neighbor(Point { x, y }).map(item_to_tuple)
            }

            /// Returns (id, coords) or None, where coords is ndarray shape (2,)
            pub fn nearest_neighbor_np<'py>(
                &self,
                py: Python<'py>,
                xy: ($t, $t),
            ) -> PyResult<Option<Bound<'py, PyTuple>>> {
                let (x, y) = xy;
                match self.inner.nearest_neighbor(Point { x, y }) {
                    None => Ok(None),
                    Some(item) => {
                        let (id, px, py_) = item_to_tuple(item);
                        unsafe {
                            let coords_arr = PyArray1::<$t>::new(py, [2], false);
                            let mut a = coords_arr.as_array_mut();
                            a[0] = px;
                            a[1] = py_;
                            let out = PyTuple::new(py, &[id.into_pyobject(py)?.as_any(), coords_arr.as_any()])?;
                            Ok(Some(out))
                        }
                    }
                }
            }

            pub fn nearest_neighbors(&self, xy: ($t, $t), k: usize) -> Vec<(u64, $t, $t)> {
                let (x, y) = xy;
                self.inner
                    .nearest_neighbors(Point { x, y }, k)
                    .into_iter()
                    .map(item_to_tuple)
                    .collect()
            }

            /// Returns (ids, coords) where ids is ndarray shape (k,) and coords is ndarray shape (k, 2)
            pub fn nearest_neighbors_np<'py>(
                &self,
                py: Python<'py>,
                xy: ($t, $t),
                k: usize,
            ) -> PyResult<Bound<'py, PyTuple>> {
                let (x, y) = xy;
                let (ids_vec, xs_vec, ys_vec) = py.detach(|| {
                    let items = self.inner.nearest_neighbors(Point { x, y }, k);
                    let n = items.len();
                    let mut ids = Vec::with_capacity(n);
                    let mut xs = Vec::with_capacity(n);
                    let mut ys = Vec::with_capacity(n);
                    for item in items {
                        let (id, px, py_) = item_to_tuple(item);
                        ids.push(id);
                        xs.push(px);
                        ys.push(py_);
                    }
                    (ids, xs, ys)
                });

                let n = ids_vec.len();
                let ids_arr = PyArray1::<u64>::from_vec(py, ids_vec);

                unsafe {
                    let coords_arr = PyArray2::<$t>::new(py, [n, 2], false);
                    let mut a = coords_arr.as_array_mut();
                    for i in 0..n {
                        a[[i, 0]] = xs_vec[i];
                        a[[i, 1]] = ys_vec[i];
                    }
                    let out = PyTuple::new(py, &[ids_arr.as_any(), coords_arr.as_any()])?;
                    Ok(out)
                }
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

            pub fn get_max_depth(&self) -> usize {
                self.inner.get_max_depth()
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
                    Some(d) => RectQuadTree::new(rect, capacity, d),
                    None => RectQuadTree::new(
                        rect,
                        capacity,
                        default_max_depth_for::<$t>(),
                    ),
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

            /// Returns (ids: np.ndarray[u64], rects: np.ndarray[Nx4])
            pub fn query_np<'py>(
                &self,
                py: Python<'py>,
                rect: ($t, $t, $t, $t),
            ) -> PyResult<Bound<'py, PyTuple>> {
                let (min_x, min_y, max_x, max_y) = rect;

                // Run the Rust search without the GIL and collect into flat vectors
                let (ids_vec, mins_x, mins_y, maxs_x, maxs_y) = py.detach(|| {
                    let hits = self.inner.query(Rect { min_x, min_y, max_x, max_y });
                    let n = hits.len();
                    let mut ids    = Vec::with_capacity(n);
                    let mut v_minx = Vec::with_capacity(n);
                    let mut v_miny = Vec::with_capacity(n);
                    let mut v_maxx = Vec::with_capacity(n);
                    let mut v_maxy = Vec::with_capacity(n);

                    for (id, r) in hits {
                        ids.push(id);
                        v_minx.push(r.min_x);
                        v_miny.push(r.min_y);
                        v_maxx.push(r.max_x);
                        v_maxy.push(r.max_y);
                    }
                    (ids, v_minx, v_miny, v_maxx, v_maxy)
                });

                let n = ids_vec.len();

                // ids: shape (N,)
                let ids_arr = PyArray1::<u64>::from_vec(py, ids_vec);

                // rects: shape (N,4) with columns [min_x, min_y, max_x, max_y]
                unsafe {
                    let rects_arr = PyArray2::<$t>::new(py, [n, 4], false);
                    let mut a = rects_arr.as_array_mut();
                    for i in 0..n {
                        a[[i, 0]] = mins_x[i];
                        a[[i, 1]] = mins_y[i];
                        a[[i, 2]] = maxs_x[i];
                        a[[i, 3]] = maxs_y[i];
                    }
                    // Return a Python tuple (ids, rects)
                    let out = PyTuple::new(py, &[ids_arr.as_any(), rects_arr.as_any()])?;
                    Ok(out)
                }
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

            /// Returns np.ndarray[u64] of ids only
            pub fn query_ids_np<'py>(
                &self,
                py: Python<'py>,
                rect: ($t, $t, $t, $t),
            ) -> PyResult<Bound<'py, PyArray1<u64>>> {
                let (min_x, min_y, max_x, max_y) = rect;

                // Run the search without the GIL and collect ids
                let ids: Vec<u64> = py.detach(|| {
                    self.inner
                        .query(Rect { min_x, min_y, max_x, max_y })
                        .into_iter()
                        .map(|(id, _r)| id)
                        .collect()
                });

                // Materialize as a NumPy array
                Ok(PyArray1::<u64>::from_vec(py, ids))
            }
            
            /// RectQuadTree version: query and return ObjStore Items (from ObjStore._arr) in hit order.
            /// Usage from Python:
            ///   items = rect_tree.query_items(rect, objstore)
            #[pyo3(signature = (rect, arr_list))]
            pub fn query_items<'py>(
                &self,
                py: Python<'py>,
                rect: ($t, $t, $t, $t),
                arr_list: &Bound<'py, PyList>,
            ) -> PyResult<Bound<'py, PyList>> {
                let (min_x, min_y, max_x, max_y) = rect;

                // 1) Run the rect quadtree query without the GIL and collect ids
                let ids: Vec<u64> = py.detach(|| {
                    self.inner
                        .query(Rect { min_x, min_y, max_x, max_y })
                        .into_iter()
                        .map(|(id, _r)| id)
                        .collect()
                });

                // 3) Build output list by indexing arr_list in C
                unsafe {
                    let n = ids.len();
                    let out_ptr = ffi::PyList_New(n as isize);
                    if out_ptr.is_null() {
                        ffi::PyErr_NoMemory();
                        return Err(PyErr::fetch(py));
                    }

                    let storage_len = ffi::PyList_Size(arr_list.as_ptr()) as usize;

                    for (i, &id_u64) in ids.iter().enumerate() {
                        if id_u64 > (usize::MAX as u64) {
                            return Err(PyValueError::new_err("id does not fit usize"));
                        }
                        let id = id_u64 as usize;
                        if id >= storage_len {
                            return Err(PyValueError::new_err(format!(
                                "id {id} out of bounds for ObjStore._arr len {storage_len}"
                            )));
                        }

                        // Borrowed ref from _arr[id]
                        let item_ptr = ffi::PyList_GetItem(arr_list.as_ptr(), id as isize);
                        if item_ptr.is_null() {
                            return Err(PyErr::fetch(py));
                        }

                        // Optional strictness: reject holes (None). Delete this block to allow None results.
                        if item_ptr == ffi::Py_None() {
                            return Err(PyValueError::new_err(format!(
                                "ObjStore has no item at id {id} (hole/None)"
                            )));
                        }

                        // PyList_SetItem steals a reference, so INCREF first
                        ffi::Py_INCREF(item_ptr);
                        let rc = ffi::PyList_SetItem(out_ptr, i as isize, item_ptr);
                        if rc != 0 {
                            return Err(PyErr::fetch(py));
                        }
                    }

                    Ok(Bound::from_owned_ptr(py, out_ptr).downcast_into_unchecked::<PyList>())
                }
            }

            /// Returns a single nearest neighbor (id, min_x, min_y, max_x, max_y)
            pub fn nearest_neighbor(&self, xy: ($t, $t)) -> Option<(u64, $t, $t, $t, $t)> {
                let (x, y) = xy;
                self.inner.nearest_neighbor(Point { x, y }).map(|item| (item.id, item.rect.min_x, item.rect.min_y, item.rect.max_x, item.rect.max_y))
            }

            /// Returns (id, coords) or None, where coords is ndarray shape (4,)
            pub fn nearest_neighbor_np<'py>(
                &self,
                py: Python<'py>,
                xy: ($t, $t),
            ) -> PyResult<Option<Bound<'py, PyTuple>>> {
                let (x, y) = xy;
                match self.inner.nearest_neighbor(Point { x, y }) {
                    None => Ok(None),
                    Some(item) => {
                        let id = item.id;
                        unsafe {
                            let coords_arr = PyArray1::<$t>::new(py, [4], false);
                            let mut a = coords_arr.as_array_mut();
                            a[0] = item.rect.min_x;
                            a[1] = item.rect.min_y;
                            a[2] = item.rect.max_x;
                            a[3] = item.rect.max_y;
                            let out = PyTuple::new(py, &[id.into_pyobject(py)?.as_any(), coords_arr.as_any()])?;
                            Ok(Some(out))
                        }
                    }
                }
            }

            /// Returns K nearest neighbors as a list[(id, min_x, min_y, max_x, max_y)]
            pub fn nearest_neighbors(&self, xy: ($t, $t), k: usize) -> Vec<(u64, $t, $t, $t, $t)> {
                let (x, y) = xy;
                self.inner
                    .nearest_neighbors(Point { x, y }, k)
                    .into_iter()
                    .map(|item| (item.id, item.rect.min_x, item.rect.min_y, item.rect.max_x, item.rect.max_y))
                    .collect()
            }

            /// Returns (ids, coords) where ids is ndarray shape (k,) and coords is ndarray shape (k, 4)
            pub fn nearest_neighbors_np<'py>(
                &self,
                py: Python<'py>,
                xy: ($t, $t),
                k: usize,
            ) -> PyResult<Bound<'py, PyTuple>> {
                let (x, y) = xy;
                let (ids_vec, mins_x, mins_y, maxs_x, maxs_y) = py.detach(|| {
                    let items = self.inner.nearest_neighbors(Point { x, y }, k);
                    let n = items.len();
                    let mut ids = Vec::with_capacity(n);
                    let mut v_minx = Vec::with_capacity(n);
                    let mut v_miny = Vec::with_capacity(n);
                    let mut v_maxx = Vec::with_capacity(n);
                    let mut v_maxy = Vec::with_capacity(n);
                    for item in items {
                        ids.push(item.id);
                        v_minx.push(item.rect.min_x);
                        v_miny.push(item.rect.min_y);
                        v_maxx.push(item.rect.max_x);
                        v_maxy.push(item.rect.max_y);
                    }
                    (ids, v_minx, v_miny, v_maxx, v_maxy)
                });

                let n = ids_vec.len();
                let ids_arr = PyArray1::<u64>::from_vec(py, ids_vec);

                unsafe {
                    let coords_arr = PyArray2::<$t>::new(py, [n, 4], false);
                    let mut a = coords_arr.as_array_mut();
                    for i in 0..n {
                        a[[i, 0]] = mins_x[i];
                        a[[i, 1]] = mins_y[i];
                        a[[i, 2]] = maxs_x[i];
                        a[[i, 3]] = maxs_y[i];
                    }
                    let out = PyTuple::new(py, &[ids_arr.as_any(), coords_arr.as_any()])?;
                    Ok(out)
                }
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

            pub fn get_max_depth(&self) -> usize {
                self.inner.get_max_depth()
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
