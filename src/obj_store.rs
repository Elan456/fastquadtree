// src/obj_store.rs
//
// Pure-Rust dense id <-> (geom, obj) store, except that it can hold opaque Python objects.
//
// Design goals you asked for:
// - Core API is "pure Rust": no `Python<'_>`, no `PyResult`, no Python exceptions.
// - Python objects are treated opaquely: stored as `Py<PyAny>` and keyed by identity (the raw pointer).
// - Supports:
//   - Dense ids with holes
//   - LIFO free-list for id reuse
//   - Reverse identity map: object identity -> ids (same object can appear at multiple ids)
//   - Deterministic lookup: min id for an object
//   - pop by id, pop all by object
//   - query by id / query by object
//
// Notes:
// - This module never checks "is this Python None?" because that requires the GIL.
//   If you want "None means no payload", do that normalization at the PyO3 boundary
//   (in lib.rs) and pass `None` (Option) into this store.
// - Methods that *construct Python objects* (lists/tuples) should live in lib.rs.
//   This file only stores and returns `Py<PyAny>` (owned references).

use pyo3::prelude::*;
use pyo3::{ffi};
use std::collections::HashMap;

/// Identity key for a Python object, equivalent to Python's `id(obj)` within a process.
/// We use the raw pointer address. This is stable for the lifetime of the object.
#[inline]
fn obj_key_ptr(ptr: *mut ffi::PyObject) -> usize {
    ptr as usize
}

/// Small, Rust-native error type for store operations.
///
/// The store stays "pure Rust": no PyErr, no PyResult.
/// Convert to PyErr in lib.rs where needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreError {
    /// id doesn't fit in usize (relevant on 32-bit or if user passes huge u64)
    IdDoesNotFitUsize,
    /// id was out of bounds for current dense length
    IdOutOfBounds,
    /// Caller attempted a non-dense insert without allowing gap filling
    OutOfOrderId,
}

pub type StoreResult<T> = Result<T, StoreError>;

#[inline]
fn u64_to_usize(id: u64) -> StoreResult<usize> {
    if id > (usize::MAX as u64) {
        Err(StoreError::IdDoesNotFitUsize)
    } else {
        Ok(id as usize)
    }
}

/// Stored record for a given id (id is implied by index in the dense vector).
pub struct Entry<G> {
    pub geom: G,
    pub obj: Option<Py<PyAny>>, // None means "no payload"
}

impl<G> Entry<G> {
    #[inline]
    fn obj_ptr(&self) -> Option<*mut ffi::PyObject> {
        self.obj.as_ref().map(|o| o.as_ptr())
    }
}

/// Dense store of entries indexed by id (Vec index).
///
/// Invariants:
/// - `entries.len()` is the dense length (valid ids are < dense_len()).
/// - Each live entry increments `live_len`.
/// - Removed ids go into `free` (LIFO) for reuse.
/// - Reverse map tracks identity(ptr) -> ids where the entry has `obj: Some(...)`.
pub struct ObjStore<G> {
    entries: Vec<Option<Entry<G>>>,
    free: Vec<usize>,
    obj_to_ids: HashMap<usize, Vec<usize>>,
    live_len: usize,
}

impl<G> Default for ObjStore<G> {
    fn default() -> Self {
        Self::new()
    }
}

impl<G> ObjStore<G> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            free: Vec::new(),
            obj_to_ids: HashMap::new(),
            live_len: 0,
        }
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            entries: Vec::with_capacity(n),
            free: Vec::new(),
            obj_to_ids: HashMap::new(),
            live_len: 0,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.live_len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.live_len == 0
    }

    /// Current dense length (valid ids are < dense_len()).
    #[inline]
    pub fn dense_len(&self) -> usize {
        self.entries.len()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.free.clear();
        self.obj_to_ids.clear();
        self.live_len = 0;
    }

    /// True if id is in-bounds and has a live entry.
    pub fn contains_id(&self, id: u64) -> bool {
        let Ok(i) = u64_to_usize(id) else { return false };
        i < self.entries.len() && self.entries[i].is_some()
    }

    /// True if this exact Python object identity exists in the reverse map.
    ///
    /// This takes `Py<PyAny>` (opaque owned handle) instead of `Bound<PyAny>`,
    /// so no GIL is needed.
    pub fn contains_obj(&self, obj: &Py<PyAny>) -> bool {
        let key = obj_key_ptr(obj.as_ptr());
        self.obj_to_ids.contains_key(&key)
    }

    /// Allocate a reusable dense id. Uses free-list else appends at tail.
    #[inline]
    pub fn alloc_id(&mut self) -> u64 {
        if let Some(id) = self.free.pop() {
            id as u64
        } else {
            self.entries.len() as u64
        }
    }

    /// Insert with auto id allocation. Returns the id used.
    pub fn insert(&mut self, geom: G, obj: Option<Py<PyAny>>) -> u64 {
        let id = self.alloc_id();
        // This can't fail because alloc_id always returns <= entries.len()
        let _ = self.insert_at(id, geom, obj, false);
        id
    }

    /// Insert or replace mapping at a specific id.
    ///
    /// - handle_out_of_order=false enforces dense ids (id must be <= dense_len()).
    /// - handle_out_of_order=true fills gaps with holes up to id.
    pub fn insert_at(
        &mut self,
        id: u64,
        geom: G,
        obj: Option<Py<PyAny>>,
        handle_out_of_order: bool,
    ) -> StoreResult<()> {
        let id_ = u64_to_usize(id)?;

        if id_ > self.entries.len() {
            if !handle_out_of_order {
                return Err(StoreError::OutOfOrderId);
            }
            while self.entries.len() < id_ {
                self.entries.push(None);
            }
        }

        let new_entry = Entry { geom, obj };

        if id_ == self.entries.len() {
            // append
            if let Some(ptr) = new_entry.obj_ptr() {
                self.add_rev_mapping(ptr, id_);
            }
            self.entries.push(Some(new_entry));
            self.live_len += 1;
            return Ok(());
        }

        // replace or fill a hole
        let was_hole = self.entries[id_].is_none();

        if let Some(old) = self.entries[id_].take() {
            if let Some(ptr) = old.obj_ptr() {
                self.remove_rev_mapping(ptr, id_);
            }
        }

        if let Some(ptr) = new_entry.obj_ptr() {
            self.add_rev_mapping(ptr, id_);
        }

        self.entries[id_] = Some(new_entry);
        if was_hole {
            self.live_len += 1;
        }
        Ok(())
    }

    /// Borrow entry by id.
    pub fn get(&self, id: u64) -> Option<&Entry<G>> {
        let i = u64_to_usize(id).ok()?;
        self.entries.get(i).and_then(|x| x.as_ref())
    }

    /// Borrow entry by id (mutable).
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Entry<G>> {
        let i = u64_to_usize(id).ok()?;
        self.entries.get_mut(i).and_then(|x| x.as_mut())
    }

    /// Get a reference to the stored object handle by id (if present).
    pub fn get_obj(&self, id: u64) -> Option<&Py<PyAny>> {
        self.get(id).and_then(|e| e.obj.as_ref())
    }

    /// Remove by id. Dense ids go to the free-list for reuse.
    pub fn pop_id(&mut self, id: u64) -> Option<Entry<G>> {
        let i = u64_to_usize(id).ok()?;
        if i >= self.entries.len() {
            return None;
        }

        let old = self.entries[i].take()?;
        if let Some(ptr) = old.obj_ptr() {
            self.remove_rev_mapping(ptr, i);
        }

        self.free.push(i);
        self.live_len = self.live_len.saturating_sub(1);
        Some(old)
    }

    /// Deterministic: return the lowest id for this object identity.
    pub fn min_id_for_obj(&self, obj: &Py<PyAny>) -> Option<u64> {
        let key = obj_key_ptr(obj.as_ptr());
        let ids = self.obj_to_ids.get(&key)?;
        let min_id = *ids.iter().min()?;
        Some(min_id as u64)
    }

    /// Return all ids for this object identity, sorted.
    pub fn ids_for_obj_sorted(&self, obj: &Py<PyAny>) -> Vec<u64> {
        let key = obj_key_ptr(obj.as_ptr());
        let mut ids = match self.obj_to_ids.get(&key) {
            Some(v) => v.clone(),
            None => return Vec::new(),
        };
        ids.sort_unstable();
        ids.into_iter().map(|i| i as u64).collect()
    }

    /// Remove all entries associated with this object identity.
    /// Returns removed entries as (id, entry) sorted by id.
    pub fn pop_by_object_all(&mut self, obj: &Py<PyAny>) -> Vec<(u64, Entry<G>)> {
        let key = obj_key_ptr(obj.as_ptr());
        let mut ids = match self.obj_to_ids.remove(&key) {
            Some(v) => v,
            None => return Vec::new(),
        };
        ids.sort_unstable();

        let mut out = Vec::with_capacity(ids.len());
        for i in ids {
            if i < self.entries.len() {
                if let Some(entry) = self.entries[i].take() {
                    // reverse-map already removed as a whole above
                    self.free.push(i);
                    self.live_len = self.live_len.saturating_sub(1);
                    out.push((i as u64, entry));
                }
            }
        }
        out
    }

    /// Convenience: remove only the deterministic "first" id for this object identity.
    /// Returns (id, entry) if removed.
    pub fn pop_by_object_min(&mut self, obj: &Py<PyAny>) -> Option<(u64, Entry<G>)> {
        let id = self.min_id_for_obj(obj)?;
        let entry = self.pop_id(id)?;
        Some((id, entry))
    }

    /// Gather a Vec of object handles corresponding to ids, preserving order.
    ///
    /// This is a Rust-only replacement for your previous `gather_objects_list`.
    /// Build the Python list in lib.rs if you want that output.
    ///
    /// - If `strict_no_holes` is true, returns Err if any id is out of bounds or has no obj.
    /// - If `strict_no_holes` is false, returns `None` for missing entries or missing payloads.
    pub fn gather_objects_ref<'a>(
        &'a self,
        ids: &[u64],
        strict_no_holes: bool,
    ) -> StoreResult<Vec<Option<&'a Py<PyAny>>>> {
        let mut out = Vec::with_capacity(ids.len());
        let storage_len = self.entries.len();

        for &id_u64 in ids {
            let id_ = u64_to_usize(id_u64)?;
            if id_ >= storage_len {
                return Err(StoreError::IdOutOfBounds);
            }

            match &self.entries[id_] {
                Some(e) => {
                    if strict_no_holes && e.obj.is_none() {
                        return Err(StoreError::IdOutOfBounds);
                    }
                    out.push(e.obj.as_ref());
                }
                None => {
                    if strict_no_holes {
                        return Err(StoreError::IdOutOfBounds);
                    }
                    out.push(None);
                }
            }
        }
        Ok(out)
    }

    /// Bulk lookup of object pointers for a set of ids, preserving order.
    ///
    /// Returned pointers are *borrowed* (no refcount change here).
    /// This is ideal for the fast path in lib.rs where you build a Python list via:
    ///   - PyList_New(n)
    ///   - for each ptr: Py_INCREF(ptr); PyList_SET_ITEM(...)
    ///
    /// If `strict_no_holes` is true:
    /// - Err if any id is out of bounds, is a hole (no entry), or has obj=None.
    ///
    /// If `strict_no_holes` is false:
    /// - Missing entries or obj=None become Py_None() pointers.
    pub fn bulk_obj_ptrs(
        &self,
        ids: &[u64],
        strict_no_holes: bool,
    ) -> StoreResult<Vec<*mut ffi::PyObject>> {
        let mut out = Vec::with_capacity(ids.len());
        let storage_len = self.entries.len();

        for &id_u64 in ids {
            let i = u64_to_usize(id_u64)?;
            if i >= storage_len {
                return Err(StoreError::IdOutOfBounds);
            }

            match &self.entries[i] {
                Some(e) => match e.obj.as_ref() {
                    Some(o) => out.push(o.as_ptr()),
                    None => {
                        if strict_no_holes {
                            return Err(StoreError::IdOutOfBounds);
                        }
                        out.push(unsafe { ffi::Py_None() });
                    }
                },
                None => {
                    if strict_no_holes {
                        return Err(StoreError::IdOutOfBounds);
                    }
                    out.push(unsafe { ffi::Py_None() });
                }
            }
        }

        Ok(out)
    }


    // ----------------------------
    // Reverse-map helpers
    // ----------------------------

    fn remove_rev_mapping(&mut self, obj_ptr: *mut ffi::PyObject, id_: usize) {
        let key = obj_key_ptr(obj_ptr);
        if let Some(v) = self.obj_to_ids.get_mut(&key) {
            v.retain(|&x| x != id_);
            if v.is_empty() {
                self.obj_to_ids.remove(&key);
            }
        }
    }

    fn add_rev_mapping(&mut self, obj_ptr: *mut ffi::PyObject, id_: usize) {
        let key = obj_key_ptr(obj_ptr);
        let v = self.obj_to_ids.entry(key).or_insert_with(Vec::new);
        if !v.iter().any(|&x| x == id_) {
            v.push(id_);
        }
    }
}
