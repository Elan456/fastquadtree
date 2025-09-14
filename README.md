# quadtree-rs

Rust-optimized quadtree with a simple Python API.

- Python package: **`quadtree_rs`**
- Python ≥ 3.8
- Import path: `from quadtree_rs import QuadTree`

## Install

```bash
pip install quadtree_rs
````

If you are developing locally:

```bash
# optimized dev install
maturin develop --release
```

## Quickstart

```python
from quadtree_rs import QuadTree

# Bounds are (min_x, min_y, max_x, max_y)
qt = QuadTree(bounds=(0, 0, 1000, 1000), capacity=20)  # max_depth is optional

# Insert points with auto ids
id1 = qt.insert((10, 10))
id2 = qt.insert((200, 300))
id3 = qt.insert((999, 500), id=42)  # you can supply your own id

# Axis-aligned rectangle query
hits = qt.query((0, 0, 250, 350))  # returns [(id, x, y), ...] by default
print(hits)  # e.g. [(1, 10.0, 10.0), (2, 200.0, 300.0)]

# Nearest neighbor
best = qt.nearest_neighbor((210, 310))  # -> (id, x, y) or None
print(best)

# k-nearest neighbors
top3 = qt.nearest_neighbors((210, 310), 3)
print(top3)  # list of up to 3 (id, x, y) tuples
```

### Working with Python objects

You can keep the tree pure and manage your own id → object map, or let the wrapper manage it.

**Option A: Manage your own map**

```python
from quadtree_rs import QuadTree

qt = QuadTree((0, 0, 1000, 1000), capacity=16)
objects: dict[int, object] = {}

def add(obj) -> int:
    obj_id = qt.insert(obj.position)  # auto id
    objects[obj_id] = obj
    return obj_id

# Later, resolve ids back to objects
ids = [obj_id for (obj_id, x, y) in qt.query((100, 100, 300, 300))]
selected = [objects[i] for i in ids]
```

**Option B: Ask the wrapper to track objects**

```python
from quadtree_rs import QuadTree

qt = QuadTree((0, 0, 1000, 1000), capacity=16, track_objects=True)

# Store the object alongside the point
qt.insert((25, 40), obj={"name": "apple"})

# Ask for Item objects so you can access .obj lazily
items = qt.query((0, 0, 100, 100), as_items=True)
for it in items:
    print(it.id, it.x, it.y, it.obj)
```

You can also attach or replace an object later:

```python
qt.attach(123, my_object)  # binds object to id 123
```

## API

### `QuadTree(bounds, capacity, *, max_depth=None, track_objects=False, start_id=1)`

* `bounds` — tuple `(min_x, min_y, max_x, max_y)` covering all points you will insert
* `capacity` — max number of points kept in a leaf before splitting
* `max_depth` — optional depth cap. If omitted, the tree can keep splitting as needed
* `track_objects` — if `True`, the wrapper maintains an id → object map
* `start_id` — starting value for auto-assigned ids

### Methods

* `insert(xy: tuple[float, float], *, id: int | None = None, obj: object | None = None) -> int`
  Insert a point. Returns the id used. Raises `ValueError` if the point is outside `bounds`.
  If `track_objects=True` and `obj` is provided, the object is stored under that id.

* `insert_many_points(points: Iterable[tuple[float, float]]) -> int`
  Bulk insert points with auto ids. Returns count inserted.

* `attach(id: int, obj: object) -> None`
  Attach or replace an object for an existing id. If `track_objects` was false, a map is created on first use.

* `query(rect: tuple[float, float, float, float], *, as_items: bool = False) -> list[(id, x, y)] | list[Item]`
  Return all points whose coordinates lie inside the rectangle. Use `as_items=True` to get `Item` wrappers with lazy `.obj`.

* `nearest_neighbor(xy: tuple[float, float], *, as_item: bool = False) -> (id, x, y) | Item | None`
  Return the closest point to `xy`, or `None` if empty.

* `nearest_neighbors(xy: tuple[float, float], k: int, *, as_items: bool = False) -> list[(id, x, y)] | list[Item]`
  Return up to `k` nearest points.

* `get(id: int) -> object | None`
  Get the object associated with `id` if tracking is enabled.

* `__len__() -> int`
  Number of successful inserts made through this wrapper.

* `NativeQuadTree`
  Reference to the underlying Rust class `quadtree_rs._native.QuadTree` for power users.

### `Item` (returned when `as_items=True`)

* Attributes: `id`, `x`, `y`, and a lazy `obj` property
* Accessing `obj` performs a dictionary lookup only if tracking is enabled

### Geometric conventions

* Rectangles are `(min_x, min_y, max_x, max_y)`.
* Containment rule is open on the min edge and closed on the max edge
  `(x > min_x and x <= max_x and y > min_y and y <= max_y)`.
  This only matters for points exactly on edges.

## Performance tips

* Choose `capacity` so that leaves keep a small batch of points. Typical values are 8 to 64.
* If your data is very skewed, set a `max_depth` to prevent long chains.
* For fastest local runs, use `maturin develop --release`.
* The wrapper keeps Python overhead low: raw tuple results by default, `Item` wrappers only when requested.

## Benchmarks

quadtree-rs outperforms all other quadtree python packages (at least all the ones I could find and install via pip.)

### Library comparison

Generated with `benchmarks/benchmark_plotly.py` in this repo.

* 100k points, 500 queries, capacity 20, max depth 10
* Median over 3 runs per size

![Total time](assets/quadtree_bench_time.png)
![Throughput](assets/quadtree_bench_throughput.png)

### Summary (largest dataset, PyQtree baseline)
- Points: **500,000**, Queries: **500**
--------------------
- Fastest total: **quadtree-rs** at **2.288 s**
- PyQtree total: **9.717 s**
- quadtree-rs total: **2.288 s**
- e-pyquadtree total: **13.504 s**
- Brute force total: **20.450 s**
--------------------

| Library | Build (s) | Query (s) | Total (s) | Speed vs PyQtree |
|---|---:|---:|---:|---:|
| quadtree-rs  | 0.330 | 1.958 | 2.288 | 4.25× |
| PyQtree      | 4.479 | 5.238 | 9.717 | 1.00× |
| e-pyquadtree | 2.821 | 10.683 | 13.504 | 0.72× |
| Brute force  | nan | 20.450 | 20.450 | 0.48× |
| nontree-QuadTree | 1.687 | 7.803 | 9.490 | 1.02× |
| quads        | 3.977 | 9.070 | 13.046 | 0.74× |
| Rtree        | 1.676 | 4.805 | 6.481 | 1.50× |

### Native vs shim

**Setup**

* Points: 100,000
* Queries: 500
* Repeats: 5

**Timing (seconds)**

| Variant           | Build | Query | Total |
| ----------------- | ----: | ----: | ----: |
| Native            | 0.038 | 0.317 | 0.354 |
| Shim (no map)     | 0.051 | 0.309 | 0.360 |
| Shim (track+objs) | 0.057 | 0.321 | 0.379 |

**Overhead vs native**

* No map: build 1.35x, query 0.98x, total 1.01x
* Track + objs: build 1.53x, query 1.01x, total 1.07x

## FAQ

**What happens if I insert the same id more than once?**
Allowed. For k-nearest, duplicates are de-duplicated by id. For range queries you will see every inserted point.

**Can I store rectangles or circles?**
The core stores points. To index objects with extent, insert whatever representative point you choose. For rectangles you can insert centers or build an AABB tree separately.

**Threading**
Use one tree per thread if you need heavy parallel inserts from Python.

## License

MIT. See `LICENSE`.

## Acknowledgments

* Python libraries compared: [PyQtree], [e-pyquadtree]
* Built with [PyO3] and [maturin]

[PyQtree]: https://pypi.org/project/pyqtree/
[e-pyquadtree]: https://pypi.org/project/e-pyquadtree/
[PyO3]: https://pyo3.rs/
[maturin]: https://www.maturin.rs/