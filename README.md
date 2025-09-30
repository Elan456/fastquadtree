# quadtree-rs

![Interactive_V2_Screenshot](https://raw.githubusercontent.com/Elan456/quadtree-rs/main/assets/interactive_v2_screenshot.png)


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

# Delete items by ID and location
deleted = qt.delete(id2, (200, 300))  # True if found and deleted
print(f"Deleted: {deleted}")
print(f"Remaining items: {qt.count_items()}")

# For object tracking with track_objects=True
qt_tracked = QuadTree((0, 0, 1000, 1000), capacity=4, track_objects=True)
player1 = {"name": "Alice", "score": 100}
player2 = {"name": "Bob", "score": 200}

id1 = qt_tracked.insert((50, 50), obj=player1)
id2 = qt_tracked.insert((150, 150), obj=player2)

# Delete by object reference (O(1) lookup!)
deleted = qt_tracked.delete_by_object(player1, (50, 50))
print(f"Deleted player: {deleted}")  # True
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

* `bounds` — tuple `(min_x, min_y, max_x, max_y)` defines the 2D area covered by the quadtree
* `capacity` — max number of points kept in a leaf before splitting
* `max_depth` — optional depth cap. If omitted, the tree can keep splitting as needed
* `track_objects` — if `True`, the wrapper maintains an id → object map for convenience.
* `start_id` — starting value for auto-assigned ids

### Core Methods

Full docs are in the docstrings of the [Python Shim](pysrc/quadtree_rs/__init__.py)

- `insert(xy, *, id=None, obj=None) -> int`

- `insert_many_points(points) -> int`

- `query(rect, *, as_items=False) -> list`

- `nearest_neighbor(xy, *, as_item=False) -> (id, x, y) | Item | None`

- `nearest_neighbors(xy, k, *, as_items=False) -> list`

- `delete(id, xy) -> bool`

- `delete_by_object(obj, xy) -> bool (requires track_objects=True)`

- `attach(id, obj) -> None`

- `count_items() -> int`

- `get(id) -> object | None`

- `get_all_rectangles() -> list[tuple] (for visualization)`

### `Item` (returned when `as_items=True`)

* Attributes: `id`, `x`, `y`, and a lazy `obj` property
* Accessing `obj` performs a dictionary lookup only if tracking is enabled

### Geometric conventions

* Rectangles are `(min_x, min_y, max_x, max_y)`.
* Containment rule is closed on the min edge and open on the max edge
  `(x >= min_x and x < max_x and y >= min_y and y < max_y)`.
  This only matters for points exactly on edges.

## Performance tips

* Choose `capacity` so that leaves keep a small batch of points. Typical values are 8 to 64.
* If your data is very skewed, set a `max_depth` to prevent long chains.
* For fastest local runs, use `maturin develop --release`.
* The wrapper keeps Python overhead low: raw tuple results by default, `Item` wrappers only when requested.

## Benchmarks

quadtree-rs outperforms all other quadtree python packages (at least all the ones I could find and install via pip.)

### Library comparison

![Total time](https://raw.githubusercontent.com/Elan456/quadtree-rs/main/assets/quadtree_bench_time.png)
![Throughput](https://raw.githubusercontent.com/Elan456/quadtree-rs/main/assets/quadtree_bench_throughput.png)

### Summary (largest dataset, PyQtree baseline)
- Points: **500,000**, Queries: **500**
- Fastest total: **quadtree-rs** at **2.207 s**

| Library | Build (s) | Query (s) | Total (s) | Speed vs PyQtree |
|---|---:|---:|---:|---:|
| quadtree-rs  | 0.321 | 1.885 | 2.207 | 4.27× |
| Rtree        | 1.718 | 4.376 | 6.095 | 1.55× |
| nontree-QuadTree | 1.617 | 7.643 | 9.260 | 1.02× |
| PyQtree      | 4.349 | 5.082 | 9.431 | 1.00× |
| quads        | 3.874 | 9.058 | 12.932 | 0.73× |
| e-pyquadtree | 2.732 | 10.598 | 13.330 | 0.71× |
| Brute force  | 0.019 | 19.986 | 20.005 | 0.47× |

### Native vs Shim

**Setup**
- Points: 500,000
- Queries: 500
- Repeats: 5

**Timing (seconds)**

| Variant | Build | Query | Total |
|---|---:|---:|---:|
| Native | 0.483 | 4.380 | 4.863 |
| Shim (no map) | 0.668 | 4.167 | 4.835 |
| Shim (track+objs) | 1.153 | 4.458 | 5.610 |

**Overhead vs Native**

- No map: build 1.38x, query 0.95x, total 0.99x  
- Track + objs: build 2.39x, query 1.02x, total 1.15x

### Run benchmarks
To run the benchmarks yourself, first install the dependencies:

```bash
pip install -r benchmarks/requirements.txt
```

Then run:

```bash
python benchmarks/cross_library_bench.py
python benchmarks/benchmark_native_vs_shim.py 
```

## Run Visualizer
A visualizer is included to help you understand how the quadtree subdivides space.

```bash
pip install -r interactive/requirements.txt
python interactive/interactive_v2.py
```

Check the CLI arguments for the cross-library benchmark in `benchmarks/quadtree_bench/main.py`.

## FAQ

**What happens if I insert the same id more than once?**
Allowed. For k-nearest, duplicates are de-duplicated by id. For range queries you will see every inserted point.

**Can I delete items from the quadtree?**
Yes! Use `delete(id, xy)` to remove specific items. You must provide both the ID and exact location for precise deletion. This handles cases where multiple items exist at the same location. If you're using `track_objects=True`, you can also use `delete_by_object(obj, xy)` for convenient object-based deletion with O(1) lookup. The tree automatically merges nodes when item counts drop below capacity.

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