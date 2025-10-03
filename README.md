# fastquadtree

[![PyPI version](https://img.shields.io/pypi/v/fastquadtree.svg)](https://pypi.org/project/fastquadtree/)
[![Python versions](https://img.shields.io/pypi/pyversions/fastquadtree.svg)](https://pypi.org/project/fastquadtree/)
[![Wheels](https://img.shields.io/pypi/wheel/fastquadtree.svg)](https://pypi.org/project/fastquadtree/#files)
[![License: MIT](https://img.shields.io/pypi/l/fastquadtree.svg)](LICENSE)

[![PyPI Downloads](https://static.pepy.tech/personalized-badge/fastquadtree?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=Total+Downloads)](https://pepy.tech/projects/fastquadtree)

[![Build](https://github.com/Elan456/fastquadtree/actions/workflows/release.yml/badge.svg)](https://github.com/Elan456/fastquadtree/actions/workflows/ci.yml)
[![Codecov](https://codecov.io/gh/Elan456/fastquadtree/branch/main/graph/badge.svg)](https://codecov.io/gh/Elan456/fastquadtree)

[![Rust core via PyO3](https://img.shields.io/badge/Rust-core%20via%20PyO3-orange)](https://pyo3.rs/)
[![Built with maturin](https://img.shields.io/badge/Built%20with-maturin-1f6feb)](https://www.maturin.rs/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)



![Interactive_V2_Screenshot](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/interactive_v2_screenshot.png)


Rust-optimized quadtree with a simple Python API.

- Python package: **`fastquadtree`**
- Python ≥ 3.8
- Import path: `from fastquadtree import QuadTree`

## Benchmarks

fastquadtree **outperforms** all other quadtree Python packages, including the Rtree spatial index.

### Library comparison

![Total time](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_time.png)
![Throughput](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_throughput.png)

### Summary (largest dataset, PyQtree baseline)
- Points: **250,000**, Queries: **500**
--------------------
- Fastest total: **fastquadtree** at **0.120 s**

| Library | Build (s) | Query (s) | Total (s) | Speed vs PyQtree |
|---|---:|---:|---:|---:|
| fastquadtree | 0.031 | 0.089 | 0.120 | 14.64× |
| Shapely STRtree | 0.179 | 0.100 | 0.279 | 6.29× |
| nontree-QuadTree | 0.595 | 0.605 | 1.200 | 1.46× |
| Rtree        | 0.961 | 0.300 | 1.261 | 1.39× |
| e-pyquadtree | 1.005 | 0.660 | 1.665 | 1.05× |
| PyQtree      | 1.492 | 0.263 | 1.755 | 1.00× |
| quads        | 1.407 | 0.484 | 1.890 | 0.93× |

#### Benchmark Configuration
| Parameter | Value |
|---|---:|
| Bounds | (0, 0, 1000, 1000) |
| Max points per node | 128 |
| Max depth | 16 |
| Queries per experiment | 500 |

## Install

```bash
pip install fastquadtree
````

If you are developing locally:

```bash
# optimized dev install
maturin develop --release
```

## Quickstart

```python
from fastquadtree import QuadTree

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
deleted = qt_tracked.delete_by_object(player1)
print(f"Deleted player: {deleted}")  # True
```

### Working with Python objects

You can keep the tree pure and manage your own id → object map, or let the wrapper manage it.

**Wrapper Managed Objects**

```python
from fastquadtree import QuadTree

qt = QuadTree((0, 0, 1000, 1000), capacity=16, track_objects=True)

# Store the object alongside the point
qt.insert((25, 40), obj={"name": "apple"})

# Ask for Item objects within a bounding box
items = qt.query((0, 0, 100, 100), as_items=True)
for it in items:
    print(it.id, it.x, it.y, it.obj)
```

You can also attach or replace an object later:

```python
qt.attach(123, my_object)  # binds object to id 123
```

## API

### `QuadTree(bounds, capacity, max_depth=None, track_objects=False, start_id=1)`

* `bounds` — tuple `(min_x, min_y, max_x, max_y)` defines the 2D area covered by the quadtree
* `capacity` — max number of points kept in a leaf before splitting
* `max_depth` — optional depth cap. If omitted, the tree can keep splitting as needed
* `track_objects` — if `True`, the wrapper maintains an id → object map for convenience.
* `start_id` — starting value for auto-assigned ids

### Core Methods

Full docs are in the docstrings of the [Python Shim](pysrc/fastquadtree/__init__.py)

- `insert(xy, *, id=None, obj=None) -> int`

- `insert_many_points(points) -> int`

- `query(rect, *, as_items=False) -> list`

- `nearest_neighbor(xy, *, as_item=False) -> (id, x, y) | Item | None`

- `nearest_neighbors(xy, k, *, as_items=False) -> list`

- `delete(id, xy) -> bool`

- `delete_by_object(obj) -> bool (requires track_objects=True)`

- `attach(id, obj) -> None (requires track_objects=True)`

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


### Native vs Shim Benchmark

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
Yes! Use `delete(id, xy)` to remove specific items. You must provide both the ID and exact location for precise deletion. This handles cases where multiple items exist at the same location. If you're using `track_objects=True`, you can also use `delete_by_object(obj)` for convenient object-based deletion with O(1) lookup. The tree automatically merges nodes when item counts drop below capacity.

**Can I store rectangles or circles?**
The core stores points. To index objects with extent, insert whatever representative point you choose. For rectangles you can insert centers or build an AABB tree separately.

**Threading**
Use one tree per thread if you need heavy parallel inserts from Python.

## License

MIT. See `LICENSE`.

## Acknowledgments

* Python libraries compared: [PyQtree], [e-pyquadtree], [Rtree], [nontree], [quads], [Shapely]
* Built with [PyO3] and [maturin]

[PyQtree]: https://pypi.org/project/pyqtree/
[e-pyquadtree]: https://pypi.org/project/e-pyquadtree/
[PyO3]: https://pyo3.rs/
[maturin]: https://www.maturin.rs/
[Rtree]: https://pypi.org/project/Rtree/
[nontree]: https://pypi.org/project/nontree/
[quads]: https://pypi.org/project/quads/
[Shapely]: https://pypi.org/project/Shapely/
