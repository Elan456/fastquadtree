# quadtree-rs

Rust-accelerated quadtree with a simple Python API.

* Python package: **`quadtree_rs`**
* Python ≥ 3.8
* Import path: `from quadtree_rs import QuadTree`

## Install

```bash
pip install quadtree_rs
```

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

# Insert points. You supply an integer id and a point (x, y)
for i, (x, y) in enumerate([(10, 10), (200, 300), (999, 500)]):
    ok = qt.insert(i, (x, y))
    assert ok  # False means the point was outside the bounds

# Axis-aligned rectangle query
hits = qt.query((0, 0, 250, 350))
# hits is a list of tuples: (id, x, y)
print(hits)  # e.g. [(0, 10.0, 10.0), (1, 200.0, 300.0)]

# Nearest neighbor
best = qt.nearest_neighbor((210, 310))
print(best)  # e.g. (1, 200.0, 300.0)

# k-nearest neighbors
top3 = qt.nearest_neighbors((210, 310), 3)
print(top3)  # list of up to 3 (id, x, y) tuples
```

### Mapping your Python objects

The quadtree stores ids and points for speed. Keep a side map in Python:

```python
from quadtree_rs import QuadTree

qt = QuadTree((0, 0, 1000, 1000), capacity=16)
objects = {}  # id -> your object

def add(obj_id: int, obj):
    objects[obj_id] = obj
    qt.insert(obj_id, obj.position)  # where obj.position is (x, y)

# Later, resolve ids back to objects
ids = [obj_id for (obj_id, x, y) in qt.query((100, 100, 300, 300))]
selected = [objects[i] for i in ids]
```

## API

### `QuadTree(bounds, capacity, *, max_depth=None)`

* `bounds` – tuple `(min_x, min_y, max_x, max_y)` covering all points you will insert
* `capacity` – max number of items kept in a leaf before a split
* `max_depth` – optional depth cap. If omitted, the tree can grow until data distribution stops forcing splits

### Methods

* `insert(id: int, xy: tuple[float, float]) -> bool`
  Inserts a point with your integer id. Returns `False` if the point is outside `bounds`.

* `query(rect: tuple[float, float, float, float]) -> list[tuple[int, float, float]]`
  Returns all items whose point lies inside the axis-aligned rectangle. Each item is `(id, x, y)`.

* `nearest_neighbor(xy: tuple[float, float]) -> tuple[int, float, float] | None`
  Returns the closest point to `xy`, or `None` if the tree is empty.

* `nearest_neighbors(xy: tuple[float, float], k: int) -> list[tuple[int, float, float]]`
  Returns up to `k` nearest points to `xy`.

### Geometric conventions

* Rectangles are `(min_x, min_y, max_x, max_y)`.
* Containment rule is open on the min edge and closed on the max edge
  `(x > min_x and x <= max_x and y > min_y and y <= max_y)`.
  This only matters for points exactly on edges. Decide your own tie-break if you care about boundaries.

### Performance tips

* Choose `capacity` so that leaves keep a small batch of points. Typical values are 8 to 64.
* If your data is very skewed, set a `max_depth` to prevent degenerate chains.
* For best runtime on your local dev build install with `maturin develop --release`.

## Benchmarks

Generated with `benchmarks/bench_plotly.py` in this repo.

* 100k points, 500 queries, capacity 20, max depth 10
* Median over 3 runs per size

![Total time](assets/quadtree_bench_time.png)
![Throughput](assets/quadtree_bench_throughput.png)

Example summary (PyQtree baseline) — numbers will vary by machine:

| Library | Build (s) | Query (s) | Total (s) |
|---|---:|---:|---:|
| Brute force  | - | 4.068 | 4.068 |
| e-pyquadtree | 0.447 | 1.951 | 2.398 |
| PyQtree      | 0.686 | 0.685 | 1.371 |
| quadtree-rs  | 0.038 | 0.285 | 0.323 |

## FAQ

**What happens if I insert the same id more than once?**
Allowed. For k-nearest, duplicates are de-duplicated by id. For range queries you will see every inserted point.

**Can I store rectangles or circles?**
The core stores points. To index objects with extent, insert the representative point you choose. For rectangles, you can either insert their centers or build a separate AABB tree.

**Threading**
The Python wrapper is a single `QuadTree` object with interior mutability handled in Rust. Use one tree per thread if you do heavy parallel inserts from Python.

## License

MIT. See `LICENSE`.

## Acknowledgments

* Python libraries compared: [PyQtree], [e-pyquadtree]
* Built with [PyO3] and [maturin]

[PyQtree]: https://pypi.org/project/pyqtree/
[e-pyquadtree]: https://pypi.org/project/e-pyquadtree/
[PyO3]: https://pyo3.rs/
[maturin]: https://www.maturin.rs/
