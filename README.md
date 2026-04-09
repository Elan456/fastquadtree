# fastquadtree

<img src="https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/interactive_v2_screenshot.png"
     alt="Interactive Screenshot" align="right" width="420">

Rust-optimized quadtree with a clean Python API

👉 **Check out the Docs:** https://elan456.github.io/fastquadtree/

[![PyPI](https://img.shields.io/pypi/v/fastquadtree.svg)](https://pypi.org/project/fastquadtree/)
[![Python versions](https://img.shields.io/pypi/pyversions/fastquadtree.svg)](https://pypi.org/project/fastquadtree/)
[![Downloads](https://static.pepy.tech/personalized-badge/fastquadtree?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=Total%20Downloads)](https://pepy.tech/projects/fastquadtree)
[![Build](https://github.com/Elan456/fastquadtree/actions/workflows/release.yml/badge.svg)](https://github.com/Elan456/fastquadtree/actions/workflows/release.yml)
![No runtime deps](https://img.shields.io/badge/deps-none-success)

[![PyO3](https://img.shields.io/badge/Rust-core%20via%20PyO3-orange)](https://pyo3.rs/)
[![maturin](https://img.shields.io/badge/Built%20with-maturin-1f6feb)](https://www.maturin.rs/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![Docs](https://img.shields.io/badge/docs-online-brightgreen)](https://elan456.github.io/fastquadtree/)
[![Wheels](https://img.shields.io/pypi/wheel/fastquadtree.svg)](https://pypi.org/project/fastquadtree/#files)
[![Coverage](https://codecov.io/gh/Elan456/fastquadtree/branch/main/graph/badge.svg)](https://codecov.io/gh/Elan456/fastquadtree)
[![License: MIT](https://img.shields.io/pypi/l/fastquadtree.svg)](LICENSE)

<br clear="right"/>

## Why use fastquadtree

- Just pip install: prebuilt wheels for Windows, macOS, and Linux (no Rust or compiler needed)
- The fastest quadtree Python package ([>10x faster](https://elan456.github.io/fastquadtree/benchmark/) than pyqtree)
- Clean [Python API](https://elan456.github.io/fastquadtree/api/quadtree/) with **no external dependencies** and modern typing hints
- Support for [inserting bounding boxes](https://elan456.github.io/fastquadtree/api/rect_quadtree/) or points
- Fast KNN and range queries
- Optional object tracking for id ↔ object mapping
- Fast [serialization](https://elan456.github.io/fastquadtree/benchmark/#serialization-vs-rebuild) to/from bytes
- Support for multiple data types (f32, f64, i32, i64) for coordinates
- [100% test coverage](https://codecov.io/gh/Elan456/fastquadtree) and CI on GitHub Actions
- Offers a drop-in [pyqtree shim](https://elan456.github.io/fastquadtree/benchmark/#pyqtree-drop-in-shim-performance-gains) that is ~10x faster while keeping the same API

----

## Install
```bash
pip install fastquadtree
```

```python
from fastquadtree import QuadTree # Or any of the other classes mentioned below
from fastquadtree.pyqtree import Index  # Drop-in pyqtree shim (~10x faster while keeping the same API)
```

### Which class should I use?

| Class | Stores | Object tracking | Best when |
|---|---|---|---|
| `QuadTree` | Points | No | Fast point indexing with IDs only. |
| `RectQuadTree` | Bounding boxes | No | Fast rectangle overlap queries. |
| `QuadTreeObjects` | Points | Yes | Point indexing with attached Python objects. |
| `RectQuadTreeObjects` | Bounding boxes | Yes | Rectangle indexing with attached objects. |
| `fastquadtree.pyqtree.Index` | Bounding boxes | Yes | pyqtree-compatible API, much faster. |


## Quickstart

### 1. Point indexing (`QuadTree`)

```python
from fastquadtree import QuadTree

qt = QuadTree((0.0, 0.0, 1000.0, 1000.0), capacity=16)

# Insert points with explicit IDs (or omit id_ for auto IDs)
qt.insert((100.0, 200.0), id_=1)
qt.insert((120.0, 260.0), id_=2)
qt.insert((700.0, 800.0), id_=3)

# Range query returns (id, x, y)
hits = qt.query((0.0, 0.0, 500.0, 500.0))
print(hits)  # [(1, 100.0, 200.0), (2, 120.0, 260.0)]
```

### 2. Object tracking (`QuadTreeObjects`)

```python
from fastquadtree import QuadTreeObjects

qto = QuadTreeObjects((0.0, 0.0, 1000.0, 1000.0), capacity=16)

player = {"name": "player-1", "team": "blue"}
enemy = {"name": "enemy-9", "team": "red"}

qto.insert((100.0, 200.0), obj=player)
qto.insert((130.0, 220.0), obj=enemy)

# Query returns PointItem objects with id_, x, y, geom, and obj
for item in qto.query((0.0, 0.0, 300.0, 300.0)):
    print(item.id_, item.x, item.y, item.obj["name"]) # 0 100.0 200.0 player-1
                                                      # 1 130.0 220.0 enemy-9

# Remove entries by object identity
qto.delete_by_object(player)  # returns 1 (number of items deleted)
```

[See the quickstart guide](https://elan456.github.io/fastquadtree/quickstart/) or the [interactive demos](https://elan456.github.io/fastquadtree/runnables/) for more details.
## Benchmarks

fastquadtree **outperforms** all other quadtree Python packages, including the Rtree spatial index.

### Library comparison

![Total time](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_time.png)
![Throughput](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_throughput.png)

### Summary (PyQtree baseline, sorted by total time)
- Points: **500,000**, Queries: **500**

| Library | Build (s) | Query (s) | Total (s) | Speed vs PyQtree |
|---|---:|---:|---:|---:|
| fastquadtree (np)[^fqtnp] | 0.052 | 0.017 | 0.068 | 42.52× |
| fastquadtree[^fqt] | 0.054 | 0.231 | 0.285 | 10.20× |
| Shapely STRtree[^npreturn] | 0.200 | 0.110 | 0.309 | 9.40× |
| fastquadtree (obj tracking)[^fqto] | 0.263 | 0.093 | 0.356 | 8.17× |
| nontree-QuadTree | 0.826 | 0.844 | 1.670 | 1.74× |
| Rtree        | 1.805 | 0.546 | 2.351 | 1.24× |
| e-pyquadtree | 1.530 | 0.941 | 2.471 | 1.18× |
| quads        | 1.907 | 0.759 | 2.667 | 1.09× |
| PyQtree      | 2.495 | 0.414 | 2.909 | 1.00× |

[^fqtnp]: Uses `query_np` for Numpy array return values rather than Python lists.  
[^fqt]: Uses standard `query` method returning Python lists.  
[^npreturn]: Uses Shapely STRtree with Numpy array points and returns.  
[^fqto]: Uses QuadTreeObjects with object association.  

See the [benchmark section](https://elan456.github.io/fastquadtree/benchmark/) for details, including configurations, system info, and native vs shim benchmarks.

## API

[See the full API](https://elan456.github.io/fastquadtree/api/quadtree/)

### `QuadTree(bounds, capacity, max_depth=None, dtype="f32")`

* `bounds` — tuple `(min_x, min_y, max_x, max_y)` defines the 2D area covered by the quadtree
* `capacity` — max number of points kept in a leaf before splitting
* `max_depth` — optional depth cap. If omitted, the tree can keep splitting as needed
* `dtype` — data type for coordinates, e.g., `"f32"`, `"f64"`, `"i32"`, `"i64"`

### Key Methods

- `insert(xy, id_=None) -> int`

- `query(rect) -> list[tuple[int, float, float]]`

- `nearest_neighbor(xy) -> tuple[int, float, float] | None`

- `delete(id, x, y) -> bool`

For object tracking, use `QuadTreeObjects` instead. See the [docs](https://elan456.github.io/fastquadtree/api/quadtree/) for more methods.

### Geometric conventions

* Rectangles are `(min_x, min_y, max_x, max_y)`.
* Containment rule is closed on the min edge and open on the max edge
  `(x >= min_x and x < max_x and y >= min_y and y < max_y)`.
  This only matters for points exactly on edges.

## Performance tips

* Choose `capacity` so that leaves keep a small batch of points. Typical values are 8 to 64.
* If your data is very skewed, set a `max_depth` to prevent long chains.
* For fastest local runs, use `maturin develop --release`.
* Use `QuadTree` when you only need spatial indexing. Use `QuadTreeObjects` when you need to store Python objects with your points.
* Refer to the [Native vs Shim Benchmark](https://elan456.github.io/fastquadtree/benchmark/#native-vs-shim-benchmark) for overhead details.

### Pygame Ball Pit Demo

![Ballpit_Demo_Screenshot](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/ballpit.png)

A simple demo of moving objects with collision detection using **fastquadtree**. 
You can toggle between fastquadtree, pyqtree, and brute-force mode to see the performance difference.
I typically see an FPS of ~70 with fastquadtree, ~25 with pyqtree, and <1 FPS with brute-force on my machine with 1500 balls. 

See the [runnables guide](https://elan456.github.io/fastquadtree/runnables/) for setup instructions.

## FAQ

**Can I delete items from the quadtree?**
Yes! Use `delete(id, x, y)` to remove specific items. You must provide both the ID and exact location for precise deletion. This handles cases where multiple items exist at the same location. If you're using `QuadTreeObjects`, you can also use `delete_by_object(obj)` for convenient object-based deletion with O(1) lookup. The tree automatically merges nodes when item counts drop below capacity.

**Can I store rectangles or circles?**
Yes, you can store rectangles using the `RectQuadTree` class. Circles can be approximated with bounding boxes. See the [RectQuadTree docs](https://elan456.github.io/fastquadtree/api/rect_quadtree/) for details.

**Do I need NumPy installed?**
No, NumPy is a fully optional dependency. If you do have NumPy installed, you can use methods such as `query_np` and `insert_many_np` for better performance. Note that `insert_many` raises `TypeError` on NumPy input—you must use `insert_many_np` explicitly for NumPy arrays. The Rust core is able to handle NumPy arrays faster than Python lists, so there's a lot of time savings in utilizing the NumPy functions. See the [Native vs Shim benchmark](https://elan456.github.io/fastquadtree/benchmark/#native-vs-shim) for details on how returing NumPy arrays can speed up queries.

```python
# Using Python lists
qt.insert_many([(10, 20), (30, 40), (50, 60)])

# Using NumPy arrays (requires NumPy)
import numpy as np
points = np.array([[10, 20], [30, 40], [50, 60]])
qt.insert_many_np(points)  # Use insert_many_np for NumPy arrays
```

**Does fastquadtree support multiprocessing?**
Yes, fastquadtree objects can be serialized to bytes using the `to_bytes()` method and deserialized back using `from_bytes()`. This allows you to share quadtree data across processes and even cache prebuilt trees to disk. When using `QuadTreeObjects` or `RectQuadTreeObjects`, you must pass `include_objects=True` to `to_bytes()` to serialize Python objects, and `allow_objects=True` to `from_bytes()` when loading. By default, objects are skipped for safety, as deserializing untrusted Python objects can be unsafe. See the [interactive v2 demo](https://github.com/Elan456/fastquadtree/blob/main/interactive/interactive_v2.py) for an example of saving and loading a quadtree, and the [QuadTreeObjects API docs](https://elan456.github.io/fastquadtree/api/quadtree_objects/#fastquadtree.QuadTreeObjects.to_bytes) for full details on the serialization methods.

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
