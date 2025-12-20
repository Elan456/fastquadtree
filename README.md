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
- Offers a drop-in [pyqtree shim](https://elan456.github.io/fastquadtree/benchmark/#pyqtree-drop-in-shim-performance-gains) that is 6.791x faster while keeping the same API

----


## Examples
See examples of how fastquadtree can be used in the [runnables](https://elan456.github.io/fastquadtree/runnables/) section.


## Install
```bash
pip install fastquadtree
```

```python
from fastquadtree import QuadTree  # Point handling
from fastquadtree import RectQuadTree  # Bounding box handling
from fastquadtree import QuadTreeObjects  # Point handling with object tracking
from fastquadtree import RectQuadTreeObjects  # Bounding box handling with object tracking
from fastquadtree.pyqtree import Index  # Drop-in pyqtree shim (6.791x faster while keeping the same API)
```

## Benchmarks

fastquadtree **outperforms** all other quadtree Python packages, including the Rtree spatial index.

### Library comparison

![Total time](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_time.png)
![Throughput](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_throughput.png)

### Summary (largest dataset, PyQtree baseline)
- Points: **500,000**, Queries: **500**
- Fastest total: **fastquadtree** at **0.067 s**

| Library | Build (s) | Query (s) | Total (s) | Speed vs PyQtree |
|---|---:|---:|---:|---:|
| fastquadtree | 0.050 | 0.017 | 0.067 | 44.95× |
| Shapely STRtree | 0.209 | 0.103 | 0.312 | 9.58× |
| fastquadtree (obj tracking) | 0.289 | 0.178 | 0.467 | 6.41× |
| nontree-QuadTree | 0.878 | 1.030 | 1.908 | 1.57× |
| Rtree        | 1.742 | 0.523 | 2.265 | 1.32× |
| e-pyquadtree | 1.425 | 0.975 | 2.400 | 1.25× |
| quads        | 2.094 | 0.782 | 2.876 | 1.04× |
| PyQtree      | 2.607 | 0.385 | 2.992 | 1.00× |

See the [benchmark section](https://elan456.github.io/fastquadtree/benchmark/) for details, including configurations, system info, and native vs shim benchmarks.

## Quickstart
[See the quickstart guide](https://elan456.github.io/fastquadtree/quickstart/)

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
You can toggle between quadtree mode and brute-force mode to see the performance difference.

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
