# fastquadtree

<img src="https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/interactive_v2_screenshot.png"
     alt="Interactive Screenshot" align="right" width="420">

Rust-optimized quadtree with a clean Python API

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

- Clean [Python API](https://elan456.github.io/fastquadtree/api/quadtree/) with no external dependencies and modern typing hints
- The fastest quadtree Python package ([>10x faster](https://elan456.github.io/fastquadtree/benchmark/) than pyqtree)
- Prebuilt wheels for Windows, macOS, and Linux
- Support for [inserting bounding boxes](https://elan456.github.io/fastquadtree/api/rect_quadtree/) or points
- Fast KNN and range queries
- Optional object tracking for id ↔ object mapping
- [100% test coverage](https://codecov.io/gh/Elan456/fastquadtree) and CI on GitHub Actions
- Offers a drop-in [pyqtree shim](https://elan456.github.io/fastquadtree/benchmark/#pyqtree-drop-in-shim-performance-gains) that is 6.567x faster while keeping the same API

----

👉 **Docs:** https://elan456.github.io/fastquadtree/

## Examples
See examples of how fastquadtree can be used in the [runnables](https://elan456.github.io/fastquadtree/runnables/) section.


## Install
```bash
pip install fastquadtree
```

```python
from fastquadtree import QuadTree  # Point handling
from fastquadtree import RectQuadTree  # Bounding box handling
from fastquadtree.pyqtree import Index  # Drop-in pyqtree shim (6.567x faster while keeping the same API)
```

## Benchmarks

fastquadtree **outperforms** all other quadtree Python packages, including the Rtree spatial index.

### Library comparison

![Total time](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_time.png)
![Throughput](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/quadtree_bench_throughput.png)

### Summary (largest dataset, PyQtree baseline)
- Points: **250,000**, Queries: **500**
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

See the [benchmark section](https://elan456.github.io/fastquadtree/benchmark/) for details, including configurations, system info, and native vs shim benchmarks.

## Quickstart
[See the quickstart guide](https://elan456.github.io/fastquadtree/quickstart/)

## API

[See the full API](https://elan456.github.io/fastquadtree/api/quadtree/)

### `QuadTree(bounds, capacity, max_depth=None, track_objects=False, start_id=1)`

* `bounds` — tuple `(min_x, min_y, max_x, max_y)` defines the 2D area covered by the quadtree
* `capacity` — max number of points kept in a leaf before splitting
* `max_depth` — optional depth cap. If omitted, the tree can keep splitting as needed
* `track_objects` — if `True`, the wrapper maintains an id → object map for convenience.
* `start_id` — starting value for auto-assigned ids

### Key Methods

- `insert(xy, *, id=None, obj=None) -> int`

- `query(rect, *, as_items=False) -> list`

- `nearest_neighbor(xy, *, as_item=False) -> (id, x, y) | Item | None`

- `delete(id, xy) -> bool`

There are more methods and object tracking versions in the [docs](https://elan456.github.io/fastquadtree/api/quadtree/).

### Geometric conventions

* Rectangles are `(min_x, min_y, max_x, max_y)`.
* Containment rule is closed on the min edge and open on the max edge
  `(x >= min_x and x < max_x and y >= min_y and y < max_y)`.
  This only matters for points exactly on edges.

## Performance tips

* Choose `capacity` so that leaves keep a small batch of points. Typical values are 8 to 64.
* If your data is very skewed, set a `max_depth` to prevent long chains.
* For fastest local runs, use `maturin develop --release`.
* The wrapper maintains an object map only if the quadtree was constructed with `track_objects=True`. If you don't need it, leave it off for best performance.
* Refer to the [Native vs Shim Benchmark](https://elan456.github.io/fastquadtree/benchmark/#native-vs-shim-benchmark) for overhead details.

### Pygame Ball Pit Demo

![Ballpit_Demo_Screenshot](https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/ballpit.png)

A simple demo of moving objects with collision detection using **fastquadtree**. 
You can toggle between quadtree mode and brute-force mode to see the performance difference.

See the [runnables guide](https://elan456.github.io/fastquadtree/runnables/) for setup instructions.

## FAQ

**What happens if I insert the same id more than once?**
Allowed. For k-nearest, duplicates are de-duplicated by id. For range queries you will see every inserted point.

**Can I delete items from the quadtree?**
Yes! Use `delete(id, xy)` to remove specific items. You must provide both the ID and exact location for precise deletion. This handles cases where multiple items exist at the same location. If you're using `track_objects=True`, you can also use `delete_by_object(obj)` for convenient object-based deletion with O(1) lookup. The tree automatically merges nodes when item counts drop below capacity.

**Can I store rectangles or circles?**
Yes, you can store rectangles using the `RectQuadTree` class. Circles can be approximated with bounding boxes. See the [RectQuadTree docs](https://elan456.github.io/fastquadtree/api/rect_quadtree/) for details.

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
