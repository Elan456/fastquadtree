<!-- Hero header -->
<div style="display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap;">
  <div>
    <h1 style="margin-bottom:0.25rem;">fastquadtree</h1>
    <p style="margin-top:0; font-size:1.05rem;">Rust-optimized quadtree with a clean Python API</p>
    <p style="margin:0.5rem 0 0;">
      <a href="https://pypi.org/project/fastquadtree/">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/fastquadtree.svg">
      </a>
      <a href="https://pypi.org/project/fastquadtree/">
        <img alt="Python versions" src="https://img.shields.io/pypi/pyversions/fastquadtree.svg">
      </a>
      <a href="https://pepy.tech/projects/fastquadtree">
        <img alt="Downloads" src="https://static.pepy.tech/personalized-badge/fastquadtree?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=Total+Downloads">
      </a>
      <a href="https://github.com/Elan456/fastquadtree/actions/workflows/release.yml">
        <img alt="Build" src="https://github.com/Elan456/fastquadtree/actions/workflows/release.yml/badge.svg">
      </a>
      <a href="https://codecov.io/gh/Elan456/fastquadtree">
        <img alt="Coverage" src="https://codecov.io/gh/Elan456/fastquadtree/branch/main/graph/badge.svg">
      </a>
    </p>
    <p style="margin:0.5rem 0 0;">
      <a href="https://pyo3.rs/"><img alt="PyO3" src="https://img.shields.io/badge/Rust-core%20via%20PyO3-orange"></a>
      <a href="https://www.maturin.rs/"><img alt="maturin" src="https://img.shields.io/badge/Built%20with-maturin-1f6feb"></a>
    </p>
    <p style="margin-top:0.75rem;">
  </div>
  <div style="min-width:260px; max-width:420px; flex:1;">
    <img alt="Interactive Screenshot" src="https://raw.githubusercontent.com/Elan456/fastquadtree/main/assets/interactive_v2_screenshot.png">
  </div>
</div>

---

## Why use fastquadtree

- Clean [Python API](api/quadtree.md) with no external dependencies and modern typing hints
- The fastest quadtree Python package ([>10x faster](benchmark.md) than pyqtree)
- Prebuilt wheels for Windows, macOS, and Linux
- Support for [inserting bounding boxes](api/rect_quadtree.md) or points
- Fast KNN and range queries
- Optional object tracking for id ↔ object mapping
- [100% test coverage](https://codecov.io/gh/Elan456/fastquadtree) and CI on GitHub Actions

## Examples
See examples of how fastquadtree can be used in the [runnables](runnables.md) section.


## Install
```bash
pip install fastquadtree
```

## Import

```python
from fastquadtree import QuadTree  # Point handling
from fastquadtree import RectQuadTree  # Bounding box handling
from fastquadtree.pyqtree import Index # Drop-in replacement for pyqtree (6.567x faster while keeping the same API)
```