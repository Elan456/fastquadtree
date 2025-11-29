# Using `fastquadtree` directly from Rust (no Python)

`fastquadtree` is primarily a Python package on PyPI, but you can also utilize its Rust implementation directly in your Rust projects.

Right now the easiest way to consume it from Rust is via a Git dependency (crates.io publishing may come later).

## Add the dependency

In your project’s `Cargo.toml`:

```toml
[dependencies]
fastquadtree = { git = "https://github.com/Elan456/fastquadtree" }
```

Then build normally:

```bash
cargo build
```

## Minimal example

Create `src/main.rs`:

```rust
use fastquadtree::{Point, Rect, Item, QuadTree};

fn main() {
    let boundary: Rect<f32> = Rect {
        min_x: 0.0,
        min_y: 0.0,
        max_x: 100.0,
        max_y: 100.0,
    };

    // QuadTree::new(boundary, capacity, max_depth)
    let mut qt: QuadTree<f32> = QuadTree::new(boundary, 16, 4);

    let item: Item<f32> = Item {
        id: 1,
        point: Point { x: 10.0, y: 10.0 },
    };

    qt.insert(item);

    let range: Rect<f32> = Rect {
        min_x: 5.0,
        min_y: 5.0,
        max_x: 15.0,
        max_y: 15.0,
    };

    let found_items = qt.query(range);
    println!("Found items: {:?}", found_items);
}
```

Run it:

```bash
cargo run
```

## Notes

* This uses the Rust “core” types directly: `QuadTree`, `RectQuadTree`, `Rect`, `Point`, `Item`.
* You can find their implementations in the [Rust source code](https://github.com/Elan456/fastquadtree/tree/main/src)
* The crate currently includes Python-related dependencies because it is also used to build the PyPI module. This is expected for now.
* In the future, dependency size will likely improve via Cargo feature flags or a split between a pure Rust `fastquadtree-core` crate and a Python wrapper crate.
* The semantic versioning of fastquadtree for PyPi is currently tied to the Python API, so breaking changes to the Rust core may not always align with major version bumps. Be sure to check the changelog for any breaking changes when updating.

## Pinning to a specific commit 

**recommended for reproducibility**

If you want your build to be fully reproducible, pin the dependency to a commit:

```toml
[dependencies]
fastquadtree = { git = "https://github.com/Elan456/fastquadtree", rev = "<commit-sha>" }
```

(Replace `<commit-sha>` with the commit you want to lock to.)
