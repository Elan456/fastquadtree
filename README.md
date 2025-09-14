# quadtree-rs
Python bindings to a rust-accelerated quadtree implementation


## Peformance

### Summary (largest dataset, PyQtree baseline)
- Points: **100,000**, Queries: **500**
--------------------
- Brute force total: **4.194 s**
- e-pyquadtree total: **2.232 s**
- PyQtree total: **1.317 s**
- quadtree-rs total: **0.309 s**
--------------------

| Library | Build (s) | Query (s) | Total (s) |
|---|---:|---:|---:|
| Brute force  | - | 4.194 | 4.194 |
| e-pyquadtree | 0.430 | 1.803 | 2.232 |
| PyQtree      | 0.688 | 0.629 | 1.317 |
| quadtree-rs  | 0.039 | 0.270 | 0.309 |