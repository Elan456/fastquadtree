# Quickstart

Meet **fastquadtree** — a Rust powered spatial index for Python

> TLDR: create a tree, insert points, insert boxes, query ranges or nearest neighbors.

## Installation

```bash
pip install fastquadtree
```

---

## 30-second demo

```python
from fastquadtree import QuadTree

# 1) Make a tree that covers your world
qt = QuadTree(bounds=(0, 0, 1000, 1000), capacity=20)

# 2) Add some stuff (a, b, and c are auto-generated ids)
a = qt.insert((10, 10))
b = qt.insert((200, 300))
c = qt.insert((999, 500))

# 3) Ask spatial questions
print("Range hits:", qt.query((0, 0, 250, 350)))  # -> [(id, x, y), ...]

print("Nearest to (210, 310):", qt.nearest_neighbor((210, 310)))
# -> (1, 200.0, 300.0)

print("Top 3 near (210, 310):", qt.nearest_neighbors((210, 310), 3))
# -> [(1, 200.0, 300.0), (0, 10.0, 10.0), (2, 999.0, 500.0)]

# 4) Delete by id and exact location
print("Deleted:", qt.delete(b, 200, 300))
print("Count:", len(qt))  # -> 2

# 5) Update position by id and exact location
success = qt.update(a, 10, 10, 35, 35)  # Move point a to (35, 35)
print("Update success:", success)  # -> True
```

## Range queries that feel natural

```python
# Think of it like a camera frustum in 2D
viewport = (100, 200, 400, 600)
for id_, x, y in qt.query(viewport):
    print(f"Visible: id={id_} at ({x:.1f}, {y:.1f})")
```

Use this for viewport culling, collision broad-phase, spatial filtering, and quick “what is inside this box” checks.

---

## Nearest neighbor for snapping and picking

```python
cursor = (212, 305)
hit = qt.nearest_neighbor(cursor)
if hit:
    id_, x, y = hit
    print(f"Closest to cursor is id={id_} at ({x:.1f}, {y:.1f})")
```

Need more than one neighbor

```python
for id_, x, y in qt.nearest_neighbors(cursor, k=5):
    print(id_, x, y)
```

---

## Track Python objects when you need them

Use `QuadTreeObjects` to bind your own objects to spatial coordinates. Object lookups for deletion are O(1).

```python
from fastquadtree import QuadTreeObjects

qt = QuadTreeObjects((0, 0, 1000, 1000), capacity=16)

player = {"name": "Alice", "hp": 100}
enemy  = {"name": "Boblin", "hp": 60}

pid = qt.insert((50, 50), obj=player)
eid = qt.insert((80, 60), obj=enemy)

# Query returns Item objects with both coordinates and the stored object
items = qt.query((0, 0, 200, 200))
for item in items:
    print(item.id_, item.x, item.y, item.obj)

# Remove by object identity (returns deletion count)
deleted = qt.delete_by_object(player)  # 1
```

Tip: Use `QuadTree` instead of `QuadTreeObjects` for max speed when you do not need object tracking.

---

## Reset between runs without breaking references

Keep the same `QuadTree` instance alive for UIs or game loops. Wipe contents and optionally reset ids.

```python
qt.clear()  # tree is empty, auto ids start again at 0, all objects forgotten
```

---

## Tiny benchmark sketch

```python
import random, time
from fastquadtree import QuadTree

N = 200_000
pts = [(random.random()*1000, random.random()*1000) for _ in range(N)]
qt = QuadTree((0, 0, 1000, 1000), capacity=32)

t0 = time.perf_counter()
qt.insert_many(pts)
t1 = time.perf_counter()

hits = qt.query((250, 250, 750, 750))
t2 = time.perf_counter()

print(f"Build: {(t1-t0):.3f}s  Query: {(t2-t1):.3f}s  Hits: {len(hits)}")
```

---

## Common patterns

* **Use `capacity` 8 to 64** for most workloads
  If data is highly skewed, set a `max_depth` to avoid very deep trees.
* **Use `clear()` to reset** when most points are moving rather than deleting and reinserting.
* **Use `insert_many()`** to bulk load a large batch of points at once.