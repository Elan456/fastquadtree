# fastquadtree.pygame

Optional pygame sprite collision integration backed by `RectQuadTreeObjects`.

!!! note "Optional dependency"
    Core `fastquadtree` APIs do not require pygame. This page documents the
    optional `fastquadtree.pygame` integration, which requires pygame or a
    compatible package such as `pygame-ce` when imported at runtime.

Use `Group.query_rect(...)` to index sprites intersecting a
rectangle. This is useful for broadphase collision culling and viewport culling.

```python
visible = blocks.query_rect(camera_rect)
nearby = blocks.query_rect((0, 0, 640, 480), sync=False)
```

::: fastquadtree.pygame
