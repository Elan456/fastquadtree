# fastquadtree.QuadTree

A spatial index for 2D points without object association.

!!! note "Parameter Name Clarification"
    In the inherited methods below, the parameter `geom` refers to a **point** coordinate tuple `(x, y)`.

    For example:

    - `insert(geom)` means `insert((x, y))`
    - `insert_many(geoms)` means `insert_many([(x1, y1), (x2, y2), ...])`

::: fastquadtree.QuadTree
    options:
        inherited_members: true
