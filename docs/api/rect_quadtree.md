# fastquadtree.RectQuadTree

A spatial index for axis-aligned rectangles without object association.

!!! note "Parameter Name Clarification"
    In the inherited methods below, the parameter `geom` refers to a **rectangle** coordinate tuple `(min_x, min_y, max_x, max_y)`.

    For example:

    - `insert(geom)` means `insert((min_x, min_y, max_x, max_y))`
    - `insert_many(geoms)` means `insert_many([(min_x1, min_y1, max_x1, max_y1), ...])`

::: fastquadtree.RectQuadTree
    options:
        inherited_members: true