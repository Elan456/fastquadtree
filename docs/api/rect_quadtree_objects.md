# fastquadtree.RectQuadTreeObjects

A spatial index for axis-aligned rectangles with automatic Python object association.

!!! note "Parameter Name Clarification"
    In the inherited methods below, the parameter `geom` refers to a **rectangle** coordinate tuple `(min_x, min_y, max_x, max_y)`.

    For example:

    - `insert(geom, obj)` means `insert((min_x, min_y, max_x, max_y), obj)` - inserts a rectangle with an associated object
    - `insert_many(geoms)` means `insert_many([(min_x1, min_y1, max_x1, max_y1), ...])` when using coordinates only

::: fastquadtree.RectQuadTreeObjects
    options:
        inherited_members: true
