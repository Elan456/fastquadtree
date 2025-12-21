# fastquadtree.QuadTreeObjects

A spatial index for 2D points with automatic Python object association.

!!! note "Parameter Name Clarification"
    In the inherited methods below, the parameter `geom` refers to a **point** coordinate tuple `(x, y)`.

    For example:

    - `insert(geom, obj)` means `insert((x, y), obj)` - inserts a point with an associated object
    - `insert_many(geoms)` means `insert_many([(x1, y1), (x2, y2), ...])` when using coordinates only

::: fastquadtree.QuadTreeObjects
    options:
        inherited_members: true
