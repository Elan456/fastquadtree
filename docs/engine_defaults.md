# Engine Defaults

## max_depth {#max-depth}

When `max_depth=None`, the native Python APIs use a dtype-specific engine
default. These defaults apply to `QuadTree`, `RectQuadTree`,
`QuadTreeObjects`, `RectQuadTreeObjects`, and the `fastquadtree.pygame`
integration when `max_depth` is omitted.

| dtype | Engine default `max_depth` |
|---|---:|
| `f32` | 24 |
| `f64` | 53 |
| `i32` | 32 |
| `i64` | 64 |

Use `get_inner_max_depth()` on a constructed tree to inspect the resolved
engine value.

### Why These Values?

Infinite subdivision was the default behavior prior to version 1.4.3. With
infinite subdivision, inserting two points at the same location could keep
splitting forever and eventually exhaust memory, as seen in
[issue #10](https://github.com/Elan456/fastquadtree/issues/10). The engine
default now caps subdivision before it can become unbounded.

One alternative would be for the engine to check for duplicate coordinates
before splitting. However, doing so would add work to each insert and make
splitting behavior harder to define for cells that contain a mixture of
duplicate and nearby non-duplicate points. Letting duplicated points create a
depth of 24 or 53 in some parts of the tree can also hurt traversal time, so the
default is a compromise rather than a perfect answer for every workload.

The default caps are based on a practical starting point for each dtype. Each
quadtree level halves a node's width and height:

```text
cell_width_at_depth_d  = world_width / 2^d
cell_height_at_depth_d = world_height / 2^d
```

The chosen defaults are roughly aligned with the binary precision available for
each coordinate type:

- `f32` uses 24 because single-precision floats have about 24 bits of
  significand precision.
- `f64` uses 53 because double-precision floats have about 53 bits of
  significand precision.
- `i32` uses 32 and `i64` uses 64 because integer coordinates cannot provide
  more distinct binary subdivision steps than their bit width.

These values are safety defaults, not hard limits on useful subdivision. World
size and data distribution still matter. For example, with `f32` and a world
width of `1_000_000_000`, depth 24 still leaves cells about 60 units wide:

```text
1_000_000_000 / 2^24 ~= 59.6
```

If most of your points are clustered near the origin and you need to separate
features smaller than that, more than 24 splits can still be meaningful. In that
case, pass an explicit `max_depth` larger than the engine default.

For `i32` and `i64`, there is generally no situation where you would need more
than 32 and 64 subdivisions, respectively.

The defaults are intended to be a relatively safe starting point. Use tighter
world bounds, coordinate normalization, or an explicit `max_depth` when your
world is very large compared with the area where points actually cluster.

For best performance, choose a `max_depth` that keeps leaf cells near the scale
of your typical queries and data spacing.
