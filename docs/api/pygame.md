# fastquadtree.pygame

`fastquadtree.pygame` is an optional pygame integration for games and
simulations with many sprite `rect`s. It provides:

- `Group`, a mostly drop-in replacement for `pygame.sprite.Group`
- `spritecollide(...)`, `spritecollideany(...)`, and `groupcollide(...)`
  helpers shaped like pygame's collision APIs
- `Group.query_rect(...)` for viewport culling or broadphase rectangle queries
  without creating a temporary sprite

The group keeps a `RectQuadTreeObjects` index of each sprite's `rect`. Collision
helpers use that index as a broadphase, so queries can skip most sprites before
running the final pygame rectangle collision check.

!!! note "Optional dependency"
    Core `fastquadtree` APIs do not require pygame. This page documents the
    optional `fastquadtree.pygame` integration, which requires pygame or a
    compatible package such as `pygame-ce` when imported at runtime.

```bash
pip install fastquadtree pygame-ce
```

## API Reference

::: fastquadtree.pygame
