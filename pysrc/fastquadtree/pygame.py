"""
pygame sprite-group integration backed by a rectangle quadtree.

This module provides pygame-facing spatial-index utilities for accelerating
sprite collision checks: ``Group``, ``spritecollide(...)``,
``spritecollideany(...)``, and ``groupcollide(...)``.

When the accelerated broadphase is used, collision result order follows the
quadtree query traversal and is not guaranteed to match pygame group iteration
order.

Core ``fastquadtree`` APIs do not require pygame. Importing this integration
module does require pygame or a compatible package such as ``pygame-ce``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable
from typing import Any, cast

try:
    import pygame as _pygame
except ImportError as exc:
    raise ImportError(
        "Pygame is not installed. Install pygame to use the fastquadtree.pygame integration features."
    ) from exc

from ._common import Bounds, Point, QuadTreeDType, validate_bounds
from .rect_quadtree_objects import RectQuadTreeObjects

__all__ = ["Group", "groupcollide", "spritecollide", "spritecollideany"]


_Collided = Callable[[Any, Any], bool]
_ATOMIC_ITERABLES = (str, bytes, bytearray, memoryview)


def _looks_like_bounds(value: Any) -> bool:
    if not isinstance(value, (list, tuple)):
        return False
    return len(value) == 4 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in value
    )


def _rect_to_bounds(rect: _pygame.Rect) -> Bounds:
    return (rect.left, rect.top, rect.right, rect.bottom)


def _is_indexable_rect(rect: _pygame.Rect) -> bool:
    return rect.left < rect.right and rect.top < rect.bottom


def _sprite_rect(sprite: Any) -> _pygame.Rect | None:
    rect = getattr(sprite, "rect", None)
    return rect if isinstance(rect, _pygame.Rect) else None


def _sprite_bounds(sprite: Any) -> Bounds | None:
    rect = _sprite_rect(sprite)
    if rect is None or not _is_indexable_rect(rect):
        return None
    return _rect_to_bounds(rect)


def _positional_bounds(value: Any) -> Bounds | None:
    if isinstance(value, _pygame.Rect):
        return validate_bounds(_rect_to_bounds(value))
    if _looks_like_bounds(value):
        return validate_bounds(value)
    return None


def _query_to_bounds(rect: Any) -> Bounds | None:
    if isinstance(rect, _pygame.Rect) and _is_indexable_rect(rect):
        return _rect_to_bounds(rect)
    if _looks_like_bounds(rect):
        try:
            return validate_bounds(rect)
        except ValueError:
            return None
    return None


def _contains_rect(bounds: Bounds, rect: Bounds) -> bool:
    min_x, min_y, max_x, max_y = bounds
    left, top, right, bottom = rect
    return min_x <= left and min_y <= top and right <= max_x and bottom <= max_y


def _intersect_bounds(left: Bounds, right: Bounds) -> Bounds | None:
    min_x = max(left[0], right[0])
    min_y = max(left[1], right[1])
    max_x = min(left[2], right[2])
    max_y = min(left[3], right[3])
    if min_x >= max_x or min_y >= max_y:
        return None
    return (min_x, min_y, max_x, max_y)


def _is_integer_dtype(dtype: QuadTreeDType) -> bool:
    return dtype.startswith("i")


def _pad_bounds(bounds: Bounds, dtype: QuadTreeDType = "f32") -> Bounds:
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    if _is_integer_dtype(dtype):
        pad = max(1, math.ceil(max(width, height, 1) * 0.1))
    else:
        pad = max(width, height, 1) * 0.1
    return (min_x - pad, min_y - pad, max_x + pad, max_y + pad)


def _bounds_from_rects(rects: Iterable[Bounds], dtype: QuadTreeDType = "f32") -> Bounds:
    rect_list = list(rects)
    if not rect_list:
        if _is_integer_dtype(dtype):
            return (-1, -1, 1, 1)
        return (-1.0, -1.0, 1.0, 1.0)

    min_x = min(rect[0] for rect in rect_list)
    min_y = min(rect[1] for rect in rect_list)
    max_x = max(rect[2] for rect in rect_list)
    max_y = max(rect[3] for rect in rect_list)

    if min_x == max_x:
        min_x -= 1
        max_x += 1
    if min_y == max_y:
        min_y -= 1
        max_y += 1

    return _pad_bounds((min_x, min_y, max_x, max_y), dtype=dtype)


def _expanded_bounds(
    current: Bounds | None, rect: Bounds, dtype: QuadTreeDType = "f32"
) -> Bounds:
    if current is None:
        return _bounds_from_rects([rect], dtype=dtype)

    min_x, min_y, max_x, max_y = current
    left, top, right, bottom = rect
    return _pad_bounds(
        (
            min(min_x, left),
            min(min_y, top),
            max(max_x, right),
            max(max_y, bottom),
        ),
        dtype=dtype,
    )


class Group(_pygame.sprite.Group):
    """
    pygame sprite group with a ``RectQuadTreeObjects`` spatial index.

    This is a sprite-oriented ``RectQuadTreeObjects`` integration: indexed
    entries use each sprite's ``rect`` as geometry and the sprite itself as the
    associated object. The index is primarily a collision broadphase: collision
    helpers first query the quadtree for nearby rects, then run pygame's normal
    rect collision checks only on those candidates instead of scanning every
    sprite in the group.

    ``Group`` behaves like ``pygame.sprite.Group`` for normal operations such as
    ``add``, ``remove``, ``empty``, iteration, membership checks, ``update``,
    and ``draw``. In parallel, it indexes sprites with usable ``rect`` bounds to
    accelerate collision-heavy scenes where each query touches only a small part
    of the world. The same index can also be queried directly with methods such
    as ``query(...)`` and ``nearest_neighbors(...)``.

    For stable runtime behavior, pass explicit world bounds. If ``bounds`` is
    omitted, the group infers bounds from initial sprites and expands/rebuilds
    when later sprites fall outside the current index.

    Args:
        *sprites: Initial sprites, pygame groups, or sprite iterables to add.
        bounds: Optional world bounds as ``(min_x, min_y, max_x, max_y)``.
        capacity: Maximum indexed rectangles per quadtree node before splitting.
        max_depth: Optional maximum quadtree depth.
        dtype: Coordinate data type (``"f32"``, ``"f64"``, ``"i32"``,
            ``"i64"``). Default: ``"f32"``.
        rebuild_on_update: When true, ``Group.update(...)`` rebuilds the whole
            index after updating sprites instead of incrementally syncing each
            sprite. This is useful when most indexed sprites move every frame.

    Example:
        ```python
        import fastquadtree.pygame as fpygame

        enemies = fpygame.Group(bounds=(0, 0, 2000, 2000))
        enemies.add(enemy_sprites)
        enemies.add(boss)

        hits = fpygame.spritecollide(player, enemies, dokill=False)
        first_hit = fpygame.spritecollideany(player, enemies)

        visible = enemies.query_rect(camera_rect)
        nearest = enemies.nearest_neighbors(player.rect.center, k=5)
        ```
    """

    def __init__(
        self,
        *sprites: Any,
        bounds: Bounds | None = None,
        capacity: int = 16,
        max_depth: int | None = None,
        dtype: QuadTreeDType = "f32",
        rebuild_on_update: bool = False,
    ):
        if isinstance(bounds, _pygame.Rect):
            bounds = _rect_to_bounds(bounds)

        if bounds is None and sprites:
            bounds = _positional_bounds(sprites[0])
            if bounds is not None:
                sprites = sprites[1:]

        initial_sprites = _flatten_sprites(sprites)

        _pygame.sprite.AbstractGroup.__init__(self)
        self._capacity = capacity
        self._max_depth = max_depth
        self._dtype: QuadTreeDType = dtype
        self._rebuild_on_update = rebuild_on_update
        self._bounds: Bounds | None = validate_bounds(bounds) if bounds else None
        self._tree: RectQuadTreeObjects | None = None
        self._indexed_rects: dict[Any, Bounds] = {}
        self._sprites_without_rect: set[Any] = set()
        self._defer_index_sync = False

        if self._bounds is not None:
            self._tree = self._new_tree(self._bounds)
        else:
            initial_rects = [
                rect_bounds
                for sprite in initial_sprites
                if (rect_bounds := _sprite_bounds(sprite)) is not None
            ]
            if initial_rects:
                self._bounds = _bounds_from_rects(initial_rects, dtype=self._dtype)
                self._tree = self._new_tree(self._bounds)

        self.add(initial_sprites)

    @property
    def bounds(self) -> Bounds | None:
        """Current quadtree bounds, or ``None`` before an index is built."""
        return self._bounds

    @property
    def indexed_count(self) -> int:
        """Number of sprites currently indexed by usable rect bounds."""
        return len(self._indexed_rects)

    def copy(self) -> Group:
        """Return a new indexed group with the same sprites and index settings."""
        return self.__class__(
            self.sprites(),
            bounds=self._bounds,
            capacity=self._capacity,
            max_depth=self._max_depth,
            dtype=self._dtype,
            rebuild_on_update=self._rebuild_on_update,
        )

    def _new_tree(self, bounds: Bounds) -> RectQuadTreeObjects:
        return RectQuadTreeObjects(
            bounds,
            capacity=self._capacity,
            max_depth=self._max_depth,
            dtype=self._dtype,
        )

    def add(self, *sprites: Any) -> None:
        """Add sprites and synchronize the quadtree index once for the batch."""
        flattened = _flatten_sprites(sprites)
        self._defer_index_sync = True
        try:
            for sprite in flattened:
                if not self.has_internal(sprite):
                    self.add_internal(sprite)
                    sprite.add_internal(self)
        finally:
            self._defer_index_sync = False
        self._sync_many(flattened)

    def add_internal(self, sprite: _pygame.sprite.Sprite, layer: Any = None) -> None:
        """Add one sprite through pygame's internal group hook and sync the index."""
        super().add_internal(sprite, layer)
        if self._defer_index_sync:
            return
        self.sync(sprite)

    def remove_internal(self, sprite: _pygame.sprite.Sprite) -> None:
        """Remove one sprite through pygame's internal group hook and unindex it."""
        self._unindex_sprite(sprite)
        super().remove_internal(sprite)

    def empty(self) -> None:
        """Remove all sprites and clear the quadtree index."""
        super().empty()
        self._clear_index()

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Run sprite updates, then synchronize or rebuild the quadtree index."""
        super().update(*args, **kwargs)
        if self._rebuild_on_update:
            self.rebuild()
        else:
            self.sync()

    def set_bounds(self, bounds: Bounds | _pygame.Rect, rebuild: bool = True) -> None:
        """
        Replace the group's world bounds.

        Args:
            bounds: New world bounds as ``(min_x, min_y, max_x, max_y)`` or a
                ``pygame.Rect``.
            rebuild: When true, rebuild the index immediately from current
                group membership.
        """
        if isinstance(bounds, _pygame.Rect):
            bounds = _rect_to_bounds(bounds)
        self._bounds = validate_bounds(bounds)
        self._tree = self._new_tree(self._bounds)
        self._clear_index(preserve_tree=True)
        if rebuild:
            self.rebuild()

    def rebuild(self) -> None:
        """Rebuild the quadtree from current group membership and sprite rects."""
        if self._bounds is None:
            rects = [
                rect_bounds
                for sprite in self.sprites()
                if (rect_bounds := _sprite_bounds(sprite)) is not None
            ]
            self._bounds = _bounds_from_rects(rects, dtype=self._dtype)

        self._tree = self._new_tree(self._bounds)
        self._clear_index(preserve_tree=True)
        for sprite in self.sprites():
            self._index_sprite(sprite)

    def sync(self, sprite: _pygame.sprite.Sprite | None = None) -> None:
        """
        Synchronize the quadtree with sprite rect changes.

        Pass one sprite to sync only that sprite, or omit ``sprite`` to sync the
        full group. ``Group.update(...)`` automatically calls ``sync()`` after
        pygame updates the sprites.

        Sprites without a usable ``rect`` remain in the pygame group but are not
        indexed. Collision helpers fall back to pygame-compatible behavior when
        needed to preserve pygame semantics.

        Args:
            sprite: Optional sprite to sync. If omitted, all group sprites are
                synced.
        """
        if sprite is None:
            for group_sprite in self.sprites():
                self._sync_sprite(group_sprite)
            return

        if not self.has_internal(sprite):
            return

        self._sync_sprite(sprite)

    def _sync_sprite(self, sprite: _pygame.sprite.Sprite) -> None:
        rect = _sprite_rect(sprite)
        if rect is None or not _is_indexable_rect(rect):
            self._unindex_sprite(sprite)
            self._sprites_without_rect.add(sprite)
            return

        self._sprites_without_rect.discard(sprite)
        new_bounds = _rect_to_bounds(rect)
        old_bounds = self._indexed_rects.get(sprite)
        if old_bounds == new_bounds:
            return

        self._ensure_tree_contains(new_bounds)
        if self._indexed_rects.get(sprite) == new_bounds:
            return

        old_bounds = self._indexed_rects.get(sprite)
        if old_bounds is None:
            self._index_sprite(sprite, rect_bounds=new_bounds)
        else:
            assert self._tree is not None
            if not self._tree.update_by_object(sprite, *new_bounds):
                self._indexed_rects.pop(sprite, None)
                self._index_sprite(sprite, rect_bounds=new_bounds)
                return
            self._indexed_rects[sprite] = new_bounds

    def _sync_many(self, sprites: Iterable[_pygame.sprite.Sprite]) -> None:
        pending: list[tuple[_pygame.sprite.Sprite, Bounds | None, Bounds]] = []

        for sprite in sprites:
            rect_bounds = _sprite_bounds(sprite)
            if rect_bounds is None:
                self._unindex_sprite(sprite)
                self._sprites_without_rect.add(sprite)
                continue

            self._sprites_without_rect.discard(sprite)
            old_bounds = self._indexed_rects.get(sprite)
            if old_bounds != rect_bounds:
                pending.append((sprite, old_bounds, rect_bounds))

        if not pending:
            return

        if self._bounds is None or self._tree is None:
            self._bounds = _bounds_from_rects(
                (rect_bounds for _, _, rect_bounds in pending), dtype=self._dtype
            )
            self.rebuild()
            return

        new_bounds = self._bounds
        for _, _, rect_bounds in pending:
            if not _contains_rect(new_bounds, rect_bounds):
                new_bounds = _expanded_bounds(
                    new_bounds, rect_bounds, dtype=self._dtype
                )

        if new_bounds != self._bounds:
            self._bounds = new_bounds
            self.rebuild()
            return

        for sprite, old_bounds, rect_bounds in pending:
            if old_bounds is None:
                self._index_sprite(sprite, rect_bounds=rect_bounds)
            else:
                assert self._tree is not None
                if not self._tree.update_by_object(sprite, *rect_bounds):
                    self._indexed_rects.pop(sprite, None)
                    self._index_sprite(sprite, rect_bounds=rect_bounds)
                else:
                    self._indexed_rects[sprite] = rect_bounds

    def query(
        self, rect: _pygame.Rect | Bounds, *, sync: bool = True
    ) -> list[_pygame.sprite.Sprite]:
        """
        Return indexed sprites whose rects intersect ``rect``.

        This is a direct spatial query over the group's internal quadtree.
        It returns pygame sprites, not quadtree IDs or item wrappers.

        Args:
            rect: Query rectangle. Accepts a ``pygame.Rect`` or bounds tuple
                ``(min_x, min_y, max_x, max_y)``.
            sync: When true, synchronize the group before querying.

        Returns:
            list[pygame.sprite.Sprite]: Sprites whose indexed rects intersect
                ``rect``. Returns an empty list if the query rectangle cannot
                be interpreted or does not overlap the current world bounds.
        """
        if sync:
            self.sync()

        rect_bounds = _query_to_bounds(rect)
        if rect_bounds is None:
            return []

        if self._tree is None:
            return []

        clipped_bounds = _intersect_bounds(cast(Bounds, self._bounds), rect_bounds)
        if clipped_bounds is None:
            return []

        return [
            cast(_pygame.sprite.Sprite, item.obj)
            for item in self._tree.query(clipped_bounds)
            if item.obj is not None and self.has_internal(item.obj)
        ]

    def query_rect(
        self, rect: _pygame.Rect | Bounds, *, sync: bool = True
    ) -> list[_pygame.sprite.Sprite]:
        """
        Return indexed sprites whose rects intersect ``rect``.

        This method is kept for backward compatibility with fastquadtree 2.3.0.
        For new code, prefer ``query(...)``.

        Args:
            rect: Query rectangle. Accepts a ``pygame.Rect`` or a bounds tuple
                ``(min_x, min_y, max_x, max_y)``.
            sync: When true, synchronize the group before querying.

        Returns:
            list[pygame.sprite.Sprite]: Sprites whose indexed rects intersect
                ``rect``. Returns an empty list if the query rectangle cannot
                be interpreted or does not overlap the current world bounds.
                Queries that partially extend outside the world bounds are
                clamped before searching.
        """
        return self.query(rect, sync=sync)

    def nearest_neighbor(
        self, point: Point, *, sync: bool = True
    ) -> _pygame.sprite.Sprite | None:
        """
        Return the sprite with the nearest indexed rect to ``point``.

        Distance is measured like ``RectQuadTreeObjects``: Euclidean distance
        from the point to the nearest edge of each rectangle, or zero when the
        point is inside a rectangle.

        Args:
            point: Query point as ``(x, y)``.
            sync: When true, synchronize the group before querying.

        Returns:
            pygame.sprite.Sprite | None: Nearest indexed sprite, or ``None`` if
                no sprites are indexed.
        """
        neighbors = self.nearest_neighbors(point, 1, sync=sync)
        return neighbors[0] if neighbors else None

    def nearest_neighbors(
        self, point: Point, k: int, *, sync: bool = True
    ) -> list[_pygame.sprite.Sprite]:
        """
        Return sprites with the ``k`` nearest indexed rects to ``point``.

        Args:
            point: Query point as ``(x, y)``.
            k: Number of neighbors to return.
            sync: When true, synchronize the group before querying.

        Returns:
            list[pygame.sprite.Sprite]: Sprites in increasing distance order.
        """
        if sync:
            self.sync()

        if self._tree is None:
            return []
        if k <= 0:
            return []
        if not self._indexed_rects:
            return []

        return [
            cast(_pygame.sprite.Sprite, item.obj)
            for item in self._tree.nearest_neighbors(point, k)
            if item.obj is not None and self.has_internal(item.obj)
        ]

    def _clear_index(self, *, preserve_tree: bool = False) -> None:
        self._indexed_rects.clear()
        self._sprites_without_rect.clear()
        if not preserve_tree and self._tree is not None:
            self._tree.clear()

    def _unindex_sprite(self, sprite: _pygame.sprite.Sprite) -> None:
        old_bounds = self._indexed_rects.pop(sprite, None)
        self._sprites_without_rect.discard(sprite)
        if old_bounds is None or self._tree is None:
            return

        self._tree.delete_one_by_object(sprite)

    def _index_sprite(
        self, sprite: _pygame.sprite.Sprite, rect_bounds: Bounds | None = None
    ) -> None:
        rect = _sprite_rect(sprite)
        if rect is None or not _is_indexable_rect(rect):
            self._sprites_without_rect.add(sprite)
            return

        rect_bounds = rect_bounds or _rect_to_bounds(rect)
        self._ensure_tree_contains(rect_bounds)
        if sprite in self._indexed_rects:
            return

        assert self._tree is not None
        self._tree.insert(rect_bounds, obj=sprite)
        self._indexed_rects[sprite] = rect_bounds
        self._sprites_without_rect.discard(sprite)

    def _ensure_tree_contains(self, rect: Bounds) -> None:
        if self._bounds is not None and _contains_rect(self._bounds, rect):
            if self._tree is None:
                self._tree = self._new_tree(self._bounds)
            return

        self._bounds = _expanded_bounds(self._bounds, rect, dtype=self._dtype)
        self.rebuild()


def _flatten_sprites(values: Iterable[Any]) -> list[Any]:
    flattened: list[Any] = []
    for value in values:
        if isinstance(value, _pygame.sprite.Sprite):
            flattened.append(value)
            continue

        if isinstance(value, _pygame.sprite.AbstractGroup):
            flattened.extend(value.sprites())
            continue

        if isinstance(value, _ATOMIC_ITERABLES):
            flattened.append(value)
            continue

        try:
            flattened.extend(_flatten_sprites(value))
        except (TypeError, AttributeError):
            if hasattr(value, "_spritegroup"):
                flattened.extend(value.sprites())
            else:
                flattened.append(value)
    return flattened


def _warn_custom_collided() -> None:
    warnings.warn(
        "fastquadtree.pygame does not accelerate custom collided callbacks; "
        "falling back to pygame-compatible full scans. The quadtree broadphase "
        "is used only when collided is None and the query sprite has a usable "
        "rect attribute.",
        RuntimeWarning,
        stacklevel=3,
    )


def spritecollide(
    sprite: _pygame.sprite.Sprite,
    group: Any,
    dokill: bool,
    collided: _Collided | None = None,
    *,
    sync: bool = True,
) -> list[Any]:
    """
    Find sprites in a group that collide with another sprite.

    This mirrors ``pygame.sprite.spritecollide``. When ``group`` is a
    ``fastquadtree.pygame.Group``, ``collided`` is ``None``, and ``sprite.rect``
    is a usable ``pygame.Rect``, the helper queries the quadtree for
    rect-overlap candidates before applying the final pygame rect collision
    check.

    Plain ``pygame.sprite.Group`` instances, custom ``collided`` callbacks, and
    sprites without usable ``pygame.Rect`` attributes fall back to pygame's native
    implementation.

    Args:
        sprite: Sprite to test against ``group``.
        group: Target sprite group.
        dokill: If true, kill each collided target sprite.
        collided: Optional custom collision callback.
        sync: When true, synchronize indexed groups before querying. Set to
            false only if you have already called ``Group.sync(...)`` after the
            latest sprite rect changes.

    Returns:
        list[Any]: Collided sprites. When the quadtree broadphase is used,
            result order is not deterministic and may differ from pygame group
            iteration order.
    """
    if not isinstance(group, Group):
        return _pygame.sprite.spritecollide(sprite, group, dokill, collided)

    if collided is not None:
        _warn_custom_collided()
        return _pygame.sprite.spritecollide(sprite, group, dokill, collided)

    query_rect = _sprite_rect(sprite)
    if query_rect is None:
        return _pygame.sprite.spritecollide(sprite, group, dokill, collided)

    if sync:
        group.sync()

    if group._sprites_without_rect:
        return _pygame.sprite.spritecollide(sprite, group, dokill, collided)

    candidates = group.query_rect(query_rect, sync=False)
    collided_sprites = [
        candidate for candidate in candidates if sprite.rect.colliderect(candidate.rect)
    ]

    if dokill:
        for candidate in collided_sprites:
            candidate.kill()

    return collided_sprites


def spritecollideany(
    sprite: _pygame.sprite.Sprite,
    group: Any,
    collided: _Collided | None = None,
    *,
    sync: bool = True,
) -> Any | None:
    """
    Return one sprite in a group that collides with another sprite.

    This mirrors ``pygame.sprite.spritecollideany``. When ``group`` is a
    ``fastquadtree.pygame.Group``, ``collided`` is ``None``, and ``sprite.rect``
    is a usable ``pygame.Rect``, indexed groups use the quadtree to narrow
    candidates before checking collisions. Plain pygame groups, custom
    ``collided`` callbacks, and sprites without usable ``pygame.Rect``
    attributes fall back to pygame's native behavior.

    Args:
        sprite: Sprite to test against ``group``.
        group: Target sprite group.
        collided: Optional custom collision callback.
        sync: When true, synchronize indexed groups before querying. Set to
            false only if you have already called ``Group.sync(...)`` after the
            latest sprite rect changes.

    Returns:
        Any | None: The first collided sprite, or ``None``. When the quadtree
            broadphase is used, "first" means first in quadtree query order,
            not pygame group iteration order.
    """

    if not isinstance(group, Group):
        return _pygame.sprite.spritecollideany(sprite, group, collided)

    if collided is not None:
        _warn_custom_collided()
        return _pygame.sprite.spritecollideany(sprite, group, collided)

    query_rect = _sprite_rect(sprite)

    if sync:
        group.sync()

    if query_rect is None or group._sprites_without_rect:
        return _pygame.sprite.spritecollideany(sprite, group, collided)

    for candidate in group.query_rect(query_rect, sync=False):
        if sprite.rect.colliderect(candidate.rect):
            return candidate
    return None


def groupcollide(
    groupa: Any,
    groupb: Any,
    dokilla: bool,
    dokillb: bool,
    collided: _Collided | None = None,
    *,
    sync: bool = True,
) -> dict[Any, list[Any]]:
    """
    Detect collisions between two sprite groups.

    This mirrors ``pygame.sprite.groupcollide``. The indexed broadphase is used
    when ``groupb`` is a ``fastquadtree.pygame.Group``, ``collided`` is
    ``None``, and each queried sprite has a usable ``rect`` attribute.

    Plain pygame groups, custom ``collided`` callbacks, sprites without usable
    ``rect`` attributes, and target groups containing unindexed sprites fall
    back to pygame's native implementation.

    Args:
        groupa: Source sprite group.
        groupb: Target sprite group.
        dokilla: If true, kill collided sprites from ``groupa``.
        dokillb: If true, kill collided sprites from ``groupb``.
        collided: Optional custom collision callback.
        sync: When true, synchronize indexed groups before querying. Set to
            false only if you have already called ``Group.sync(...)`` after the
            latest sprite rect changes.

    Returns:
        dict[Any, list[Any]]: Mapping from each collided sprite in ``groupa`` to
            its collided sprites from ``groupb``. When the quadtree broadphase
            is used, each list follows quadtree query order, not pygame group
            iteration order.

    Example:
        ```python
        hits = fpygame.groupcollide(players, enemies, False, False)
        ```
    """
    if not isinstance(groupb, Group):
        return _pygame.sprite.groupcollide(groupa, groupb, dokilla, dokillb, collided)

    if collided is not None:
        _warn_custom_collided()
        return _pygame.sprite.groupcollide(groupa, groupb, dokilla, dokillb, collided)

    if sync:
        groupb.sync()

    if groupb._sprites_without_rect:
        return _pygame.sprite.groupcollide(groupa, groupb, dokilla, dokillb, collided)
    collided_sprites: dict[Any, list[Any]] = {}

    for group_a_sprite in groupa:
        collisions = spritecollide(
            group_a_sprite, groupb, dokillb, collided, sync=False
        )
        if collisions:
            collided_sprites[group_a_sprite] = collisions

    if dokilla:
        for group_a_sprite in collided_sprites:
            group_a_sprite.kill()

    return collided_sprites
