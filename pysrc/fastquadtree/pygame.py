"""
pygame sprite-group integration backed by a rectangle quadtree.

This module provides ``fastquadtree.pygame.Group``, a replacement for
``pygame.sprite.Group`` that keeps a ``RectQuadTreeObjects`` index of each
sprite's ``rect``. The collision helpers mirror pygame's sprite collision APIs
and use the index as a broadphase when it is safe to do so.

Core ``fastquadtree`` APIs do not require pygame. Importing this integration
module does require pygame or a compatible package such as ``pygame-ce``.

Example:
    ```python
    import fastquadtree.pygame as fpygame

    enemies = fpygame.Group(bounds=(0, 0, 2000, 2000))
    enemies.add(enemy_sprites)

    hits = fpygame.spritecollide(player, enemies, dokill=False)
    first_hit = fpygame.spritecollideany(player, enemies)
    ```
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from typing import Any, cast

try:
    import pygame as _pygame
except ImportError as exc:
    raise ImportError(
        "Pygame is not installed. Install pygame to use the fastquadtree.pygame integration features."
    ) from exc

from ._common import Bounds, QuadTreeDType, validate_bounds
from .rect_quadtree_objects import RectQuadTreeObjects

__all__ = ["Group", "groupcollide", "spritecollide", "spritecollideany"]


_Collided = Callable[[Any, Any], bool]


def _looks_like_bounds(value: Any) -> bool:
    try:
        values = tuple(value)
    except TypeError:
        return False
    return len(values) == 4 and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in values
    )


def _rect_to_bounds(rect: Any) -> Bounds:
    return (rect.left, rect.top, rect.right, rect.bottom)


def _has_rect(sprite: Any) -> bool:
    return hasattr(sprite, "rect")


def _is_rect_like(rect: Any) -> bool:
    return all(hasattr(rect, attr) for attr in ("left", "top", "right", "bottom"))


def _is_indexable_rect(rect: Any) -> bool:
    if not _is_rect_like(rect):
        return False
    return rect.left < rect.right and rect.top < rect.bottom


def _query_to_bounds(rect: Any) -> Bounds | None:
    if _is_indexable_rect(rect):
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


def _pad_bounds(bounds: Bounds) -> Bounds:
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    pad = max(width, height, 1) * 0.1
    return (min_x - pad, min_y - pad, max_x + pad, max_y + pad)


def _bounds_from_rects(rects: Iterable[Bounds]) -> Bounds:
    rect_list = list(rects)
    if not rect_list:
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

    return _pad_bounds((min_x, min_y, max_x, max_y))


def _expanded_bounds(current: Bounds | None, rect: Bounds) -> Bounds:
    if current is None:
        return _bounds_from_rects([rect])

    min_x, min_y, max_x, max_y = current
    left, top, right, bottom = rect
    return _pad_bounds(
        (
            min(min_x, left),
            min(min_y, top),
            max(max_x, right),
            max(max_y, bottom),
        )
    )


class Group(_pygame.sprite.Group):
    """
    pygame sprite group with a ``RectQuadTreeObjects`` broadphase index.

    ``Group`` behaves like ``pygame.sprite.Group`` for normal operations such as
    ``add``, ``remove``, ``empty``, iteration, membership checks, ``update``,
    and ``draw``. In parallel, it indexes sprites with usable ``rect`` bounds so
    the module-level collision helpers can avoid scanning every sprite.

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
        group = fpygame.Group(bounds=(0, 0, 1000, 1000))
        group.add(enemies)
        hits = fpygame.spritecollide(player, group, dokill=False)
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
        if bounds is None and sprites and _looks_like_bounds(sprites[0]):
            bounds = validate_bounds(sprites[0])
            sprites = sprites[1:]

        _pygame.sprite.AbstractGroup.__init__(self)
        self._capacity = capacity
        self._max_depth = max_depth
        self._dtype: QuadTreeDType = dtype
        self._rebuild_on_update = rebuild_on_update
        self._bounds: Bounds | None = validate_bounds(bounds) if bounds else None
        self._tree: RectQuadTreeObjects | None = None
        self._indexed_rects: dict[Any, Bounds] = {}
        self._sprites_without_rect: set[Any] = set()

        if self._bounds is not None:
            self._tree = self._new_tree(self._bounds)
        else:
            initial_rects = [
                _rect_to_bounds(sprite.rect)
                for sprite in _flatten_sprites(sprites)
                if _has_rect(sprite) and _is_indexable_rect(sprite.rect)
            ]
            if initial_rects:
                self._bounds = _bounds_from_rects(initial_rects)
                self._tree = self._new_tree(self._bounds)

        self.add(*sprites)

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

    def add_internal(self, sprite: Any, layer: Any = None) -> None:
        super().add_internal(sprite, layer)
        self.sync(sprite)

    def remove_internal(self, sprite: Any) -> None:
        self._unindex_sprite(sprite)
        super().remove_internal(sprite)

    def empty(self) -> None:
        super().empty()
        self._clear_index()

    def update(self, *args: Any, **kwargs: Any) -> None:
        super().update(*args, **kwargs)
        if self._rebuild_on_update:
            self.rebuild()
        else:
            self.sync()

    def set_bounds(self, bounds: Bounds, rebuild: bool = True) -> None:
        """
        Replace the group's world bounds.

        Args:
            bounds: New world bounds as ``(min_x, min_y, max_x, max_y)``.
            rebuild: When true, rebuild the index immediately from current
                group membership.
        """
        self._bounds = validate_bounds(bounds)
        self._tree = self._new_tree(self._bounds)
        self._clear_index(preserve_tree=True)
        if rebuild:
            self.rebuild()

    def rebuild(self) -> None:
        """Rebuild the quadtree from current group membership and sprite rects."""
        if self._bounds is None:
            rects = [
                _rect_to_bounds(sprite.rect)
                for sprite in self.sprites()
                if _has_rect(sprite) and _is_indexable_rect(sprite.rect)
            ]
            self._bounds = _bounds_from_rects(rects)

        self._tree = self._new_tree(self._bounds)
        self._clear_index(preserve_tree=True)
        for sprite in self.sprites():
            self._index_sprite(sprite)

    def sync(self, sprite: Any | None = None) -> None:
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
                self.sync(group_sprite)
            return

        if not _has_rect(sprite) or not _is_rect_like(sprite.rect):
            self._unindex_sprite(sprite)
            self._sprites_without_rect.add(sprite)
            return

        rect = sprite.rect
        if not _is_indexable_rect(rect):
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
            id_ = self._find_sprite_id(sprite, old_bounds)
            if id_ is None:
                self._indexed_rects.pop(sprite, None)
                self._index_sprite(sprite, rect_bounds=new_bounds)
                return
            self._tree.update(id_, *new_bounds)
            self._indexed_rects[sprite] = new_bounds

    def query_rect(self, rect: Any, *, sync: bool = True) -> list[Any]:
        """
        Return indexed sprites whose rects intersect ``rect``.

        Args:
            rect: Query rectangle. Accepts a pygame rect-like object or a bounds
                tuple ``(min_x, min_y, max_x, max_y)``.
            sync: When true, synchronize the group before querying.

        Returns:
            Sprites whose indexed rects intersect ``rect``. Returns an empty
            list if the query rectangle cannot be interpreted or does not
            overlap the current world bounds. Queries that partially extend
            outside the world bounds are clamped before searching.
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
        rect_bounds = clipped_bounds

        objects: list[Any] = []
        for id_ in self._tree.query_ids(rect_bounds):
            obj = self._tree.get(id_)
            if obj is not None:
                objects.append(obj)
        return objects

    def _clear_index(self, *, preserve_tree: bool = False) -> None:
        self._indexed_rects.clear()
        self._sprites_without_rect.clear()
        if not preserve_tree and self._tree is not None:
            self._tree.clear()

    def _unindex_sprite(self, sprite: Any) -> None:
        old_bounds = self._indexed_rects.pop(sprite, None)
        self._sprites_without_rect.discard(sprite)
        if old_bounds is None or self._tree is None:
            return

        id_ = self._find_sprite_id(sprite, old_bounds)
        if id_ is not None:
            self._tree.delete(id_)

    def _index_sprite(self, sprite: Any, rect_bounds: Bounds | None = None) -> None:
        if not _has_rect(sprite) or not _is_rect_like(sprite.rect):
            self._sprites_without_rect.add(sprite)
            return

        rect = sprite.rect
        if not _is_indexable_rect(rect):
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

    def _find_sprite_id(self, sprite: Any, rect_bounds: Bounds) -> int | None:
        if self._tree is None:
            return None

        for item in self._tree.query(rect_bounds):
            if item.obj is sprite and item.geom == rect_bounds:
                return item.id_
        return None

    def _ensure_tree_contains(self, rect: Bounds) -> None:
        if self._bounds is not None and _contains_rect(self._bounds, rect):
            if self._tree is None:
                self._tree = self._new_tree(self._bounds)
            return

        self._bounds = _expanded_bounds(self._bounds, rect)
        self.rebuild()


def _flatten_sprites(values: Iterable[Any]) -> list[Any]:
    flattened: list[Any] = []
    for value in values:
        if isinstance(value, _pygame.sprite.Sprite):
            flattened.append(value)
            continue

        if hasattr(value, "_spritegroup"):
            flattened.extend(value.sprites())
            continue

        try:
            flattened.extend(_flatten_sprites(value))
        except TypeError:
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
    sprite: Any,
    group: Any,
    dokill: bool,
    collided: _Collided | None = None,
    *,
    sync: bool = True,
) -> list[Any]:
    """
    Find sprites in a group that collide with another sprite.

    This mirrors ``pygame.sprite.spritecollide``. When ``group`` is a
    ``fastquadtree.pygame.Group``, ``collided`` is ``None``, and ``sprite`` has
    a usable ``rect`` attribute, the helper queries the quadtree for
    rect-overlap candidates before applying the final pygame rect collision
    check.

    Plain ``pygame.sprite.Group`` instances, custom ``collided`` callbacks, and
    sprites without usable ``rect`` attributes fall back to pygame's native
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
        List of collided sprites.
    """
    if not isinstance(group, Group):
        return _pygame.sprite.spritecollide(sprite, group, dokill, collided)

    if collided is not None:
        _warn_custom_collided()
        return _pygame.sprite.spritecollide(sprite, group, dokill, collided)

    query_rect = getattr(sprite, "rect", None)
    if query_rect is None:
        return _pygame.sprite.spritecollide(sprite, group, dokill, collided)

    if group._sprites_without_rect:
        return _pygame.sprite.spritecollide(sprite, group, dokill, collided)

    if not _is_rect_like(query_rect):
        return _pygame.sprite.spritecollide(sprite, group, dokill, collided)

    candidates = group.query_rect(query_rect, sync=sync)
    collided_sprites = [
        candidate for candidate in candidates if sprite.rect.colliderect(candidate.rect)
    ]

    if dokill:
        for candidate in collided_sprites:
            candidate.kill()

    return collided_sprites


def spritecollideany(
    sprite: Any,
    group: Any,
    collided: _Collided | None = None,
    *,
    sync: bool = True,
) -> Any | None:
    """
    Return one sprite in a group that collides with another sprite.

    This mirrors ``pygame.sprite.spritecollideany``. When ``group`` is a
    ``fastquadtree.pygame.Group``, ``collided`` is ``None``, and ``sprite`` has
    a usable ``rect`` attribute, indexed groups use the quadtree to narrow
    candidates before checking collisions. Plain pygame groups, custom
    ``collided`` callbacks, and sprites without usable ``rect`` attributes fall
    back to pygame's native behavior.

    Args:
        sprite: Sprite to test against ``group``.
        group: Target sprite group.
        collided: Optional custom collision callback.
        sync: When true, synchronize indexed groups before querying. Set to
            false only if you have already called ``Group.sync(...)`` after the
            latest sprite rect changes.

    Returns:
        The first collided sprite, or ``None``.
    """

    query_rect = getattr(sprite, "rect", None)

    if (
        not isinstance(group, Group)
        or query_rect is None
        or not _is_rect_like(query_rect)
    ):
        return _pygame.sprite.spritecollideany(sprite, group, collided)

    if collided is not None:
        _warn_custom_collided()
        return _pygame.sprite.spritecollideany(sprite, group, collided)

    if group._sprites_without_rect:
        return _pygame.sprite.spritecollideany(sprite, group, collided)

    for candidate in group.query_rect(query_rect, sync=sync):
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
        Dictionary mapping each collided sprite from ``groupa`` to a list of
        collided sprites from ``groupb``.

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

    if groupb._sprites_without_rect:
        return _pygame.sprite.groupcollide(groupa, groupb, dokilla, dokillb, collided)

    if sync:
        groupb.sync()
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
