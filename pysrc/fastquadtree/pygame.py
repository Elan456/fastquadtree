"""
pygame integration plan.

Goal
----
Provide ``fastquadtree.pygame.Group`` as a drop-in replacement for
``pygame.sprite.Group``. Normal sprite-group behavior should stay compatible
with pygame, while collision helpers can use a quadtree broadphase to avoid
checking every sprite against every other sprite.

Primary user path
-----------------
Existing pygame code should be able to switch from:

    group = pygame.sprite.Group()
    hits = pygame.sprite.spritecollide(player, group, dokill=False)

to:

    group = fastquadtree.pygame.Group(bounds)
    hits = fastquadtree.pygame.spritecollide(player, group, dokill=False)

The replacement group should still support the usual ``add``, ``remove``,
``empty``, ``sprites``, iteration, membership, ``update``, and ``draw`` flows.
When callers pass a plain ``pygame.sprite.Group`` to the module-level
collision functions, fall back to pygame's implementation.

Public API to provide
---------------------
- ``Group(*sprites, bounds=None, capacity=16, max_depth=None, dtype="f32")``
  subclasses ``pygame.sprite.Group`` and owns a parallel
  ``RectQuadTreeObjects`` index keyed by each sprite's ``rect``.
- ``spritecollide(sprite, group, dokill, collided=None)`` mirrors
  ``pygame.sprite.spritecollide``. With ``collided=None`` it queries only
  sprites whose rects overlap the query sprite's rect before running the final
  rect check. With arbitrary custom ``collided`` callbacks, preserve pygame
  semantics by default instead of silently applying an unsafe broadphase.
- ``spritecollideany(sprite, group, collided=None)`` mirrors
  ``pygame.sprite.spritecollideany`` and returns after the first confirmed hit.
- ``groupcollide(groupa, groupb, dokilla, dokillb, collided=None)`` mirrors
  ``pygame.sprite.groupcollide`` and uses the indexed side as the broadphase.
- Optional methods on ``Group``:
  ``rebuild()``, ``sync(sprite=None)``, ``query_rect(rect)``,
  ``set_bounds(bounds, rebuild=True)``, and read-only diagnostics such as
  ``indexed_count`` or ``bounds``.

Indexing model
--------------
- Store each indexed sprite by object identity in ``RectQuadTreeObjects`` using
  ``(rect.left, rect.top, rect.right, rect.bottom)``.
- Maintain ``_indexed_rects[sprite]`` with each sprite's last indexed rect.
  When deleting or updating, query that exact rect, find the item whose object
  is the same sprite by identity, and then call ID-based ``delete`` or
  ``update``. Do not use ``delete_by_object`` so a duplicate-object bug cannot
  remove multiple entries accidentally.
- Override group mutation hooks used by pygame's ``Group`` implementation
  (``add_internal`` and ``remove_internal``) so membership and quadtree state
  stay aligned for all standard add/remove paths.
- ``update(*args, **kwargs)`` should call the pygame implementation, then sync
  all indexed rects because pygame sprites commonly mutate ``rect`` during
  their own update methods.
- Sprites without a usable ``rect`` should remain in the pygame group but be
  skipped by the quadtree. Collision helpers must still handle them through a
  compatibility fallback when exact pygame semantics require it.

Bounds strategy
---------------
- Prefer explicit world bounds from the user. They make the quadtree stable and
  avoid surprise rebuilds during gameplay.
- If ``bounds`` is omitted, infer bounds from the initial sprites and grow the
  index when a later sprite falls outside the current bounds.
- Growing should rebuild the tree with padded bounds instead of failing during
  insertion. Keep this path correct first; later benchmarks can tune the growth
  factor.

Collision semantics
-------------------
- Default ``collided=None`` must preserve pygame behavior, meaning rect
  collision is the final narrowphase.
- Custom ``collided(sprite, candidate)`` callbacks need special handling.
  Pygame calls custom callbacks against every sprite in the target group. A
  quadtree broadphase based on rect overlap, distance, or any other envelope can
  produce false negatives for an arbitrary callback that reports collisions
  outside that envelope. Therefore the default behavior for custom callbacks
  should be:
  1. fall back to pygame-compatible full scans, and
  2. emit a ``RuntimeWarning`` when the caller is using ``Group`` and expected
     acceleration but supplied an opaque custom callback.
- Provide an opt-in fast path for custom callbacks only when the broadphase is
  explicit. Options to evaluate:
  ``collided.fastquadtree_bounds(sprite) -> rect``, a separate
  ``query_rect=``/``broadphase_rect=`` argument on our helpers, or documented
  helper constructors for common pygame callbacks such as circle and mask
  collision. In those cases the quadtree limits the candidate set, then the
  custom callback still decides the final hit result.
- Include ``sprite`` itself in ``spritecollide`` results when the query sprite
  is also present in the target group and its rect collides, matching pygame's
  behavior.
- Apply ``dokill``/``dokilla``/``dokillb`` by calling ``kill`` or removing from
  the relevant groups in a way that also updates the quadtree.
- Preserve result shapes exactly:
  ``spritecollide`` returns a list, ``spritecollideany`` returns one sprite or
  ``None``, and ``groupcollide`` returns ``{sprite: [colliding_sprites]}``.

Implementation phases
---------------------
1. Add a thin ``Group`` subclass that behaves exactly like
   ``pygame.sprite.Group`` and keeps an empty/private quadtree in sync with
   add/remove/update.
2. Implement rect conversion, bounds validation/growth, full rebuild, and
   per-sprite sync using ID-based delete/update after resolving the ID from the
   last indexed rect. Add focused tests that compare membership and query
   results against brute-force rect checks, including a guard that removing one
   sprite only deletes that sprite's indexed entry.
3. Implement ``spritecollide`` and ``spritecollideany`` with pygame fallbacks
   for non-indexed groups and for opaque custom ``collided`` callbacks. Test
   against pygame for default and custom callbacks, including self-collision,
   warning behavior, and ``dokill``.
4. Implement ``groupcollide`` using the smaller or indexed side as the query
   driver where possible. Preserve dictionary shape and kill behavior.
5. Add benchmarks and the interactive sprite demo path showing pygame's native
   collision helper versus the fastquadtree-backed helper at high sprite counts.
6. Document limits: rect-based broadphase only, users should provide stable
   world bounds for best performance, and moving sprites require ``update`` or
   explicit ``sync`` before collision queries.

Testing checklist
-----------------
- Compare all public collision helpers with pygame's outputs for small random
  scenes.
- Verify add/remove/empty/kill keep the quadtree and group membership aligned.
- Verify moved and resized rects are reindexed after ``update`` or ``sync``.
- Verify out-of-bounds sprites trigger growth/rebuild or documented fallback.
- Verify plain pygame groups passed to these helpers still behave correctly.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from typing import Any

import pygame as _pygame

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
    try:
        rect.left
        rect.top
        rect.right
        rect.bottom
    except AttributeError:
        return False
    return True


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
    """A pygame sprite group with a RectQuadTreeObjects broadphase index."""

    def __init__(
        self,
        *sprites: Any,
        bounds: Bounds | None = None,
        capacity: int = 16,
        max_depth: int | None = None,
        dtype: QuadTreeDType = "f32",
    ):
        if bounds is None and sprites and _looks_like_bounds(sprites[0]):
            bounds = validate_bounds(sprites[0])
            sprites = sprites[1:]

        _pygame.sprite.AbstractGroup.__init__(self)
        self._capacity = capacity
        self._max_depth = max_depth
        self._dtype = dtype
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
        return self._bounds

    @property
    def indexed_count(self) -> int:
        return len(self._indexed_rects)

    def copy(self) -> Group:
        return self.__class__(
            self.sprites(),
            bounds=self._bounds,
            capacity=self._capacity,
            max_depth=self._max_depth,
            dtype=self._dtype,
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
        self.sync()

    def set_bounds(self, bounds: Bounds, rebuild: bool = True) -> None:
        self._bounds = validate_bounds(bounds)
        self._tree = self._new_tree(self._bounds)
        self._clear_index(preserve_tree=True)
        if rebuild:
            self.rebuild()

    def rebuild(self) -> None:
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
                self._index_sprite(sprite, rect_bounds=new_bounds)
                return
            self._tree.update(id_, *new_bounds)
            self._indexed_rects[sprite] = new_bounds

    def query_rect(self, rect: Any, *, sync: bool = True) -> list[Any]:
        if sync:
            self.sync()

        rect_bounds = _query_to_bounds(rect)
        if rect_bounds is None:
            return []

        if self._tree is None:
            return []
        if self._bounds is not None and not _contains_rect(self._bounds, rect_bounds):
            # Queries outside the tree bounds may still intersect indexed sprites
            # near the edge. Grow/rebuild so the native query can represent it.
            self._ensure_tree_contains(rect_bounds)

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
        "fastquadtree.pygame cannot safely accelerate an opaque custom "
        "collided callback; falling back to pygame-compatible full scans. "
        "Attach a fastquadtree_bounds(sprite) callable to the callback to opt "
        "into quadtree broadphase filtering.",
        RuntimeWarning,
        stacklevel=3,
    )


def _custom_query_bounds(collided: _Collided | None, sprite: Any) -> Any | None:
    if collided is None:
        return getattr(sprite, "rect", None)

    bounds_getter = getattr(collided, "fastquadtree_bounds", None)
    if not callable(bounds_getter):
        return None
    return bounds_getter(sprite)


def _has_custom_query_bounds(collided: _Collided | None) -> bool:
    return collided is not None and callable(
        getattr(collided, "fastquadtree_bounds", None)
    )


def _collides(sprite: Any, candidate: Any, collided: _Collided | None) -> bool:
    if collided is None:
        return sprite.rect.colliderect(candidate.rect)
    return bool(collided(sprite, candidate))


def spritecollide(
    sprite: Any,
    group: Any,
    dokill: bool,
    collided: _Collided | None = None,
) -> list[Any]:
    """Find sprites in a group that collide with another sprite."""
    if not isinstance(group, Group):
        return _pygame.sprite.spritecollide(sprite, group, dokill, collided)

    query_rect = _custom_query_bounds(collided, sprite)
    if query_rect is None:
        _warn_custom_collided()
        return _pygame.sprite.spritecollide(sprite, group, dokill, collided)

    if group._sprites_without_rect:
        return _pygame.sprite.spritecollide(sprite, group, dokill, collided)

    if collided is None and (not _has_rect(sprite) or not _is_rect_like(sprite.rect)):
        return _pygame.sprite.spritecollide(sprite, group, dokill, collided)

    candidates = group.query_rect(query_rect)
    collided_sprites = [
        candidate for candidate in candidates if _collides(sprite, candidate, collided)
    ]

    if dokill:
        for candidate in collided_sprites:
            candidate.kill()

    return collided_sprites


def spritecollideany(
    sprite: Any,
    group: Any,
    collided: _Collided | None = None,
) -> Any | None:
    """Return one sprite in a group that collides with another sprite."""
    if not isinstance(group, Group):
        return _pygame.sprite.spritecollideany(sprite, group, collided)

    query_rect = _custom_query_bounds(collided, sprite)
    if query_rect is None:
        _warn_custom_collided()
        return _pygame.sprite.spritecollideany(sprite, group, collided)

    if group._sprites_without_rect:
        return _pygame.sprite.spritecollideany(sprite, group, collided)

    if collided is None and (not _has_rect(sprite) or not _is_rect_like(sprite.rect)):
        return _pygame.sprite.spritecollideany(sprite, group, collided)

    for candidate in group.query_rect(query_rect):
        if _collides(sprite, candidate, collided):
            return candidate
    return None


def groupcollide(
    groupa: Any,
    groupb: Any,
    dokilla: bool,
    dokillb: bool,
    collided: _Collided | None = None,
) -> dict[Any, list[Any]]:
    """Detect collisions between two sprite groups."""
    if not isinstance(groupb, Group):
        return _pygame.sprite.groupcollide(groupa, groupb, dokilla, dokillb, collided)

    if collided is not None and not _has_custom_query_bounds(collided):
        _warn_custom_collided()
        return _pygame.sprite.groupcollide(groupa, groupb, dokilla, dokillb, collided)

    if groupb._sprites_without_rect:
        return _pygame.sprite.groupcollide(groupa, groupb, dokilla, dokillb, collided)

    groupb.sync()
    collided_sprites: dict[Any, list[Any]] = {}

    for group_a_sprite in groupa:
        collisions = spritecollide(group_a_sprite, groupb, dokillb, collided)
        if collisions:
            collided_sprites[group_a_sprite] = collisions

    if dokilla:
        for group_a_sprite in collided_sprites:
            group_a_sprite.kill()

    return collided_sprites
