from __future__ import annotations

import random
from typing import Any

import pytest

pygame = pytest.importorskip("pygame")
fpygame = pytest.importorskip("fastquadtree.pygame")


class RectSprite(pygame.sprite.Sprite):
    def __init__(self, rect: tuple[int, int, int, int] | None = None, label: str = ""):
        super().__init__()
        if rect is not None:
            self.rect = pygame.Rect(rect)
        self.label = label
        self.update_calls = 0

    def update(self, dx: int = 0, dy: int = 0) -> None:
        self.update_calls += 1
        if hasattr(self, "rect"):
            self.rect.move_ip(dx, dy)


def _same_sprites(left: list[Any], right: list[Any]) -> bool:
    return set(left) == set(right)


def _same_groupcollide(left: dict[Any, list[Any]], right: dict[Any, list[Any]]) -> bool:
    return {key: set(value) for key, value in left.items()} == {
        key: set(value) for key, value in right.items()
    }


def test_group_is_drop_in_for_core_group_operations():
    a = RectSprite((0, 0, 10, 10))
    b = RectSprite((20, 0, 10, 10))
    c = RectSprite((40, 0, 10, 10))

    pg_group = pygame.sprite.Group(a, b)
    fq_group = fpygame.Group((0, 0, 100, 100), a, b)

    assert len(fq_group) == len(pg_group)
    assert bool(fq_group) is True
    assert set(fq_group.sprites()) == set(pg_group.sprites())
    assert a in fq_group
    assert fq_group.has(a, b)
    assert repr(fq_group) == "<Group(2 sprites)>"
    assert fq_group.indexed_count == 2

    fq_group.add([c])
    pg_group.add([c])
    assert set(fq_group) == set(pg_group)
    assert fq_group.indexed_count == 3

    fq_group.remove([b])
    pg_group.remove([b])
    assert set(fq_group.sprites()) == set(pg_group.sprites())
    assert fq_group.indexed_count == 2

    fq_group.update(1, 2)
    assert [sprite.update_calls for sprite in fq_group] == [1, 1]
    assert _same_sprites(
        fpygame.spritecollide(a, fq_group, False),
        pygame.sprite.spritecollide(a, pg_group, False),
    )

    clone = fq_group.copy()
    assert isinstance(clone, fpygame.Group)
    assert clone.bounds == fq_group.bounds
    assert set(clone.sprites()) == set(fq_group.sprites())

    fq_group.empty()
    pg_group.empty()
    assert len(fq_group) == len(pg_group) == 0
    assert bool(fq_group) is False
    assert fq_group.indexed_count == 0


def test_spritecollide_matches_pygame_for_random_rect_scenes():
    for seed in range(40):
        random.seed(seed)
        sprites = [
            RectSprite(
                (
                    random.randint(-50, 50),
                    random.randint(-50, 50),
                    random.randint(1, 20),
                    random.randint(1, 20),
                )
            )
            for _ in range(20)
        ]
        query = sprites[0]
        pg_group = pygame.sprite.Group(sprites)
        fq_group = fpygame.Group((-100, -100, 100, 100), sprites)

        assert _same_sprites(
            fpygame.spritecollide(query, fq_group, False),
            pygame.sprite.spritecollide(query, pg_group, False),
        )

        pg_any = pygame.sprite.spritecollideany(query, pg_group)
        fq_any = fpygame.spritecollideany(query, fq_group)
        assert (fq_any is None) == (pg_any is None)
        if fq_any is not None:
            assert fq_any in pygame.sprite.spritecollide(query, pg_group, False)

        assert _same_groupcollide(
            fpygame.groupcollide(fq_group, fq_group, False, False),
            pygame.sprite.groupcollide(pg_group, pg_group, False, False),
        )


def test_collision_helpers_fall_back_for_plain_pygame_groups():
    player = RectSprite((0, 0, 10, 10))
    hit = RectSprite((5, 5, 10, 10))
    miss = RectSprite((50, 50, 10, 10))
    group = pygame.sprite.Group(hit, miss)

    assert fpygame.spritecollide(player, group, False) == pygame.sprite.spritecollide(
        player, group, False
    )
    assert fpygame.spritecollideany(player, group) is pygame.sprite.spritecollideany(
        player, group
    )
    assert fpygame.groupcollide(
        pygame.sprite.Group(player), group, False, False
    ) == pygame.sprite.groupcollide(pygame.sprite.Group(player), group, False, False)


def test_dokill_matches_pygame_for_spritecollide_and_groupcollide():
    pg_player = RectSprite((0, 0, 10, 10), "player")
    pg_hit = RectSprite((5, 5, 10, 10), "hit")
    pg_miss = RectSprite((50, 50, 10, 10), "miss")
    fq_player = RectSprite((0, 0, 10, 10), "player")
    fq_hit = RectSprite((5, 5, 10, 10), "hit")
    fq_miss = RectSprite((50, 50, 10, 10), "miss")
    pg_group = pygame.sprite.Group(pg_player, pg_hit, pg_miss)
    fq_group = fpygame.Group((0, 0, 100, 100), fq_player, fq_hit, fq_miss)

    fq_hits = fpygame.spritecollide(fq_player, fq_group, True)
    pg_hits = pygame.sprite.spritecollide(pg_player, pg_group, True)
    assert [sprite.label for sprite in fq_hits] == [sprite.label for sprite in pg_hits]
    assert {sprite.label for sprite in fq_group} == {
        sprite.label for sprite in pg_group
    }
    assert fq_group.indexed_count == len(pg_group)

    pg_a_hit = RectSprite((0, 0, 10, 10), "a-hit")
    pg_a_miss = RectSprite((50, 50, 10, 10), "a-miss")
    pg_b_hit = RectSprite((5, 5, 10, 10), "b-hit")
    fq_a_hit = RectSprite((0, 0, 10, 10), "a-hit")
    fq_a_miss = RectSprite((50, 50, 10, 10), "a-miss")
    fq_b_hit = RectSprite((5, 5, 10, 10), "b-hit")
    pg_a = pygame.sprite.Group(pg_a_hit, pg_a_miss)
    pg_b = pygame.sprite.Group(pg_b_hit)
    fq_a = fpygame.Group((0, 0, 100, 100), fq_a_hit, fq_a_miss)
    fq_b = fpygame.Group((0, 0, 100, 100), fq_b_hit)

    fq_collisions = fpygame.groupcollide(fq_a, fq_b, True, True)
    pg_collisions = pygame.sprite.groupcollide(pg_a, pg_b, True, True)
    assert {sprite.label for sprite in fq_collisions} == {
        sprite.label for sprite in pg_collisions
    }
    assert [[sprite.label for sprite in hits] for hits in fq_collisions.values()] == [
        [sprite.label for sprite in hits] for hits in pg_collisions.values()
    ]
    assert {sprite.label for sprite in fq_a} == {sprite.label for sprite in pg_a}
    assert {sprite.label for sprite in fq_b} == {sprite.label for sprite in pg_b}
    assert fq_a.indexed_count == len(pg_a)
    assert fq_b.indexed_count == len(pg_b)


def test_duplicate_same_rect_removal_deletes_only_that_sprite():
    first = RectSprite((10, 10, 10, 10))
    second = RectSprite((10, 10, 10, 10))
    group = fpygame.Group((0, 0, 100, 100), first, second)

    group.remove(first)

    assert first not in group
    assert second in group
    assert group.indexed_count == 1
    assert fpygame.spritecollide(second, group, False) == [second]


def test_sync_reindexes_moved_and_resized_sprites():
    query = RectSprite((0, 0, 10, 10))
    moving = RectSprite((50, 50, 5, 5))
    group = fpygame.Group((0, 0, 100, 100), query, moving)

    assert fpygame.spritecollide(query, group, False) == [query]

    moving.rect.update(5, 5, 20, 20)
    group.sync(moving)

    assert set(fpygame.spritecollide(query, group, False)) == {query, moving}


def test_inferred_bounds_growth_set_bounds_rebuild_and_query_rect():
    near = RectSprite((0, 0, 10, 10))
    far = RectSprite((500, 500, 10, 10))
    group = fpygame.Group([near])

    assert group.bounds is not None
    assert group.query_rect((0, 0, 10, 10)) == [near]

    group.add(far)
    assert group.bounds is not None
    assert group.bounds[2] > 500
    assert group.query_rect(far.rect) == [far]

    group.set_bounds((-1000, -1000, 1000, 1000))
    assert group.bounds == (-1000, -1000, 1000, 1000)
    assert group.indexed_count == 2

    group.rebuild()
    assert set(group.query_rect(pygame.Rect(-1, -1, 20, 20))) == {near}
    assert group.query_rect((0, 0, 0, 10)) == []


def test_unusable_rects_preserve_pygame_fallback_behavior():
    query = RectSprite((0, 0, 10, 10))
    zero_width = RectSprite((0, 0, 0, 10))
    missing_rect = RectSprite()

    pg_zero = pygame.sprite.Group(zero_width)
    fq_zero = fpygame.Group((0, 0, 100, 100), zero_width)
    assert fpygame.spritecollide(query, fq_zero, False) == pygame.sprite.spritecollide(
        query, pg_zero, False
    )
    assert fpygame.spritecollideany(query, fq_zero) == pygame.sprite.spritecollideany(
        query, pg_zero
    )

    pg_missing = pygame.sprite.Group(missing_rect)
    fq_missing = fpygame.Group((0, 0, 100, 100), missing_rect)
    with pytest.raises(TypeError):
        pygame.sprite.spritecollide(query, pg_missing, False)
    with pytest.raises(TypeError):
        fpygame.spritecollide(query, fq_missing, False)


def test_opaque_custom_collided_warns_and_matches_pygame_full_scan():
    query = RectSprite((0, 0, 10, 10))
    far = RectSprite((1000, 1000, 10, 10))
    group = fpygame.Group((0, 0, 2000, 2000), far)
    pg_group = pygame.sprite.Group(far)

    def always_collide(left: Any, right: Any) -> bool:
        return True

    with pytest.warns(RuntimeWarning, match="cannot safely accelerate"):
        assert fpygame.spritecollide(query, group, False, always_collide) == (
            pygame.sprite.spritecollide(query, pg_group, False, always_collide)
        )

    with pytest.warns(RuntimeWarning, match="cannot safely accelerate"):
        assert fpygame.spritecollideany(query, group, always_collide) is far

    with pytest.warns(RuntimeWarning, match="cannot safely accelerate"):
        assert fpygame.groupcollide(
            pygame.sprite.Group(query), group, False, False, always_collide
        ) == pygame.sprite.groupcollide(
            pygame.sprite.Group(query), pg_group, False, False, always_collide
        )


def test_custom_collided_with_explicit_broadphase_bounds_is_accelerated():
    query = RectSprite((0, 0, 10, 10))
    near = RectSprite((30, 0, 10, 10))
    far = RectSprite((1000, 0, 10, 10))
    group = fpygame.Group((0, 0, 2000, 2000), near, far)

    def near_center(left: Any, right: Any) -> bool:
        return abs(left.rect.centerx - right.rect.centerx) < 50

    near_center.fastquadtree_bounds = lambda sprite: (
        sprite.rect.left - 50,
        sprite.rect.top - 50,
        sprite.rect.right + 50,
        sprite.rect.bottom + 50,
    )

    assert fpygame.spritecollide(query, group, False, near_center) == [near]
    assert fpygame.spritecollideany(query, group, near_center) is near
    assert fpygame.groupcollide(
        pygame.sprite.Group(query), group, False, False, near_center
    ) == {query: [near]}


def test_invalid_constructor_and_query_inputs_are_handled():
    with pytest.raises(ValueError):
        fpygame.Group((0, 0, 0, 10))

    group = fpygame.Group()
    assert group.bounds is None
    assert group.indexed_count == 0
    assert group.query_rect(object()) == []


def test_internal_edge_paths_keep_public_behavior_stable():
    sprite = RectSprite((1, 1, 5, 5))
    missing = RectSprite()
    zero = RectSprite((1, 1, 0, 5))

    empty = fpygame.Group()
    empty.rebuild()
    assert empty.bounds == (-1.0, -1.0, 1.0, 1.0)

    grows_from_no_bounds = fpygame.Group()
    grows_from_no_bounds.add(sprite)
    assert grows_from_no_bounds.bounds is not None
    assert grows_from_no_bounds.indexed_count == 1

    no_rebuild = fpygame.Group((0, 0, 10, 10), sprite)
    no_rebuild.set_bounds((-100, -100, 100, 100), rebuild=False)
    assert no_rebuild.indexed_count == 0
    no_rebuild.rebuild()
    assert no_rebuild.indexed_count == 1

    assert fpygame.Group().query_rect(pygame.Rect(0, 0, 1, 1)) == []
    assert no_rebuild.query_rect(sprite.rect, sync=False) == [sprite]

    no_rebuild._tree.insert((50, 50, 51, 51), obj=None)
    assert no_rebuild.query_rect((50, 50, 51, 51), sync=False) == []

    no_rebuild._tree.clear()
    sprite.rect.move_ip(1, 1)
    no_rebuild.sync(sprite)
    assert no_rebuild.query_rect(sprite.rect) == [sprite]

    stale = fpygame.Group((0, 0, 10, 10), sprite)
    stale._tree.clear()
    stale.remove(sprite)
    assert stale.indexed_count == 0

    rebuild_with_unusable = fpygame.Group((0, 0, 10, 10), missing, zero)
    rebuild_with_unusable.rebuild()
    assert rebuild_with_unusable.indexed_count == 0

    group = fpygame.Group((0, 0, 10, 10), sprite)
    group._index_sprite(sprite)
    assert group.indexed_count == 1
    group._tree = None
    assert group._find_sprite_id(sprite, (1, 1, 6, 6)) is None
    group._ensure_tree_contains((1, 1, 6, 6))
    assert group._tree is not None

    other = RectSprite((1, 1, 5, 5))
    assert group._find_sprite_id(other, (1, 1, 6, 6)) is None

    first_same_rect = RectSprite((4, 4, 6, 6))
    second_same_rect = RectSprite((4, 4, 6, 6))
    same_rect_group = fpygame.Group((0, 0, 20, 20), first_same_rect, second_same_rect)
    assert same_rect_group._find_sprite_id(second_same_rect, (4, 4, 10, 10)) is not None

    nested_group = pygame.sprite.Group(sprite)
    inferred_from_group = fpygame.Group(nested_group)
    assert inferred_from_group.indexed_count == 1

    assert (
        fpygame.spritecollideany(RectSprite((500, 500, 5, 5)), inferred_from_group)
        is None
    )

    query_without_rect = RectSprite()
    with pytest.raises(AttributeError):
        fpygame.spritecollide(query_without_rect, inferred_from_group, False)
    with pytest.raises(AttributeError):
        fpygame.spritecollideany(query_without_rect, inferred_from_group)

    query_with_bad_rect = RectSprite()
    query_with_bad_rect.rect = object()
    with pytest.raises(AttributeError):
        fpygame.spritecollide(query_with_bad_rect, inferred_from_group, False)
    with pytest.raises(AttributeError):
        fpygame.spritecollideany(query_with_bad_rect, inferred_from_group)

    with_unusable = fpygame.Group((0, 0, 10, 10), missing)
    with pytest.raises(TypeError):
        pygame.sprite.groupcollide(
            pygame.sprite.Group(sprite), pygame.sprite.Group(missing), False, False
        )
    with pytest.raises(TypeError):
        fpygame.groupcollide(pygame.sprite.Group(sprite), with_unusable, False, False)


def test_private_helpers_cover_degenerate_inputs():
    assert fpygame._bounds_from_rects([(2, 3, 2, 3)]) == (0.8, 1.8, 3.2, 4.2)
    assert fpygame._flatten_sprites([object()])
