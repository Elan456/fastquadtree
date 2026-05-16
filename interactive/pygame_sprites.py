"""Interactive example of fastquadtree's integration with pygame sprites."""

import argparse
import random
from collections.abc import Iterable
from typing import Literal

import pygame

import fastquadtree.pygame as fpygame

SCREEN_SIZE = (400, 400)
WORLD_MIN_X = -4000
WORLD_MIN_Y = -4000
WORLD_MAX_X = 4000
WORLD_MAX_Y = 4000
WORLD_WIDTH = WORLD_MAX_X - WORLD_MIN_X
WORLD_HEIGHT = WORLD_MAX_Y - WORLD_MIN_Y
BLOCK_COUNT = 40000
BORDER_COLOR = (25, 25, 25)
BORDER_WIDTH = 4
GROUP_BACKEND = Literal["pygame", "fastquadtree"]
OVERLAY_TEXT_COLOR = (20, 20, 20)
OVERLAY_BACKGROUND_COLOR = (255, 255, 255)
OVERLAY_PADDING = 6
FQT_GROUP_BOUNDS = (
    WORLD_MIN_X,
    WORLD_MIN_Y,
    WORLD_MAX_X,
    WORLD_MAX_Y,
)


class Camera:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0


class Block(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        width = random.randint(10, 30)
        height = random.randint(10, 30)

        self.x = random.randint(WORLD_MIN_X, WORLD_MAX_X - width)
        self.y = random.randint(WORLD_MIN_Y, WORLD_MAX_Y - height)

        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.topleft = (self.x, self.y)


class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()

        self.image = pygame.Surface([20, 20])
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect()
        self.x = 0
        self.y = 0
        self.sync_to_world()

    def update(self):
        keys = pygame.key.get_pressed()
        assert self.rect is not None, "Player sprite must have a rect attribute"
        if keys[pygame.K_LEFT]:
            self.x -= 5
        if keys[pygame.K_RIGHT]:
            self.x += 5
        if keys[pygame.K_UP]:
            self.y -= 5
        if keys[pygame.K_DOWN]:
            self.y += 5

        self.x = max(WORLD_MIN_X, min(self.x, WORLD_MAX_X - self.rect.width))
        self.y = max(WORLD_MIN_Y, min(self.y, WORLD_MAX_Y - self.rect.height))

        self.sync_to_world()

    def sync_to_world(self):
        self.rect.x = self.x
        self.rect.y = self.y


def draw_world_border(screen: pygame.Surface, camera: Camera):
    border_rect = pygame.Rect(
        WORLD_MIN_X - camera.x,
        WORLD_MIN_Y - camera.y,
        WORLD_WIDTH,
        WORLD_HEIGHT,
    )
    pygame.draw.rect(screen, BORDER_COLOR, border_rect, BORDER_WIDTH)


def draw_sprites(
    screen: pygame.Surface,
    sprites: Iterable[pygame.sprite.Sprite],
    camera: Camera,
):
    for sprite in sprites:
        screen.blit(sprite.image, (sprite.rect.x - camera.x, sprite.rect.y - camera.y))


def draw_sprite(screen: pygame.Surface, sprite: pygame.sprite.Sprite, camera: Camera):
    screen.blit(sprite.image, (sprite.rect.x - camera.x, sprite.rect.y - camera.y))


def draw_group_backend_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    backend: GROUP_BACKEND,
    fps: float,
    visible_count: int,
):
    lines = [f"Group: {backend}", f"FPS: {fps:.1f}", f"Visible: {visible_count}"]
    rendered_lines = [font.render(line, True, OVERLAY_TEXT_COLOR) for line in lines]
    width = max(line.get_width() for line in rendered_lines)
    height = sum(line.get_height() for line in rendered_lines)
    background = pygame.Rect(8, 8, width, height).inflate(
        OVERLAY_PADDING * 2, OVERLAY_PADDING * 2
    )
    pygame.draw.rect(screen, OVERLAY_BACKGROUND_COLOR, background)
    pygame.draw.rect(screen, BORDER_COLOR, background, 1)
    y = background.top + OVERLAY_PADDING
    for line in rendered_lines:
        screen.blit(line, (background.left + OVERLAY_PADDING, y))
        y += line.get_height()


def create_group(backend: GROUP_BACKEND, sprites: Iterable[pygame.sprite.Sprite]):
    if backend == "fastquadtree":
        return fpygame.Group(*sprites, bounds=FQT_GROUP_BOUNDS)
    return pygame.sprite.Group(*sprites)


def viewport_rect(camera: Camera) -> pygame.Rect:
    left = max(WORLD_MIN_X, int(camera.x))
    top = max(WORLD_MIN_Y, int(camera.y))
    right = min(WORLD_MAX_X, int(camera.x) + SCREEN_SIZE[0])
    bottom = min(WORLD_MAX_Y, int(camera.y) + SCREEN_SIZE[1])
    return pygame.Rect(left, top, max(0, right - left), max(0, bottom - top))


def visible_blocks(
    backend: GROUP_BACKEND,
    group: pygame.sprite.Group,
    viewport: pygame.Rect,
) -> list[pygame.sprite.Sprite]:
    if backend == "fastquadtree":
        assert isinstance(group, fpygame.Group)
        return group.query_rect(viewport, sync=False)
    return [sprite for sprite in group if sprite.rect.colliderect(viewport)]


def collide_blocks(
    backend: GROUP_BACKEND,
    player: Player,
    group: pygame.sprite.Group,
):
    if backend == "fastquadtree":
        return fpygame.spritecollide(player, group, False, sync=False)
    return pygame.sprite.spritecollide(player, group, False)


def next_backend(backend: GROUP_BACKEND) -> GROUP_BACKEND:
    return "pygame" if backend == "fastquadtree" else "fastquadtree"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pygame.sprite.Group with fastquadtree.pygame.Group."
    )
    parser.add_argument(
        "--group",
        choices=("pygame", "fastquadtree"),
        default="fastquadtree",
        help="Sprite group backend to start with. Press G in the demo to switch.",
    )
    return parser.parse_args()


def main(group_backend: GROUP_BACKEND = "fastquadtree"):
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption(
        f"Pygame Sprites with {group_backend} group - press G to switch"
    )
    font = pygame.font.Font(None, 24)

    camera = Camera()  # camera follows the player
    player = Player()
    blocks: list[pygame.sprite.Sprite] = []

    for _ in range(BLOCK_COUNT):
        block = Block()
        blocks.append(block)

    block_group = create_group(group_backend, blocks)

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                group_backend = next_backend(group_backend)
                block_group = create_group(group_backend, blocks)
                pygame.display.set_caption(
                    f"Pygame Sprites with {group_backend} group - press G to switch"
                )

        player.update()

        camera.x = player.x - screen.get_width() // 2
        camera.y = player.y - screen.get_height() // 2

        viewport = viewport_rect(camera)
        visible = visible_blocks(group_backend, block_group, viewport)
        touched_blocks = collide_blocks(group_backend, player, block_group)

        screen.fill((255, 255, 255))
        draw_world_border(screen, camera)
        draw_sprites(screen, visible, camera)
        draw_sprite(screen, player, camera)
        draw_group_backend_overlay(
            screen, font, group_backend, clock.get_fps(), len(visible)
        )

        for block in touched_blocks:
            highlight_rect = block.rect.move(-camera.x, -camera.y)
            pygame.draw.rect(screen, (255, 0, 0), highlight_rect, 2)

        pygame.display.flip()
        clock.tick(240)


if __name__ == "__main__":
    args = parse_args()
    pygame.init()
    main(args.group)
    pygame.quit()
