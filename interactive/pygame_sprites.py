"""
Interactive example of fastquadtree's integration with pygame's spirtes.

Includes simple rectangle sprites that move around the screen and collide with each other.
(fastquadtree integration is WIP and upcoming, so this is just a placeholder for now)
"""

import random

import pygame

SCREEN_SIZE = (400, 400)
WORLD_MIN_X = -1000
WORLD_MIN_Y = -1000
WORLD_MAX_X = 1000
WORLD_MAX_Y = 1000
WORLD_WIDTH = WORLD_MAX_X - WORLD_MIN_X
WORLD_HEIGHT = WORLD_MAX_Y - WORLD_MIN_Y
BLOCK_COUNT = 250
BORDER_COLOR = (25, 25, 25)
BORDER_WIDTH = 4


class Camera:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0


class Block(pygame.sprite.Sprite):
    def __init__(self, camera: Camera):
        super().__init__()

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        width = random.randint(10, 30)
        height = random.randint(10, 30)

        self.x = random.randint(WORLD_MIN_X, WORLD_MAX_X - width)
        self.y = random.randint(WORLD_MIN_Y, WORLD_MAX_Y - height)

        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.camera = camera

    def update(self):
        assert self.rect is not None, "Block sprite must have a rect attribute"
        self.sync_to_camera()

    def sync_to_camera(self):
        self.rect.x = self.x - self.camera.x
        self.rect.y = self.y - self.camera.y


class Player(pygame.sprite.Sprite):
    def __init__(self, camera: Camera):
        super().__init__()

        self.image = pygame.Surface([20, 20])
        self.image.fill((0, 255, 0))
        self.rect = self.image.get_rect()
        self.camera = camera
        self.x = 0
        self.y = 0

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

        self.sync_to_camera()

    def sync_to_camera(self):
        self.rect.x = self.x - self.camera.x
        self.rect.y = self.y - self.camera.y


def draw_world_border(screen: pygame.Surface, camera: Camera):
    border_rect = pygame.Rect(
        WORLD_MIN_X - camera.x,
        WORLD_MIN_Y - camera.y,
        WORLD_WIDTH,
        WORLD_HEIGHT,
    )
    pygame.draw.rect(screen, BORDER_COLOR, border_rect, BORDER_WIDTH)


def main():
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption("Pygame Sprites with FastQuadtree")

    default_group = pygame.sprite.Group()
    camera = Camera()  # camera follows the player
    player = Player(camera)
    default_group.add(player)

    for _ in range(BLOCK_COUNT):
        block = Block(camera)
        default_group.add(block)

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        default_group.update()

        camera.x = player.x - screen.get_width() // 2
        camera.y = player.y - screen.get_height() // 2
        for sprite in default_group:
            sprite.sync_to_camera()

        touched_blocks = pygame.sprite.spritecollide(player, default_group, False)

        screen.fill((255, 255, 255))
        draw_world_border(screen, camera)
        default_group.draw(screen)

        if len(touched_blocks) > 1:
            # highlight the touched blocks (excluding the player itself)
            for block in touched_blocks:
                if block is not player:
                    pygame.draw.rect(screen, (255, 0, 0), block.rect, 2)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    pygame.init()
    main()
    pygame.quit()
