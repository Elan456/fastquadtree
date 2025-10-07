import contextlib
import math
import random
from typing import List, Set, Tuple

import pygame

from fastquadtree import QuadTree

# ---------------------------- Ball object ---------------------------- #


class Ball:
    __slots__ = ("color", "mass", "r", "restitution", "vx", "vy", "x", "y")

    def __init__(
        self,
        x: float,
        y: float,
        r: int = 10,
        color: Tuple[int, int, int] = (255, 0, 0),
        vx: float = 0.0,
        vy: float = 0.0,
        mass: float = 1.0,
        restitution: float = 0.7,
    ):
        self.x = float(x)
        self.y = float(y)
        self.vx = float(vx)
        self.vy = float(vy)
        self.r = int(r)
        self.color = color
        self.mass = float(mass)
        self.restitution = float(restitution)

    def aabb(self) -> Tuple[float, float, float, float]:
        return (self.x - self.r, self.y - self.r, self.x + self.r, self.y + self.r)

    def integrate(self, ax: float, ay: float, dt: float):
        self.vx += ax * dt
        self.vy += ay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def clamp_to_bounds(self, w: int, h: int):
        # Floor and ceiling
        if self.y + self.r > h:
            self.y = h - self.r
            self.vy = -self.vy * self.restitution
        if self.y - self.r < 0:
            self.y = self.r
            self.vy = -self.vy * self.restitution
        # Walls
        if self.x - self.r < 0:
            self.x = self.r
            self.vx = -self.vx * self.restitution
        if self.x + self.r > w:
            self.x = w - self.r
            self.vx = -self.vx * self.restitution

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.r)


# ------------------------- Collision utilities ------------------------- #


def resolve_ball_ball(a: Ball, b: Ball):
    """Generic elastic collision with positional correction and restitution."""
    dx = b.x - a.x
    dy = b.y - a.y
    dist_sq = dx * dx + dy * dy
    rsum = a.r + b.r
    if dist_sq <= 0 or dist_sq > rsum * rsum:
        return  # no collision

    dist = math.sqrt(dist_sq)
    # Normalized normal
    nx = dx / dist if dist != 0 else 1.0
    ny = dy / dist if dist != 0 else 0.0

    # Positional correction to remove overlap
    overlap = rsum - dist if dist != 0 else rsum
    # Split correction by mass proportion
    inv_ma = 0.0 if a.mass == 0 else 1.0 / a.mass
    inv_mb = 0.0 if b.mass == 0 else 1.0 / b.mass
    inv_sum = inv_ma + inv_mb if (inv_ma + inv_mb) != 0 else 1.0

    corr_a = overlap * inv_ma / inv_sum
    corr_b = overlap * inv_mb / inv_sum

    a.x -= nx * corr_a
    a.y -= ny * corr_a
    b.x += nx * corr_b
    b.y += ny * corr_b

    # Relative velocity along the normal
    rvx = b.vx - a.vx
    rvy = b.vy - a.vy
    vel_along_normal = rvx * nx + rvy * ny
    if vel_along_normal > 0:
        return  # separating

    # Restitution
    e = min(a.restitution, b.restitution)

    # Impulse scalar
    j = -(1 + e) * vel_along_normal
    j /= inv_sum

    # Apply impulse
    impulse_x = j * nx
    impulse_y = j * ny
    a.vx -= impulse_x * inv_ma
    a.vy -= impulse_y * inv_ma
    b.vx += impulse_x * inv_mb
    b.vy += impulse_y * inv_mb


# ------------------------------ BallPit ------------------------------ #


class BallPit:
    def __init__(self, screen, width, height):
        self.screen = screen
        self.width = width
        self.height = height
        self.qt = QuadTree((0, 0, width, height), 16, track_objects=True)
        self.balls: List[Ball] = []
        self.use_quadtree = True
        self.pair_checks = 0  # updated each frame

    def add_ball(self, x, y, radius=10, color=(255, 0, 0)):
        vx = (random.random() - 0.5) * 300.0  # px/s
        vy = (random.random() - 0.5) * 300.0  # px/s
        ball = Ball(
            x, y, r=radius, color=color, vx=vx, vy=vy, mass=1.0, restitution=0.7
        )
        self.balls.append(ball)
        with contextlib.suppress(ValueError):
            self.qt.insert((ball.x, ball.y), obj=ball)

    def rebuild_quadtree(self):
        # Clear quadtree and re-insert all balls
        # This is faster when most of the balls move each frame compared to updating individual points
        self.qt.clear()
        for b in self.balls:
            with contextlib.suppress(ValueError):
                self.qt.insert((b.x, b.y), obj=b)

    def update(self, dt: float):
        # 1) Integrate motion
        ax, ay = 0.0, 0.0
        for b in self.balls:
            b.integrate(ax, ay, dt)
            b.clamp_to_bounds(self.width, self.height)

        self.pair_checks = 0

        if self.use_quadtree:
            # 2) Rebuild spatial index
            self.rebuild_quadtree()

            # 3) Quadtree neighborhood checks
            processed: Set[Tuple[int, int]] = set()
            for b in self.balls:
                # Query a box that guarantees catching overlaps
                x0 = b.x - 2 * b.r
                y0 = b.y - 2 * b.r
                x1 = b.x + 2 * b.r
                y1 = b.y + 2 * b.r
                for item in self.qt.query((x0, y0, x1, y1), as_items=True):
                    other = item.obj
                    if other is b or other is None:
                        continue
                    a_id = id(b)
                    o_id = id(other)
                    key = (a_id, o_id) if a_id < o_id else (o_id, a_id)
                    if key in processed:
                        continue
                    processed.add(key)
                    self.pair_checks += 1
                    resolve_ball_ball(b, other)

            # 4) Update quadtree after resolution
            self.rebuild_quadtree()

        else:
            # Brute-force O(n^2) checks
            n = len(self.balls)
            for i in range(n):
                a = self.balls[i]
                for j in range(i + 1, n):
                    b = self.balls[j]
                    self.pair_checks += 1
                    resolve_ball_ball(a, b)

    def draw(self, fps: float):
        for ball in self.balls:
            ball.draw(self.screen)

        # HUD
        font = pygame.font.SysFont(None, 20)
        mode = "Quadtree" if self.use_quadtree else "Brute force"
        hud_lines = [
            f"FPS: {fps:.1f}",
            f"Mode: {mode} (press Q to toggle)",
            f"Balls: {len(self.balls)}",
            f"Pair checks this frame: {self.pair_checks}",
        ]

        # Draw a semi-transparent background for the HUD
        hud_bg_height = len(hud_lines) * 18 + 6
        hud_bg = pygame.Surface((250, hud_bg_height), pygame.SRCALPHA)
        hud_bg.fill((255, 255, 255, 200))  # White
        self.screen.blit(hud_bg, (0, 0))

        y = 6
        for line in hud_lines:
            surf = font.render(line, True, (20, 20, 20))
            self.screen.blit(surf, (6, y))
            y += 18


# ------------------------------- main ------------------------------- #


def main():
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("BallPit: Quadtree vs Brute Force")
    clock = pygame.time.Clock()
    ball_pit = BallPit(screen, width, height)

    # Pre-seed a few balls so the FPS change is obvious
    for _ in range(500):
        x, y = random.randint(40, width - 40), random.randint(40, height - 40)
        r = random.randint(8, 18)
        color = (
            random.randint(80, 255),
            random.randint(80, 255),
            random.randint(80, 255),
        )
        ball_pit.add_ball(x, y, radius=r, color=color)

    running = True
    while running:
        dt_ms = clock.tick(60)  # target 60 fps
        dt = dt_ms / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    ball_pit.use_quadtree = not ball_pit.use_quadtree
                elif event.key == pygame.K_c:
                    # Clear all balls (optional helper)
                    ball_pit.balls.clear()
                    ball_pit.rebuild_quadtree()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                r = random.randint(8, 18)
                color = (
                    random.randint(80, 255),
                    random.randint(80, 255),
                    random.randint(80, 255),
                )
                ball_pit.add_ball(x, y, radius=r, color=color)

        ball_pit.update(dt)

        screen.fill((255, 255, 255))
        fps = clock.get_fps()
        ball_pit.draw(fps)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
