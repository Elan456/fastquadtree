import math
import random
import time
from typing import Iterable, List, Set, Tuple

import pygame
from pyqtree import Index as PyQIndex

# Spatial backends -------------------------------------------------------------
from fastquadtree import Quadtree

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
    """Elastic collision with positional correction and restitution."""
    dx = b.x - a.x
    dy = b.y - a.y
    dist_sq = dx * dx + dy * dy
    rsum = a.r + b.r
    if dist_sq <= 0 or dist_sq > rsum * rsum:
        return  # no collision

    dist = math.sqrt(dist_sq)
    nx = dx / dist if dist != 0 else 1.0
    ny = dy / dist if dist != 0 else 0.0

    overlap = rsum - dist if dist != 0 else rsum
    inv_ma = 0.0 if a.mass == 0 else 1.0 / a.mass
    inv_mb = 0.0 if b.mass == 0 else 1.0 / b.mass
    inv_sum = inv_ma + inv_mb if (inv_ma + inv_mb) != 0 else 1.0

    corr_a = overlap * inv_ma / inv_sum
    corr_b = overlap * inv_mb / inv_sum

    a.x -= nx * corr_a
    a.y -= ny * corr_a
    b.x += nx * corr_b
    b.y += ny * corr_b

    rvx = b.vx - a.vx
    rvy = b.vy - a.vy
    vel_along_normal = rvx * nx + rvy * ny
    if vel_along_normal > 0:
        return  # separating

    e = min(a.restitution, b.restitution)
    j = -(1 + e) * vel_along_normal
    j /= inv_sum

    impulse_x = j * nx
    impulse_y = j * ny
    a.vx -= impulse_x * inv_ma
    a.vy -= impulse_y * inv_ma
    b.vx += impulse_x * inv_mb
    b.vy += impulse_y * inv_mb


# -------------------------- Spatial strategies -------------------------- #


class SpatialBase:
    name = "base"

    def rebuild(self, balls: List[Ball], width: int, height: int) -> None:
        raise NotImplementedError

    def neighbors(self, b: Ball) -> Iterable[Ball]:
        raise NotImplementedError


class FastQTIndex(SpatialBase):
    name = "fastquadtree"

    def __init__(self, width: int, height: int, capacity: int = 16):
        self.width = width
        self.height = height
        self.capacity = capacity
        self.qt = Quadtree((0, 0, width, height), capacity)

    def rebuild(self, balls: List[Ball], width: int, height: int) -> None:
        if width != self.width or height != self.height:
            self.width, self.height = width, height
            self.qt.clear()
        else:
            self.qt.clear()

        self.qt.insert_many([(b.x, b.y) for b in balls])

    def neighbors(self, b: Ball, balls: List[Ball]) -> Iterable[Ball]:
        r2 = 2 * b.r
        min_x, min_y, max_x, max_y = b.x - r2, b.y - r2, b.x + r2, b.y + r2
        neighbors = self.qt.query((min_x, min_y, max_x, max_y))
        for it in neighbors:
            other = balls[it[0]]
            if other is not None and other is not b:
                yield other


class PyQTreeIndex(SpatialBase):
    name = "pyqtree"

    def __init__(
        self, width: int, height: int, max_items: int = 16, max_depth: int = 20
    ):
        self.width = width
        self.height = height
        self.max_items = max_items
        self.max_depth = max_depth
        self.idx = PyQIndex(
            bbox=(0, 0, width, height), max_items=max_items, max_depth=max_depth
        )

    def rebuild(self, balls: List[Ball], width: int, height: int) -> None:
        if width != self.width or height != self.height:
            self.width, self.height = width, height
            self.idx = PyQIndex(
                bbox=(0, 0, width, height),
                max_items=self.max_items,
                max_depth=self.max_depth,
            )
        else:
            # pyqtree has no clear, so rebuild a fresh tree
            self.idx = PyQIndex(
                bbox=(0, 0, width, height),
                max_items=self.max_items,
                max_depth=self.max_depth,
            )
        for b in balls:
            self.idx.insert(b, b.aabb())

    def neighbors(self, b: Ball) -> Iterable[Ball]:
        r2 = 2 * b.r
        qbox = (b.x - r2, b.y - r2, b.x + r2, b.y + r2)
        # pyqtree returns stored objects directly
        for other in self.idx.intersect(qbox):
            if other is not b:
                yield other


class BruteIndex(SpatialBase):
    name = "bruteforce"

    def __init__(self, balls_ref: List[Ball]):
        self._balls = balls_ref

    def rebuild(self, balls: List[Ball], width: int, height: int) -> None:
        # Nothing to build
        pass

    def neighbors(self, b: Ball) -> Iterable[Ball]:
        return (o for o in self._balls if o is not b)


# ------------------------------ BallPit ------------------------------ #


class BallPit:
    MODES = ("fastquadtree", "pyqtree", "bruteforce")

    def __init__(self, screen, width, height):
        self.screen = screen
        self.width = width
        self.height = height

        self.balls: List[Ball] = []
        self.pair_checks = 0  # updated each frame

        # Spatial backends
        self.backends: dict[str, SpatialBase] = {}
        self.backends["fastquadtree"] = FastQTIndex(width, height)
        self.backends["pyqtree"] = PyQTreeIndex(width, height)
        self.backends["bruteforce"] = BruteIndex(self.balls)

        self.mode_idx = (
            0
            if "fastquadtree" in self.backends
            else 1
            if "pyqtree" in self.backends
            else 2
        )
        self.mode = self.MODES[self.mode_idx]
        if self.mode not in self.backends:
            # Fallback if some backends are unavailable
            self.mode = next(iter(self.backends.keys()))

    def set_mode(self, name: str):
        if name in self.backends:
            self.mode = name

    def cycle_mode(self):
        # Cycle only through available backends
        order = [m for m in self.MODES if m in self.backends]
        cur = order.index(self.mode)
        self.mode = order[(cur + 1) % len(order)]

    def add_ball(self, x, y, radius=10, color=(255, 0, 0)):
        vx = (random.random() - 0.5) * 300.0  # px/s
        vy = (random.random() - 0.5) * 300.0  # px/s
        ball = Ball(
            x, y, r=radius, color=color, vx=vx, vy=vy, mass=1.0, restitution=0.7
        )
        self.balls.append(ball)

    def _backend(self) -> SpatialBase:
        return self.backends[self.mode]

    def update(self, dt: float) -> tuple[float, float]:
        # 1) Integrate motion
        ax, ay = 0.0, 0.0
        for b in self.balls:
            b.integrate(ax, ay, dt)
            b.clamp_to_bounds(self.width, self.height)

        # 2) Rebuild spatial index
        backend = self._backend()
        start = time.perf_counter()
        backend.rebuild(self.balls, self.width, self.height)
        end = time.perf_counter()

        tree_build_time = end - start

        total_neighbor_collection_time = 0.0

        # 3) Neighborhood checks with dedup on object id pairs
        self.pair_checks = 0
        processed: Set[Tuple[int, int]] = set()
        for a in self.balls:
            start = time.perf_counter()
            if self.mode == "fastquadtree":
                neighbors = backend.neighbors(a, self.balls)  # type: ignore
            else:
                neighbors = backend.neighbors(a)
            end = time.perf_counter()
            total_neighbor_collection_time += end - start
            for other in neighbors:
                a_id = id(a)
                o_id = id(other)
                key = (a_id, o_id) if a_id < o_id else (o_id, a_id)
                if key in processed:
                    continue
                processed.add(key)
                self.pair_checks += 1
                resolve_ball_ball(a, other)
        return (total_neighbor_collection_time, tree_build_time)

    def draw(
        self, fps: float, total_neighbor_collection_time: float, tree_build_time: float
    ):
        for ball in self.balls:
            ball.draw(self.screen)

        # HUD
        font = pygame.font.SysFont(None, 20)
        hud_lines = [
            f"FPS: {fps:.1f}",
            f"Build time: {tree_build_time * 1000:.2f} ms",
            f"Query time: {total_neighbor_collection_time * 1000:.2f} ms",
            f"Mode: {self.mode} (Tab to cycle, 1/2/3 to select)",
            f"Balls: {len(self.balls)}",
            f"Pair checks this frame: {self.pair_checks}",
        ]

        # Draw a semi-transparent background for the HUD
        hud_bg_height = len(hud_lines) * 18 + 6
        hud_bg = pygame.Surface((400, hud_bg_height), pygame.SRCALPHA)
        hud_bg.fill((255, 255, 255, 200))
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
    pygame.display.set_caption("BallPit: fastquadtree vs pyqtree vs brute force")
    clock = pygame.time.Clock()
    ball_pit = BallPit(screen, width, height)

    metrics = {}

    # Pre-seed a few balls so the FPS change is obvious
    for _ in range(1500):
        x, y = random.randint(40, width - 40), random.randint(40, height - 40)
        r = random.randint(8, 10)
        color = (
            random.randint(80, 255),
            random.randint(80, 255),
            random.randint(80, 255),
        )
        ball_pit.add_ball(x, y, radius=r, color=color)

    running = True
    while running:
        dt_ms = clock.tick(200)  # target 200 FPS
        dt = dt_ms / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    ball_pit.cycle_mode()
                elif event.key == pygame.K_1:
                    ball_pit.set_mode("fastquadtree")
                elif event.key == pygame.K_2:
                    ball_pit.set_mode("pyqtree")
                elif event.key == pygame.K_3:
                    ball_pit.set_mode("bruteforce")
                elif event.key == pygame.K_c:
                    ball_pit.balls.clear()
                elif event.key == pygame.K_q:
                    # legacy toggle: fastquadtree <-> brute (kept for convenience)
                    ball_pit.set_mode(
                        "bruteforce"
                        if ball_pit.mode == "fastquadtree"
                        else "fastquadtree"
                    )
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                r = random.randint(8, 18)
                color = (
                    random.randint(80, 255),
                    random.randint(80, 255),
                    random.randint(80, 255),
                )
                ball_pit.add_ball(x, y, radius=r, color=color)

        stats = ball_pit.update(dt)

        screen.fill((255, 255, 255))
        fps = clock.get_fps()

        if ball_pit.mode not in metrics:
            metrics[ball_pit.mode] = {"count": 0, "avg_fps": 0.0}

        metrics[ball_pit.mode]["count"] += 1
        metrics[ball_pit.mode]["avg_fps"] = (
            metrics[ball_pit.mode]["avg_fps"] * (metrics[ball_pit.mode]["count"] - 1)
            + fps
        ) / metrics[ball_pit.mode]["count"]

        ball_pit.draw(fps, stats[0], stats[1])
        pygame.display.flip()

    pygame.quit()

    print("\nAverage FPS per backend mode:")
    for mode, data in metrics.items():
        print(f"  {mode}: {data['avg_fps']:.2f} FPS over {data['count']} frames")


if __name__ == "__main__":
    main()
