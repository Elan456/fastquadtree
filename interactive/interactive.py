"""
fastquadtree v2 demo (non-object QuadTree)

Showcases:
- insert((x, y)) auto IDs and insert((x, y), id_=custom)
- insert_many / InsertResult
- update(id_, old_x, old_y, new_x, new_y)
- delete(id_, x, y) and delete_tuple((id_, x, y))
- query(rect) and query_np(rect) (optional if NumPy installed)
- nearest_neighbor / nearest_neighbors
- __contains__
- __iter__
- get_all_node_boundaries()
- len(qt)
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass

import pygame

from fastquadtree import QuadTree

# -----------------------------
# Config
# -----------------------------
SCREEN_W, SCREEN_H = 1000, 1000
WORLD_BOUNDS = (-1000.0, -1000.0, 1000.0, 1000.0)
INIT_BALLS = 500
MAX_SPEED = 5.0

# UI / interaction
QUERY_SIZE = 220.0
QUERY_SPEED = 6.0
CAM_SPEED = 8.0

# Controls:
#   WASD: move camera
#   Arrow keys: move query rect
#   Left click: add a ball at mouse world position (auto id)
#   Right click: add a ball at mouse world position (custom id == list index)
#   Space: delete nearest to mouse (delete_tuple)
#   Backspace: delete all balls inside query rect (delete(id_, x, y))
#   R: rebuild the tree using insert_many (demonstrates bulk insert)
#   C: clear everything
#   T: toggle quadtree boundary drawing
#   N: toggle nearest-neighbor lines
#   K: toggle kNN (k=6) lines
#   P: toggle numpy query highlight (if numpy installed)


# -----------------------------
# Helpers
# -----------------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def world_from_screen(
    mx: int, my: int, cam_x: float, cam_y: float
) -> tuple[float, float]:
    return (mx + cam_x, my + cam_y)


def screen_from_world(
    x: float, y: float, cam_x: float, cam_y: float
) -> tuple[int, int]:
    return (int(x - cam_x), int(y - cam_y))


def rect_from_center(
    cx: float, cy: float, size: float
) -> tuple[float, float, float, float]:
    half = size * 0.5
    return (cx - half, cy - half, cx + half, cy + half)


def rect_to_pygame(
    rect: tuple[float, float, float, float], cam_x: float, cam_y: float
) -> pygame.Rect:
    min_x, min_y, max_x, max_y = rect
    return pygame.Rect(
        int(min_x - cam_x),
        int(min_y - cam_y),
        int(max_x - min_x),
        int(max_y - min_y),
    )


# -----------------------------
# Ball payload store (external)
# -----------------------------
@dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float
    radius: int
    alive: bool = True

    def update(self) -> None:
        self.x += self.vx
        self.y += self.vy

        min_x, min_y, max_x, max_y = WORLD_BOUNDS

        # bounce with radius-aware bounds
        if self.x - self.radius < min_x or self.x + self.radius > max_x:
            self.vx *= -1.0
            self.x = clamp(self.x, min_x + self.radius, max_x - self.radius)
        if self.y - self.radius < min_y or self.y + self.radius > max_y:
            self.vy *= -1.0
            self.y = clamp(self.y, min_y + self.radius, max_y - self.radius)


def make_ball(x: float, y: float) -> Ball:
    vx = random.uniform(-MAX_SPEED, MAX_SPEED)
    vy = random.uniform(-MAX_SPEED, MAX_SPEED)
    r = random.randint(4, 18)
    return Ball(x=x, y=y, vx=vx, vy=vy, radius=r)


# -----------------------------
# Demo setup
# -----------------------------
pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 22)

qt = QuadTree(WORLD_BOUNDS, capacity=6, max_depth=10)

# Camera in world coords (top-left of the screen in world space)
camera_x = -500.0
camera_y = -500.0

# Query rect stored as min/max
query_rect = rect_from_center(0.0, 0.0, QUERY_SIZE)

# External payload store: list indexed by ID
balls: list[Ball] = []


def rebuild_tree_bulk() -> None:
    """Rebuild qt from alive balls using insert_many (demonstrates InsertResult)."""
    qt.clear()

    pts: list[tuple[float, float]] = []
    id_map: list[int] = []

    for i, b in enumerate(balls):
        if b.alive:
            pts.append((b.x, b.y))
            id_map.append(i)

    # insert_many in v2 assigns contiguous IDs, so it doesn't preserve our list indices.
    # For demos that want list-index IDs, we reinsert with custom IDs instead.
    #
    # But we still demo insert_many here by rebuilding into a fresh qt with dense IDs
    # AND keeping a mapping from dense->list index.
    #
    # To keep the rest of the demo simple, we use custom IDs for all “real” behavior
    # and only use insert_many as a one-time showcase, then immediately reinsert with custom IDs.
    if pts:
        qt.insert_many(pts)
        # r.ids gives range(start..end). We'll just show it in the UI; then discard and reinsert.
        # Now immediately rebuild with custom IDs (real use case):
        qt.clear()
        for idx, b in enumerate(balls):
            if b.alive:
                qt.insert((b.x, b.y), id_=idx)
    else:
        qt.clear()


def seed() -> None:
    balls.clear()
    qt.clear()

    # create payloads first
    for _ in range(INIT_BALLS):
        x = random.uniform(WORLD_BOUNDS[0] + 1, WORLD_BOUNDS[2] - 1)
        y = random.uniform(WORLD_BOUNDS[1] + 1, WORLD_BOUNDS[3] - 1)
        balls.append(make_ball(x, y))

    # Insert using custom IDs == list indices (recommended for non-object QuadTree)
    for idx, b in enumerate(balls):
        qt.insert((b.x, b.y), id_=idx)


seed()

draw_bounds = True
draw_nn = True
draw_knn = False
knn_k = 6
use_numpy_highlight = False


def draw_quadtree_boundaries() -> None:
    if not draw_bounds:
        return
    for bbox in qt.get_all_node_boundaries():
        rect = rect_to_pygame(bbox, camera_x, camera_y)
        pygame.draw.rect(screen, (255, 255, 255), rect, 1)


def draw_hud(
    fps: float, nn: tuple[int, float, float] | None, mouse_world: tuple[float, float]
) -> None:
    mx, my = mouse_world
    lines = [
        "fastquadtree QuadTree (non-object) demo",
        f"len(qt)={len(qt)}  alive={sum(1 for b in balls if b.alive)}  balls_total={len(balls)}",
        f"mouse world=({mx:.1f}, {my:.1f})  contains? {((mx, my) in qt)}",
        f"NN={nn[0] if nn else None}",
        "WASD camera | Arrows move query | LMB add(auto id) | RMB add(custom id=list index)",
        "Space delete nearest | Backspace delete all in query | R bulk-rebuild demo | C clear",
        "T toggle tree | N toggle NN | K toggle kNN | P toggle numpy query highlight",
    ]
    y = 8
    for s in lines:
        surf = font.render(s, True, (220, 220, 220))
        screen.blit(surf, (8, y))
        y += 20


def try_numpy_highlight(rect: tuple[float, float, float, float]) -> None:
    if not use_numpy_highlight:
        return
    try:
        _ids, coords = qt.query_np(rect)
    except ImportError:
        return

    # highlight with blue outline
    for x, y in coords:
        pygame.draw.circle(
            screen,
            (80, 160, 255),
            screen_from_world(float(x), float(y), camera_x, camera_y),
            10,
            2,
        )


# -----------------------------
# Main loop
# -----------------------------
while True:
    dt = clock.tick(120) / 1000.0
    fps = clock.get_fps()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_t:
                draw_bounds = not draw_bounds
            elif event.key == pygame.K_n:
                draw_nn = not draw_nn
            elif event.key == pygame.K_k:
                draw_knn = not draw_knn
            elif event.key == pygame.K_p:
                use_numpy_highlight = not use_numpy_highlight
            elif event.key == pygame.K_r:
                # demo bulk rebuild behavior
                rebuild_tree_bulk()
            elif event.key == pygame.K_c:
                balls.clear()
                qt.clear()

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            wx, wy = world_from_screen(mx, my, camera_x, camera_y)

            if event.button == 1:
                # LMB: add with auto-id (shows insert() auto-id)
                b = make_ball(wx, wy)
                new_id = qt.insert((b.x, b.y))
                # We still rely on list storage, so we must make list index == id.
                # For auto IDs, we extend list to fit.
                if new_id == len(balls):
                    balls.append(b)
                else:
                    # If user previously inserted with custom IDs and gaps exist,
                    # auto-id could land somewhere else. Make it work by growing list.
                    while len(balls) <= new_id:
                        balls.append(Ball(0, 0, 0, 0, 1, alive=False))
                    balls[new_id] = b

            if event.button == 3:
                # RMB: add with custom id_ == list index (canonical pattern)
                b = make_ball(wx, wy)
                new_id = len(balls)
                balls.append(b)
                qt.insert((b.x, b.y), id_=new_id)

    # Continuous input
    keys = pygame.key.get_pressed()

    # camera movement
    if keys[pygame.K_w]:
        camera_y -= CAM_SPEED
    if keys[pygame.K_s]:
        camera_y += CAM_SPEED
    if keys[pygame.K_a]:
        camera_x -= CAM_SPEED
    if keys[pygame.K_d]:
        camera_x += CAM_SPEED

    # move query rect
    qmin_x, qmin_y, qmax_x, qmax_y = query_rect
    if keys[pygame.K_RIGHT]:
        qmin_x += QUERY_SPEED
        qmax_x += QUERY_SPEED
    if keys[pygame.K_LEFT]:
        qmin_x -= QUERY_SPEED
        qmax_x -= QUERY_SPEED
    if keys[pygame.K_UP]:
        qmin_y -= QUERY_SPEED
        qmax_y -= QUERY_SPEED
    if keys[pygame.K_DOWN]:
        qmin_y += QUERY_SPEED
        qmax_y += QUERY_SPEED
    query_rect = (qmin_x, qmin_y, qmax_x, qmax_y)

    # Update simulation + keep quadtree in sync using update()
    for idx, b in enumerate(balls):
        if not b.alive:
            continue

        old_x, old_y = b.x, b.y
        b.update()

        # update quadtree
        qt.update_tuple(idx, (old_x, old_y), (b.x, b.y))

    # Mouse world position
    mx, my = pygame.mouse.get_pos()
    mouse_world = world_from_screen(mx, my, camera_x, camera_y)

    # Nearest neighbor to mouse
    nn_mouse = qt.nearest_neighbor(mouse_world)

    # Space: delete nearest-to-mouse using delete_tuple
    if keys[pygame.K_SPACE] and nn_mouse is not None:
        id_, x, y = nn_mouse
        if 0 <= id_ < len(balls) and balls[id_].alive:
            qt.delete_tuple(nn_mouse)  # <-- feature: delete_tuple((id,x,y))
            balls[id_].alive = False

    # Backspace: delete all inside query rect using query + delete(id_, x, y)
    if keys[pygame.K_BACKSPACE]:
        hits = qt.query(query_rect)
        for id_, x, y in hits:
            if 0 <= id_ < len(balls) and balls[id_].alive:
                qt.delete(id_, x, y)  # <-- feature: delete(id_, x, y)
                balls[id_].alive = False

    # Rendering
    screen.fill((0, 0, 0))

    # Draw quadtree boundaries
    draw_quadtree_boundaries()

    # Draw query rect
    pygame.draw.rect(
        screen, (255, 0, 255), rect_to_pygame(query_rect, camera_x, camera_y), 2
    )

    # Draw balls (payload store)
    for b in balls:
        if not b.alive:
            continue
        pygame.draw.circle(
            screen,
            (0, 220, 0),
            screen_from_world(b.x, b.y, camera_x, camera_y),
            b.radius,
        )

    # Highlight query hits using query()
    hits = qt.query(query_rect)
    for _id, x, y in hits:
        pygame.draw.circle(
            screen, (255, 0, 255), screen_from_world(x, y, camera_x, camera_y), 8, 5
        )

    # Optional numpy highlight for same rect
    try_numpy_highlight(query_rect)

    # Draw nearest-neighbor line to mouse
    if draw_nn and nn_mouse is not None:
        id_, x, y = nn_mouse
        if 0 <= id_ < len(balls) and balls[id_].alive:
            pygame.draw.line(
                screen,
                (255, 255, 255),
                screen_from_world(mouse_world[0], mouse_world[1], camera_x, camera_y),
                screen_from_world(x, y, camera_x, camera_y),
                2,
            )

    # Draw kNN lines from mouse (nearest_neighbors)
    if draw_knn:
        knn = qt.nearest_neighbors(mouse_world, knn_k)
        mpx, mpy = screen_from_world(mouse_world[0], mouse_world[1], camera_x, camera_y)
        for _id, x, y in knn:
            pygame.draw.line(
                screen,
                (120, 200, 255),
                (mpx, mpy),
                screen_from_world(x, y, camera_x, camera_y),
                1,
            )

    # Demonstrate __iter__ by sampling a few points and drawing tiny white dots
    # (Iterating the whole tree every frame is expensive; keep it small.)
    # Still shows that `for id_, x, y in qt:` works.
    for it_count, (_id, x, y) in enumerate(qt):
        pygame.draw.circle(
            screen, (200, 200, 200), screen_from_world(x, y, camera_x, camera_y), 2
        )
        if it_count >= 30:
            break

    draw_hud(fps, nn_mouse, mouse_world)

    pygame.display.set_caption(f"FPS: {fps:.1f}")
    pygame.display.flip()
