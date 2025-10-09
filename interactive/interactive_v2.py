import math
import random
from collections import deque

import pygame

from fastquadtree import QuadTree

pygame.init()
pygame.font.init()

# ------------------------------
# Window and world configuration
# ------------------------------
W, H = 1920, 1080
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("fastquadtree showcase")
clock = pygame.time.Clock()

WORLD_MIN_X, WORLD_MIN_Y = -1200, -800
WORLD_MAX_X, WORLD_MAX_Y = 1200, 800

# Camera
camera_x = -(W // 2)
camera_y = -(H // 2)
zoom = 1.0
zoom_target = 0.17
ZOOM_MIN, ZOOM_MAX = 0.01, 2.5
ZOOM_FACTOR = 1.15  # per key press or mouse wheel notch
ZOOM_SMOOTH = 10.0  # higher = snappier smoothing

# Pan speed in world units per second (at zoom 1.0)
PAN_SPEED = 1200.0


# ------------------------------
# Helpers for coords and drawing
# ------------------------------
def world_to_screen(x, y):
    return int((x - camera_x) * zoom), int((y - camera_y) * zoom)


def screen_rect_from_world(b):
    # b = (min_x, min_y, max_x, max_y)
    x0, y0 = world_to_screen(b[0], b[1])
    x1, y1 = world_to_screen(b[2], b[3])
    return pygame.Rect(x0, y0, x1 - x0, y1 - y0)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ------------------------------
# Pretty colors
# ------------------------------
COL_BG_1 = (8, 8, 12)
COL_BG_2 = (18, 18, 28)
COL_GRID = (28, 28, 42)
COL_NODE = (90, 90, 140)
COL_NODE_HOT = (150, 170, 255)
COL_QUERY_RECT = (255, 100, 255)
COL_QUERY_CIRC = (120, 200, 255)
COL_POINT = (120, 255, 170)
COL_POINT_DIM = (60, 160, 120)
COL_NN = (255, 255, 255)
COL_TEXT = (220, 230, 240)
COL_TRAIL = (90, 210, 180)

font = pygame.font.SysFont("Consolas", 12)

# ------------------------------
# Quadtree and actors
# ------------------------------
qtree = QuadTree(
    (WORLD_MIN_X, WORLD_MIN_Y, WORLD_MAX_X + 1, WORLD_MAX_Y + 1),
    6,
    max_depth=12,
    track_objects=True,
)


class Particle:
    __slots__ = ("r", "trail", "vx", "vy", "x", "y")

    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        ang = random.uniform(0, math.tau)
        # speed in world units per second
        spd = random.uniform(60.0, 160.0)
        self.vx = math.cos(ang) * spd
        self.vy = math.sin(ang) * spd
        self.r = random.randint(8, 16)
        self.trail = deque(maxlen=14)

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt

        if self.x < WORLD_MIN_X or self.x > WORLD_MAX_X:
            self.vx *= -1
            self.x = clamp(self.x, WORLD_MIN_X, WORLD_MAX_X)
        if self.y < WORLD_MIN_Y or self.y > WORLD_MAX_Y:
            self.vy *= -1
            self.y = clamp(self.y, WORLD_MIN_Y, WORLD_MAX_Y)

        self.trail.append((self.x, self.y))

    def draw(self):
        # trail (draw to alpha surface that we fully rebuild every frame)
        if len(self.trail) > 1:
            pts = [world_to_screen(px, py) for (px, py) in self.trail]
            n = len(pts)
            for i in range(1, n):
                t = i / n
                alpha = int(50 + 180 * t)
                col = (COL_TRAIL[0], COL_TRAIL[1], COL_TRAIL[2], alpha)
                pygame.draw.line(
                    trail_surf, col, pts[i - 1], pts[i], max(1, int(2 * zoom))
                )

        sx, sy = world_to_screen(self.x, self.y)
        pygame.draw.circle(screen, COL_POINT, (sx, sy), max(1, int(self.r * zoom)))


# Seed particles
random.seed(2)
initial_particles = [
    Particle(
        random.uniform(WORLD_MIN_X * 0.9, WORLD_MAX_X * 0.9),
        random.uniform(WORLD_MIN_Y * 0.9, WORLD_MAX_Y * 0.9),
    )
    for _ in range(60)
]

for p in initial_particles:
    qtree.insert((p.x, p.y), obj=p)

# Offscreen surfaces
trail_surf = pygame.Surface((W, H), pygame.SRCALPHA)
node_surf = pygame.Surface((W, H), pygame.SRCALPHA)


# ------------------------------
# Controllable rectangle
# ------------------------------
# Rectangle size (half-width and half-height)
rect_half_width = 200.0
rect_half_height = 150.0
RECT_MIN_SIZE = 50.0
RECT_MAX_SIZE = 800.0
RECT_RESIZE_SPEED = 200.0  # pixels per second


def screen_to_world(screen_x, screen_y):
    """Convert screen coordinates to world coordinates"""
    world_x = screen_x / max(zoom, 1e-6) + camera_x
    world_y = screen_y / max(zoom, 1e-6) + camera_y
    return world_x, world_y


def get_mouse_rect():
    """Get rectangle centered at mouse position in world coordinates"""
    mx, my = pygame.mouse.get_pos()
    cx, cy = screen_to_world(mx, my)
    return (
        cx - rect_half_width,
        cy - rect_half_height,
        cx + rect_half_width,
        cy + rect_half_height,
    )


def draw_grid():
    step = 100
    s = step * zoom
    if s < 30:
        return
    ox = -int((camera_x * zoom) % s)
    oy = -int((camera_y * zoom) % s)
    for x in range(ox, W, int(s)):
        pygame.draw.line(screen, COL_GRID, (x, 0), (x, H), 1)
    for y in range(oy, H, int(s)):
        pygame.draw.line(screen, COL_GRID, (0, y), (W, y), 1)


def draw_nodes(query_rect):
    rects = qtree.get_all_node_boundaries()
    qminx, qminy, qmaxx, qmaxy = query_rect
    lw = max(1, int(1.25 * zoom))
    for r in rects:
        srect = screen_rect_from_world(r[:4])
        inter = not (r[2] < qminx or r[0] > qmaxx or r[3] < qminy or r[1] > qmaxy)
        base = COL_NODE_HOT if inter else COL_NODE
        # soft fill plus brighter outline
        fill_a = 50 if inter else 30
        line_a = 220 if inter else 140
        pygame.draw.rect(node_surf, (*base, fill_a), srect)
        pygame.draw.rect(node_surf, (*base, line_a), srect, lw)


def draw_query_rect(rect, blink):
    srect = screen_rect_from_world(rect)
    w = max(2, int(2 * zoom)) if blink else max(1, int(1 * zoom))
    pygame.draw.rect(screen, COL_QUERY_RECT, srect, w)


def draw_query_circle(cx, cy, r, blink):
    sx, sy = world_to_screen(cx, cy)
    sr = max(2, int(r * zoom))
    w = max(2, int(2 * zoom)) if blink else max(1, int(1 * zoom))
    pygame.draw.circle(screen, COL_QUERY_CIRC, (sx, sy), sr, w)


def draw_nn_rays():
    for obj in list(qtree.get_all_objects()):
        wx, wy = obj.x, obj.y
        out = qtree.nearest_neighbors((wx, wy), 2, as_items=True)
        if len(out) < 2:
            continue
        nn = out[1].obj  # second nearest (first is self)
        if nn is None:
            continue
        x0, y0 = world_to_screen(wx, wy)
        x1, y1 = world_to_screen(nn.x, nn.y)
        pygame.draw.line(screen, COL_NN, (x0, y0), (x1, y1), 2)


def hud(text_lines):
    y = 8
    for s in text_lines:
        img = font.render(s, True, COL_TEXT)
        screen.blit(img, (10, y))
        y += 12


# ------------------------------
# Event and input handling
# ------------------------------
def handle_events(running, paused, show_nodes, show_nn, show_trails, zoom_target):
    """Handle discrete events like key presses and mouse clicks."""
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            running = False
        elif ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_ESCAPE:
                running = False
            elif ev.key == pygame.K_SPACE:
                paused = not paused
            elif ev.key == pygame.K_1:
                show_nodes = not show_nodes
            elif ev.key == pygame.K_2:
                show_nn = not show_nn
            elif ev.key == pygame.K_3:
                show_trails = not show_trails
            elif ev.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                zoom_target = clamp(zoom_target * ZOOM_FACTOR, ZOOM_MIN, ZOOM_MAX)
            elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                zoom_target = clamp(zoom_target / ZOOM_FACTOR, ZOOM_MIN, ZOOM_MAX)

        elif ev.type == pygame.MOUSEWHEEL:
            if ev.y != 0:
                factor = ZOOM_FACTOR**ev.y
                zoom_target = clamp(zoom_target * factor, ZOOM_MIN, ZOOM_MAX)

        # Left click
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mx, my = ev.pos
            wx = mx / max(zoom, 1e-6) + camera_x
            wy = my / max(zoom, 1e-6) + camera_y
            if WORLD_MIN_X <= wx < WORLD_MAX_X and WORLD_MIN_Y <= wy < WORLD_MAX_Y:
                p = Particle(wx, wy)
                qtree.insert((p.x, p.y), obj=p)

        # Right click
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 3:
            mx, my = ev.pos
            wx = mx / max(zoom, 1e-6) + camera_x
            wy = my / max(zoom, 1e-6) + camera_y
            nn = qtree.nearest_neighbor((wx, wy), as_item=True)
            if nn is not None:
                qtree.delete_by_object(nn.obj)

    return running, paused, show_nodes, show_nn, show_trails, zoom_target


def handle_continuous_input(
    dt, zoom, zoom_target, camera_x, camera_y, rect_half_width, rect_half_height
):
    """Handle continuous input like held keys and mouse buttons."""
    mx, my = pygame.mouse.get_pos()

    # Smooth zoom toward target and keep the mouse-anchored world point fixed
    wx_anchor = camera_x + mx / max(zoom, 1e-6)
    wy_anchor = camera_y + my / max(zoom, 1e-6)

    # Add a point instantly with left mouse button held + shift
    if pygame.mouse.get_pressed(3)[0] and pygame.key.get_mods() & pygame.KMOD_SHIFT:
        wx = mx / max(zoom, 1e-6) + camera_x
        wy = my / max(zoom, 1e-6) + camera_y
        if WORLD_MIN_X <= wx < WORLD_MAX_X and WORLD_MIN_Y <= wy < WORLD_MAX_Y:
            p = Particle(wx, wy)
            qtree.insert((p.x, p.y), obj=p)

    # Remove nearest point with right click
    if pygame.mouse.get_pressed(3)[2] and pygame.key.get_mods() & pygame.KMOD_SHIFT:
        wx = mx / max(zoom, 1e-6) + camera_x
        wy = my / max(zoom, 1e-6) + camera_y
        nn = qtree.nearest_neighbor((wx, wy), as_item=True)
        if nn is not None:
            qtree.delete_by_object(nn.obj)

    zoom += (zoom_target - zoom) * min(1.0, ZOOM_SMOOTH * dt)
    zoom = clamp(zoom, ZOOM_MIN, ZOOM_MAX)

    camera_x = wx_anchor - mx / max(zoom, 1e-6)
    camera_y = wy_anchor - my / max(zoom, 1e-6)

    # Rectangle size controls with arrow keys
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        rect_half_height = min(RECT_MAX_SIZE, rect_half_height + RECT_RESIZE_SPEED * dt)
    if keys[pygame.K_DOWN]:
        rect_half_height = max(RECT_MIN_SIZE, rect_half_height - RECT_RESIZE_SPEED * dt)
    if keys[pygame.K_RIGHT]:
        rect_half_width = min(RECT_MAX_SIZE, rect_half_width + RECT_RESIZE_SPEED * dt)
    if keys[pygame.K_LEFT]:
        rect_half_width = max(RECT_MIN_SIZE, rect_half_width - RECT_RESIZE_SPEED * dt)

    # Camera pan with WASD only (dt and zoom awareness)
    dx = keys[pygame.K_d] - keys[pygame.K_a]
    dy = keys[pygame.K_s] - keys[pygame.K_w]
    if dx or dy:
        camera_x += (dx * PAN_SPEED * dt) / max(zoom, 1e-6)
        camera_y += (dy * PAN_SPEED * dt) / max(zoom, 1e-6)

    return zoom, camera_x, camera_y, rect_half_width, rect_half_height


# ------------------------------
# Main loop
# ------------------------------
def main():
    global camera_x, camera_y, zoom, zoom_target, rect_half_width, rect_half_height
    running = True
    paused = False
    show_nodes = True
    show_nn = False
    show_trails = True
    t = 0.0

    while running:
        dt = clock.tick(1e6) / 1000.0  # seconds
        fps = clock.get_fps()
        t += dt

        # Handle discrete events
        running, paused, show_nodes, show_nn, show_trails, zoom_target = handle_events(
            running, paused, show_nodes, show_nn, show_trails, zoom_target
        )

        # Handle continuous input
        zoom, camera_x, camera_y, rect_half_width, rect_half_height = (
            handle_continuous_input(
                dt,
                zoom,
                zoom_target,
                camera_x,
                camera_y,
                rect_half_width,
                rect_half_height,
            )
        )

        # Update particles and quadtree
        if not paused:
            objs = list(qtree.get_all_objects())
            for p in objs:
                p.update(dt)
                qtree.delete_by_object(p)
                qtree.insert((p.x, p.y), obj=p)

        # Mouse-following rectangle
        rect_q = get_mouse_rect()
        rect_hits = qtree.query(rect_q, as_items=True)

        # ------------ Rendering ------------
        screen.fill(COL_BG_1)
        pygame.draw.rect(screen, COL_BG_2, pygame.Rect(0, 0, W, H), 0)

        # Node layer
        node_surf.fill((0, 0, 0, 0))
        if show_nodes:
            draw_nodes(rect_q)
            screen.blit(node_surf, (0, 0))

        # Trails layer: rebuild every frame in world space
        trail_surf.fill((0, 0, 0, 0))

        # Highlight hits under queries first
        for item in rect_hits:
            if item.obj is None:
                continue
            sx, sy = world_to_screen(item.x, item.y)
            radius = (item.obj.r + 8) * zoom
            pygame.draw.circle(
                screen,
                COL_QUERY_RECT,
                (sx, sy),
                max(1, int(radius)),
                max(1, int(3 * zoom)),
            )

        # Draw particles and their trails
        for p in qtree.get_all_objects():
            p.draw()
        if show_trails:
            screen.blit(trail_surf, (0, 0))

        # Query shapes
        blink = (pygame.time.get_ticks() // 400) % 2 == 0
        draw_query_rect(rect_q, blink)

        # Optional NN rays
        if show_nn:
            draw_nn_rays()

        # HUD
        hud(
            [
                f"FPS: {fps:.1f}",
                f"particles: {len(qtree.get_all_objects())}",
                f"rect hits: {len(rect_hits)}",
                f"rect size: {rect_half_width * 2:.0f}x{rect_half_height * 2:.0f}",
                f"zoom: {zoom:.2f}  target: {zoom_target:.2f}",
                "WASD to pan. Mouse wheel or +/- to zoom.",
                "arrow keys to resize rectangle.",
                "1 nodes. 2 NN rays. 3 trails. SPACE pause. L-click add. R-click remove (shift to repeat)",
            ]
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
