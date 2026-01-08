import numpy as np
import pytest

from fastquadtree.point_quadtree import QuadTree


def test_f32_containment():
    bounds = (0.0, 0.0, 1000.0, 1000.0)
    qt = QuadTree(bounds, capacity=4, dtype="f32")
    pts = [
        (50, 50),
        (100, 100),
        (150, 150),
        (200, 200),
        (350.0, 260.0),
    ]
    qt.insert_many(pts)

    got = qt.query((50.0, 50.0, 51.0, 51.0))
    assert len(got) == 1

    got = qt.query((350.0, 260.0, 351.0, 261.0))
    assert len(got) == 1


def test_f32_can_round_bbox_min_up_and_exclude_integer_point():
    offset = 1e8
    x = 261.0  # integer point, exactly representable in f32 and f64

    # Simulate "world -> local" by subtracting a large origin.
    # In float32, this suffers catastrophic cancellation and rounds to a multiple of 8 here.
    min_x32 = np.float32(offset + x) - np.float32(offset)  # typically becomes 264.0
    min_x64 = np.float64(offset + x) - np.float64(offset)  # stays 261.0

    assert float(min_x64) == x
    assert float(min_x32) != x  # demonstrates the f32 rounding drift

    # BBox that "should" start at x, but in f32 it starts at 264 and excludes x=261
    bbox64 = (float(min_x64), 0.0, float(min_x64) + 10.0, 10.0)
    bbox32 = (float(min_x32), 0.0, float(min_x32) + 10.0, 10.0)

    qt64 = QuadTree(bbox64, 1, max_depth=8, dtype="f64")
    qt64.insert((x, 5.0))  # should succeed

    qt32 = QuadTree(bbox32, 1, max_depth=8, dtype="f32")
    with pytest.raises(ValueError):
        qt32.insert((np.float32(x), np.float32(5.0)))  # should fail (x < min_x32)
