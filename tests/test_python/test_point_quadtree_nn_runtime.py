import pytest

from fastquadtree._base_quadtree import Bounds
from fastquadtree.point_quadtree import QuadTree as PointQT


class _DummyNN:
    """Native stub that only implements NN calls."""

    def __init__(self, nn=None, knn=None):
        self._nn = nn
        self._knn = knn

    def nearest_neighbor(self, xy):
        return self._nn

    def nearest_neighbors(self, xy, k):
        return list(self._knn or [])


B: Bounds = (0.0, 0.0, 10.0, 10.0)
Q = (5.0, 5.0)


def test_nearest_neighbor_missing_tracked_item_raises_runtime(monkeypatch):
    # Force the native to return an ID that is not tracked by the Python store
    def _fake_new_native(self, bounds, capacity, max_depth):
        return _DummyNN(nn=(42, 1.0, 2.0))  # id 42 is not in the store

    monkeypatch.setattr(PointQT, "_new_native", _fake_new_native)

    qt = PointQT(bounds=B, capacity=4, track_objects=True)
    with pytest.raises(RuntimeError, match="missing tracked item"):
        _ = qt.nearest_neighbor(Q, as_item=True)


def test_nearest_neighbors_missing_tracked_item_raises_runtime(monkeypatch):
    # Force the native to return a list with an untracked id
    def _fake_new_native(self, bounds, capacity, max_depth):
        return _DummyNN(knn=[(7, 9.0, 9.0), (8, 1.0, 1.0)])

    monkeypatch.setattr(PointQT, "_new_native", _fake_new_native)

    qt = PointQT(bounds=B, capacity=4, track_objects=True)
    with pytest.raises(RuntimeError, match="missing tracked item"):
        _ = qt.nearest_neighbors(Q, 2, as_items=True)
