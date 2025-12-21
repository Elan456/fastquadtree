import pytest
from tests.test_python.conftest import (
    corrupt_magic,
    get_bounds_for_dtype,
    inflate_core_length,
    truncate_bytes,
)

from fastquadtree._common import SerializationError
from fastquadtree.point_quadtree import QuadTree


def test_to_bytes_from_bytes_round_trip_preserves_state(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTree(bounds_use, capacity=4, max_depth=5, dtype=dtype)
    pt1 = (1, 1) if dtype.startswith("i") else (1.0, 1.0)
    pt2 = (2, 2) if dtype.startswith("i") else (2.0, 2.0)
    qt.insert(pt1)
    qt.insert(pt2)
    # Advance next_id with custom id + auto to ensure value persists
    pt_custom = (9, 9) if dtype.startswith("i") else (9.0, 9.0)
    qt.insert(pt_custom, id_=50)
    pt3 = (3, 3) if dtype.startswith("i") else (3.0, 3.0)
    qt.insert(pt3)

    data = qt.to_bytes()
    clone = QuadTree.from_bytes(data)

    assert clone._dtype == dtype
    assert clone._bounds == bounds_use
    assert clone._capacity == 4
    assert clone._max_depth == 5
    assert clone._next_id == qt._next_id
    assert len(clone) == len(qt)
    expected = {
        (float(pt1[0]), float(pt1[1])),
        (float(pt2[0]), float(pt2[1])),
        (float(pt3[0]), float(pt3[1])),
        (float(pt_custom[0]), float(pt_custom[1])),
    }
    assert {pt[1:] for pt in clone.query(bounds_use)} == expected


def test_from_bytes_rejects_corrupted_payload(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTree(bounds_use, capacity=2, dtype=dtype)
    qt.insert((1, 1) if dtype.startswith("i") else (1.0, 1.0))
    data = qt.to_bytes()

    for bad in (
        corrupt_magic(data),
        truncate_bytes(data, 5),
        inflate_core_length(data, extra=10),
    ):
        with pytest.raises(SerializationError):
            QuadTree.from_bytes(bad)
