import pytest
from tests.test_python.conftest import (
    corrupt_magic,
    get_bounds_for_dtype,
    inflate_core_length,
    truncate_bytes,
)

from fastquadtree._common import (
    FLAG_CORE_CODEC_WINCODE,
    SERIALIZATION_FORMAT_VERSION,
    UNSUPPORTED_BINCODE_MESSAGE,
    SerializationError,
    build_container,
    parse_container,
)
from fastquadtree.rect_quadtree import RectQuadTree


def test_serialization_round_trip(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTree(bounds_use, capacity=5, max_depth=7, dtype=dtype)
    r1 = (1, 1, 2, 2) if dtype.startswith("i") else (1.0, 1.0, 2.0, 2.0)
    r2 = (3, 3, 4, 4) if dtype.startswith("i") else (3.0, 3.0, 4.0, 4.0)
    r3 = (9, 9, 10, 10) if dtype.startswith("i") else (9.0, 9.0, 10.0, 10.0)

    rqt.insert(r1)
    rqt.insert(r2)
    rqt.insert(r3, id_=99)

    data = rqt.to_bytes()
    parsed = parse_container(data)
    assert parsed["fmt_ver"] == SERIALIZATION_FORMAT_VERSION
    assert parsed["flags"] & FLAG_CORE_CODEC_WINCODE

    clone = RectQuadTree.from_bytes(data)

    assert clone._dtype == dtype
    assert clone._bounds == bounds_use
    assert clone._capacity == 5
    assert clone._max_depth == 7
    assert clone._next_id == rqt._next_id
    assert len(clone) == len(rqt)
    expected = {
        tuple(map(float, r1)),
        tuple(map(float, r2)),
        tuple(map(float, r3)),
    }
    assert {tuple(r[1:]) for r in clone.query(bounds_use)} == expected


def test_serialization_errors(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTree(bounds_use, capacity=2, dtype=dtype)
    rect = (1, 1, 2, 2) if dtype.startswith("i") else (1.0, 1.0, 2.0, 2.0)
    rqt.insert(rect)
    data = rqt.to_bytes()

    for bad in (
        corrupt_magic(data),
        truncate_bytes(data, 2),
        inflate_core_length(data, extra=3),
    ):
        with pytest.raises(SerializationError):
            RectQuadTree.from_bytes(bad)


def test_from_bytes_rejects_legacy_bincode_container(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    data = build_container(
        fmt_ver=1,
        dtype=dtype,
        flags=0,
        capacity=2,
        max_depth=None,
        next_id=0,
        count=0,
        bounds=bounds_use,
        core=b"legacy-bincode-core",
    )

    with pytest.raises(SerializationError, match=UNSUPPORTED_BINCODE_MESSAGE):
        RectQuadTree.from_bytes(data)
