import struct
from typing import Any, Iterable, Sequence, Tuple

import numpy as np
import pytest

from fastquadtree._common import (
    DTYPE_BOUNDS_SIZE_BYTES,
    Bounds,
    QuadTreeDType,
    parse_container,
)

DEFAULT_BOUNDS = (0.0, 0.0, 100.0, 100.0)
DTYPE_TO_NP = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64}


def get_bounds_for_dtype(bounds: Bounds, dtype: QuadTreeDType) -> Bounds:
    """Convert bounds to appropriate type for dtype."""
    if dtype.startswith("i"):
        bound_use = tuple(map(int, bounds))
        assert len(bound_use) == 4
        return bound_use
    return bounds


@pytest.fixture
def bounds() -> Tuple[float, float, float, float]:
    return DEFAULT_BOUNDS


@pytest.fixture(params=["f32", "f64", "i32", "i64"])
def dtype(request: pytest.FixtureRequest) -> str:
    return request.param


def insert_point_grid(
    qt: Any,
    *,
    rows: int = 3,
    cols: int = 3,
    offset: float = 0.5,
    spacing: float = 1.0,
    with_objects: bool = False,
) -> list[tuple]:
    """
    Insert a dense grid of points. Returns inserted metadata:
    - non-object trees: (id, (x, y))
    - object trees: (id, (x, y), obj)
    """
    out: list[tuple] = []
    for i in range(rows):
        for j in range(cols):
            x = offset + i * spacing
            y = offset + j * spacing
            if with_objects:
                obj = {"i": i, "j": j}
                rid = qt.insert((x, y), obj)
                out.append((rid, (x, y), obj))
            else:
                rid = qt.insert((x, y))
                out.append((rid, (x, y)))
    return out


def insert_rect_grid(
    qt: Any,
    *,
    rows: int = 2,
    cols: int = 2,
    size: float = 1.0,
    spacing: float = 2.0,
    offset: float = 0.5,
    with_objects: bool = False,
) -> list[tuple]:
    """
    Insert a small grid of rectangles. Returns inserted metadata:
    - non-object trees: (id, bounds)
    - object trees: (id, bounds, obj)
    """
    out: list[tuple] = []
    for i in range(rows):
        for j in range(cols):
            min_x = offset + i * spacing
            min_y = offset + j * spacing
            rect = (min_x, min_y, min_x + size, min_y + size)
            if with_objects:
                obj = {"i": i, "j": j}
                rid = qt.insert(rect, obj)
                out.append((rid, rect, obj))
            else:
                rid = qt.insert(rect)
                out.append((rid, rect))
    return out


def assert_query_matches_np(qt: Any, rect: Sequence[float]) -> None:
    """
    Assert that query() list output matches query_np() output for IDs and coords.
    Works for both tuple-returning and Item-returning classes.
    """
    results = qt.query(rect)
    ids_list: list[int] = []
    coords_list: list[tuple] = []
    for item in results:
        if hasattr(item, "id_"):
            ids_list.append(int(item.id_))
            coords_list.append(tuple(item.geom))  # type: ignore[arg-type]
        else:
            ids_list.append(int(item[0]))
            coords_list.append(tuple(item[1:]))  # type: ignore[index]

    ids_np, coords_np = qt.query_np(rect)
    assert list(ids_np) == ids_list
    coords_np_list = [tuple(map(float, row)) for row in coords_np.tolist()]
    coords_list_float = [tuple(map(float, row)) for row in coords_list]
    assert coords_np_list == coords_list_float


def make_np_coords(dtype: str, rows: Iterable[Sequence[float]]) -> np.ndarray:
    """Build a NumPy array with dtype matching the quadtree dtype."""
    return np.array(list(rows), dtype=DTYPE_TO_NP[dtype])


def corrupt_magic(data: bytes) -> bytes:
    """Flip the serialization magic header to trigger parse errors."""
    return b"BADC" + data[4:]


def truncate_bytes(data: bytes, count: int = 1) -> bytes:
    """Drop the last `count` bytes from a payload."""
    return data[:-count] if count else data


def inflate_core_length(data: bytes, extra: int = 1) -> bytes:
    """
    Increase the encoded core length without adding bytes to force truncation errors.
    """
    parsed = parse_container(data)
    off = 30  # fixed header through count
    if parsed["max_depth"] is not None:
        off += 4
    off += DTYPE_BOUNDS_SIZE_BYTES[parsed["dtype"]]
    buf = bytearray(data)
    (core_len,) = struct.unpack_from("<I", buf, off)
    struct.pack_into("<I", buf, off, core_len + extra)
    return bytes(buf)
