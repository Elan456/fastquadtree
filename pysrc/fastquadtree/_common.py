"""Common utilities and constants shared across quadtree implementations."""

from __future__ import annotations

import math
from typing import Any, Final, Literal, Union, overload

SERIALIZATION_FORMAT_VERSION: Final[int] = 1

# Serialization container constants (shared by Objects and non-Objects trees)
SERIALIZATION_MAGIC: Final[bytes] = b"FQT0"  # fastquadtree container v0
SERIALIZATION_ENDIANNESS: Final[str] = "<"  # little-endian
SECTION_ITEMS: int = 1  # safe: ids + geometry only
SECTION_OBJECTS: int = 2  # unsafe: pickle payload (opt-in)

# Type aliases
Bounds = tuple[
    Union[float, int], Union[float, int], Union[float, int], Union[float, int]
]
"""Type alias for axis-aligned rectangle bounds as (min_x, min_y, max_x, max_y)."""

Point = tuple[Union[float, int], Union[float, int]]
"""Type alias for 2D point coordinates as (x, y)."""

# Dtype mappings
QuadTreeDType = Literal["f32", "f64", "i32", "i64"]

QUADTREE_DTYPE_TO_NP_DTYPE: Final[dict[QuadTreeDType, str]] = {
    "f32": "float32",
    "f64": "float64",
    "i32": "int32",
    "i64": "int64",
}
"""Mapping from quadtree dtype strings to NumPy dtype strings."""

# Dtype codes for on-disk format
DTYPE_TO_CODE: Final[dict[QuadTreeDType, int]] = {
    "f32": 0,
    "f64": 1,
    "i32": 2,
    "i64": 3,
}
CODE_TO_DTYPE: Final[dict[int, QuadTreeDType]] = {
    0: "f32",
    1: "f64",
    2: "i32",
    3: "i64",
}

# Binary sizes for bounds encoding
DTYPE_BOUNDS_SIZE_BYTES: Final[dict[QuadTreeDType, int]] = {
    "f32": 16,
    "f64": 32,
    "i32": 16,
    "i64": 32,
}


def _is_np_array(x: Any) -> bool:
    """
    Check if an object is a NumPy array without importing NumPy.

    This allows dtype checking without forcing NumPy as a hard dependency.

    Args:
        x: Object to check.

    Returns:
        True if x appears to be a NumPy array.
    """
    mod = getattr(x.__class__, "__module__", "")
    return mod.startswith("numpy") and hasattr(x, "ndim") and hasattr(x, "shape")


def validate_bounds(bounds: Any) -> Bounds:
    """
    Validate and normalize bounds to a tuple.

    Args:
        bounds: Bounds specification to validate.

    Returns:
        Normalized bounds tuple (min_x, min_y, max_x, max_y).

    Raises:
        ValueError: If bounds is not a 4-element sequence of finite numbers with min < max.
    """
    if type(bounds) is not tuple:
        bounds = tuple(bounds)
    if len(bounds) != 4:
        raise ValueError(
            "bounds must be a tuple of four numeric values (min_x, min_y, max_x, max_y)"
        )

    numeric_vals: list[float | int] = []
    floats_for_checks: list[float] = []
    for v in bounds:
        if not isinstance(v, (int, float)):
            raise ValueError("bounds must contain numeric values")
        numeric_vals.append(
            int(v) if isinstance(v, int) and not isinstance(v, bool) else float(v)
        )
        floats_for_checks.append(float(v))

    if not all(math.isfinite(v) for v in floats_for_checks):
        raise ValueError("bounds must be finite numbers")

    min_x, min_y, max_x, max_y = floats_for_checks
    if min_x >= max_x or min_y >= max_y:
        raise ValueError("bounds must satisfy min < max for both axes")

    return tuple(numeric_vals)  # type: ignore[return-value]


def validate_np_dtype(geoms: Any, expected_dtype: QuadTreeDType) -> None:
    """
    Validate that a NumPy array's dtype matches the expected dtype.

    Args:
        geoms: NumPy array to validate.
        expected_dtype: Expected quadtree dtype string.

    Raises:
        TypeError: If the array's dtype doesn't match the expected dtype.
    """
    expected_np_dtype = QUADTREE_DTYPE_TO_NP_DTYPE.get(expected_dtype)
    if expected_np_dtype is None:
        raise TypeError(f"Unknown quadtree dtype {expected_dtype!r}")
    if str(getattr(geoms, "dtype", None)) != expected_np_dtype:
        raise TypeError(
            f"NumPy array dtype {getattr(geoms, 'dtype', None)} does not match quadtree dtype {expected_dtype}"
        )


# ---------------------------
# Serialization helpers
# ---------------------------


class SerializationError(ValueError):
    """Exception raised when serialized data is malformed or incompatible with the current version."""


def dtype_to_code(dtype: str) -> int:
    try:
        return DTYPE_TO_CODE[dtype]  # type: ignore[index]
    except KeyError:
        raise SerializationError(f"Unsupported dtype: {dtype!r}") from None


def code_to_dtype(code: int) -> QuadTreeDType:
    try:
        return CODE_TO_DTYPE[int(code)]
    except KeyError:
        raise SerializationError(f"Unknown dtype code: {code!r}") from None


@overload
def pack_bounds(bounds: Bounds, dtype: Literal["f32"]) -> bytes: ...
@overload
def pack_bounds(bounds: Bounds, dtype: Literal["f64"]) -> bytes: ...
@overload
def pack_bounds(bounds: Bounds, dtype: Literal["i32"]) -> bytes: ...
@overload
def pack_bounds(bounds: Bounds, dtype: Literal["i64"]) -> bytes: ...


def pack_bounds(bounds: Bounds, dtype: QuadTreeDType) -> bytes:
    """
    Encode bounds as binary data using the specified dtype.

    This ensures bounds round-trip consistently with dtype semantics in little-endian format.

    Args:
        bounds: Bounds tuple (min_x, min_y, max_x, max_y).
        dtype: Data type for encoding ('f32', 'f64', 'i32', 'i64').

    Returns:
        Binary representation of the bounds.

    Raises:
        SerializationError: If dtype is unsupported.
    """
    import struct

    b0, b1, b2, b3 = bounds
    if dtype == "f32":
        return struct.pack(
            f"{SERIALIZATION_ENDIANNESS}4f", float(b0), float(b1), float(b2), float(b3)
        )
    if dtype == "f64":
        return struct.pack(
            f"{SERIALIZATION_ENDIANNESS}4d", float(b0), float(b1), float(b2), float(b3)
        )
    if dtype == "i32":
        return struct.pack(
            f"{SERIALIZATION_ENDIANNESS}4i", int(b0), int(b1), int(b2), int(b3)
        )
    if dtype == "i64":
        return struct.pack(
            f"{SERIALIZATION_ENDIANNESS}4q", int(b0), int(b1), int(b2), int(b3)
        )
    raise SerializationError(f"Unsupported dtype for bounds packing: {dtype!r}")


def unpack_bounds(
    buf: memoryview, offset: int, dtype: QuadTreeDType
) -> tuple[Bounds, int]:
    """
    Decode bounds from binary data.

    Args:
        buf: Memory buffer containing the serialized data.
        offset: Starting position in the buffer.
        dtype: Data type for decoding ('f32', 'f64', 'i32', 'i64').

    Returns:
        Tuple of (bounds, new_offset) where bounds is the decoded tuple
        and new_offset is the position after reading.

    Raises:
        SerializationError: If data is too short or dtype is unsupported.
    """
    import struct

    size = DTYPE_BOUNDS_SIZE_BYTES[dtype]
    if len(buf) < offset + size:
        raise SerializationError("Data too short while reading bounds")

    if dtype == "f32":
        vals = struct.unpack_from(f"{SERIALIZATION_ENDIANNESS}4f", buf, offset)
        bounds: Bounds = (
            float(vals[0]),
            float(vals[1]),
            float(vals[2]),
            float(vals[3]),
        )
    elif dtype == "f64":
        vals = struct.unpack_from(f"{SERIALIZATION_ENDIANNESS}4d", buf, offset)
        bounds = (float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]))
    elif dtype == "i32":
        vals = struct.unpack_from(f"{SERIALIZATION_ENDIANNESS}4i", buf, offset)
        bounds = (
            float(int(vals[0])),
            float(int(vals[1])),
            float(int(vals[2])),
            float(int(vals[3])),
        )
    elif dtype == "i64":
        vals = struct.unpack_from(f"{SERIALIZATION_ENDIANNESS}4q", buf, offset)
        bounds = (
            float(int(vals[0])),
            float(int(vals[1])),
            float(int(vals[2])),
            float(int(vals[3])),
        )
    else:
        raise SerializationError(f"Unsupported dtype for bounds unpacking: {dtype!r}")

    return bounds, offset + size


def build_container(
    *,
    fmt_ver: int,
    dtype: QuadTreeDType,
    flags: int,
    capacity: int,
    max_depth: int | None,
    next_id: int,
    count: int,
    bounds: Bounds,
    core: bytes,
    extra_sections: list[tuple[int, bytes]] | None = None,
) -> bytes:
    """
    Build the fastquadtree serialization container.

    This function creates a binary container with header, metadata, and optional sections.

    Args:
        fmt_ver: Serialization format version.
        dtype: Quadtree data type.
        flags: Serialization flags bitfield.
        capacity: Tree capacity value.
        max_depth: Optional maximum tree depth.
        next_id: Next auto-assigned ID.
        count: Number of items in the tree.
        bounds: Tree bounds.
        core: Core tree data as bytes.
        extra_sections: Optional list of (section_type, payload_bytes) for additional data.

    Returns:
        Complete serialized container as bytes.

    Raises:
        SerializationError: If data is invalid or too large.
    """
    import struct

    if not isinstance(core, (bytes, bytearray, memoryview)):
        raise TypeError("core must be bytes-like")
    core_bytes = bytes(core)
    if len(core_bytes) > 0xFFFFFFFF:
        raise SerializationError("core payload too large (>4GiB)")

    if extra_sections is None:
        extra_sections = []

    if len(extra_sections) > 0xFFFF:
        raise SerializationError("too many extra sections")

    header = struct.pack(
        f"{SERIALIZATION_ENDIANNESS}4sHHBBIQQ",
        SERIALIZATION_MAGIC,
        int(fmt_ver),
        int(flags),
        int(dtype_to_code(dtype)),
        0,  # reserved
        int(capacity),
        int(next_id),
        int(count),
    )

    md_bytes = b""
    if flags & 1:
        if max_depth is None:
            raise SerializationError("max_depth flag set but max_depth is None")
        if int(max_depth) < 0:
            raise SerializationError("max_depth must be >= 0")
        md_bytes = struct.pack(f"{SERIALIZATION_ENDIANNESS}i", int(max_depth))

    bounds_bytes = pack_bounds(bounds, dtype)
    core_len = struct.pack(f"{SERIALIZATION_ENDIANNESS}I", len(core_bytes))

    # sections: u16 num_sections then repeated (u16 type, u32 len, bytes payload)
    sections_blob = b""
    if extra_sections:
        parts: list[bytes] = [
            struct.pack(f"{SERIALIZATION_ENDIANNESS}H", len(extra_sections))
        ]
        for section_type, payload in extra_sections:
            payload_b = bytes(payload)
            if len(payload_b) > 0xFFFFFFFF:
                raise SerializationError("section payload too large (>4GiB)")
            parts.append(
                struct.pack(
                    f"{SERIALIZATION_ENDIANNESS}HI", int(section_type), len(payload_b)
                )
            )
            parts.append(payload_b)
        sections_blob = b"".join(parts)
    else:
        sections_blob = struct.pack(f"{SERIALIZATION_ENDIANNESS}H", 0)

    return b"".join(
        [header, md_bytes, bounds_bytes, core_len, core_bytes, sections_blob]
    )


def parse_container(data: bytes) -> dict[str, Any]:
    """
    Parse a fastquadtree serialization container.

    Args:
        data: Serialized container bytes.

    Returns:
        Dictionary containing parsed fields:
            - magic: Magic header bytes
            - fmt_ver: Format version
            - flags: Flags bitfield
            - dtype: Quadtree data type
            - capacity: Tree capacity
            - max_depth: Optional maximum depth
            - next_id: Next auto-assigned ID
            - count: Number of items
            - bounds: Tree bounds
            - core: Core tree data
            - sections: List of (section_type, payload) tuples

    Raises:
        SerializationError: If data is malformed or has invalid magic header.
    """
    import struct

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    buf = memoryview(data)

    # Must at least contain fixed header fields up through count
    if len(buf) < 30:
        raise SerializationError("Data too short to be a fastquadtree serialization")

    off = 0
    magic = bytes(buf[off : off + 4])
    off += 4
    if magic != SERIALIZATION_MAGIC:
        raise SerializationError("Invalid magic header for fastquadtree serialization")

    (fmt_ver,) = struct.unpack_from(f"{SERIALIZATION_ENDIANNESS}H", buf, off)
    off += 2
    (flags,) = struct.unpack_from(f"{SERIALIZATION_ENDIANNESS}H", buf, off)
    off += 2
    (dtype_code,) = struct.unpack_from(f"{SERIALIZATION_ENDIANNESS}B", buf, off)
    off += 1
    off += 1  # reserved
    (capacity,) = struct.unpack_from(f"{SERIALIZATION_ENDIANNESS}I", buf, off)
    off += 4
    (next_id,) = struct.unpack_from(f"{SERIALIZATION_ENDIANNESS}Q", buf, off)
    off += 8
    (count,) = struct.unpack_from(f"{SERIALIZATION_ENDIANNESS}Q", buf, off)
    off += 8

    dtype = code_to_dtype(int(dtype_code))

    max_depth = None
    if flags & 1:
        if len(buf) < off + 4:
            raise SerializationError("Data too short while reading max_depth")
        (md,) = struct.unpack_from(f"{SERIALIZATION_ENDIANNESS}i", buf, off)
        off += 4
        if md < 0:
            raise SerializationError("Invalid max_depth in serialization")
        max_depth = int(md)

    bounds, off = unpack_bounds(buf, off, dtype)

    if len(buf) < off + 4:
        raise SerializationError("Data too short while reading core length")
    (core_len,) = struct.unpack_from(f"{SERIALIZATION_ENDIANNESS}I", buf, off)
    off += 4
    if len(buf) < off + core_len:
        raise SerializationError("Data too short while reading core payload")
    core = bytes(buf[off : off + core_len])
    off += core_len

    if len(buf) < off + 2:
        raise SerializationError("Data too short while reading section count")
    (n_sections,) = struct.unpack_from(f"{SERIALIZATION_ENDIANNESS}H", buf, off)
    off += 2

    sections: list[tuple[int, bytes]] = []
    for _ in range(int(n_sections)):
        if len(buf) < off + 6:
            raise SerializationError("Data too short while reading section header")
        (stype, slen) = struct.unpack_from(f"{SERIALIZATION_ENDIANNESS}HI", buf, off)
        off += 6
        if len(buf) < off + slen:
            raise SerializationError("Data too short while reading section payload")
        payload = bytes(buf[off : off + slen])
        off += slen
        sections.append((int(stype), payload))

    # Trailing bytes are currently ignored (future extension space).

    return {
        "magic": magic,
        "fmt_ver": int(fmt_ver),
        "flags": int(flags),
        "dtype": dtype,
        "capacity": int(capacity),
        "max_depth": max_depth,
        "next_id": int(next_id),
        "count": int(count),
        "bounds": bounds,
        "core": core,
        "sections": sections,
    }
