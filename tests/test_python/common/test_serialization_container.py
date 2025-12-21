import pytest
from tests.test_python.conftest import (
    corrupt_magic,
    truncate_bytes,
)

from fastquadtree._common import (
    DTYPE_BOUNDS_SIZE_BYTES,
    SECTION_ITEMS,
    QuadTreeDType,
    SerializationError,
    build_container,
    parse_container,
)


@pytest.mark.parametrize("dtype", ["f32", "f64", "i32", "i64"])
@pytest.mark.parametrize("with_max_depth", [False, True])
def test_build_and_parse_round_trip(dtype: QuadTreeDType, with_max_depth: bool):
    flags = 1 if with_max_depth else 0
    max_depth = 3 if with_max_depth else None
    core = b"core-" + dtype.encode()
    sections = [(SECTION_ITEMS, b"items"), (255, b"ignored")]  # unknown preserved

    blob = build_container(
        fmt_ver=1,
        dtype=dtype,
        flags=flags,
        capacity=16,
        max_depth=max_depth,
        next_id=5,
        count=4,
        bounds=(0.0, 0.0, 10.0, 10.0),
        core=core,
        extra_sections=sections,
    )
    parsed = parse_container(blob)

    assert parsed["fmt_ver"] == 1
    assert parsed["flags"] == flags
    assert parsed["dtype"] == dtype
    assert parsed["capacity"] == 16
    assert parsed["max_depth"] == (max_depth if with_max_depth else None)
    assert parsed["next_id"] == 5
    assert parsed["count"] == 4
    assert parsed["bounds"] == (0.0, 0.0, 10.0, 10.0)
    assert parsed["core"] == core
    assert parsed["sections"] == sections


def test_parse_rejects_bad_magic_and_truncation():
    blob = build_container(
        fmt_ver=1,
        dtype="f32",
        flags=0,
        capacity=1,
        max_depth=None,
        next_id=0,
        count=0,
        bounds=(0, 0, 1, 1),
        core=b"x",
        extra_sections=None,
    )

    with pytest.raises(SerializationError):
        parse_container(corrupt_magic(blob))

    with pytest.raises(SerializationError):
        parse_container(truncate_bytes(blob, 3))


def test_build_container_validates_max_depth_and_types():
    # Flag requires max_depth
    with pytest.raises(SerializationError):
        build_container(
            fmt_ver=1,
            dtype="f32",
            flags=1,
            capacity=1,
            max_depth=None,
            next_id=0,
            count=0,
            bounds=(0, 0, 1, 1),
            core=b"",
        )

    # Negative max_depth rejected
    with pytest.raises(SerializationError):
        build_container(
            fmt_ver=1,
            dtype="f32",
            flags=1,
            capacity=1,
            max_depth=-1,
            next_id=0,
            count=0,
            bounds=(0, 0, 1, 1),
            core=b"",
        )

    # Non-bytes core rejected
    with pytest.raises(TypeError):
        build_container(
            fmt_ver=1,
            dtype="f32",
            flags=0,
            capacity=1,
            max_depth=None,
            next_id=0,
            count=0,
            bounds=(0, 0, 1, 1),
            core="not-bytes",  # type: ignore[arg-type]
        )


def test_build_container_limits_extra_sections_count_and_sizes():
    sections = [(1, b"")] * (0xFFFF + 1)
    with pytest.raises(SerializationError):
        build_container(
            fmt_ver=1,
            dtype="f32",
            flags=0,
            capacity=1,
            max_depth=None,
            next_id=0,
            count=0,
            bounds=(0, 0, 1, 1),
            core=b"",
            extra_sections=sections,
        )

    too_big_payload = b"x" * (0xFFFFFFFF + 1)
    with pytest.raises(SerializationError):
        build_container(
            fmt_ver=1,
            dtype="f32",
            flags=0,
            capacity=1,
            max_depth=None,
            next_id=0,
            count=0,
            bounds=(0, 0, 1, 1),
            core=b"",
            extra_sections=[(2, too_big_payload)],
        )


def test_parse_fails_on_inflated_core_length_and_truncated_section():
    blob = build_container(
        fmt_ver=1,
        dtype="i32",
        flags=0,
        capacity=1,
        max_depth=None,
        next_id=0,
        count=0,
        bounds=(0, 0, 1, 1),
        core=b"abc",
        extra_sections=[(9, b"payload")],
    )

    with pytest.raises(SerializationError):
        parse_container(truncate_bytes(blob, 1))

    # Truncated header before sections count
    header_end = 30 + DTYPE_BOUNDS_SIZE_BYTES["i32"] + 4 + len(b"abc")
    with pytest.raises(SerializationError):
        parse_container(blob[: header_end + 1])
