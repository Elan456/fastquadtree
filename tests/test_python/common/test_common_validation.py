import numpy as np
import pytest

from fastquadtree._common import (
    CODE_TO_DTYPE,
    DTYPE_BOUNDS_SIZE_BYTES,
    QUADTREE_DTYPE_TO_NP_DTYPE,
    QuadTreeDType,
    SerializationError,
    _is_np_array,
    code_to_dtype,
    dtype_to_code,
    pack_bounds,
    unpack_bounds,
    validate_bounds,
    validate_np_dtype,
)


def test_validate_bounds_normalizes_and_errors_on_bad_length():
    bounds_list = [0.0, 1.0, 2.0, 3.0]
    normalized = validate_bounds(bounds_list)
    assert normalized == (0.0, 1.0, 2.0, 3.0)
    assert isinstance(normalized, tuple)

    with pytest.raises(ValueError):
        validate_bounds((0.0, 1.0, 2.0))  # too short


def test_validate_bounds_rejects_inverted_and_non_finite():
    with pytest.raises(ValueError):
        validate_bounds((10.0, 10.0, 0.0, 0.0))  # inverted

    with pytest.raises(ValueError):
        validate_bounds((0.0, 0.0, float("inf"), 1.0))

    with pytest.raises(ValueError):
        validate_bounds((0.0, 0.0, 0.0, 1.0))  # zero-width

    with pytest.raises(ValueError):
        validate_bounds(("a", 0.0, 1.0, 2.0))


def test_is_np_array_detects_numpy_and_rejects_non_numpy():
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    assert _is_np_array(arr) is True
    assert _is_np_array([1, 2, 3]) is False

    class FakeNp:
        __module__ = "numpy.fake"

        def __init__(self):
            self.ndim = 2
            self.shape = (1, 2)

    assert _is_np_array(FakeNp()) is True

    class NotNp:
        __module__ = "fastquadtree.fake"

        def __init__(self):
            self.ndim = 2
            self.shape = (1, 2)

    assert _is_np_array(NotNp()) is False


@pytest.mark.parametrize("dtype", ["f32", "f64", "i32", "i64"])
def test_validate_np_dtype_allows_matching_dtype(dtype: QuadTreeDType):
    np_dtype = QUADTREE_DTYPE_TO_NP_DTYPE[dtype]
    arr = np.zeros((2, 2), dtype=np_dtype)
    validate_np_dtype(arr, dtype)  # should not raise


def test_validate_np_dtype_rejects_mismatch_and_unknown_dtype():
    arr = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(TypeError):
        validate_np_dtype(arr, "f64")

    with pytest.raises(TypeError):
        validate_np_dtype(arr, "bad_dtype")  # type: ignore # unknown dtype string


def test_dtype_code_mappings_and_errors():
    for code, dtype in CODE_TO_DTYPE.items():
        assert dtype_to_code(dtype) == code
        assert code_to_dtype(code) == dtype

    with pytest.raises(SerializationError):
        dtype_to_code("bad")  # type: ignore[arg-type]
    with pytest.raises(SerializationError):
        code_to_dtype(99)


@pytest.mark.parametrize("dtype", ["f32", "f64", "i32", "i64"])
def test_pack_and_unpack_bounds_round_trip(dtype: QuadTreeDType):
    bounds = (1.5, 2.5, 3.5, 4.5)
    if dtype in ("i32", "i64"):
        # Int dtypes cast inputs to ints during packing
        bounds = (1.0, 2.0, 3.0, 4.0)
    packed = pack_bounds(bounds, dtype)
    assert len(packed) == DTYPE_BOUNDS_SIZE_BYTES[dtype]
    unpacked, new_off = unpack_bounds(memoryview(packed), 0, dtype)
    assert unpacked == bounds
    assert new_off == len(packed)


def test_pack_bounds_rejects_unknown_dtype():
    with pytest.raises(SerializationError):
        pack_bounds((0, 0, 1, 1), "bad")  # type: ignore[arg-type]
