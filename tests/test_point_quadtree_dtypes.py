import pytest

from fastquadtree import QuadTree


def test_unsupported_dtype():
    """Test that providing an unsupported dtype raises ValueError."""
    with pytest.raises(TypeError):
        QuadTree((0, 0, 100, 100), capacity=4, track_objects=True, dtype="f128")  # type: ignore

    # From bytes
    qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=True, dtype="f32")
    data = qt.to_bytes()
    with pytest.raises(TypeError):
        QuadTree.from_bytes(data, dtype="f128")  # type: ignore


def test_all_other_dtypes():
    """Test that all supported dtypes can be used without error."""
    for dtype in ["f32", "f64", "i32", "i64"]:
        print("Testing dtype:", dtype)
        qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=True, dtype=dtype)
        id1 = qt.insert((10, 10), obj="test")
        assert id1 == 0
        data = qt.to_bytes()
        qt2 = QuadTree.from_bytes(data, dtype=dtype)
        items = qt2.get_all_items()
        assert len(items) == 1
        assert items[0].obj == "test"

        # Run some queries
        results = qt2.query((5, 5, 15, 15), as_items=True)
        assert len(results) == 1
        assert results[0].obj == "test"


def test_all_other_dtypes_implicit():
    """Test that all supported dtypes can be used without error."""
    # Leave it at as f32
    qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=True)
    id1 = qt.insert((10, 10), obj="test_int")
    id2 = qt.insert((20.5, 20.5), obj="test_float")
    assert id1 == 0
    assert id2 == 1
    data = qt.to_bytes()
    qt2 = QuadTree.from_bytes(data)
    items = qt2.get_all_items()
    assert len(items) == 2
    assert items[0].obj == "test_int"
    assert items[1].obj == "test_float"

    # Run some queries
    results = qt2.query((5, 5, 20.6, 20.6), as_items=True)
    assert len(results) == 2
    assert results[0].obj == "test_int"
    assert results[1].obj == "test_float"


def test_from_bytes_with_mismatched_dtype():
    """Test that providing a different dtype in from_bytes raises ValueError."""
    qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=True, dtype="f32")
    id1 = qt.insert((10, 10), obj="test")
    assert id1 == 0
    data = qt.to_bytes()  # To bytes a f32
    with pytest.raises(ValueError):
        QuadTree.from_bytes(data, dtype="f64")  # type: ignore  # From bytes with f64 should fail


def test_integer_out_of_bounds_insertion():
    """Test that inserting points out of bounds with integer dtypes raises ValueError."""
    qt = QuadTree((0, 0, 100, 100), capacity=4, track_objects=True, dtype="i32")
    with pytest.raises(ValueError):
        qt.insert((150, 150), obj="out_of_bounds")  # type: ignore

    with pytest.raises(ValueError):
        qt.insert((-10, -10), obj="out_of_bounds")  # type: ignore


def test_integer_overflow_bounding_box():
    """Test that inserting points that would cause integer overflow raises ValueError."""
    qt = QuadTree(
        (0, 0, 2**31 - 1, 2**31 - 1), capacity=4, track_objects=True, dtype="i32"
    )
    with pytest.raises(OverflowError):
        qt.insert((2**31, 2**31), obj="overflow")  # type: ignore

    with pytest.raises(OverflowError):
        qt.insert((-(2**31) - 1, -(2**31) - 1), obj="underflow")  # type: ignore
