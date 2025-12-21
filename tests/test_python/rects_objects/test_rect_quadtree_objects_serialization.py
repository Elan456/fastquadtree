import pytest
from tests.test_python.conftest import get_bounds_for_dtype, truncate_bytes

from fastquadtree._base_quadtree_objects import _encode_items_section
from fastquadtree._common import SerializationError
from fastquadtree.rect_quadtree_objects import RectQuadTreeObjects


def rect_for_dtype(dtype: str, coords: tuple[float, float, float, float]) -> tuple:
    return tuple(map(int, coords)) if dtype.startswith("i") else coords


def test_serialization_with_and_without_objects(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTreeObjects(bounds_use, capacity=4, dtype=dtype)
    obj = {"k": "v"}
    rqt.insert(rect_for_dtype(dtype, (1.0, 1.0, 2.0, 2.0)), obj=obj)
    rqt.insert(rect_for_dtype(dtype, (3.0, 3.0, 4.0, 4.0)))

    data_no_obj = rqt.to_bytes()
    clone = RectQuadTreeObjects.from_bytes(data_no_obj)
    assert [it.obj for it in clone.query(bounds_use)] == [None, None]

    data_with_obj = rqt.to_bytes(include_objects=True)
    clone_with = RectQuadTreeObjects.from_bytes(data_with_obj, allow_objects=True)
    assert any(isinstance(it.obj, dict) for it in clone_with.query(bounds_use))


def test_free_list_restored_after_deserialize(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTreeObjects(bounds_use, capacity=6, dtype=dtype)
    ids = [
        rqt.insert(rect_for_dtype(dtype, (1.0, 1.0, 2.0, 2.0))),
        rqt.insert(rect_for_dtype(dtype, (3.0, 3.0, 4.0, 4.0))),
    ]
    rqt.delete(ids[0])
    data = rqt.to_bytes(include_objects=True)

    clone = RectQuadTreeObjects.from_bytes(data, allow_objects=True)
    new_id = clone.insert(rect_for_dtype(dtype, (5.0, 5.0, 6.0, 6.0)))
    assert new_id == ids[0]  # reused free slot


def test_encode_items_section_mixed_geometry_error():
    from fastquadtree._item import PointItem, RectItem

    items = [RectItem(0, (0.0, 0.0, 1.0, 1.0)), PointItem(1, (0.0, 0.0))]
    with pytest.raises(SerializationError):
        _encode_items_section(items)  # type: ignore[arg-type]


def test_truncated_items_section_raises(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTreeObjects(bounds_use, capacity=4, dtype=dtype)
    rqt.insert(rect_for_dtype(dtype, (1.0, 1.0, 2.0, 2.0)), obj="x")
    data = rqt.to_bytes(include_objects=True)
    with pytest.raises(SerializationError):
        RectQuadTreeObjects.from_bytes(truncate_bytes(data, 5), allow_objects=True)
