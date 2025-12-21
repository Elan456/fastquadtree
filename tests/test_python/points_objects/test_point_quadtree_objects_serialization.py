import pytest
from tests.test_python.conftest import get_bounds_for_dtype

from fastquadtree._base_quadtree_objects import _encode_items_section
from fastquadtree._common import SerializationError
from fastquadtree.point_quadtree_objects import QuadTreeObjects


def test_serialization_include_objects_gate(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTreeObjects(bounds_use, capacity=4, dtype=dtype)
    obj = {"a": 1}
    qt.insert((1, 1) if dtype.startswith("i") else (1.0, 1.0), obj=obj)
    qt.insert((2, 2) if dtype.startswith("i") else (2.0, 2.0))

    data_without = qt.to_bytes()
    clone = QuadTreeObjects.from_bytes(data_without)
    # Object payload ignored by default
    query_rect = bounds_use if dtype.startswith("i") else bounds
    assert [it.obj for it in clone.query(query_rect)] == [None, None]

    data_with = qt.to_bytes(include_objects=True)
    clone_with = QuadTreeObjects.from_bytes(data_with, allow_objects=True)
    assert any(isinstance(it.obj, dict) for it in clone_with.query(query_rect))


def test_free_list_restored_after_load(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTreeObjects(bounds_use, capacity=4, dtype=dtype)
    ids = [
        qt.insert((1, 1) if dtype.startswith("i") else (1.0, 1.0)),
        qt.insert((2, 2) if dtype.startswith("i") else (2.0, 2.0)),
        qt.insert((3, 3) if dtype.startswith("i") else (3.0, 3.0)),
    ]
    qt.delete(ids[1])  # creates a hole at id 1
    data = qt.to_bytes(include_objects=True)

    clone = QuadTreeObjects.from_bytes(data, allow_objects=True)
    # Free slot 1 should be reusable
    new_id = clone.insert((4, 4) if dtype.startswith("i") else (4.0, 4.0))
    assert new_id == 1


def test_encode_items_section_rejects_mixed_geometry():
    from fastquadtree._item import PointItem, RectItem

    items = [PointItem(0, (0.0, 0.0)), RectItem(1, (0.0, 0.0, 1.0, 1.0))]
    with pytest.raises(SerializationError):
        _encode_items_section(items)  # type: ignore[arg-type]
