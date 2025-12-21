import numpy as np
import pytest

from fastquadtree.point_quadtree import QuadTree
from fastquadtree.point_quadtree_objects import QuadTreeObjects
from fastquadtree.rect_quadtree import RectQuadTree
from fastquadtree.rect_quadtree_objects import RectQuadTreeObjects


def test_numpy_into_non_np_methods_raise(bounds):
    qt = QuadTree(bounds, capacity=4, dtype="f32")
    rqt = RectQuadTree(bounds, capacity=4, dtype="f32")
    arr_points = np.array([[1.0, 1.0]], dtype=np.float32)
    arr_rects = np.array([[1.0, 1.0, 2.0, 2.0]], dtype=np.float32)
    with pytest.raises(TypeError):
        qt.insert_many(arr_points)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        rqt.insert_many(arr_rects)  # type: ignore[arg-type]


def test_objects_class_rejects_custom_id(bounds):
    qt_obj = QuadTreeObjects(bounds, capacity=4)
    with pytest.raises(TypeError):
        qt_obj.insert((1.0, 1.0), id_=5)  # type: ignore[call-arg]


def test_serialization_objects_allow_flag(bounds):
    qt_obj = QuadTreeObjects(bounds, capacity=4)
    obj = {"danger": True}
    qt_obj.insert((1.0, 1.0), obj=obj)

    data = qt_obj.to_bytes(include_objects=True)
    safe = QuadTreeObjects.from_bytes(data)
    assert [it.obj for it in safe.query(bounds)] == [None]

    unsafe = QuadTreeObjects.from_bytes(data, allow_objects=True)
    assert any(
        isinstance(it.obj, dict) and it.obj == obj for it in unsafe.query(bounds)
    )


def test_unsafe_load_allows_pickled_objects(bounds):
    rqt_obj = RectQuadTreeObjects(bounds, capacity=4)
    marker = {"marker": True}
    rqt_obj.insert((1.0, 1.0, 2.0, 2.0), obj=marker)
    data = rqt_obj.to_bytes(include_objects=True)
    loaded = RectQuadTreeObjects.from_bytes(data, allow_objects=True)
    objs = loaded.get_all_objects()
    assert any(isinstance(o, dict) and o.get("marker") for o in objs)
