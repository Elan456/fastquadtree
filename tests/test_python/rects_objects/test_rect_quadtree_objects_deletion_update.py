from tests.test_python.conftest import get_bounds_for_dtype

from fastquadtree.rect_quadtree_objects import RectQuadTreeObjects


def rect_for_dtype(dtype: str, coords: tuple[float, float, float, float]) -> tuple:
    return tuple(map(int, coords)) if dtype.startswith("i") else coords


def test_delete_and_delete_at(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTreeObjects(bounds_use, capacity=6, dtype=dtype)
    rect = rect_for_dtype(dtype, (5.0, 5.0, 6.0, 6.0))
    rqt.insert(rect)
    rqt.insert(rect)
    assert rqt.delete_at(*rect) is True  # lowest id first
    assert rqt.delete_at(*rect) is True
    assert rqt.delete_at(*rect) is False
    assert rqt.delete(999) is False

    rid3 = rqt.insert(rect_for_dtype(dtype, (7.0, 7.0, 8.0, 8.0)))
    assert rqt.delete(rid3) is True


def test_delete_by_object_and_attach(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTreeObjects(bounds_use, capacity=6, dtype=dtype)
    obj = object()
    rqt.insert(rect_for_dtype(dtype, (1.0, 1.0, 2.0, 2.0)), obj=obj)
    rqt.insert(rect_for_dtype(dtype, (3.0, 3.0, 4.0, 4.0)), obj=obj)
    rqt.insert(rect_for_dtype(dtype, (5.0, 5.0, 6.0, 6.0)), obj="other")

    assert rqt.delete_one_by_object(obj) is True
    remaining = rqt.query(rect_for_dtype(dtype, (0.0, 0.0, 10.0, 10.0)))
    assert len([it for it in remaining if it.obj is obj]) == 1

    assert rqt.delete_by_object(obj) == 1
    assert rqt.delete_by_object("missing") == 0

    rid = rqt.insert(rect_for_dtype(dtype, (7.0, 7.0, 8.0, 8.0)), obj="old")
    rqt.attach(rid, "new")
    items = rqt.query(rect_for_dtype(dtype, (0.0, 0.0, 10.0, 10.0)))
    updated = next(it for it in items if it.id_ == rid)
    assert updated.obj == "new"


def test_update_and_update_by_object(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    rqt = RectQuadTreeObjects(bounds_use, capacity=4, dtype=dtype)
    initial = rect_for_dtype(dtype, (1.0, 1.0, 2.0, 2.0))
    rid = rqt.insert(initial, obj="obj")
    updated = rect_for_dtype(dtype, (3.0, 3.0, 4.0, 4.0))
    assert rqt.update(rid, *updated) is True
    assert updated in rqt

    obj2 = object()
    rqt.attach(rid, obj2)
    moved = rect_for_dtype(dtype, (5.0, 5.0, 6.0, 6.0))
    assert rqt.update_by_object(obj2, *moved) is True
    assert moved in rqt

    missing_obj = object()
    missing_target = rect_for_dtype(dtype, (7.0, 7.0, 8.0, 8.0))
    assert rqt.update_by_object(missing_obj, *missing_target) is False
