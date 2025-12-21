from tests.test_python.conftest import get_bounds_for_dtype

from fastquadtree.point_quadtree_objects import QuadTreeObjects


def test_delete_and_delete_at_with_duplicates(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTreeObjects(bounds_use, capacity=6, dtype=dtype)
    pt = (5, 5) if dtype.startswith("i") else (5.0, 5.0)
    rid1 = qt.insert(pt)
    rid2 = qt.insert(pt)

    # delete_at removes lowest id
    if not dtype.startswith("i"):
        _ = qt.delete_at(pt[0], pt[1])
        _ = qt.delete_at(pt[0], pt[1])  # second one
        assert qt.delete_at(pt[0], pt[1]) is False
    # Ensure they are gone regardless of delete_at outcome
    qt.delete(rid1)
    qt.delete(rid2)
    assert qt.delete(999) is False

    # Reinsert and delete by id path
    rid3 = qt.insert((6, 6) if dtype.startswith("i") else (6.0, 6.0))
    assert qt.delete(rid3) is True


def test_delete_by_object_variants(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTreeObjects(bounds_use, capacity=6, dtype=dtype)
    obj = {"k": 1}
    qt.insert((1, 1) if dtype.startswith("i") else (1.0, 1.0), obj=obj)
    qt.insert((2, 2) if dtype.startswith("i") else (2.0, 2.0), obj=obj)
    qt.insert((3, 3) if dtype.startswith("i") else (3.0, 3.0), obj="other")

    assert qt.delete_one_by_object(obj) is True
    # Only one of the duplicates removed, lowest id first
    rect = (0, 0, 5, 5) if dtype.startswith("i") else (0.0, 0.0, 5.0, 5.0)
    remaining = qt.query(rect)
    assert len([it for it in remaining if it.obj == obj]) == 1

    assert qt.delete_by_object(obj) == 1  # remove remaining match
    assert qt.delete_by_object("missing") == 0


def test_attach_replaces_object_and_update_paths(bounds, dtype):
    bounds_use = get_bounds_for_dtype(bounds, dtype)
    qt = QuadTreeObjects(bounds_use, capacity=4, dtype=dtype)
    rid = qt.insert((1, 1) if dtype.startswith("i") else (1.0, 1.0), obj="old")
    qt.attach(rid, "new")
    rect = (0, 0, 5, 5) if dtype.startswith("i") else (0.0, 0.0, 5.0, 5.0)
    it = qt.query(rect)[0]
    assert it.obj == "new"

    new_val = 2 if dtype.startswith("i") else 2.0
    assert qt.update(rid, new_val, new_val) is True
    if not dtype.startswith("i"):
        _ = (2.0, 2.0) in qt

    obj2 = object()
    qt.attach(rid, obj2)
    new_val = 3 if dtype.startswith("i") else 3.0
    assert qt.update_by_object(obj2, new_val, new_val) is True
    if not dtype.startswith("i"):
        _ = (float(new_val), float(new_val)) in qt

    missing_obj = object()
    assert qt.update_by_object(missing_obj, 4.0, 4.0) is False
