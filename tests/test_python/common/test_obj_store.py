from fastquadtree._item import Item
from fastquadtree._obj_store import ObjStore


def _mk(id_: int, geom=(0.0, 0.0), obj=None) -> Item:
    return Item(id_, geom, obj)


def test_alloc_free_reuse_and_pop_id():
    store = ObjStore[Item]()
    i0 = store.alloc_id()
    assert i0 == 0
    store.add(_mk(i0, obj="a"))
    assert len(store) == 1

    popped = store.pop_id(i0)
    assert popped is not None
    assert popped.id_ == 0
    assert len(store) == 0
    assert store._free == [0]

    i1 = store.alloc_id()
    assert i1 == 0  # reused from free list
    store.add(_mk(i1, obj="b"))
    assert store.contains_id(0) is True
    assert store.contains_obj("b") is True


def test_add_replaces_object_and_updates_reverse_map():
    store = ObjStore[Item]()
    obj1 = {"k": 1}
    obj2 = {"k": 2}
    store.add(_mk(0, obj=obj1))
    store.add(_mk(0, obj=obj2))  # replace same id with new object

    assert store.by_obj(obj1) is None
    found = store.by_obj(obj2)
    assert found is not None
    assert found.id_ == 0
    assert found.obj == obj2


def test_handle_out_of_order_fill_and_free_list_population():
    store = ObjStore[Item]()
    store.add(_mk(3, obj="late"), handle_out_of_order=True)
    assert len(store._arr) == 4
    assert store._arr[0] is None
    assert store._arr[3] is not None
    assert store._arr[3].obj == "late"
    assert store._free == []  # holes not added until explicit population

    # Simulate post-processing that records holes
    for idx, it in enumerate(store._arr):
        if it is None:
            store._free.append(idx)
    assert sorted(store._free) == [0, 1, 2]


def test_by_obj_lowest_id_and_by_obj_all_sorted():
    store = ObjStore[Item]()
    obj = object()
    store.add(_mk(0, obj=obj))
    store.add(_mk(1, obj=obj))
    store.add(_mk(2, obj=obj))

    lowest = store.by_obj(obj)
    assert lowest is not None
    assert lowest.id_ == 0
    all_items = store.by_obj_all(obj)
    assert [it.id_ for it in all_items] == [0, 1, 2]


def test_get_many_by_ids_and_get_many_objects_ordering():
    store = ObjStore[Item]()
    for i in range(5):
        store.add(_mk(i, geom=(i, i), obj=f"o{i}"))

    ids = [4, 1, 3, 0]
    items = store.get_many_by_ids(ids, chunk=2)
    assert [(it.id_, it.geom) for it in items] == [
        (4, (4, 4)),
        (1, (1, 1)),
        (3, (3, 3)),
        (0, (0, 0)),
    ]
    objs = store.get_many_objects(ids, chunk=3)
    assert objs == ["o4", "o1", "o3", "o0"]


def test_clear_resets_all_state():
    store = ObjStore[Item]()
    store.add(_mk(0, obj="x"))
    store.pop_id(0)
    store.clear()
    assert len(store) == 0
    assert store._arr == []
    assert store._objs == []
    assert store._obj_to_ids == {}
    assert store._free == []


def test_replace_with_none_removes_old_object_mapping():
    store = ObjStore[Item]()
    obj = object()
    store.add(_mk(0, obj=obj))
    assert store.contains_obj(obj) is True

    store.add(_mk(0, obj=None))
    assert store.contains_obj(obj) is False
    assert store._objs[0] is None
