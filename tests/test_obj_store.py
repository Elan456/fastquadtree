import pytest

from fastquadtree._item import Item
from fastquadtree._obj_store import ObjStore


class DummyItem(Item):
    """Minimal Item-like object: only id_ and obj are used by ObjStore."""

    __slots__ = ("geom", "id_", "obj")

    def __init__(self, id_, obj=None, geom=None):
        self.id_ = id_
        self.obj = obj
        self.geom = geom


def test_init_with_items_and_basic_lookup():
    it0 = DummyItem(0, "a")
    it1 = DummyItem(1, {"b": 2})
    s = ObjStore([it0, it1])

    assert len(s) == 2
    assert s.by_id(0) is it0
    assert s.by_id(1) is it1
    assert s.by_id(-1) is None
    assert s.by_id(99) is None

    # reverse map
    assert s.by_obj("a") is it0
    assert s.by_obj({"b": 2}) is None  # different identity
    assert s.contains_obj("a") is True
    assert s.contains_obj("not-there") is False


def test_add_append_fill_hole_and_replace_removes_old_reverse_map():
    s = ObjStore()

    # append at tail
    s.add(DummyItem(0, "a"))
    s.add(DummyItem(1, "b"))
    assert len(s) == 2
    assert s.contains_id(0)
    assert s.contains_id(1)

    # replace at same id where old had an object and new also has object
    s.add(DummyItem(1, "b2"))
    assert len(s) == 2
    assert s.by_obj("b") is None
    assert s.by_obj("b2") is s.by_id(1)

    # pop to create a hole at 0, then fill the hole (old had obj, new has None)
    popped = s.pop_id(0)
    assert popped.obj == "a"  # type: ignore
    assert len(s) == 1
    assert s.by_obj("a") is None
    assert s.contains_id(0) is False

    # fill hole with None obj
    s.add(DummyItem(0, None))
    assert len(s) == 2
    assert s.contains_id(0) is True
    assert s.by_obj("a") is None  # old reverse mapping stayed removed

    # adding the exact same instance again is a no-op for reverse mapping semantics
    same = s.by_id(0)
    s.add(same)
    assert len(s) == 2


def test_add_out_of_order_id_raises():
    s = ObjStore()
    s.add(DummyItem(0, "x"))
    with pytest.raises(AssertionError, match="out-of-order id"):
        s.add(DummyItem(2, "skip"))  # gap is not allowed


def test_alloc_id_and_lifo_free_reuse():
    s = ObjStore()
    s.add(DummyItem(0, "a"))
    s.add(DummyItem(1, "b"))
    s.add(DummyItem(2, "c"))

    # free 2 then 1
    assert s.pop_id(2).obj == "c"  # type: ignore
    assert s.pop_id(1).obj == "b"  # type: ignore

    # alloc should reuse in LIFO order
    assert s.alloc_id() == 1
    assert s.alloc_id() == 2
    # then append
    assert s.alloc_id() == 3


def test_batch_get_many_by_ids_tuple_and_single_paths_and_empty():
    s = ObjStore()
    # ids 0..5
    for i in range(6):
        s.add(DummyItem(i, f"obj{i}"))

    # chunk forces a 4-element tuple then a 2-element tuple
    out_items = s.get_many_by_ids([0, 1, 2, 3, 4, 5], chunk=4)
    assert [it.id_ for it in out_items] == [0, 1, 2, 3, 4, 5]

    # single-element block to hit the non-tuple branch
    out_single = s.get_many_by_ids([3], chunk=4)
    assert [it.id_ for it in out_single] == [3]

    # empty path
    assert s.get_many_by_ids([]) == []


def test_batch_get_many_objects_tuple_and_single_paths_and_holes():
    s = ObjStore()
    for i in range(6):
        s.add(DummyItem(i, f"obj{i}"))

    # remove id 4 to create a hole (object becomes None)
    s.pop_id(4)

    # tuple branch then tuple with a None in it
    objs = s.get_many_objects([0, 1, 2, 3, 4, 5], chunk=4)
    assert objs == ["obj0", "obj1", "obj2", "obj3", None, "obj5"]

    # single-element branch
    assert s.get_many_objects([2], chunk=4) == ["obj2"]

    # empty path
    assert s.get_many_objects([]) == []


def test_iterators_items_and_items_by_id():
    s = ObjStore()
    s.add(DummyItem(0, "a"))
    s.add(DummyItem(1, "b"))
    s.add(DummyItem(2, "c"))
    s.pop_id(1)  # create a hole

    pairs = list(s.items_by_id())
    assert pairs[0][0] == 0
    assert isinstance(pairs[0][1], DummyItem)
    assert pairs[1][0] == 2
    assert isinstance(pairs[1][1], DummyItem)
    # id 1 is absent

    only_items = list(s.items())
    assert [it.id_ for it in only_items] == [0, 2]


def test_contains_and_clear_resets_everything():
    s = ObjStore()
    s.add(DummyItem(0, "a"))
    s.add(DummyItem(1, "b"))

    assert s.contains_id(0) is True
    assert s.contains_id(2) is False
    assert s.contains_obj("a") is True

    assert s.pop_id(-23) is None  # no error on bad pop
    assert s.pop_id(200) is None

    s.clear()
    assert len(s) == 0
    assert s.contains_id(0) is False
    assert s.contains_obj("a") is False
    assert list(s.items()) == []
    assert list(s.items_by_id()) == []
