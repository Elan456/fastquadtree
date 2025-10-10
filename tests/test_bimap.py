from fastquadtree._bimap import BiMap
from fastquadtree._item import Item


def make_item(id_, x=0.0, y=0.0, obj=None):
    return Item(id_=id_, geom=(x, y), obj=obj)


def test_init_with_items_populates_both_maps():
    i1 = make_item(1, obj={"a": 1})
    i2 = make_item(2, obj={"b": 2})
    b = BiMap([i1, i2])

    assert b.by_id(1) is i1
    assert b.by_id(2) is i2
    assert b.by_obj(i1.obj) is i1
    assert b.by_obj(i2.obj) is i2
    assert len(b) == 2
    assert b.contains_id(1)
    assert b.contains_obj(i1.obj)


def test_add_with_obj_none_only_sets_id_side():
    i = make_item(10, obj=None)
    b = BiMap()
    b.add(i)
    assert b.by_id(10) is i
    assert b.by_obj({"irrelevant": True}) is None
    assert not b.contains_obj(object())


def test_add_replaces_existing_id_and_unlinks_old_obj():
    old_obj = {"v": 1}
    i_old = make_item(7, obj=old_obj)
    i_new = make_item(7, obj={"v": 2})

    b = BiMap([i_old])
    assert b.by_obj(old_obj) is i_old

    b.add(i_new)  # same id, new item with new obj
    assert b.by_id(7) is i_new
    # old object mapping must be removed
    assert b.by_obj(old_obj) is None
    # new object mapping must exist
    assert b.by_obj(i_new.obj) is i_new


def test_add_replaces_existing_obj_and_unlinks_previous_id():
    shared_obj = {"x": 1}
    i_a = make_item(100, obj=shared_obj)
    i_b = make_item(200, obj=shared_obj)

    b = BiMap([i_a])
    assert b.by_id(100) is i_a
    assert b.by_obj(shared_obj) is i_a

    b.add(i_b)  # same obj, new item with different id
    assert b.by_obj(shared_obj) is i_b
    # previous id mapping must be removed
    assert b.by_id(100) is None
    assert b.by_id(200) is i_b


def test_add_same_item_is_idempotent():
    obj = {"z": 1}
    i = make_item(5, obj=obj)
    b = BiMap()
    b.add(i)
    before = (b.by_id(5), b.by_obj(obj))
    b.add(i)
    after = (b.by_id(5), b.by_obj(obj))
    assert before == after
    assert before[0] is i
    assert before[1] is i


def test_by_obj_uses_identity_not_equality():
    obj1 = {"k": 1}
    obj2 = {"k": 1}  # equal but different identity
    i = make_item(9, obj=obj1)
    b = BiMap([i])
    assert b.by_obj(obj1) is i
    assert b.by_obj(obj2) is None  # identity based


def test_pop_id_removes_both_sides_and_handles_missing():
    obj = {"rm": 1}
    i = make_item(11, obj=obj)
    b = BiMap([i])

    popped = b.pop_id(11)
    assert popped is i
    assert b.by_id(11) is None
    assert b.by_obj(obj) is None
    # popping again is harmless
    assert b.pop_id(11) is None


def test_pop_obj_removes_both_sides_and_handles_missing():
    obj = {"rm": 2}
    i = make_item(12, obj=obj)
    b = BiMap([i])

    popped = b.pop_obj(obj)
    assert popped is i
    assert b.by_id(12) is None
    assert b.by_obj(obj) is None
    # popping again is harmless
    assert b.pop_obj(obj) is None


def test_pop_item_removes_if_present_on_either_side():
    obj = {"p": 1}
    i = make_item(13, obj=obj)
    b = BiMap([i])

    # present on both sides
    assert b.pop_item(i) is i
    assert b.by_id(13) is None
    assert b.by_obj(obj) is None

    # add only on id side (obj is None)
    i2 = make_item(14, obj=None)
    b.add(i2)
    assert b.pop_item(i2) is i2
    assert b.by_id(14) is None


def test_len_clear_items_iterators_and_contains_helpers():
    i1 = make_item(31, obj={"a": 1})
    i2 = make_item(32, obj={"b": 2})
    b = BiMap([i1, i2])

    # __len__
    assert len(b) == 2

    # items_by_id returns (id, item) pairs
    pairs = dict(b.items_by_id())
    assert pairs[31] is i1
    assert pairs[32] is i2

    # items returns items
    vals = list(b.items())
    assert set(vals) == {i1, i2}

    # contains helpers
    assert b.contains_id(31)
    assert b.contains_obj(i2.obj)

    # clear wipes everything
    b.clear()
    assert len(b) == 0
    assert list(b.items()) == []
    assert b.by_id(31) is None
    assert b.by_obj(i1.obj) is None
