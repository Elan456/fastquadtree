import pytest

from fastquadtree._insert_result import InsertResult
from fastquadtree._item import Item, PointItem, RectItem


def test_insert_result_ids_property_empty_and_non_empty():
    empty = InsertResult(count=0, start_id=10, end_id=9)
    assert list(empty.ids) == []

    result = InsertResult(count=3, start_id=5, end_id=7)
    assert list(result.ids) == [5, 6, 7]


def test_item_to_from_dict_round_trip():
    item = Item(2, (1.0, 2.0), obj={"a": 1})
    data = item.to_dict()
    clone = Item.from_dict(data)
    assert clone.id_ == 2
    assert clone.geom == (1.0, 2.0)
    assert clone.obj == {"a": 1}


def test_point_item_and_rect_item_slots_and_attrs():
    p = PointItem(1, (3.0, 4.0), obj="payload")
    assert p.x == 3.0
    assert p.y == 4.0
    assert p.obj == "payload"
    with pytest.raises(AttributeError):
        p.__dict__  # __slots__ prevents dict  # noqa: B018

    r = RectItem(7, (0.0, 1.0, 2.0, 3.0), obj=None)
    assert (r.min_x, r.min_y, r.max_x, r.max_y) == (0.0, 1.0, 2.0, 3.0)
    assert r.obj is None
