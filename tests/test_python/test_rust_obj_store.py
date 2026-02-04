import random

import pyqtree
import pytest

from fastquadtree import QuadTreeObjects 
from fastquadtree._item import PointItem

def test_rust_obj_store():
    qt = QuadTreeObjects((0, 0, 100, 100), 5)

    id_ = qt.insert((10, 10), "obj1")
    assert id_ == 0

    id_ = qt.insert((12, 12), "obj2")
    assert id_ == 1

    out_objs = qt.query((9, 9, 11, 11))
    ep_point_item = PointItem(0, (10, 10), "obj1")
    ac_point_item = out_objs[0]
    assert ac_point_item.id_ == ep_point_item.id_
    assert ac_point_item.geom == ep_point_item.geom
    assert ac_point_item.obj == ep_point_item.obj

    # out_objs = qt.query((11, 11, 15, 15))
    # ep_point_item = PointItem(1, (12, 12), "obj2")
    # ac_point_item = out_objs[0]
    # assert ac_point_item.id_ == ep_point_item.id_
    # assert ac_point_item.geom == ep_point_item.geom
    # assert ac_point_item.obj == ep_point_item.obj