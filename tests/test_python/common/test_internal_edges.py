from __future__ import annotations

import struct
from typing import Any, Iterable

import numpy as np
import pytest

import fastquadtree._common as common
from fastquadtree._base_quadtree import _BaseQuadTree
from fastquadtree._base_quadtree_objects import (
    _BaseQuadTreeObjects,
    _decode_items_section,
    _decode_objects_section,
    _encode_items_section,
)
from fastquadtree._common import (
    SECTION_ITEMS,
    SECTION_OBJECTS,
    SERIALIZATION_FORMAT_VERSION,
    QuadTreeDType,
    SerializationError,
    build_container,
    parse_container,
    unpack_bounds,
)
from fastquadtree._item import Item, PointItem
from fastquadtree._obj_store import ObjStore
from fastquadtree.point_quadtree import QuadTree
from fastquadtree.point_quadtree_objects import QuadTreeObjects
from fastquadtree.rect_quadtree import RectQuadTree
from fastquadtree.rect_quadtree_objects import RectQuadTreeObjects


class StubNative:
    def __init__(self):
        self.insert_result = True
        self.insert_many_result: int | None = None
        self.insert_many_np_result: int | None = None
        self.delete_result = True
        self.query_result: list[Any] = []
        self.query_ids_result: list[int] = []
        self.query_np_result: tuple[Any, Any] = ("ids", "coords")
        self.nn_result: Any = None
        self.nn_np_result: Any = None
        self.nn_list_result: list[Any] = []
        self.nn_np_list_result: tuple[Any, Any] = ("ids_nn", "coords_nn")
        self.to_bytes_value = b"core"
        self.boundaries = [(0.0, 0.0, 1.0, 1.0)]
        self.max_depth = 2
        self.from_bytes_payload: bytes | None = None

    def insert(self, id_: int, geom: Any) -> bool:
        return self.insert_result

    def insert_many(self, start_id: int, geoms: Iterable[Any]) -> int:
        if self.insert_many_result is not None:
            return self.insert_many_result
        return start_id + len(list(geoms)) - 1

    def insert_many_np(self, start_id: int, geoms: Any) -> int:
        if self.insert_many_np_result is not None:
            return self.insert_many_np_result
        return start_id + len(geoms) - 1

    def delete(self, id_: int, geom: Any) -> bool:
        return self.delete_result

    def query(self, rect: Any) -> list[Any]:
        return list(self.query_result)

    def query_ids(self, rect: Any) -> list[int]:
        return list(self.query_ids_result)

    def query_np(self, rect: Any) -> tuple[Any, Any]:
        return self.query_np_result

    def nearest_neighbor(self, point: Any) -> Any:
        return self.nn_result

    def nearest_neighbor_np(self, point: Any) -> Any:
        return self.nn_np_result

    def nearest_neighbors(self, point: Any, k: int) -> list[Any]:
        return list(self.nn_list_result)

    def nearest_neighbors_np(self, point: Any, k: int) -> tuple[Any, Any]:
        return self.nn_np_list_result

    def get_all_node_boundaries(self) -> list[Any]:
        return self.boundaries

    def get_max_depth(self) -> int:
        return self.max_depth

    def to_bytes(self) -> bytes:
        return self.to_bytes_value


class StubTree(_BaseQuadTree[tuple]):
    def __init__(self, native: StubNative | None = None):
        self._stub_native = native or StubNative()
        super().__init__((0.0, 0.0, 1.0, 1.0), capacity=4, dtype="f32")

    def _new_native(
        self, bounds, capacity, max_depth, dtype
    ):  # pragma: no cover - exercised via stub
        return self._stub_native

    @classmethod
    def _new_native_from_bytes(cls, data: bytes, dtype: QuadTreeDType):
        native = StubNative()
        native.from_bytes_payload = data
        return native


class StubObjTree(_BaseQuadTreeObjects[tuple, Item]):
    def __init__(
        self,
        native: StubNative | None = None,
        *,
        max_depth: int | None = None,
        dtype: QuadTreeDType = "f32",
    ):
        self._stub_native = native or StubNative()
        super().__init__(
            (0.0, 0.0, 1.0, 1.0), capacity=4, max_depth=max_depth, dtype=dtype
        )

    def _new_native(
        self, bounds, capacity, max_depth
    ):  # pragma: no cover - exercised via stub
        return self._stub_native

    @classmethod
    def _new_native_from_bytes(cls, data: bytes, dtype: QuadTreeDType):
        native = StubNative()
        native.from_bytes_payload = data
        return native

    @staticmethod
    def _make_item(id_: int, geom: tuple, obj: Any | None) -> Item:
        return Item(id_, geom, obj)

    @staticmethod
    def _extract_coords_from_geom(geom: tuple) -> tuple:
        return geom


class HugeBytes(bytes):
    def __len__(self):  # pragma: no cover - trivial override
        return 0xFFFFFFFF + 1


def test_unpack_bounds_short_and_bad_dtype(monkeypatch):
    with pytest.raises(SerializationError):
        unpack_bounds(memoryview(b"\x00"), 0, "f32")

    monkeypatch.setitem(common.DTYPE_BOUNDS_SIZE_BYTES, "weird", 16)
    with pytest.raises(SerializationError):
        unpack_bounds(memoryview(b"\x00" * 16), 0, "weird")  # type: ignore[arg-type]


def test_build_container_rejects_oversized_core(monkeypatch):
    core = b"core"

    import builtins

    real_len = builtins.len

    def fake_len(obj):
        if isinstance(obj, (bytes, bytearray)) and obj == core:
            return 0xFFFFFFFF + 1
        return real_len(obj)

    monkeypatch.setattr("builtins.len", fake_len)
    with pytest.raises(SerializationError):
        build_container(
            fmt_ver=1,
            dtype="f32",
            flags=0,
            capacity=1,
            max_depth=None,
            next_id=0,
            count=0,
            bounds=(0, 0, 1, 1),
            core=core,
        )


def test_parse_container_type_and_truncation_errors():
    with pytest.raises(TypeError):
        parse_container(123)  # type: ignore[arg-type]

    with pytest.raises(SerializationError):
        parse_container(b"short")

    blob = build_container(
        fmt_ver=1,
        dtype="f32",
        flags=1,
        capacity=1,
        max_depth=1,
        next_id=0,
        count=0,
        bounds=(0, 0, 1, 1),
        core=b"x",
    )

    # Truncate inside max_depth field
    with pytest.raises(SerializationError):
        parse_container(blob[:32])

    # Flip max_depth to negative
    mutated = bytearray(blob)
    mutated[30:34] = struct.pack("<i", -5)
    with pytest.raises(SerializationError):
        parse_container(bytes(mutated))

    # Remove bytes before core length to trigger length check
    with pytest.raises(SerializationError):
        parse_container(blob[:40])

    # Build with a section then truncate its header
    with_section = build_container(
        fmt_ver=1,
        dtype="f32",
        flags=0,
        capacity=1,
        max_depth=None,
        next_id=0,
        count=0,
        bounds=(0, 0, 1, 1),
        core=b"xyz",
        extra_sections=[(5, b"payload")],
    )
    with pytest.raises(SerializationError):
        parse_container(with_section[:-1])


def test_parse_container_core_and_section_truncations():
    blob = build_container(
        fmt_ver=1,
        dtype="f32",
        flags=0,
        capacity=1,
        max_depth=None,
        next_id=0,
        count=0,
        bounds=(0, 0, 1, 1),
        core=b"abc",
        extra_sections=[(9, b"x")],
    )
    off = 30 + 16  # header + bounds for f32
    tampered = bytearray(blob)
    tampered[off : off + 4] = struct.pack("<I", 10)
    with pytest.raises(SerializationError):
        parse_container(bytes(tampered))

    with pytest.raises(SerializationError):
        parse_container(blob[: off + 3])

    with pytest.raises(SerializationError):
        parse_container(blob[:-2])


def test_base_quadtree_insert_and_batch_failure_paths():
    native = StubNative()
    native.insert_result = False
    qt = StubTree(native)
    with pytest.raises(ValueError):
        qt.insert((5.0, 5.0))
    assert len(qt) == 0

    res = qt.insert_many([])
    assert res.count == 0
    assert res.start_id == qt._next_id
    assert res.end_id == qt._next_id - 1

    native.insert_result = True
    native.insert_many_result = 0  # simulate partial insert
    with pytest.raises(ValueError):
        qt.insert_many([(0, 0), (1, 1)])


def test_base_quadtree_insert_many_success_and_update_failure():
    qt = StubTree()
    res = qt.insert_many([(0, 0), (1, 1)])
    assert res.count == 2
    assert qt._next_id == 2
    assert len(qt) == 2

    qt._stub_native.delete_result = False
    assert qt._update_geom(1, (0, 0), (2, 2)) is False


def test_base_quadtree_numpy_validation_and_rollbacks():
    qt = StubTree()
    with pytest.raises(TypeError):
        qt.insert_many_np([1, 2])  # type: ignore[arg-type]

    class FakeArray:
        __module__ = "numpy.testing"
        ndim = 1
        shape = ()
        dtype = "float32"

    with pytest.raises(TypeError):
        qt.insert_many_np(FakeArray())  # type: ignore[arg-type]

    native = StubNative()
    native.insert_many_np_result = 0  # only first inserted
    qt_fail = StubTree(native)
    arr = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        qt_fail.insert_many_np(arr)

    native.delete_result = True
    native.insert_result = False
    with pytest.raises(ValueError):
        qt_fail._update_geom(1, (0, 0), (2, 2))


def test_base_quadtree_from_bytes_rejects_future_version():
    data = build_container(
        fmt_ver=SERIALIZATION_FORMAT_VERSION + 1,
        dtype="f32",
        flags=0,
        capacity=1,
        max_depth=None,
        next_id=0,
        count=0,
        bounds=(0, 0, 1, 1),
        core=b"x",
    )
    with pytest.raises(SerializationError):
        StubTree.from_bytes(data)


def test_encode_items_section_edge_cases():
    assert _encode_items_section([]) == struct.pack("<BBI", 0, 0, 0)

    weird_item = Item(0, (0, 1, 2))  # type: ignore[arg-type]
    with pytest.raises(SerializationError):
        _encode_items_section([weird_item])  # type: ignore[arg-type]

    large_list = type("HugeList", (list,), {"__len__": lambda self: 0xFFFFFFFF + 1})(
        [PointItem(0, (0.0, 0.0))]
    )
    with pytest.raises(SerializationError):
        _encode_items_section(large_list)


def test_decode_items_section_error_paths():
    with pytest.raises(SerializationError):
        _decode_items_section(b"")

    empty = struct.pack("<BBI", 0, 0, 0)
    assert _decode_items_section(empty) == []

    truncated_point = struct.pack("<BBI", 0, 0, 1)
    with pytest.raises(SerializationError):
        _decode_items_section(truncated_point)

    truncated_rect = struct.pack("<BBI", 1, 0, 1)
    with pytest.raises(SerializationError):
        _decode_items_section(truncated_rect)

    with pytest.raises(SerializationError):
        _decode_items_section(struct.pack("<BBI", 5, 0, 1))


def test_decode_objects_section_rejects_bad_payloads():
    import pickle

    bad_type = pickle.dumps({"a": 1})
    with pytest.raises(SerializationError):
        _decode_objects_section(bad_type)

    bad_entries = pickle.dumps([("id", "obj")])
    with pytest.raises(SerializationError):
        _decode_objects_section(bad_entries)


def test_object_tree_insert_and_batch_failures_restore_free_list():
    native = StubNative()
    native.insert_result = False
    qt = StubObjTree(native)
    with pytest.raises(ValueError):
        qt.insert((2.0, 2.0, 3.0, 3.0))
    assert qt._store._free == [0]

    with pytest.raises(TypeError):
        qt.insert_many(np.zeros((1, 4), dtype=np.float32))  # type: ignore[arg-type]

    res = qt.insert_many([])
    assert res.count == 0
    assert res.start_id == 0
    assert res.end_id == -1

    native.insert_result = True
    native.insert_many_result = 0
    with pytest.raises(ValueError):
        qt.insert_many([(0.0, 0.0, 1.0, 1.0), (2.0, 2.0, 3.0, 3.0)])


def test_object_tree_numpy_validation_and_partial_failure():
    qt = StubObjTree()
    with pytest.raises(TypeError):
        qt.insert_many_np([(1.0, 1.0, 2.0, 2.0)])  # type: ignore[arg-type]

    class FakeArray:
        __module__ = "numpy.fake"
        ndim = 2
        shape = ()
        dtype = "float32"

    with pytest.raises(TypeError):
        qt.insert_many_np(FakeArray())  # type: ignore[arg-type]

    native = StubNative()
    native.insert_many_np_result = 0
    qt_fail = StubObjTree(native)
    coords = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        qt_fail.insert_many_np(coords)


def test_object_tree_query_and_neighbor_guardrails():
    native = StubNative()
    native.nn_result = (9, 0.0, 0.0)
    qt = StubObjTree(native)
    with pytest.raises(RuntimeError):
        qt.nearest_neighbor((0.0, 0.0))

    native.nn_list_result = [(7, 1.0, 1.0)]
    with pytest.raises(RuntimeError):
        qt.nearest_neighbors((0.0, 0.0), 1)

    native.nn_np_result = ("id", "coords")
    assert qt.nearest_neighbor_np((0.0, 0.0)) == ("id", "coords")

    native.nn_np_list_result = ("ids", "coords")
    assert qt.nearest_neighbors_np((0.0, 0.0), 2) == ("ids", "coords")


def test_object_tree_neighbor_success_and_none_paths():
    native = StubNative()
    qt = StubObjTree(native)
    assert qt.nearest_neighbor((0.0, 0.0)) is None

    item = Item(0, (0.0, 0.0), None)
    qt._store.add(item)
    native.nn_result = (0, 0.0, 0.0)
    native.nn_list_result = [(0, 0.0, 0.0)]
    assert qt.nearest_neighbor((0.0, 0.0)) is item
    assert qt.nearest_neighbors((0.0, 0.0), 1) == [item]


def test_object_tree_delete_branches_and_by_object_counts():
    native = StubNative()
    qt = StubObjTree(native)
    qt._store.add(Item(0, (0.0, 0.0), "obj"))
    qt._count = 1
    native.delete_result = False
    assert qt.delete(0) is False  # item exists but native delete failed
    assert qt.delete_by_object("obj") == 0

    qt2 = StubObjTree(native)
    fresh_id = qt2._store.alloc_id()
    qt2._store.add(Item(fresh_id, (1.0, 1.0), "obj"))
    qt2._count = 1
    native.delete_result = True
    assert qt2.delete_one_by_object("obj") is True


def test_object_tree_delete_update_and_contains_branches():
    qt = StubObjTree()
    assert qt.delete(0) is False
    assert qt.delete_one_by_object("missing") is False

    qt._store.add(Item(0, (0.0, 0.0), "obj"))
    qt._count = 1
    qt._stub_native.delete_result = True
    assert qt.delete(0) is True
    assert len(qt) == 0

    qt._stub_native.delete_result = True
    qt._stub_native.insert_result = False
    with pytest.raises(ValueError):
        qt._update_geom(1, (0, 0), (1, 1))

    qt._store.add(Item(1, (1.0, 2.0), None))
    qt._count = 1
    qt._stub_native.delete_result = True
    qt._stub_native.insert_result = True
    assert qt.get(99) is None
    with pytest.raises(KeyError):
        qt.attach(99, "x")

    qt._stub_native.query_result = [(1, 1.0, 2.0)]
    assert (1.0, 2.0) in qt

    qt._stub_native.query_result = [(2, 0.0, 0.0, 1.0, 1.0)]
    assert (0.0, 0.0, 1.0, 1.0) in qt

    qt._stub_native.delete_result = False
    assert qt._update_geom(5, (0, 0), (1, 1)) is False

    qt.clear()
    assert len(qt._store._arr) == 0


def test_object_tree_iter_helpers_and_boundaries():
    qt = StubObjTree()
    qt._store.add(Item(0, (0.0, 0.0), "a"))
    qt._store.add(Item(1, (1.0, 1.0), None))
    qt._count = 2
    assert qt.get_all_items()
    assert qt.get_all_objects() == ["a"]
    assert list(qt)  # __iter__
    assert qt.get_all_node_boundaries() == qt._stub_native.boundaries
    assert qt.get_inner_max_depth() == qt._stub_native.max_depth


def test_object_tree_serialization_sections_and_missing_items_section():
    native = StubNative()
    qt = StubObjTree(native, max_depth=3)
    qt._store.add(Item(0, (0.0, 0.0), obj={"k": "v"}))
    qt._count = 1

    data = qt.to_bytes(include_objects=True)
    parsed = parse_container(data)
    assert any(stype == SECTION_OBJECTS for stype, _ in parsed["sections"])

    with pytest.raises(SerializationError):
        StubObjTree.from_bytes(
            build_container(
                fmt_ver=SERIALIZATION_FORMAT_VERSION + 1,
                dtype="f32",
                flags=0,
                capacity=1,
                max_depth=None,
                next_id=0,
                count=0,
                bounds=(0, 0, 1, 1),
                core=b"c",
                extra_sections=[(1, b"\x00")],
            )
        )

    missing_items = build_container(
        fmt_ver=SERIALIZATION_FORMAT_VERSION,
        dtype="f32",
        flags=0,
        capacity=1,
        max_depth=None,
        next_id=0,
        count=0,
        bounds=(0, 0, 1, 1),
        core=b"c",
        extra_sections=None,
    )
    with pytest.raises(SerializationError):
        StubObjTree.from_bytes(missing_items)


def test_object_tree_from_bytes_ignores_objects_when_not_allowed():
    qt = StubObjTree(max_depth=1)
    qt._store.add(Item(0, (0.0, 0.0), obj={"s": "t"}))
    qt._count = 1
    data = qt.to_bytes(include_objects=True)
    clone = StubObjTree.from_bytes(data)
    assert all(it.obj is None for it in clone._store.items())

    clone_with = StubObjTree.from_bytes(data, allow_objects=True)
    assert any(it.obj is not None for it in clone_with._store.items())


def test_object_tree_from_bytes_with_only_objects_section_triggers_missing_items():
    data = build_container(
        fmt_ver=SERIALIZATION_FORMAT_VERSION,
        dtype="f32",
        flags=0,
        capacity=1,
        max_depth=None,
        next_id=0,
        count=0,
        bounds=(0, 0, 1, 1),
        core=b"c",
        extra_sections=[(SECTION_OBJECTS, b"pickles")],
    )
    with pytest.raises(SerializationError):
        StubObjTree.from_bytes(data)


def test_object_tree_from_bytes_sections_loop_hits_objects_branch():
    items = _encode_items_section([PointItem(0, (0.0, 0.0))])
    empty_objs = b"\x80\x04]"  # pickle for empty list
    data = build_container(
        fmt_ver=SERIALIZATION_FORMAT_VERSION,
        dtype="f32",
        flags=0,
        capacity=1,
        max_depth=None,
        next_id=0,
        count=1,
        bounds=(0, 0, 1, 1),
        core=b"c",
        extra_sections=[
            (SECTION_ITEMS, items),
            (SECTION_OBJECTS, empty_objs),
            (255, b"junk"),
        ],
    )
    clone = StubObjTree.from_bytes(data)
    assert len(clone._store._arr) == 1


def test_obj_store_edges_and_reverse_mapping_cleanup():
    store = ObjStore([PointItem(0, (0.0, 0.0)), PointItem(2, (1.0, 1.0))])
    assert store._free == [1]

    payload = store.to_dict()
    rebuilt = ObjStore.from_dict(payload, lambda i, g, o: PointItem(i, g, o))
    assert len(rebuilt) == len(store)

    with pytest.raises(AssertionError):
        store.add(PointItem(5, (5.0, 5.0)))

    store.add(PointItem(2, (1.0, 1.0), obj="a"))
    store.add(PointItem(2, (1.0, 1.0), obj="b"))
    assert store.contains_obj("a") is False
    assert store.contains_obj("b") is True

    assert store.pop_id(10) is None
    store._arr.append(None)
    store._objs.append(None)
    assert store.pop_id(len(store._arr) - 1) is None

    ids = [store.alloc_id(), store.alloc_id()]
    store.add(PointItem(ids[0], (2.0, 2.0)))
    store.add(PointItem(ids[1], (3.0, 3.0)))
    objs = ["o1", "o2"]
    store.add(PointItem(ids[1], (0.0, 0.0), obj=objs[1]))  # pyright: ignore[reportArgumentType]
    assert store.get_many_by_ids(ids, chunk=1)[0].id_ == ids[0]
    assert store.get_many_objects(ids) == [None, objs[1]]
    popped = store.pop_id(ids[1])
    assert popped
    assert popped.obj == objs[1]

    assert store.contains_id(ids[0]) is True
    assert store.contains_id(ids[1]) is False
    assert store.contains_obj(objs[1]) is False
    assert list(store.items_by_id())
    assert list(store.items())
    store.clear()
    assert len(store) == 0
    assert not store._free


def test_obj_store_handles_stale_reverse_mappings_and_missing_map():
    store = ObjStore()
    store._obj_to_ids[id("ghost")] = {0}
    store._arr.append(None)
    store._objs.append(None)
    assert store.by_obj_all("ghost") == []

    store.add(PointItem(0, (0.0, 0.0), obj="x"))
    # Remove reverse map manually to exercise branch where old obj not tracked
    store._obj_to_ids.clear()
    store.add(PointItem(0, (1.0, 1.0), obj="y"))
    store._obj_to_ids.clear()
    popped = store.pop_id(0)
    assert popped
    assert popped.obj == "y"


def test_obj_store_replaces_object_and_clears_reverse_map():
    store = ObjStore()
    store.add(PointItem(0, (0.0, 0.0), obj="first"))
    assert store.contains_obj("first") is True
    store.add(PointItem(0, (1.0, 1.0), obj="second"))
    assert store.contains_obj("first") is False
    assert store.contains_obj("second") is True


def test_obj_store_replace_branch_with_missing_reverse_entry():
    store = ObjStore()
    store.add(PointItem(0, (0.0, 0.0), obj="keep"))
    store._obj_to_ids.clear()
    store.add(PointItem(0, (0.0, 0.0), obj="new"))
    assert store.contains_obj("new") is True


def test_obj_store_replacement_removes_old_object_mapping():
    store = ObjStore()
    obj1 = object()
    obj2 = object()
    store.add(PointItem(0, (0.0, 0.0), obj=obj1))
    assert store.contains_obj(obj1) is True
    store.add(PointItem(0, (1.0, 1.0), obj=obj2))
    assert store.contains_obj(obj1) is False
    assert store.contains_obj(obj2) is True


def test_obj_store_replacement_preserves_shared_object_mapping():
    store = ObjStore()
    shared = object()
    store.add(PointItem(0, (0.0, 0.0), obj=shared))
    store.add(PointItem(1, (1.0, 1.0), obj=shared))
    store.add(PointItem(0, (2.0, 2.0), obj="other"))
    assert store.contains_obj(shared) is True


@pytest.mark.parametrize(
    ("cls", "args"),
    [
        (QuadTree, {"bounds": (0, 0, 1, 1), "capacity": 1}),
        (RectQuadTree, {"bounds": (0, 0, 1, 1), "capacity": 1}),
        (QuadTreeObjects, {"bounds": (0, 0, 1, 1), "capacity": 1}),
        (RectQuadTreeObjects, {"bounds": (0, 0, 1, 1), "capacity": 1}),
    ],
)
def test_public_classes_reject_unknown_dtype(cls, args):
    with pytest.raises(TypeError):
        cls(dtype="bad", **args)


def test_new_native_from_bytes_rejects_bad_dtype():
    with pytest.raises(TypeError):
        QuadTree._new_native_from_bytes(b"x", "bad")
    with pytest.raises(TypeError):
        RectQuadTree._new_native_from_bytes(b"x", "bad")
    with pytest.raises(TypeError):
        QuadTreeObjects._new_native_from_bytes(b"x", "bad")
    with pytest.raises(TypeError):
        RectQuadTreeObjects._new_native_from_bytes(b"x", "bad")


def test_rect_quadtree_contains_branch():
    rqt = RectQuadTree((0.0, 0.0, 10.0, 10.0), capacity=4, dtype="f32")
    rect = (1.0, 1.0, 2.0, 2.0)
    rqt.insert(rect)
    assert rect in rqt


def test_point_object_update_branches(monkeypatch):
    qt = QuadTreeObjects((0.0, 0.0, 10.0, 10.0), capacity=4)
    assert qt.update(999, 1.0, 1.0) is False
    rid = qt.insert((1.0, 1.0), obj="x")
    monkeypatch.setattr(qt, "_update_geom", lambda *_, **__: False)
    assert qt.update(rid, 2.0, 2.0) is False


def test_rect_object_update_branches(monkeypatch):
    rqt = RectQuadTreeObjects((0.0, 0.0, 10.0, 10.0), capacity=4)
    assert rqt.update(123, 1.0, 1.0, 2.0, 2.0) is False
    rid = rqt.insert((1.0, 1.0, 2.0, 2.0))
    monkeypatch.setattr(rqt, "_update_geom", lambda *_, **__: False)
    assert rqt.update(rid, 3.0, 3.0, 4.0, 4.0) is False
