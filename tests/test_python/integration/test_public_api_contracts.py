import pytest

import fastquadtree as fqt


def test_public_api_all_exports():
    expected = [
        "InsertResult",
        "Item",
        "PointItem",
        "QuadTree",
        "QuadTreeObjects",
        "RectItem",
        "RectQuadTree",
        "RectQuadTreeObjects",
        "Quadtree",  # Lowercase added in 2.0.2
        "QuadtreeObjects",
        "Rectquadtree",
        "RectquadtreeObjects",
    ]
    assert sorted(fqt.__all__) == sorted(expected)


@pytest.mark.parametrize(
    "cls",
    [
        fqt.QuadTree,
        fqt.RectQuadTree,
        fqt.QuadTreeObjects,
        fqt.RectQuadTreeObjects,
    ],
)
def test_no_track_objects_param(cls, bounds):
    with pytest.raises(TypeError):
        cls(bounds, capacity=1, track_objects=True)  # type: ignore[arg-type]


def test_query_has_no_as_items_and_custom_id_warning_not_enforced(bounds):
    qt = fqt.QuadTree(bounds, capacity=4)
    qt.insert((1.0, 1.0))
    with pytest.raises(TypeError):
        qt.query(bounds, as_items=True)  # type: ignore[arg-type]

    # Custom ID collisions allowed (documented warning)
    qt.insert((2.0, 2.0), id_=0)
    ids = [pt[0] for pt in qt.query(bounds)]
    assert ids.count(0) == 2
