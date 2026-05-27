use fastquadtree::serialization::{
    NATIVE_FORMAT_VERSION, NATIVE_KIND_POINT, NATIVE_KIND_RECT, NATIVE_MAGIC,
};
use fastquadtree::{Item, Point, QuadTree, Rect, RectItem, RectQuadTree};

#[test]
fn quadtree_roundtrip_bytes() {
    // Build a small tree
    let mut qt = QuadTree::new(
        Rect { min_x: 0.0, min_y: 0.0, max_x: 10.0, max_y: 10.0 },
        4,
        8
    );
    for (i, (x, y)) in [(1.0, 1.0), (2.0, 3.0), (7.5, 8.5), (9.0, 0.5)].into_iter().enumerate() {
        qt.insert(Item { id: i as u64 + 1, point: Point { x, y } });
    }

    // Serialize
    let bytes = qt.to_bytes().expect("serialize quadtree");
    assert_eq!(&bytes[..4], NATIVE_MAGIC);
    assert_eq!(
        u16::from_le_bytes([bytes[4], bytes[5]]),
        NATIVE_FORMAT_VERSION
    );
    assert_eq!(bytes[6], NATIVE_KIND_POINT);
    assert_eq!(bytes[7], 0);

    // Deserialize
    let qt2 = QuadTree::from_bytes(&bytes).expect("deserialize quadtree");

    // Basic invariants
    assert_eq!(qt.count_items(), qt2.count_items());

    // Query equality for a region
    let rect = Rect { min_x: 0.0, min_y: 0.0, max_x: 5.0, max_y: 5.0 };
    let a: Vec<_> = qt.query(rect).into_iter().map(|(id, _, _)| id).collect();
    let b: Vec<_> = qt2.query(rect).into_iter().map(|(id, _, _)| id).collect();
    assert_eq!(a, b);

    // Nearest neighbor equality
    let nn1 = qt.nearest_neighbor(Point { x: 1.2, y: 1.1 }).map(|it| it.id);
    let nn2 = qt2.nearest_neighbor(Point { x: 1.2, y: 1.1 }).map(|it| it.id);
    assert_eq!(nn1, nn2);
}


#[test]
fn rectquadtree_roundtrip_bytes() {
    // Build a small tree
    let mut qt = RectQuadTree::new(
        Rect { min_x: 0.0, min_y: 0.0, max_x: 10.0, max_y: 10.0 },
        4,
        8
    );
    for (i, (x, y)) in [(1.0, 1.0), (2.0, 3.0), (7.5, 8.5), (9.0, 0.5)].into_iter().enumerate() {
        qt.insert(RectItem { id: i as u64 + 1, rect: Rect { min_x: x, min_y: y, max_x: x + 1.0, max_y: y + 1.0 } });
    }

    // Serialize
    let bytes = qt.to_bytes().expect("serialize quadtree");
    assert_eq!(&bytes[..4], NATIVE_MAGIC);
    assert_eq!(
        u16::from_le_bytes([bytes[4], bytes[5]]),
        NATIVE_FORMAT_VERSION
    );
    assert_eq!(bytes[6], NATIVE_KIND_RECT);
    assert_eq!(bytes[7], 0);

    // Deserialize
    let qt2 = RectQuadTree::from_bytes(&bytes).expect("deserialize quadtree");

    // Basic invariants
    assert_eq!(qt.count_items(), qt2.count_items());

    // Query equality for a region
    let rect = Rect { min_x: 0.0, min_y: 0.0, max_x: 5.0, max_y: 5.0 };
    let a: Vec<_> = qt.query(rect).into_iter().map(|it| it.0).collect();
    let b: Vec<_> = qt2.query(rect).into_iter().map(|it| it.0).collect();
    assert_eq!(a, b);
}

#[test]
fn quadtree_rejects_invalid_serialization_payloads() {
    let mut qt = QuadTree::new(
        Rect { min_x: 0.0, min_y: 0.0, max_x: 10.0, max_y: 10.0 },
        4,
        8,
    );
    qt.insert(Item { id: 1, point: Point { x: 1.0, y: 1.0 } });

    let bytes = qt.to_bytes().expect("serialize quadtree");

    assert!(QuadTree::<f64>::from_bytes(b"not-fqtw").is_err());

    let mut unsupported_version = bytes.clone();
    unsupported_version[4..6].copy_from_slice(&2u16.to_le_bytes());
    assert!(QuadTree::<f64>::from_bytes(&unsupported_version).is_err());

    let mut wrong_kind = bytes.clone();
    wrong_kind[6] = NATIVE_KIND_RECT;
    assert!(QuadTree::<f64>::from_bytes(&wrong_kind).is_err());

    let mut unsupported_flags = bytes.clone();
    unsupported_flags[7] = 1;
    assert!(QuadTree::<f64>::from_bytes(&unsupported_flags).is_err());

    assert!(QuadTree::<f64>::from_bytes(&bytes[..bytes.len() - 1]).is_err());

    let mut trailing = bytes;
    trailing.push(0);
    assert!(QuadTree::<f64>::from_bytes(&trailing).is_err());
}

#[test]
fn quadtree_decode_preallocation_limit_can_be_overridden() {
    let mut qt = QuadTree::new(
        Rect { min_x: 0.0, min_y: 0.0, max_x: 200.0, max_y: 200.0 },
        1_000,
        8,
    );
    for id in 0..100 {
        qt.insert(Item {
            id,
            point: Point { x: id as f64, y: id as f64 },
        });
    }

    let bytes = qt.to_bytes().expect("serialize quadtree");
    assert!(QuadTree::<f64>::from_bytes_with_preallocation_limit::<1024>(&bytes).is_err());
    assert!(QuadTree::<f64>::from_bytes(&bytes).is_ok());
    assert!(QuadTree::<f64>::from_bytes_unlimited(&bytes).is_ok());
}
