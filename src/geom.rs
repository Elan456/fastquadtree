use core::ops::{Add, Sub, Mul, Div};
use num_traits::{Zero, One};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

// Creating a set of traits for coordinate types
// Requires basic arithmetic operations and ordering
pub trait Coord:
    Copy + PartialOrd + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Zero + One + Serialize + DeserializeOwned
{}

impl<T> Coord for T where
    T: Copy + PartialOrd + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Zero + One + Serialize + DeserializeOwned
{}

// Generic mid function for all Coord types
#[inline(always)]
pub fn mid<T: Coord>(a: T, b: T) -> T {
    // a + (b - a) / 2
    a + (b - a) / (T::one() + T::one())  // 1 + 1 = 2 for Div
}

#[derive(Copy, Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct Point<T: Coord> {
    pub x: T,
    pub y: T,
}

#[derive(Copy, Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct Rect<T: Coord> {
    pub min_x: T,
    pub min_y: T,
    pub max_x: T,
    pub max_y: T,
}

impl<T: Coord> Rect<T> {
    pub fn contains(&self, point: &Point<T>) -> bool {
        return point.x >= self.min_x && point.x < self.max_x && point.y >= self.min_y && point.y < self.max_y;
    }

    // Check if two Rect overlap at all
    pub fn intersects(&self, other: &Rect<T>) -> bool {
        return self.min_x < other.max_x && self.max_x > other.min_x && self.min_y < other.max_y && self.max_y > other.min_y
    }
}

pub fn dist_sq_point_to_rect<T: Coord>(p: &Point<T>, r: &Rect<T>) -> T {
    let dx = if p.x < r.min_x {
        r.min_x - p.x
    } else if p.x > r.max_x {
        p.x - r.max_x
    } else {
        T::zero()
    };

    let dy = if p.y < r.min_y {
        r.min_y - p.y
    } else if p.y > r.max_y {
        p.y - r.max_y
    } else {
        T::zero()
    };

    dx * dx + dy * dy
}

pub fn dist_sq_points<T: Coord>(a: &Point<T>, b: &Point<T>) -> T {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}