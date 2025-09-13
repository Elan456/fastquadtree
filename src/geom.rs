#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Rect {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl Rect {
    pub fn contains(&self, point: &Point) -> bool {
        return point.x >= self.min_x && point.x < self.max_x && point.y >= self.min_y && point.y < self.max_y;
    }

    // Check if two Rect overlap at all
    pub fn intersects(&self, other: &Rect) -> bool {
        return self.min_x < other.max_x && self.max_x > other.min_x && self.min_y < other.max_y && self.max_y > other.min_y
    }
}

