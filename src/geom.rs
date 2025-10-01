#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct Rect {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
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

pub fn dist_sq_point_to_rect(p: &Point, r: &Rect) -> f32 {
    let dx = if p.x < r.min_x {
        r.min_x - p.x
    } else if p.x > r.max_x {
        p.x - r.max_x
    } else {
        0.0
    };

    let dy = if p.y < r.min_y {
        r.min_y - p.y
    } else if p.y > r.max_y {
        p.y - r.max_y
    } else {
        0.0
    };

    dx * dx + dy * dy
}

pub fn dist_sq_points(a: &Point, b: &Point) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    dx * dx + dy * dy
}