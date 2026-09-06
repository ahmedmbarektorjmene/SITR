#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenRect {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
}

impl ScreenRect {
    pub fn new(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn right(&self) -> i32 {
        self.x + self.width as i32
    }

    pub fn bottom(&self) -> i32 {
        self.y + self.height as i32
    }

    pub fn contains_point(&self, px: i32, py: i32) -> bool {
        px >= self.x && px < self.right() && py >= self.y && py < self.bottom()
    }

    pub fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    pub fn intersection(&self, other: &ScreenRect) -> Option<ScreenRect> {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = self.right().min(other.right());
        let y2 = self.bottom().min(other.bottom());

        if x2 > x1 && y2 > y1 {
            Some(ScreenRect {
                x: x1,
                y: y1,
                width: (x2 - x1) as u32,
                height: (y2 - y1) as u32,
            })
        } else {
            None
        }
    }

    pub fn contains(&self, other: &ScreenRect) -> bool {
        other.x >= self.x
            && other.y >= self.y
            && other.right() <= self.right()
            && other.bottom() <= self.bottom()
    }

    pub fn subtract(&self, other: &ScreenRect) -> Vec<ScreenRect> {
        match self.intersection(other) {
            None => vec![*self],
            Some(overlap) => {
                let mut rects = Vec::with_capacity(4);

                if overlap.x > self.x {
                    rects.push(ScreenRect {
                        x: self.x,
                        y: self.y,
                        width: (overlap.x - self.x) as u32,
                        height: self.height,
                    });
                }

                if overlap.right() < self.right() {
                    rects.push(ScreenRect {
                        x: overlap.right(),
                        y: self.y,
                        width: (self.right() - overlap.right()) as u32,
                        height: self.height,
                    });
                }

                if overlap.y > self.y {
                    rects.push(ScreenRect {
                        x: overlap.x,
                        y: self.y,
                        width: overlap.width,
                        height: (overlap.y - self.y) as u32,
                    });
                }

                if overlap.bottom() < self.bottom() {
                    rects.push(ScreenRect {
                        x: overlap.x,
                        y: overlap.bottom(),
                        width: overlap.width,
                        height: (self.bottom() - overlap.bottom()) as u32,
                    });
                }

                rects
            }
        }
    }

    pub fn pad(&self, padding: u32) -> ScreenRect {
        ScreenRect {
            x: self.x - padding as i32,
            y: self.y - padding as i32,
            width: self.width + padding * 2,
            height: self.height + padding * 2,
        }
    }

    pub fn scale(&self, sx: f32, sy: f32) -> ScreenRect {
        ScreenRect {
            x: (self.x as f32 * sx) as i32,
            y: (self.y as f32 * sy) as i32,
            width: (self.width as f32 * sx) as u32,
            height: (self.height as f32 * sy) as u32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScreenPoint {
    pub x: i32,
    pub y: i32,
}

impl ScreenPoint {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ColorRgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl ColorRgb {
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub fn from_tuple((r, g, b): (u8, u8, u8)) -> Self {
        Self { r, g, b }
    }

    pub fn to_tuple(self) -> (u8, u8, u8) {
        (self.r, self.g, self.b)
    }

    pub fn to_hex(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }

    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            return None;
        }
        let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
        Some(Self { r, g, b })
    }
}

impl Default for ColorRgb {
    fn default() -> Self {
        Self { r: 0, g: 0, b: 255 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MonitorInfo {
    pub id: u32,
    pub bounds: ScreenRect,
    pub work_area: ScreenRect,
    pub is_primary: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersection() {
        let a = ScreenRect::new(0, 0, 100, 100);
        let b = ScreenRect::new(50, 50, 100, 100);
        let inter = a.intersection(&b).unwrap();
        assert_eq!(inter, ScreenRect::new(50, 50, 50, 50));
    }

    #[test]
    fn test_no_intersection() {
        let a = ScreenRect::new(0, 0, 10, 10);
        let b = ScreenRect::new(20, 20, 10, 10);
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_subtract() {
        let a = ScreenRect::new(0, 0, 100, 100);
        let b = ScreenRect::new(25, 25, 50, 50);
        let parts = a.subtract(&b);
        assert_eq!(parts.len(), 4);
    }

    #[test]
    fn test_contains() {
        let outer = ScreenRect::new(0, 0, 100, 100);
        let inner = ScreenRect::new(10, 10, 20, 20);
        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
    }
}
