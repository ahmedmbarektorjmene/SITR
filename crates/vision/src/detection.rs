use crate::geometry::ScreenRect;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObjectClass {
    Male = 0,
    Female = 1,
}

impl ObjectClass {
    pub fn from_id(id: i32) -> Option<Self> {
        match id {
            0 => Some(Self::Male),
            1 => Some(Self::Female),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Male => "Male",
            Self::Female => "Female",
        }
    }

    pub fn all() -> &'static [ObjectClass] {
        &[Self::Male, Self::Female]
    }
}

#[derive(Debug, Clone)]
pub struct Detection {
    pub class: ObjectClass,
    pub confidence: f32,
    pub screen_rect: ScreenRect,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverMode {
    Blur,
    SolidColor,
    BackgroundColor,
}

#[derive(Debug, Clone)]
pub struct CoverRect {
    pub screen_rect: ScreenRect,
    pub mode: CoverMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DetectionState {
    Active,
    #[default]
    Sleep,
}

#[derive(Debug, Clone)]
pub struct FrameData {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub data: Vec<u8>,
    pub format: PixelFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    Bgr,
    Rgb,
    Bgra,
    Rgba,
}

impl PixelFormat {
    pub fn bytes_per_pixel(self) -> u32 {
        match self {
            PixelFormat::Bgr | PixelFormat::Rgb => 3,
            PixelFormat::Bgra | PixelFormat::Rgba => 4,
        }
    }
}

impl FrameData {
    pub fn new_bgr(width: u32, height: u32, data: Vec<u8>) -> Self {
        let stride = width * 3;
        Self {
            width,
            height,
            stride,
            data,
            format: PixelFormat::Bgr,
        }
    }

    pub fn new_rgb(width: u32, height: u32, data: Vec<u8>) -> Self {
        let stride = width * 3;
        Self {
            width,
            height,
            stride,
            data,
            format: PixelFormat::Rgb,
        }
    }

    pub fn new_with_stride(
        width: u32,
        height: u32,
        stride: u32,
        data: Vec<u8>,
        format: PixelFormat,
    ) -> Self {
        Self {
            width,
            height,
            stride,
            data,
            format,
        }
    }

    pub fn pixel_at(&self, x: u32, y: u32) -> Option<[u8; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let bpp = self.format.bytes_per_pixel();
        let row_offset = (y * self.stride + x * bpp) as usize;
        if row_offset + 2 >= self.data.len() {
            return None;
        }
        match self.format {
            PixelFormat::Bgr => Some([
                self.data[row_offset + 2],
                self.data[row_offset + 1],
                self.data[row_offset],
            ]),
            PixelFormat::Rgb => Some([
                self.data[row_offset],
                self.data[row_offset + 1],
                self.data[row_offset + 2],
            ]),
            _ => None,
        }
    }

    pub fn region(&self, rect: &ScreenRect) -> Option<FrameData> {
        let x = rect.x.max(0) as u32;
        let y = rect.y.max(0) as u32;
        if x + rect.width > self.width || y + rect.height > self.height {
            return None;
        }

        let bpp = self.format.bytes_per_pixel();
        let row_bytes = rect.width * bpp;
        let mut region_data = Vec::with_capacity((rect.height * row_bytes) as usize);
        for row in y..y + rect.height {
            let start = (row * self.stride + x * bpp) as usize;
            let end = start + row_bytes as usize;
            if end <= self.data.len() {
                region_data.extend_from_slice(&self.data[start..end]);
            }
        }

        Some(FrameData {
            width: rect.width,
            height: rect.height,
            stride: row_bytes,
            data: region_data,
            format: self.format,
        })
    }
}

pub fn extract_dominant_color(
    frame: &FrameData,
    rect: &ScreenRect,
) -> Option<crate::geometry::ColorRgb> {
    let region = frame.region(rect)?;

    if region.data.is_empty() {
        return None;
    }

    let mut histogram = [[0u32; 32]; 32 * 32];
    let _pixel_count = (region.width * region.height) as usize;

    for i in (0..region.data.len()).step_by(3) {
        let b = region.data[i] as usize;
        let g = region.data[i + 1] as usize;
        let r = region.data[i + 2] as usize;

        let ri = r >> 3;
        let gi = g >> 3;
        let bi = b >> 3;
        let idx = (ri * 32 + gi) * 32 + bi;
        histogram[idx][0] += 1;
        histogram[idx][1] += r as u32;
        histogram[idx][2] += g as u32;
    }

    let mut best_idx = 0;
    let mut best_count = 0;
    for (i, entry) in histogram.iter().enumerate() {
        if entry[0] > best_count {
            best_count = entry[0];
            best_idx = i;
        }
    }

    if best_count == 0 {
        return None;
    }

    let ri = best_idx / (32 * 32);
    let gi = (best_idx / 32) % 32;
    let bi = best_idx % 32;

    let r = ((ri as u32 * 32 + 16) * 255 / (31 * 32 + 16)).min(255) as u8;
    let g = ((gi as u32 * 32 + 16) * 255 / (31 * 32 + 16)).min(255) as u8;
    let b = ((bi as u32 * 32 + 16) * 255 / (31 * 32 + 16)).min(255) as u8;

    Some(crate::geometry::ColorRgb::new(r, g, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::ColorRgb;

    #[test]
    fn test_object_class() {
        assert_eq!(ObjectClass::from_id(0), Some(ObjectClass::Male));
        assert_eq!(ObjectClass::from_id(1), Some(ObjectClass::Female));
        assert_eq!(ObjectClass::from_id(2), None);
    }

    #[test]
    fn test_color_rgb_hex() {
        let c = ColorRgb::new(255, 128, 0);
        assert_eq!(c.to_hex(), "#ff8000");
        let c2 = ColorRgb::from_hex("#ff8000").unwrap();
        assert_eq!(c, c2);
    }
}
