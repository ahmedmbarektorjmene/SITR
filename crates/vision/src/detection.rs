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
    /// Resolved solid/dominant color (ARGB 217 alpha handled by renderer).
    pub resolved_color: Option<crate::geometry::ColorRgb>,
    /// Blurred pixel data (BGR, width*height*3) for blur mode.
    pub blur_data: Option<Vec<u8>>,
}

impl CoverRect {
    pub fn new_solid(rect: ScreenRect, color: crate::geometry::ColorRgb) -> Self {
        Self {
            screen_rect: rect,
            mode: CoverMode::SolidColor,
            resolved_color: Some(color),
            blur_data: None,
        }
    }
    pub fn new_dominant(rect: ScreenRect, color: crate::geometry::ColorRgb) -> Self {
        Self {
            screen_rect: rect,
            mode: CoverMode::BackgroundColor,
            resolved_color: Some(color),
            blur_data: None,
        }
    }
    pub fn new_blur(rect: ScreenRect, blurred_bgr: Vec<u8>) -> Self {
        Self {
            screen_rect: rect,
            mode: CoverMode::Blur,
            resolved_color: None,
            blur_data: Some(blurred_bgr),
        }
    }
    /// Fallback: mode only (used by legacy tests where color not checked)
    pub fn new_mode_only(rect: ScreenRect, mode: CoverMode) -> Self {
        Self {
            screen_rect: rect,
            mode,
            resolved_color: None,
            blur_data: None,
        }
    }
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

/// Port of Python `Porda-AI-python/main.py:655-659`:
/// ```python
/// pixels = frame[y+20:y+80:3, x:x+50:3, ::-1].reshape(-1, 3)
/// unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
/// dominant_color = unique_colors[np.argmax(counts)]
/// ```
/// Exact subsampled region (60x50 window at top-left of detection) -> exact RGB counting.
/// Returns `None` if the subsampled window yields no pixels (e.g. tiny / off-screen rect),
/// caller should fallback to solid.
pub fn extract_dominant_color(
    frame: &FrameData,
    rect: &ScreenRect,
) -> Option<crate::geometry::ColorRgb> {
    use std::collections::HashMap;

    // Python fixed window: y+20..y+80 step3, x..x+50 step3 (exclusive end)
    let start_y = rect.y + 20;
    let end_y = rect.y + 80;
    let start_x = rect.x;
    let end_x = rect.x + 50;

    let mut counts: HashMap<[u8; 3], u32> = HashMap::with_capacity(512);

    // Collect subsampled pixels as RGB (Python `::-1` converts BGR->RGB)
    for y in (start_y..end_y).step_by(3) {
        if y < 0 || y >= frame.height as i32 {
            continue;
        }
        for x in (start_x..end_x).step_by(3) {
            if x < 0 || x >= frame.width as i32 {
                continue;
            }
            // `pixel_at` already returns RGB regardless of Bgr/Rgb format
            if let Some(rgb) = frame.pixel_at(x as u32, y as u32) {
                *counts.entry(rgb).or_insert(0) += 1;
            }
        }
    }

    if !counts.is_empty() {
        let (&dominant, _) = counts.iter().max_by_key(|(_, &c)| c).unwrap();
        return Some(crate::geometry::ColorRgb::new(
            dominant[0],
            dominant[1],
            dominant[2],
        ));
    }

    // Fallback for tiny/off-screen rects where subsampled window is empty:
    // exact counting over the whole rect (not quantized) to avoid panic,
    // still strictly from the same frame (no stale buffer).
    let region = frame.region(rect)?;
    if region.data.is_empty() {
        return None;
    }
    let mut fallback: HashMap<[u8; 3], u32> = HashMap::with_capacity(1024);
    for chunk in region.data.chunks_exact(3) {
        // region.data is BGR if frame was Bgr, else Rgb - need to map to RGB
        let is_bgr = matches!(region.format, PixelFormat::Bgr);
        let rgb = if is_bgr {
            [chunk[2], chunk[1], chunk[0]]
        } else {
            [chunk[0], chunk[1], chunk[2]]
        };
        *fallback.entry(rgb).or_insert(0) += 1;
    }
    if fallback.is_empty() {
        return None;
    }
    let (&dominant, _) = fallback.iter().max_by_key(|(_, &c)| c).unwrap();
    Some(crate::geometry::ColorRgb::new(
        dominant[0],
        dominant[1],
        dominant[2],
    ))
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
