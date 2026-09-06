use crate::geometry::ScreenRect;

use fast_image_resize::images::{Image, ImageRef};
use fast_image_resize::{FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer};
/// Mirrors Python `MainWindow.add_padding` in `Porda-AI/Porda-AI/main.py:782-801`.
///
/// Returns `(padded_data, x_ratio, y_ratio)` where `x_ratio = src_width/new_w`
/// and `y_ratio = src_height/new_h`. When the early-return optimization triggers
/// (image already close to network size), returns original data with ratios `1,1`.
pub fn resize_and_pad(
    data: &[u8],
    src_width: u32,
    src_height: u32,
    target_width: u32,
    target_height: u32,
) -> (Vec<u8>, f32, f32) {
    if src_width == 0 || src_height == 0 || target_width == 0 || target_height == 0 {
        return (vec![], 1.0, 1.0);
    }

    let scale =
        (target_height as f32 / src_height as f32).min(target_width as f32 / src_width as f32);
    let new_w = (src_width as f32 * scale) as u32;
    let new_h = (src_height as f32 * scale) as u32;

    if new_w == 0 || new_h == 0 {
        return (
            vec![0u8; (target_width * target_height * 3) as usize],
            1.0,
            1.0,
        );
    }

    let bottom = target_height - new_h;
    let right = target_width - new_w;

    if (bottom < 55 && right == 0) || (right < 70 && bottom == 0) {
        return (data.to_vec(), 1.0, 1.0);
    }

    // 1. Borrow source slice without allocation using ImageRef::new
    let src_image = ImageRef::new(src_width, src_height, data, PixelType::U8x3)
        .expect("Failed to create src image view");

    // 2. Prepare destination image
    let mut dst_image = Image::new(new_w, new_h, PixelType::U8x3);

    // 3. Configure and execute resizer
    let mut resizer = Resizer::new();
    let mut options = ResizeOptions::default();
    options.algorithm = ResizeAlg::Convolution(FilterType::Bilinear);

    resizer
        .resize(&src_image, &mut dst_image, &options)
        .expect("Resize failed");

    let resized_data = dst_image.into_vec();

    // 4. Zero-padded canvas matching OpenCV copyMakeBorder layout
    let mut padded = vec![0u8; (target_width * target_height * 3) as usize];
    for y in 0..new_h {
        let src_start = (y * new_w * 3) as usize;
        let src_end = src_start + (new_w * 3) as usize;
        let dst_start = (y * target_width * 3) as usize;
        let dst_end = dst_start + (new_w * 3) as usize;
        if src_end <= resized_data.len() && dst_end <= padded.len() {
            padded[dst_start..dst_end].copy_from_slice(&resized_data[src_start..src_end]);
        }
    }

    let x_ratio = src_width as f32 / new_w as f32;
    let y_ratio = src_height as f32 / new_h as f32;

    (padded, x_ratio, y_ratio)
}

pub fn nms(boxes: &mut Vec<ScreenRect>, scores: &mut Vec<f32>, threshold: f32) {
    if boxes.is_empty() {
        return;
    }

    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut suppressed = vec![false; indices.len()];

    for i in 0..indices.len() {
        if suppressed[i] {
            continue;
        }
        for j in (i + 1)..indices.len() {
            if suppressed[j] {
                continue;
            }
            let a = boxes[indices[i]];
            let b = boxes[indices[j]];

            if let Some(inter) = a.intersection(&b) {
                let inter_area = inter.area() as f32;
                let union_area = a.area() as f32 + b.area() as f32 - inter_area;
                let iou = if union_area > 0.0 {
                    inter_area / union_area
                } else {
                    0.0
                };

                if iou > threshold {
                    suppressed[j] = true;
                }
            }
        }
    }

    let mut write_idx = 0;
    for i in 0..indices.len() {
        if !suppressed[i] {
            boxes[write_idx] = boxes[indices[i]];
            scores[write_idx] = scores[indices[i]];
            write_idx += 1;
        }
    }
    boxes.truncate(write_idx);
    scores.truncate(write_idx);
}

/// Box blur mimicking Python `cv2.blur(image, (k,k))` where `k = w // 2`.
/// Uses integral image for O(w*h) performance regardless of kernel size.
/// Kernel is clamped to `width`/`height` and preserves BGR ordering.
pub fn blur_region(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    if data.is_empty() || width == 0 || height == 0 {
        return vec![];
    }
    if data.len() < (width * height * 3) as usize {
        return vec![];
    }

    // Python: k = w // 2 . Ensure odd-ish but OpenCV allows even.
    let mut k = (width / 2).max(1);
    // Clamp to region size to avoid degenerate windows
    k = k.min(width).min(height);
    if k <= 1 {
        // 1x1 kernel = no-op, return clone to avoid modifying original
        return data.to_vec();
    }
    // Ensure kernel >=1 and handle even: use k as window size (OpenCV blur k x k)
    // Window for pixel (x,y) is [x - k/2 , x + k/2 + k%2)
    let half = (k / 2) as i32;
    let tail = (k - k / 2) as i32; // handles even/odd correctly

    let w = width as usize;
    let h = height as usize;

    // Integral images for B,G,R (u64 to avoid overflow: max 255*1920*1200 ~589M fits u32 but use u64)
    let iw = w + 1;
    let ih = h + 1;
    let size = iw * ih;
    let mut int_b = vec![0u64; size];
    let mut int_g = vec![0u64; size];
    let mut int_r = vec![0u64; size];

    // Build integral: int[y+1][x+1] = sum of rect (0,0)-(x,y)
    for y in 0..h {
        let mut row_b: u64 = 0;
        let mut row_g: u64 = 0;
        let mut row_r: u64 = 0;
        for x in 0..w {
            let idx = (y * w + x) * 3;
            let b = data[idx] as u64;
            let g = data[idx + 1] as u64;
            let r = data[idx + 2] as u64;
            row_b += b;
            row_g += g;
            row_r += r;
            let pos = (y + 1) * iw + (x + 1);
            let above = y * iw + (x + 1);
            int_b[pos] = int_b[above] + row_b;
            int_g[pos] = int_g[above] + row_g;
            int_r[pos] = int_r[above] + row_r;
        }
    }

    let mut out = vec![0u8; data.len()];

    for y in 0..h {
        let y0 = (y as i32 - half).max(0) as usize;
        let y1 = (y as i32 + tail).min(h as i32) as usize;
        for x in 0..w {
            let x0 = (x as i32 - half).max(0) as usize;
            let x1 = (x as i32 + tail).min(w as i32) as usize;

            let area = ((x1 - x0) * (y1 - y0)) as u64;
            if area == 0 {
                continue;
            }
            // Integral rect sum: I[y1][x1] - I[y0][x1] - I[y1][x0] + I[y0][x0]
            let a = y1 * iw + x1;
            let b = y0 * iw + x1;
            let c = y1 * iw + x0;
            let d = y0 * iw + x0;
            let sum_b = int_b[a] + int_b[d] - int_b[b] - int_b[c];
            let sum_g = int_g[a] + int_g[d] - int_g[b] - int_g[c];
            let sum_r = int_r[a] + int_r[d] - int_r[b] - int_r[c];

            let idx = (y * w + x) * 3;
            out[idx] = (sum_b / area) as u8;
            out[idx + 1] = (sum_g / area) as u8;
            out[idx + 2] = (sum_r / area) as u8;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_and_pad_early_return() {
        // 100x100 -> 320x320: new_w=320,new_h=320,bottom=0,right=0 => early return
        let data = vec![128u8; 100 * 100 * 3];
        let (padded, xr, yr) = resize_and_pad(&data, 100, 100, 320, 320);
        // Early return: original data with ratios 1,1
        assert_eq!(padded.len(), 100 * 100 * 3);
        assert!((xr - 1.0).abs() < 0.01);
        assert!((yr - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_resize_and_pad_needs_padding() {
        // 800x600 -> 544x320: scale=0.533, new 426x320, bottom 0, right 118 => needs pad
        let data = vec![128u8; 800 * 600 * 3];
        let (padded, xr, yr) = resize_and_pad(&data, 800, 600, 544, 320);
        assert_eq!(padded.len(), 544 * 320 * 3);
        assert!((xr - (800.0 / 426.0)).abs() < 0.01);
        assert!((yr - (600.0 / 320.0)).abs() < 0.01);
    }

    #[test]
    fn test_resize_and_pad_1920x1200_to_544x320_early_return() {
        // 1920x1200 -> 544x320: scale 0.266, new 512x320, bottom 0,right 32 => early return
        let data = vec![128u8; 1920 * 1200 * 3];
        let (padded, xr, yr) = resize_and_pad(&data, 1920, 1200, 544, 320);
        assert_eq!(padded.len(), 1920 * 1200 * 3);
        assert!((xr - 1.0).abs() < 0.01);
        assert!((yr - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_nms_suppresses_overlapping() {
        let mut boxes = vec![
            ScreenRect::new(0, 0, 100, 100),
            ScreenRect::new(10, 10, 100, 100),
            ScreenRect::new(200, 200, 50, 50),
        ];
        let mut scores = vec![0.9, 0.8, 0.7];
        nms(&mut boxes, &mut scores, 0.5);
        // First two overlap heavily (IoU >0.5), second should be suppressed
        assert_eq!(boxes.len(), 2);
        assert_eq!(scores.len(), 2);
        // Highest score first
        assert_eq!(boxes[0], ScreenRect::new(0, 0, 100, 100));
        assert_eq!(boxes[1], ScreenRect::new(200, 200, 50, 50));
    }

    #[test]
    fn test_nms_no_suppression_high_threshold() {
        let mut boxes = vec![
            ScreenRect::new(0, 0, 100, 100),
            ScreenRect::new(50, 50, 100, 100),
        ];
        let mut scores = vec![0.9, 0.8];
        nms(&mut boxes, &mut scores, 0.9);
        // IoU ~0.14, threshold 0.9 => no suppression
        assert_eq!(boxes.len(), 2);
    }
}
