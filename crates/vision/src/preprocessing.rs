use crate::geometry::ScreenRect;
use image::{imageops::FilterType, ImageBuffer, Rgb};

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

    // Avoid division by zero if scale is 0
    if new_w == 0 || new_h == 0 {
        return (
            vec![0u8; (target_width * target_height * 3) as usize],
            1.0,
            1.0,
        );
    }

    let bottom = target_height - new_h;
    let right = target_width - new_w;

    // Python early-return optimization: avoid manual resize when already close
    // if (bottom <55 and right==0) or (right <70 and bottom==0): return frame,1,1
    if (bottom < 55 && right == 0) || (right < 70 && bottom == 0) {
        return (data.to_vec(), 1.0, 1.0);
    }

    // Resize with bilinear (INTER_LINEAR equivalent)
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_raw(src_width, src_height, data.to_vec())
            .unwrap_or_else(|| ImageBuffer::new(src_width, src_height));

    let resized = image::imageops::resize(&img, new_w, new_h, FilterType::Triangle);
    let resized_data = resized.into_raw();

    // Pad with black (0,0,0) on bottom and right only, matching
    // cv2.copyMakeBorder(resized, 0, bottom, 0, right, BORDER_CONSTANT, value=[0,0,0])
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

#[allow(clippy::manual_checked_ops)]
pub fn blur_region(data: &[u8], width: u32, height: u32) -> Vec<u8> {
    if data.is_empty() || width == 0 || height == 0 {
        return vec![];
    }

    let kernel = (width / 2).max(1);
    let mut output = vec![0u8; data.len()];

    for y in 0..height {
        for x in 0..width {
            let mut r_sum: u32 = 0;
            let mut g_sum: u32 = 0;
            let mut b_sum: u32 = 0;
            let mut count: u32 = 0;

            let ky_start = y.saturating_sub(kernel);
            let ky_end = (y + kernel + 1).min(height);
            let kx_start = x.saturating_sub(kernel);
            let kx_end = (x + kernel + 1).min(width);

            for ky in ky_start..ky_end {
                for kx in kx_start..kx_end {
                    let idx = ((ky * width + kx) * 3) as usize;
                    if idx + 2 < data.len() {
                        b_sum += data[idx] as u32;
                        g_sum += data[idx + 1] as u32;
                        r_sum += data[idx + 2] as u32;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < output.len() {
                    output[idx] = (b_sum / count) as u8;
                    output[idx + 1] = (g_sum / count) as u8;
                    output[idx + 2] = (r_sum / count) as u8;
                }
            }
        }
    }

    output
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
