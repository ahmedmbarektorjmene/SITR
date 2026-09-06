use vision::geometry::ScreenRect;

/// Candidate produced by the anchor-free postprocessor.
///
/// The detector currently exposes `class_id` for compatibility with the
/// existing inference API. The model itself is single-class.
#[derive(Clone, Debug)]
pub struct YoloCandidate {
    pub class_id: i32,
    pub confidence: f32,
    pub rect_network: ScreenRect,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Returns true when `idx` is a heatmap peak in its 3x3 neighborhood.
///
/// The comparison is intentionally done on logits rather than sigmoid
/// probabilities because sigmoid is strictly monotonic.
#[inline]
fn is_local_max(heat: &[f32], idx: usize, width: usize, height: usize) -> bool {
    let score = heat[idx];
    let x = idx % width;
    let y = idx / width;

    if x > 0 {
        if heat[idx - 1] > score {
            return false;
        }

        if y > 0 && heat[idx - width - 1] > score {
            return false;
        }

        if y + 1 < height && heat[idx + width - 1] > score {
            return false;
        }
    }

    if x + 1 < width {
        if heat[idx + 1] > score {
            return false;
        }

        if y > 0 && heat[idx - width + 1] > score {
            return false;
        }

        if y + 1 < height && heat[idx + width + 1] > score {
            return false;
        }
    }

    if y > 0 && heat[idx - width] > score {
        return false;
    }

    if y + 1 < height && heat[idx + width] > score {
        return false;
    }

    true
}

/// Decode a single anchor-free output tensor.
///
/// Expected tensor layout:
///
///     [1, 5, H, W]
///
/// Channels:
///
///     0 = heatmap logits
///     1 = x offset logits
///     2 = y offset logits
///     3 = log(width)
///     4 = log(height)
///
/// The output tensor is supplied as five channel planes:
///
///     heat
///     offset_x
///     offset_y
///     log_width
///     log_height
///
/// This function intentionally performs no allocation. `out` is reused by
/// the detector between frames.
pub fn decode_heads(
    heads: &[(&[f32], u32, u32)],
    cfg_w: u32,
    cfg_h: u32,
    _target_classes: &[i32],
    confidence_threshold: f32,
    out: &mut Vec<YoloCandidate>,
) {
    out.clear();

    if heads.len() < 5 || cfg_w == 0 || cfg_h == 0 {
        return;
    }

    let (heat, grid_w, grid_h) = heads[0];
    let (offset_x, offset_x_w, offset_x_h) = heads[1];
    let (offset_y, offset_y_w, offset_y_h) = heads[2];
    let (log_width, width_w, width_h) = heads[3];
    let (log_height, height_w, height_h) = heads[4];

    let grid_w = grid_w as usize;
    let grid_h = grid_h as usize;

    if grid_w == 0 || grid_h == 0 {
        return;
    }

    let plane_len = grid_w * grid_h;

    // All five channels must describe the same feature map.
    if heat.len() < plane_len
        || offset_x.len() < plane_len
        || offset_y.len() < plane_len
        || log_width.len() < plane_len
        || log_height.len() < plane_len
        || offset_x_w as usize != grid_w
        || offset_x_h as usize != grid_h
        || offset_y_w as usize != grid_w
        || offset_y_h as usize != grid_h
        || width_w as usize != grid_w
        || width_h as usize != grid_h
        || height_w as usize != grid_w
        || height_h as usize != grid_h
    {
        return;
    }

    let stride_x = cfg_w as f32 / grid_w as f32;
    let stride_y = cfg_h as f32 / grid_h as f32;

    for idx in 0..plane_len {
        let heat_logit = heat[idx];

        // Reject invalid model output before doing any further work.
        if !heat_logit.is_finite() {
            continue;
        }

        // Cheap threshold test in probability space.
        let confidence = sigmoid(heat_logit);
        if confidence < confidence_threshold {
            continue;
        }

        // One detection per local heatmap peak.
        if !is_local_max(heat, idx, grid_w, grid_h) {
            continue;
        }

        let offset_x_logit = offset_x[idx];
        let offset_y_logit = offset_y[idx];
        let log_w = log_width[idx];
        let log_h = log_height[idx];

        if !offset_x_logit.is_finite()
            || !offset_y_logit.is_finite()
            || !log_w.is_finite()
            || !log_h.is_finite()
        {
            continue;
        }

        // Offsets are trained as normalized values in [0, 1].
        let ox = sigmoid(offset_x_logit);
        let oy = sigmoid(offset_y_logit);

        let bw = log_w.exp();
        let bh = log_h.exp();

        if !bw.is_finite() || !bh.is_finite() || bw <= 0.0 || bh <= 0.0 {
            continue;
        }

        let gx = idx % grid_w;
        let gy = idx / grid_w;

        let cx = (gx as f32 + ox) * stride_x;
        let cy = (gy as f32 + oy) * stride_y;

        if !cx.is_finite() || !cy.is_finite() {
            continue;
        }

        // Reject boxes that are clearly unusable.
        if bw > cfg_w as f32 * 2.0 || bh > cfg_h as f32 * 2.0 {
            continue;
        }

        let x = (cx - bw * 0.5).round() as i32;
        let y = (cy - bh * 0.5).round() as i32;
        let w = bw.round() as u32;
        let h = bh.round() as u32;

        if w == 0 || h == 0 {
            continue;
        }

        out.push(YoloCandidate {
            class_id: 0,
            confidence,
            rect_network: ScreenRect::new(x, y, w, h),
        });
    }
}

/// Greedy Non-Maximum Suppression.
///
/// Candidates must be sorted by confidence descending. Since the detector
/// has a single class, no class comparison is necessary.
pub fn filter_and_nms(
    candidates: &mut Vec<YoloCandidate>,
    nms_threshold: f32,
    keep_buf: &mut Vec<bool>,
) {
    if candidates.len() <= 1 {
        return;
    }

    candidates.sort_unstable_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let len = candidates.len();

    keep_buf.clear();
    keep_buf.resize(len, true);

    for i in 0..len {
        if !keep_buf[i] {
            continue;
        }

        let current = &candidates[i];

        for j in (i + 1)..len {
            if !keep_buf[j] {
                continue;
            }

            if current.rect_network.iou(&candidates[j].rect_network) > nms_threshold {
                keep_buf[j] = false;
            }
        }
    }

    // Compact in place. No second Vec and no candidate cloning required.
    let mut dst = 0;

    for src in 0..len {
        if !keep_buf[src] {
            continue;
        }

        if dst != src {
            candidates.swap(dst, src);
        }

        dst += 1;
    }

    candidates.truncate(dst);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_max() {
        let heat = [0.1, 0.2, 0.1, 0.3, 0.9, 0.4, 0.2, 0.5, 0.1];

        assert!(is_local_max(&heat, 4, 3, 3));
        assert!(!is_local_max(&heat, 1, 3, 3));
    }

    #[test]
    fn test_local_max_corner() {
        let heat = [0.9, 0.1, 0.1, 0.1];

        assert!(is_local_max(&heat, 0, 2, 2));
    }

    #[test]
    fn test_decode_anchor_free() {
        let heat = [-10.0, 10.0, -10.0, -10.0];

        let offset_x = [0.0; 4];
        let offset_y = [0.0; 4];

        // log(2) => decoded width/height = 2.
        let log_width = [2.0_f32.ln(); 4];
        let log_height = [2.0_f32.ln(); 4];

        let heads = [
            (&heat[..], 2, 2),
            (&offset_x[..], 2, 2),
            (&offset_y[..], 2, 2),
            (&log_width[..], 2, 2),
            (&log_height[..], 2, 2),
        ];

        let mut out = Vec::new();

        decode_heads(&heads, 4, 4, &[], 0.5, &mut out);

        assert_eq!(out.len(), 1);

        let candidate = &out[0];

        assert_eq!(candidate.class_id, 0);
        assert!(candidate.confidence > 0.99);

        // Peak is at grid (1, 0), stride = 2.
        // sigmoid(0) = 0.5:
        // cx = (1 + 0.5) * 2 = 3
        // cy = (0 + 0.5) * 2 = 1
        //
        // 2x2 box => x=2, y=0.
        assert_eq!(candidate.rect_network.x, 2);
        assert_eq!(candidate.rect_network.y, 0);
        assert_eq!(candidate.rect_network.width, 2);
        assert_eq!(candidate.rect_network.height, 2);
    }

    #[test]
    fn test_nms_keeps_highest_confidence() {
        let rect = ScreenRect::new(10, 10, 20, 20);

        let mut candidates = vec![
            YoloCandidate {
                class_id: 0,
                confidence: 0.6,
                rect_network: rect,
            },
            YoloCandidate {
                class_id: 0,
                confidence: 0.9,
                rect_network: rect,
            },
        ];

        let mut keep = Vec::new();

        filter_and_nms(&mut candidates, 0.5, &mut keep);

        assert_eq!(candidates.len(), 1);
        assert!((candidates[0].confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_nms_keeps_non_overlapping() {
        let mut candidates = vec![
            YoloCandidate {
                class_id: 0,
                confidence: 0.9,
                rect_network: ScreenRect::new(0, 0, 10, 10),
            },
            YoloCandidate {
                class_id: 0,
                confidence: 0.8,
                rect_network: ScreenRect::new(100, 100, 10, 10),
            },
        ];

        let mut keep = Vec::new();

        filter_and_nms(&mut candidates, 0.5, &mut keep);

        assert_eq!(candidates.len(), 2);
    }
}
