use vision::geometry::ScreenRect;

pub const ANCHORS: [[f32; 2]; 6] = [
    [10.0, 14.0],
    [23.0, 27.0],
    [37.0, 58.0],
    [81.0, 82.0],
    [135.0, 169.0],
    [344.0, 319.0],
];

/// Masks per YOLO head: order corresponds to outputs as stored in ONNX
/// First head (small grid 10x17) uses large anchors 3,4,5
/// Second head (large grid 20x34) uses small anchors 0,1,2
pub const MASKS: [[usize; 3]; 2] = [[3, 4, 5], [0, 1, 2]];

pub const SCALE_X_Y: f32 = 1.05;
pub const NUM_CLASSES: usize = 2;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[derive(Debug, Clone)]
pub struct YoloCandidate {
    pub class_id: i32,
    pub confidence: f32,
    pub rect_network: ScreenRect, // in network coords 544x320
}

pub fn decode_heads(
    heads: &[(Vec<f32>, u32, u32)], // (data flattened NCHW, h, w)
    network_width: u32,
    network_height: u32,
) -> Vec<YoloCandidate> {
    let mut candidates = Vec::new();
    for (head_idx, (data, h, w)) in heads.iter().enumerate() {
        let mask = MASKS[head_idx];
        let grid_h = *h as usize;
        let grid_w = *w as usize;
        let channels_per_anchor = 5 + NUM_CLASSES as usize; // 7
                                                            // data layout: NCHW with N=1, C=21, H, W flattened as [c*H*W + h*W + w]
        for gh in 0..grid_h {
            for gw in 0..grid_w {
                for (a_idx, &anchor_idx) in mask.iter().enumerate() {
                    let anchor_w = ANCHORS[anchor_idx][0];
                    let anchor_h = ANCHORS[anchor_idx][1];
                    let base = a_idx * channels_per_anchor;
                    // offset = c*H*W + gh*W + gw
                    let off = |c: usize| -> f32 {
                        let idx = c * grid_h * grid_w + gh * grid_w + gw;
                        data[idx]
                    };
                    let tx = off(base);
                    let ty = off(base + 1);
                    let tw = off(base + 2);
                    let th = off(base + 3);
                    let obj = off(base + 4);
                    let cls0 = off(base + 5);
                    let cls1 = off(base + 6);

                    // decode centre
                    let bx = (sigmoid(tx) * SCALE_X_Y - 0.5 * (SCALE_X_Y - 1.0) + gw as f32)
                        / grid_w as f32
                        * network_width as f32;
                    let by = (sigmoid(ty) * SCALE_X_Y - 0.5 * (SCALE_X_Y - 1.0) + gh as f32)
                        / grid_h as f32
                        * network_height as f32;
                    let bw = (tw.exp()) * anchor_w;
                    let bh = (th.exp()) * anchor_h;

                    let obj_s = sigmoid(obj);
                    let c0 = sigmoid(cls0);
                    let c1 = sigmoid(cls1);
                    let confs = [obj_s * c0, obj_s * c1];
                    // store per class candidates will be filtered later
                    // we emit both
                    for class_id in 0..NUM_CLASSES {
                        let conf = confs[class_id];
                        // keep all for later filtering
                        let x = (bx - bw * 0.5) as i32;
                        let y = (by - bh * 0.5) as i32;
                        let w_i = bw as u32;
                        let h_i = bh as u32;
                        // filter out degenerate
                        if w_i == 0 || h_i == 0 {
                            continue;
                        }
                        candidates.push(YoloCandidate {
                            class_id: class_id as i32,
                            confidence: conf,
                            rect_network: ScreenRect::new(x, y, w_i, h_i),
                        });
                    }
                }
            }
        }
    }
    candidates
}

/// Apply confidence threshold, class filter, and NMS, returning final detections in network coords
pub fn filter_and_nms(
    mut candidates: Vec<YoloCandidate>,
    confidence_threshold: f32,
    nms_threshold: f32,
    target_classes: &[i32],
) -> Vec<YoloCandidate> {
    // confidence + class filter
    candidates
        .retain(|c| c.confidence >= confidence_threshold && target_classes.contains(&c.class_id));
    // sort by confidence descending
    candidates.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // NMS: greedy
    let mut keep: Vec<bool> = vec![true; candidates.len()];
    for i in 0..candidates.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..candidates.len() {
            if !keep[j] {
                continue;
            }
            // Only suppress same class? Spec doesn't clarify. We'll suppress across all if overlapping large.
            // But to be safe, only suppress if same class. This matches typical.
            if candidates[i].class_id != candidates[j].class_id {
                continue;
            }
            let inter = candidates[i]
                .rect_network
                .intersection(&candidates[j].rect_network);
            if let Some(inter_rect) = inter {
                let inter_area = inter_rect.area() as f32;
                let union = candidates[i].rect_network.area() as f32
                    + candidates[j].rect_network.area() as f32
                    - inter_area;
                let iou = if union > 0.0 { inter_area / union } else { 0.0 };
                if iou > nms_threshold {
                    keep[j] = false;
                }
            }
        }
    }
    let mut out = Vec::new();
    for (i, k) in keep.iter().enumerate() {
        if *k {
            out.push(candidates[i].clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_decode_empty_heads() {
        let h0 = (vec![0.0; 21 * 10 * 17], 10, 17);
        let h1 = (vec![0.0; 21 * 20 * 34], 20, 34);
        let cands = decode_heads(&[h0, h1], 544, 320);
        assert_eq!(cands.len(), (10 * 17 * 3 + 20 * 34 * 3) * 2); // per class 2
    }

    #[test]
    fn test_filter() {
        let cands = vec![
            YoloCandidate {
                class_id: 1,
                confidence: 0.9,
                rect_network: ScreenRect::new(0, 0, 100, 100),
            },
            YoloCandidate {
                class_id: 1,
                confidence: 0.8,
                rect_network: ScreenRect::new(10, 10, 100, 100),
            },
            YoloCandidate {
                class_id: 0,
                confidence: 0.9,
                rect_network: ScreenRect::new(200, 200, 50, 50),
            },
        ];
        let out = filter_and_nms(cands, 0.25, 0.1, &[1]);
        // should keep only female, and NMS suppress second overlapping female (IoU >0.1? compute ~0.68)
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].class_id, 1);
    }
}
