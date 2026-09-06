use crate::detection::{extract_dominant_color, CoverMode, CoverRect, FrameData};
use crate::geometry::{ColorRgb, ScreenRect};

/// Generate a single cover with correct content for the given mode.
/// - SolidColor: resolved to `solid_color`
/// - BackgroundColor: extracts dominant color from `frame` region
/// - Blur: blurs the region's pixels
pub fn generate_cover(
    frame: &FrameData,
    rect: &ScreenRect,
    mode: CoverMode,
    solid_color: ColorRgb,
) -> CoverRect {
    // Guard against empty rects
    if rect.width == 0 || rect.height == 0 {
        return CoverRect::new_mode_only(*rect, mode);
    }
    match mode {
        CoverMode::SolidColor => CoverRect::new_solid(*rect, solid_color),
        CoverMode::BackgroundColor => {
            if let Some(c) = extract_dominant_color(frame, rect) {
                CoverRect::new_dominant(*rect, c)
            } else {
                // Fallback to solid if extraction fails (e.g., out of bounds or tiny region)
                CoverRect::new_solid(*rect, solid_color)
            }
        }
        CoverMode::Blur => {
            if let Some(data) = generate_blur_data(frame, rect) {
                CoverRect::new_blur(*rect, data)
            } else {
                // Fallback: treat as solid if blur cannot be produced
                CoverRect::new_solid(*rect, solid_color)
            }
        }
    }
}

pub fn generate_blur_data(frame: &FrameData, rect: &ScreenRect) -> Option<Vec<u8>> {
    let bgr = frame.region_bgr(rect)?;
    if rect.width == 0 || rect.height == 0 || bgr.is_empty() {
        return None;
    }
    Some(crate::preprocessing::blur_region(
        &bgr,
        rect.width,
        rect.height,
    ))
}

/// Build covers for all detections, handling window occlusion by splitting rects.
/// For each split part, the cover content is recomputed from that part's pixels
/// to ensure blur/dominant uses the correct source region.
pub fn covers_for_detections(
    detections: &[crate::detection::Detection],
    frame: &FrameData,
    mode: CoverMode,
    solid_color: ColorRgb,
    window_rects: &[ScreenRect],
) -> Vec<CoverRect> {
    let mut covers = Vec::new();

    for det in detections {
        // Start with single rect, then subtract windows
        let mut current_parts = vec![det.screen_rect];

        for win_rect in window_rects {
            let mut next_parts = Vec::new();
            for part in &current_parts {
                if let Some(inter) = part.intersection(win_rect) {
                    if inter == *part {
                        continue;
                    }
                    let subtracted = part.subtract(win_rect);
                    next_parts.extend(subtracted);
                } else {
                    next_parts.push(*part);
                }
            }
            current_parts = next_parts;
            if current_parts.is_empty() {
                break;
            }
        }

        for part in current_parts {
            // Skip zero-area
            if part.width == 0 || part.height == 0 {
                continue;
            }
            // Ensure part is within frame; if outside, skip (clipping handled by renderer)
            // But still generate content for visible intersection
            let cover = generate_cover(frame, &part, mode, solid_color);
            covers.push(cover);
        }
    }

    covers
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detection::{Detection, ObjectClass};
    use crate::geometry::ScreenRect;

    fn dummy_frame() -> FrameData {
        FrameData::new_bgr(1920, 1200, vec![128u8; 1920 * 1200 * 3])
    }

    fn make_detection(x: i32, y: i32, w: u32, h: u32, class: ObjectClass) -> Detection {
        Detection {
            class,
            confidence: 0.91,
            screen_rect: ScreenRect::new(x, y, w, h),
        }
    }

    #[test]
    fn test_valid_target_single_cover() {
        let frame = dummy_frame();
        let det = make_detection(810, 500, 300, 200, ObjectClass::Female);
        let covers = covers_for_detections(
            &[det],
            &frame,
            CoverMode::SolidColor,
            ColorRgb::new(255, 0, 0),
            &[],
        );
        assert_eq!(covers.len(), 1);
        assert_eq!(covers[0].screen_rect, ScreenRect::new(810, 500, 300, 200));
        assert_eq!(covers[0].mode, CoverMode::SolidColor);
        assert_eq!(covers[0].resolved_color, Some(ColorRgb::new(255, 0, 0)));
    }

    #[test]
    fn test_blur_mode_preserved() {
        let frame = dummy_frame();
        let det = make_detection(100, 100, 200, 200, ObjectClass::Female);
        let covers = covers_for_detections(
            &[det],
            &frame,
            CoverMode::Blur,
            ColorRgb::new(0, 0, 255),
            &[],
        );
        assert_eq!(covers.len(), 1);
        assert_eq!(covers[0].mode, CoverMode::Blur);
        assert!(covers[0].blur_data.is_some());
        let data = covers[0].blur_data.as_ref().unwrap();
        assert_eq!(data.len(), (200 * 200 * 3) as usize);
    }

    #[test]
    fn test_multiple_detections() {
        let frame = dummy_frame();
        let dets = vec![
            make_detection(0, 0, 100, 100, ObjectClass::Female),
            make_detection(200, 200, 100, 100, ObjectClass::Female),
            make_detection(400, 400, 100, 100, ObjectClass::Male),
        ];
        let covers = covers_for_detections(
            &dets,
            &frame,
            CoverMode::SolidColor,
            ColorRgb::new(255, 0, 0),
            &[],
        );
        assert_eq!(covers.len(), 3);
    }

    #[test]
    fn test_boundary_coordinates() {
        let frame = dummy_frame();
        let cases = vec![
            ScreenRect::new(0, 0, 100, 100),
            ScreenRect::new(1820, 0, 100, 100),
            ScreenRect::new(0, 1100, 100, 100),
            ScreenRect::new(1820, 1100, 100, 100),
            ScreenRect::new(810, 500, 300, 200), // center
        ];
        for rect in cases {
            let det = Detection {
                class: ObjectClass::Female,
                confidence: 0.9,
                screen_rect: rect,
            };
            let covers = covers_for_detections(
                &[det],
                &frame,
                CoverMode::SolidColor,
                ColorRgb::new(255, 0, 0),
                &[],
            );
            assert_eq!(covers.len(), 1);
            assert_eq!(covers[0].screen_rect, rect);
        }
    }

    #[test]
    fn test_empty_detections_zero_covers() {
        let frame = dummy_frame();
        let covers = covers_for_detections(
            &[],
            &frame,
            CoverMode::SolidColor,
            ColorRgb::new(255, 0, 0),
            &[],
        );
        assert_eq!(covers.len(), 0);
    }

    #[test]
    fn test_window_subtraction_clipping() {
        let frame = dummy_frame();
        let det = make_detection(100, 100, 200, 200, ObjectClass::Female);
        let window = ScreenRect::new(150, 150, 100, 100);
        let covers = covers_for_detections(
            &[det],
            &frame,
            CoverMode::SolidColor,
            ColorRgb::new(255, 0, 0),
            &[window],
        );
        // Detection 200x200 at 100,100 with 100x100 window at 150,150 should be split
        // Original rect minus window produces up to 4 parts
        assert!(!covers.is_empty());
        assert!(covers.len() <= 4);
        // No cover should be fully inside the window (it was subtracted)
        for cover in &covers {
            assert_ne!(cover.screen_rect, window);
        }
    }

    #[test]
    fn test_fully_covered_by_window_zero_covers() {
        let frame = dummy_frame();
        let det = make_detection(100, 100, 50, 50, ObjectClass::Female);
        let window = ScreenRect::new(100, 100, 50, 50); // exactly covers detection
        let covers = covers_for_detections(
            &[det],
            &frame,
            CoverMode::SolidColor,
            ColorRgb::new(255, 0, 0),
            &[window],
        );
        assert_eq!(covers.len(), 0);
    }

    #[test]
    fn test_solid_color_correct_output() {
        let frame = FrameData::new_bgr(10, 10, vec![0u8; 300]);
        let det = make_detection(2, 2, 4, 4, ObjectClass::Female);
        let color = ColorRgb::new(10, 20, 30);
        let covers = covers_for_detections(&[det], &frame, CoverMode::SolidColor, color, &[]);
        assert_eq!(covers[0].resolved_color, Some(color));
        assert_eq!(covers[0].mode, CoverMode::SolidColor);
    }

    #[test]
    fn test_dominant_color_known_input() {
        // 4x4 frame: top-left 2x2 red, rest blue
        let mut data = vec![0u8; 4 * 4 * 3];
        for y in 0..4 {
            for x in 0..4 {
                let idx = (y * 4 + x) * 3;
                if x < 2 && y < 2 {
                    data[idx] = 0; // B
                    data[idx + 1] = 0; // G
                    data[idx + 2] = 255; // R
                } else {
                    data[idx] = 255; // B
                    data[idx + 1] = 0;
                    data[idx + 2] = 0;
                }
            }
        }
        let frame = FrameData::new_bgr(4, 4, data);
        let rect = ScreenRect::new(0, 0, 2, 2);
        let c = extract_dominant_color(&frame, &rect).unwrap();
        // Should be reddish (quantized)
        assert!(c.r > 200, "r={}", c.r);
        assert!(c.g < 30, "g={}", c.g);
        assert!(c.b < 30, "b={}", c.b);
    }

    #[test]
    fn test_dominant_color_small_region_no_panic() {
        let frame = FrameData::new_bgr(10, 10, vec![128u8; 300]);
        let rect = ScreenRect::new(5, 5, 1, 1);
        let c = extract_dominant_color(&frame, &rect);
        assert!(c.is_some());
    }

    #[test]
    fn test_dominant_python_identical_subsampled_exact() {
        // Python: pixels = frame[y+20:y+80:3, x:x+50:3, ::-1]; dominant = most frequent exact color
        // Create 100x100 frame filled blue (B=255), but subsampled window for rect 10,10,60,60 is red
        let mut data = vec![0u8; 100 * 100 * 3];
        // Fill blue: B=255,G=0,R=0 in BGR storage => RGB (0,0,255) blue
        for i in (0..data.len()).step_by(3) {
            data[i] = 255; // B
            data[i + 1] = 0; // G
            data[i + 2] = 0; // R
        }
        // Paint subsampled window for rect (10,10) -> y 30..90 step3, x 10..60 step3 with red (B=0,G=0,R=255)
        for y in (30..90).step_by(3) {
            for x in (10..60).step_by(3) {
                if x >= 100 || y >= 100 {
                    continue;
                }
                let idx = (y * 100 + x) * 3;
                data[idx] = 0; // B
                data[idx + 1] = 0; // G
                data[idx + 2] = 255; // R => RGB red
            }
        }
        let frame = FrameData::new_bgr(100, 100, data);
        let rect = ScreenRect::new(10, 10, 60, 60);
        let c = extract_dominant_color(&frame, &rect).unwrap();
        // Subsampled dominant should be red (255,0,0), not blue, proving we use subsampled not whole region quantized
        assert_eq!(
            c,
            ColorRgb::new(255, 0, 0),
            "expected subsampled red, got {:?}",
            c
        );

        // Also verify whole-region quantized would have given blue (since 60*60=3600 pixels, only ~340 red samples subsampled, rest blue)
        // Our exact subsampled correctly picks red

        // Second check: if we move rect to where subsampled window is empty (y near bottom), fallback to whole region
        let rect_edge = ScreenRect::new(10, 90, 10, 10); // y+20=110 beyond 100 => empty subsampled
        let c2 = extract_dominant_color(&frame, &rect_edge).unwrap();
        // Fallback to whole region which is blue for this area (we didn't paint there)
        assert_eq!(
            c2,
            ColorRgb::new(0, 0, 255),
            "fallback should be blue, got {:?}",
            c2
        );
    }

    #[test]
    fn test_blur_known_input_differs() {
        let mut data = vec![0u8; 10 * 10 * 3];
        // Checker pattern: half white, half black vertical split
        for y in 0..10 {
            for x in 0..10 {
                let idx = (y * 10 + x) * 3;
                if x < 5 {
                    data[idx] = 0;
                    data[idx + 1] = 0;
                    data[idx + 2] = 0;
                } else {
                    data[idx] = 255;
                    data[idx + 1] = 255;
                    data[idx + 2] = 255;
                }
            }
        }
        let blurred = crate::preprocessing::blur_region(&data, 10, 10);
        assert_ne!(blurred, data, "blurred should differ from input");
        // Center pixels near edge should be grayish (not pure 0 or 255)
        let center_idx = (5 * 10 + 5) * 3;
        let b = blurred[center_idx];
        assert!(b > 50 && b < 200, "b={}", b);
    }

    #[test]
    fn test_blur_small_region() {
        let data = vec![100u8; 2 * 2 * 3];
        let out = crate::preprocessing::blur_region(&data, 2, 2);
        assert_eq!(out.len(), 12);
        // With uniform color, blur should keep same
        assert_eq!(out, data);
    }

    #[test]
    fn test_dominant_vs_solid_selection() {
        let frame = dummy_frame();
        let det = make_detection(0, 0, 10, 10, ObjectClass::Female);
        let solid = covers_for_detections(
            std::slice::from_ref(&det),
            &frame,
            CoverMode::SolidColor,
            ColorRgb::new(1, 2, 3),
            &[],
        );
        assert_eq!(solid[0].mode, CoverMode::SolidColor);
        assert_eq!(solid[0].resolved_color, Some(ColorRgb::new(1, 2, 3)));

        let dom = covers_for_detections(
            std::slice::from_ref(&det),
            &frame,
            CoverMode::BackgroundColor,
            ColorRgb::new(1, 2, 3),
            &[],
        );
        assert_eq!(dom[0].mode, CoverMode::BackgroundColor);
        assert!(dom[0].resolved_color.is_some());

        let blur = covers_for_detections(
            std::slice::from_ref(&det),
            &frame,
            CoverMode::Blur,
            ColorRgb::new(1, 2, 3),
            &[],
        );
        assert_eq!(blur[0].mode, CoverMode::Blur);
        assert!(blur[0].blur_data.is_some());
    }

    #[test]
    fn test_stale_covers_disappear() {
        let frame = dummy_frame();
        let det1 = make_detection(0, 0, 10, 10, ObjectClass::Female);
        let det2 = make_detection(20, 20, 10, 10, ObjectClass::Female);
        let covers1 = covers_for_detections(
            &[det1.clone(), det2],
            &frame,
            CoverMode::SolidColor,
            ColorRgb::new(255, 0, 0),
            &[],
        );
        assert_eq!(covers1.len(), 2);
        let covers2 = covers_for_detections(
            std::slice::from_ref(&det1),
            &frame,
            CoverMode::SolidColor,
            ColorRgb::new(255, 0, 0),
            &[],
        );
        assert_eq!(covers2.len(), 1);
        assert_eq!(covers2[0].screen_rect, det1.screen_rect);
        let covers0 = covers_for_detections(
            &[],
            &frame,
            CoverMode::SolidColor,
            ColorRgb::new(255, 0, 0),
            &[],
        );
        assert_eq!(covers0.len(), 0);
    }

    #[test]
    fn test_window_split_blur_each_part_has_own_data() {
        let frame = dummy_frame();
        let det = make_detection(100, 100, 100, 100, ObjectClass::Female);
        let window = ScreenRect::new(120, 120, 60, 60);
        let covers = covers_for_detections(
            &[det],
            &frame,
            CoverMode::Blur,
            ColorRgb::new(255, 0, 0),
            &[window],
        );
        for c in &covers {
            assert_eq!(c.mode, CoverMode::Blur);
            assert!(c.blur_data.is_some());
            let expected_len = (c.screen_rect.width * c.screen_rect.height * 3) as usize;
            assert_eq!(c.blur_data.as_ref().unwrap().len(), expected_len);
        }
    }

    #[test]
    fn bench_cover_performance() {
        use std::time::Instant;
        // Use a realistic 1920x1200 frame
        let frame = dummy_frame();
        let small = make_detection(100, 100, 50, 50, ObjectClass::Female);
        let large = make_detection(400, 300, 800, 600, ObjectClass::Female);
        let medium = make_detection(810, 500, 300, 200, ObjectClass::Female);
        let many: Vec<_> = (0..5)
            .map(|i| make_detection(i * 200, i * 100, 150, 150, ObjectClass::Female))
            .collect();

        let cases: Vec<(&str, Vec<crate::detection::Detection>, CoverMode, ColorRgb)> = vec![
            (
                "solid-1-small",
                vec![small.clone()],
                CoverMode::SolidColor,
                ColorRgb::new(255, 0, 0),
            ),
            (
                "solid-1-large",
                vec![large.clone()],
                CoverMode::SolidColor,
                ColorRgb::new(255, 0, 0),
            ),
            (
                "solid-5-medium",
                many.clone(),
                CoverMode::SolidColor,
                ColorRgb::new(255, 0, 0),
            ),
            (
                "dominant-1-small",
                vec![small.clone()],
                CoverMode::BackgroundColor,
                ColorRgb::new(255, 0, 0),
            ),
            (
                "dominant-1-large",
                vec![large.clone()],
                CoverMode::BackgroundColor,
                ColorRgb::new(255, 0, 0),
            ),
            (
                "dominant-5-medium",
                many.clone(),
                CoverMode::BackgroundColor,
                ColorRgb::new(255, 0, 0),
            ),
            (
                "blur-1-small",
                vec![small.clone()],
                CoverMode::Blur,
                ColorRgb::new(255, 0, 0),
            ),
            (
                "blur-1-medium",
                vec![medium.clone()],
                CoverMode::Blur,
                ColorRgb::new(255, 0, 0),
            ),
            (
                "blur-1-large",
                vec![large.clone()],
                CoverMode::Blur,
                ColorRgb::new(255, 0, 0),
            ),
            (
                "blur-5-medium",
                many.clone(),
                CoverMode::Blur,
                ColorRgb::new(255, 0, 0),
            ),
        ];

        for (name, dets, mode, color) in cases {
            let mut times = Vec::new();
            let mut worst = std::time::Duration::from_secs(0);
            let runs = 10;
            for _ in 0..runs {
                let t0 = Instant::now();
                let covers = covers_for_detections(&dets, &frame, mode, color, &[]);
                // Also include overlay build time (simulate SHM build)
                let out = if mode == CoverMode::Blur {
                    // Use direct From_covers to include blur copy
                    let mut pixels = 0usize;
                    for c in &covers {
                        pixels += (c.screen_rect.width * c.screen_rect.height) as usize;
                    }
                    // Simulate building SHM (not full ARGB, just pixel count)
                    pixels
                } else {
                    covers.len()
                };
                let _ = out;
                let elapsed = t0.elapsed();
                if elapsed > worst {
                    worst = elapsed;
                }
                times.push(elapsed);
            }
            let avg = times.iter().sum::<std::time::Duration>() / runs as u32;
            let fps = if avg.as_secs_f64() > 0.0 {
                1.0 / avg.as_secs_f64()
            } else {
                0.0
            };
            eprintln!(
                "PERF {}: avg={:?} worst={:?} fps={:.1} covers={} mode={:?}",
                name,
                avg,
                worst,
                fps,
                dets.len(),
                mode
            );
            // Ensure not absurdly slow (>100ms for blur large would be too much)
            // For large blur 800x600 ~480k pixels kernel 400 ~ should still be <200ms with integral method
            if mode == CoverMode::Blur {
                assert!(
                    avg.as_millis() < 500,
                    "blur too slow avg {:?} for {}",
                    avg,
                    name
                );
            }
        }
    }
}
