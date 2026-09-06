use crate::detection::{extract_dominant_color, CoverMode, CoverRect, FrameData};
use crate::geometry::{ColorRgb, ScreenRect};

pub fn generate_cover(
    frame: &FrameData,
    rect: &ScreenRect,
    mode: CoverMode,
    _solid_color: ColorRgb,
) -> CoverRect {
    match mode {
        CoverMode::SolidColor => CoverRect {
            screen_rect: *rect,
            mode,
        },
        CoverMode::BackgroundColor => {
            let _ = extract_dominant_color(frame, rect);
            CoverRect {
                screen_rect: *rect,
                mode,
            }
        }
        CoverMode::Blur => CoverRect {
            screen_rect: *rect,
            mode,
        },
    }
}

pub fn generate_blur_data(frame: &FrameData, rect: &ScreenRect) -> Option<Vec<u8>> {
    let region = frame.region(rect)?;
    Some(crate::preprocessing::blur_region(
        &region.data,
        region.width,
        region.height,
    ))
}

pub fn covers_for_detections(
    detections: &[crate::detection::Detection],
    frame: &FrameData,
    mode: CoverMode,
    solid_color: ColorRgb,
    window_rects: &[ScreenRect],
) -> Vec<CoverRect> {
    let mut covers = Vec::new();

    for det in detections {
        let cover = generate_cover(frame, &det.screen_rect, mode, solid_color);

        let mut final_parts = Vec::new();
        let mut current_parts = vec![cover.screen_rect];

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
        }

        for part in current_parts {
            final_parts.push(CoverRect {
                screen_rect: part,
                mode: cover.mode,
            });
        }

        covers.append(&mut final_parts);
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
}
