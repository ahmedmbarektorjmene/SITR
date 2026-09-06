use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::app_state::AppState;
use crate::commands::CoreEvent;
use inference::detector::{Detector, MockDetector, OpenCvDetector};
use overlay::compositor::{CpuOverlayRenderer, OverlayRenderer};
#[cfg(not(target_os = "linux"))]
use porda_capture::capturer::{PlatformCapturer, ScreenCapturer};
use vision::cover::covers_for_detections;

pub struct Pipeline {
    state: Arc<Mutex<AppState>>,
    event_tx: std::sync::mpsc::Sender<CoreEvent>,
    running: Arc<Mutex<bool>>,
}

impl Pipeline {
    pub fn new(state: Arc<Mutex<AppState>>, event_tx: std::sync::mpsc::Sender<CoreEvent>) -> Self {
        Self {
            state,
            event_tx,
            running: Arc::new(Mutex::new(false)),
        }
    }

    pub fn start(&self) {
        {
            let mut running = self.running.lock().unwrap();
            *running = true;
        }

        let state = Arc::clone(&self.state);
        let event_tx = self.event_tx.clone();
        let running = Arc::clone(&self.running);

        std::thread::Builder::new()
            .name("porda-pipeline".to_string())
            .spawn(move || {
                tracing::info!("Pipeline thread started");

                #[cfg(target_os = "linux")]
                {
                    run_linux_pipeline(state, event_tx, running);
                }
                #[cfg(not(target_os = "linux"))]
                {
                    run_windows_pipeline(state, event_tx, running);
                }
            })
            .expect("Failed to spawn pipeline thread");
    }

    pub fn stop(&self) {
        let mut running = self.running.lock().unwrap();
        *running = false;
    }

    pub fn is_running(&self) -> bool {
        *self.running.lock().unwrap()
    }
}

#[cfg(target_os = "linux")]
fn run_linux_pipeline(
    state: Arc<Mutex<AppState>>,
    event_tx: std::sync::mpsc::Sender<CoreEvent>,
    running: Arc<Mutex<bool>>,
) {
    let onnx_path = platform::onnx_model_path();
    let (cfg_path, w_path) = platform::model_paths();
    let detector: Box<dyn Detector> = if std::env::var("PORDA_MOCK_DETECTIONS").is_ok() {
        tracing::info!("Pipeline: PORDA_MOCK_DETECTIONS set, using MockDetector for testing");
        Box::new(MockDetector)
    } else if onnx_path.exists() {
        let device = std::env::var("inference_DEVICE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(inference::detector::InferenceDevice::Auto);
        let d = OpenCvDetector::new(onnx_path.clone(), device);
        tracing::info!(
            "Pipeline: ONNX model found, using {} (onnx={:?} device={:?})",
            d.backend_name(),
            onnx_path,
            device
        );
        Box::new(d)
    } else if cfg_path.exists() && w_path.exists() {
        tracing::warn!(
            "Pipeline: ONNX not found at {:?}, falling back to legacy Darknet cfg/weights {:?}/{:?} (deprecated)",
            onnx_path, cfg_path, w_path
        );
        let onnx_fallback = cfg_path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join("porda.onnx");
        let d = OpenCvDetector::new(onnx_fallback, inference::detector::InferenceDevice::Auto);
        Box::new(d)
    } else {
        tracing::info!(
            "Pipeline: model not found (onnx={:?} exists={}, cfg={:?} exists={}, weights={:?} exists={}), using mock",
            onnx_path,
            onnx_path.exists(),
            cfg_path,
            cfg_path.exists(),
            w_path,
            w_path.exists()
        );
        Box::new(MockDetector)
    };

    let mut overlay: Box<dyn OverlayRenderer> = {
        let outputs = platform::linux::get_outputs();
        let primary = outputs
            .iter()
            .find(|o| o.focused)
            .or_else(|| outputs.first());
        let (w, h) = primary
            .map(|o| (o.geometry.width, o.geometry.height))
            .unwrap_or((1920, 1200));
        let (w, h) = if w == 1920 && h == 1080 {
            (1920, 1200)
        } else {
            (w, h)
        };
        let cfg = overlay::OverlayConfig {
            width: w,
            height: h,
            scale: primary.map(|o| o.scale_factor).unwrap_or(1.0),
            output_name: primary.map(|o| o.name.clone()),
        };
        let has_test = std::env::var("overlay_TEST_RECT").is_ok();
        let wl: Box<dyn OverlayRenderer> = if has_test {
            tracing::info!("Overlay: test rect mode enabled ({}x{} at center)", w, h);
            Box::new(overlay::WaylandOverlay::with_test_rect(cfg))
        } else {
            Box::new(overlay::WaylandOverlay::new(
                cfg,
                vision::geometry::ColorRgb::new(255, 0, 0),
            ))
        };
        match wl.capability() {
            overlay::OverlayCapability::Supported => {
                tracing::info!("Overlay: Wayland layer-shell supported, using overlay");
                wl
            }
            overlay::OverlayCapability::Unsupported(reason) => {
                tracing::warn!(
                    "Overlay: layer-shell unsupported ({}), falling back to CPU stub",
                    reason
                );
                Box::new(CpuOverlayRenderer::new(
                    vision::geometry::ColorRgb::default(),
                ))
            }
        }
    };

    tracing::info!("Linux pipeline: PipeWire capture will be initialized lazily on first Activate (no portal dialog while DEACTIVATED)");
    tracing::info!("Linux pipeline: entering main loop (DEACTIVATED idle, no capture session yet)");

    loop {
        if !*running.lock().unwrap() {
            break;
        }

        let should_detect = {
            let s = state.lock().unwrap();
            let base = s.should_run_detection();
            let force_mock = std::env::var("PORDA_MOCK_DETECTIONS").is_ok();
            let force_active = std::env::var("PORDA_FORCE_ACTIVE").is_ok();
            let result = base || force_mock || force_active;
            if result != base {
                tracing::info!(
                    "Pipeline: forcing active (base={}, force_mock={}, force_active={}, is_active={})",
                    base,
                    force_mock,
                    force_active,
                    s.is_active
                );
            }
            result
        };

        if !should_detect {
            // P1.3: Do NOT start screen capture while DEACTIVATED.
            // No Portal request, no PipeWire session, no dialog. Keep tray/UI/hotkeys alive.
            // Clear any stale covers/overlay when transitioning to inactive.
            let needs_clear = {
                let mut s = state.lock().unwrap();
                if !s.covers.is_empty() || !s.last_detections.is_empty() {
                    s.covers.clear();
                    s.last_detections.clear();
                    s.detection_state = vision::detection::DetectionState::Sleep;
                    true
                } else {
                    false
                }
            };
            if needs_clear {
                tracing::info!("Pipeline: DEACTIVATED -> clearing stale covers/overlay");
                let _ = overlay.clear();
                let _ = event_tx.send(CoreEvent::CoversUpdated(vec![]));
            }
            tracing::trace!("Pipeline: inactive (is_active=false), idle without capture");
            std::thread::sleep(Duration::from_millis(200));
            continue;
        }

        let interval_ms = {
            let state = state.lock().unwrap();
            state.detection_interval_ms()
        };

        match platform::linux_screen_capture() {
            Some((frame, desktop_rect)) => {
                tracing::info!(
                    "Capture: frame {}x{} stride={} format={:?} rect={:?}",
                    frame.width,
                    frame.height,
                    frame.stride,
                    frame.format,
                    desktop_rect
                );

                // Extract detector params with single lock to avoid dangling refs
                let (conf_thresh, nms_thresh, target_classes, net_w, net_h) = {
                    let s = state.lock().unwrap();
                    (
                        s.confidence_threshold(),
                        s.config.detection.nms_threshold,
                        s.target_classes(),
                        s.config.detection.network_width,
                        s.config.detection.network_height,
                    )
                };
                tracing::info!(
                    "Pipeline: calling detector backend={} conf_thresh={:.2} target_classes={:?}",
                    detector.backend_name(),
                    conf_thresh,
                    target_classes
                );

                let detections = match detector.detect(
                    &frame,
                    conf_thresh,
                    nms_thresh,
                    &target_classes,
                    net_w,
                    net_h,
                    &desktop_rect,
                ) {
                    Ok(d) => d,
                    Err(e) => {
                        tracing::error!("Detector error: {}", e);
                        std::thread::sleep(Duration::from_millis(interval_ms));
                        continue;
                    }
                };

                tracing::info!("Detector: {} detections", detections.len());
                for (i, det) in detections.iter().enumerate() {
                    tracing::info!(
                        "Detection[{}]: class={:?} conf={:.2} bbox=({},{},{},{})",
                        i,
                        det.class,
                        det.confidence,
                        det.screen_rect.x,
                        det.screen_rect.y,
                        det.screen_rect.width,
                        det.screen_rect.height
                    );
                }

                let (covers, cover_time) = {
                    let s = state.lock().unwrap();
                    let mode = s.cover_mode();
                    let solid = s.solid_color();
                    let wins = s.window_rects.clone();
                    drop(s);
                    let t0 = std::time::Instant::now();
                    let c = covers_for_detections(&detections, &frame, mode, solid, &wins);
                    let elapsed = t0.elapsed();
                    (c, elapsed)
                };
                let total_pixels: usize = covers
                    .iter()
                    .map(|c| (c.screen_rect.width * c.screen_rect.height) as usize)
                    .sum();
                let mode = covers
                    .first()
                    .map(|c| c.mode)
                    .unwrap_or(vision::detection::CoverMode::SolidColor);
                tracing::info!(
                    "covers_for_detections: input={} -> output={} covers mode={:?} pixels={} time={:?}",
                    detections.len(),
                    covers.len(),
                    mode,
                    total_pixels,
                    cover_time
                );
                // Lightweight debug per-cover timing already logged via tracing::debug in cover generation
                for (i, cover) in covers.iter().enumerate() {
                    tracing::info!(
                        "CoverRect[{}]: x={} y={} w={} h={} mode={:?} resolved={:?} blur={}",
                        i,
                        cover.screen_rect.x,
                        cover.screen_rect.y,
                        cover.screen_rect.width,
                        cover.screen_rect.height,
                        cover.mode,
                        cover.resolved_color,
                        cover.blur_data.is_some()
                    );
                }

                {
                    let has_detections = !detections.is_empty();
                    let mut s = state.lock().unwrap();
                    s.last_detections = detections.clone();
                    s.update_detection_state(has_detections);
                    tracing::info!("Pipeline: detection_state={:?}", s.detection_state);
                }

                tracing::info!("Overlay: sending UpdateCovers count={}", covers.len());
                let ot0 = std::time::Instant::now();
                let overlay_result = overlay.update_covers(&covers, &frame);
                let ot = ot0.elapsed();
                tracing::debug!(
                    "Overlay: update_covers took {:?} covers={}",
                    ot,
                    covers.len()
                );
                match &overlay_result {
                    Ok(_) => tracing::info!(
                        "Overlay: UpdateCovers sent successfully (overlay_time={:?})",
                        ot
                    ),
                    Err(e) => tracing::error!("Overlay: UpdateCovers failed: {}", e),
                }
                let _ = overlay_result;

                {
                    let mut s = state.lock().unwrap();
                    s.covers = covers.clone();
                }
                let _ = event_tx.send(CoreEvent::CoversUpdated(covers));
            }
            None => {
                tracing::trace!("Capture: no frame available");
            }
        }

        std::thread::sleep(Duration::from_millis(interval_ms));
    }

    tracing::info!("Linux pipeline thread stopped");
}

#[cfg(not(target_os = "linux"))]
fn run_windows_pipeline(
    state: Arc<Mutex<AppState>>,
    event_tx: std::sync::mpsc::Sender<CoreEvent>,
    running: Arc<Mutex<bool>>,
) {
    let capturer = PlatformCapturer::new();
    let onnx_path = platform::onnx_model_path();
    let detector: Box<dyn Detector> = if onnx_path.exists() {
        let device = std::env::var("inference_DEVICE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(inference::detector::InferenceDevice::Auto);
        Box::new(OpenCvDetector::new(onnx_path, device))
    } else {
        let (cfg_path, w_path) = platform::model_paths();
        if cfg_path.exists() && w_path.exists() {
            Box::new(OpenCvDetector::new_legacy_darknet(cfg_path, w_path))
        } else {
            Box::new(MockDetector)
        }
    };
    let mut overlay = CpuOverlayRenderer::new(vision::geometry::ColorRgb::default());

    loop {
        if !*running.lock().unwrap() {
            break;
        }

        let should_detect = {
            let s = state.lock().unwrap();
            s.should_run_detection()
        };
        if !should_detect {
            // P1.3: DEACTIVATED – do not capture, clear stale covers, keep pipeline/tray alive
            let needs_clear = {
                let mut s = state.lock().unwrap();
                if !s.covers.is_empty() || !s.last_detections.is_empty() {
                    s.covers.clear();
                    s.last_detections.clear();
                    s.detection_state = vision::detection::DetectionState::Sleep;
                    true
                } else {
                    false
                }
            };
            if needs_clear {
                tracing::info!("Windows pipeline: DEACTIVATED -> clearing stale covers");
                let _ = overlay.clear();
                let _ = event_tx.send(CoreEvent::CoversUpdated(vec![]));
            }
            std::thread::sleep(Duration::from_millis(500));
            continue;
        }
        let interval_ms = {
            let s = state.lock().unwrap();
            s.detection_interval_ms()
        };

        match capturer.capture_foreground(
            &state.lock().unwrap().config.windows.include_windows,
            &state.lock().unwrap().config.windows.exclude_windows,
            &state.lock().unwrap().config.windows.always_skip_windows,
        ) {
            Ok(captured) => {
                match detector.detect(
                    &captured.frame,
                    state.lock().unwrap().confidence_threshold(),
                    state.lock().unwrap().config.detection.nms_threshold,
                    &state.lock().unwrap().target_classes(),
                    state.lock().unwrap().config.detection.network_width,
                    state.lock().unwrap().config.detection.network_height,
                    &captured.window_rect,
                ) {
                    Ok(detections) => {
                        let has_detections = !detections.is_empty();

                        {
                            let mut s = state.lock().unwrap();
                            s.last_detections = detections.clone();
                            s.update_detection_state(has_detections);
                        }

                        {
                            let covers = covers_for_detections(
                                &detections,
                                &captured.frame,
                                state.lock().unwrap().cover_mode(),
                                state.lock().unwrap().solid_color(),
                                &state.lock().unwrap().window_rects,
                            );

                            let _ = overlay.update_covers(&covers, &captured.frame);

                            {
                                let mut s = state.lock().unwrap();
                                s.covers = covers.clone();
                            }

                            let _ = event_tx.send(CoreEvent::CoversUpdated(covers));
                        }
                    }
                    Err(e) => {
                        tracing::error!("Detection failed: {}", e);
                    }
                }
            }
            Err(porda_capture::capturer::CaptureError::NoForegroundWindow) => {
                std::thread::sleep(Duration::from_millis(interval_ms));
                continue;
            }
            Err(e) => {
                tracing::debug!("Capture failed: {}", e);
            }
        }

        std::thread::sleep(Duration::from_millis(interval_ms));
    }

    tracing::info!("Pipeline thread stopped");
}
