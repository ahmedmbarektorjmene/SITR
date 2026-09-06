use config::settings::PordaConfig;
use vision::detection::{Detection, DetectionState};
use vision::geometry::{ColorRgb, ScreenRect};

#[derive(Debug, Clone)]
pub struct AppState {
    pub config: PordaConfig,
    pub detection_state: DetectionState,
    pub is_active: bool,
    pub last_detections: Vec<Detection>,
    pub covers: Vec<vision::detection::CoverRect>,
    pub window_rects: Vec<ScreenRect>,
    pub last_detection_time: Option<std::time::Instant>,
    pub cpu_usage: f32,
}

impl AppState {
    pub fn new(config: PordaConfig) -> Self {
        // P1.3: Porda opens deactivated by default. Do not start PipeWire/screen sharing merely because window opened.
        Self {
            config,
            detection_state: DetectionState::Sleep,
            is_active: false,
            last_detections: Vec::new(),
            covers: Vec::new(),
            window_rects: Vec::new(),
            last_detection_time: None,
            cpu_usage: 0.0,
        }
    }

    pub fn confidence_threshold(&self) -> f32 {
        self.config.confidence_threshold()
    }

    pub fn target_classes(&self) -> Vec<i32> {
        self.config.target_classes()
    }

    pub fn cover_mode(&self) -> vision::detection::CoverMode {
        if self.config.overlay.is_blur {
            vision::detection::CoverMode::Blur
        } else if self.config.overlay.is_solid_color {
            vision::detection::CoverMode::SolidColor
        } else {
            vision::detection::CoverMode::BackgroundColor
        }
    }

    pub fn solid_color(&self) -> ColorRgb {
        self.config.overlay.rgb_color
    }

    pub fn should_run_detection(&self) -> bool {
        if !self.is_active {
            return false;
        }
        if self.config.performance.is_allow_max_cpu_limit
            && self.cpu_usage > self.config.performance.max_cpu_limit as f32
        {
            return false;
        }
        true
    }

    pub fn detection_interval_ms(&self) -> u64 {
        // P1.3: no countdown / automatic timeout. Active interval only; sleep/keep_running removed.
        self.config.detection.active_timeout_ms
    }

    pub fn update_detection_state(&mut self, has_detections: bool) {
        // P1.3: no automatic timeout or auto-deactivation. Remains Active until explicit Deactivate.
        if !self.is_active {
            self.detection_state = DetectionState::Sleep;
            return;
        }
        self.detection_state = DetectionState::Active;
        if has_detections {
            self.last_detection_time = Some(std::time::Instant::now());
        }
    }
}
