use std::sync::{Arc, Mutex};

use config::settings::PordaConfig;

#[derive(Debug, Clone)]
pub struct UiState {
    pub is_active: bool,
    pub is_blur: bool,
    pub is_bg_color: bool,
    pub is_solid_color: bool,
    pub rgb_r: u8,
    pub rgb_g: u8,
    pub rgb_b: u8,
    pub accuracy: u8,
    pub network_width: u32,
    pub network_height: u32,
    pub active_timeout_ms: u64,
    pub sleep_timeout_ms: u64,
    pub keep_running_seconds: u64,
    pub engine: String,
    pub is_detect_male: bool,
    pub is_detect_female: bool,
    pub include_windows: String,
    pub exclude_windows: String,
    pub is_all_windows: bool,
    pub is_include_window: bool,
    pub is_exclude_window: bool,
    pub auto_startup: bool,
    pub is_priority_realtime: bool,
    pub is_allow_max_cpu_limit: bool,
    pub max_cpu_limit: u8,
    pub toggle_key: String,
    pub screenshot_key: String,
    pub cpu_usage: f32,
    pub current_page: usize,
    pub detection_state: String,
}

impl Default for UiState {
    fn default() -> Self {
        let config = PordaConfig::default();
        Self::from_config(&config)
    }
}

impl UiState {
    pub fn from_config(config: &PordaConfig) -> Self {
        Self {
            is_active: false,
            is_blur: config.overlay.is_blur,
            is_bg_color: config.overlay.is_bg_color,
            is_solid_color: config.overlay.is_solid_color,
            rgb_r: config.overlay.rgb_color.r,
            rgb_g: config.overlay.rgb_color.g,
            rgb_b: config.overlay.rgb_color.b,
            accuracy: config.detection.accuracy,
            network_width: config.detection.network_width,
            network_height: config.detection.network_height,
            active_timeout_ms: config.detection.active_timeout_ms,
            sleep_timeout_ms: config.detection.sleep_timeout_ms,
            keep_running_seconds: config.detection.keep_running_seconds,
            engine: config.detection.engine.clone(),
            is_detect_male: config.detection.is_detect_male,
            is_detect_female: config.detection.is_detect_female,
            include_windows: config.windows.include_windows.join(", "),
            exclude_windows: config.windows.exclude_windows.join(", "),
            is_all_windows: config.windows.is_all_windows,
            is_include_window: config.windows.is_include_window,
            is_exclude_window: config.windows.is_exclude_window,
            auto_startup: config.startup.auto_startup,
            is_priority_realtime: config.performance.is_priority_realtime,
            is_allow_max_cpu_limit: config.performance.is_allow_max_cpu_limit,
            max_cpu_limit: config.performance.max_cpu_limit,
            toggle_key: config.hotkeys.toggle_key.clone(),
            screenshot_key: config.hotkeys.screenshot_key.clone(),
            cpu_usage: 0.0,
            current_page: 0,
            detection_state: "Sleep".to_string(),
        }
    }

    pub fn to_config(&self) -> PordaConfig {
        PordaConfig {
            detection: config::settings::DetectionConfig {
                accuracy: self.accuracy,
                network_width: self.network_width,
                network_height: self.network_height,
                engine: self.engine.clone(),
                hardware_accelerated: true,
                is_detect_male: self.is_detect_male,
                is_detect_female: self.is_detect_female,
                active_timeout_ms: self.active_timeout_ms,
                sleep_timeout_ms: self.sleep_timeout_ms,
                keep_running_seconds: self.keep_running_seconds,
                nms_threshold: 0.1,
            },
            overlay: config::settings::OverlayConfig {
                is_blur: self.is_blur,
                is_bg_color: self.is_bg_color,
                is_solid_color: self.is_solid_color,
                rgb_color: vision::geometry::ColorRgb::new(
                    self.rgb_r, self.rgb_g, self.rgb_b,
                ),
            },
            hotkeys: config::settings::HotkeyConfig {
                toggle_key: self.toggle_key.clone(),
                screenshot_key: self.screenshot_key.clone(),
            },
            performance: config::settings::PerformanceConfig {
                is_priority_realtime: self.is_priority_realtime,
                is_allow_max_cpu_limit: self.is_allow_max_cpu_limit,
                max_cpu_limit: self.max_cpu_limit,
                average_reading_interval: 30,
            },
            windows: config::settings::WindowConfig {
                is_all_windows: self.is_all_windows,
                is_include_window: self.is_include_window,
                is_exclude_window: self.is_exclude_window,
                include_windows: self
                    .include_windows
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
                exclude_windows: self
                    .exclude_windows
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
                always_skip_windows: PordaConfig::default().windows.always_skip_windows,
            },
            startup: config::settings::StartupConfig {
                auto_startup: self.auto_startup,
            },
            tracking: config::settings::TrackingConfig::default(),
        }
    }
}

pub type SharedUiState = Arc<Mutex<UiState>>;

pub fn create_shared_state() -> SharedUiState {
    let config = config::defaults::load_config().unwrap_or_default();
    Arc::new(Mutex::new(UiState::from_config(&config)))
}
