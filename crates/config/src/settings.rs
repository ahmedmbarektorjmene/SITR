use serde::{Deserialize, Serialize};

use vision::geometry::ColorRgb;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PordaConfig {
    pub detection: DetectionConfig,
    pub overlay: OverlayConfig,
    pub hotkeys: HotkeyConfig,
    pub performance: PerformanceConfig,
    pub windows: WindowConfig,
    pub startup: StartupConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    pub accuracy: u8,
    pub network_width: u32,
    pub network_height: u32,
    pub engine: String,
    pub hardware_accelerated: bool,
    pub is_detect_male: bool,
    pub is_detect_female: bool,
    pub active_timeout_ms: u64,
    pub sleep_timeout_ms: u64,
    pub keep_running_seconds: u64,
    pub nms_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlayConfig {
    pub is_blur: bool,
    pub is_bg_color: bool,
    pub is_solid_color: bool,
    pub rgb_color: ColorRgb,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotkeyConfig {
    pub toggle_key: String,
    // screenshot_key removed: global screenshot hotkey no longer registered (P0 single-shortcut requirement)
    // Keep for migration: if old config contains screenshot_key it will be ignored on load (serde default)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub is_priority_realtime: bool,
    pub is_allow_max_cpu_limit: bool,
    pub max_cpu_limit: u8,
    pub average_reading_interval: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowConfig {
    pub is_all_windows: bool,
    pub is_include_window: bool,
    pub is_exclude_window: bool,
    pub include_windows: Vec<String>,
    pub exclude_windows: Vec<String>,
    pub always_skip_windows: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupConfig {
    pub auto_startup: bool,
}

impl PordaConfig {
    pub fn confidence_threshold(&self) -> f32 {
        self.detection.accuracy as f32 / 100.0
    }

    pub fn target_classes(&self) -> Vec<i32> {
        let mut classes = Vec::new();
        if self.detection.is_detect_male {
            classes.push(0);
        }
        if self.detection.is_detect_female {
            classes.push(1);
        }
        classes
    }
}
