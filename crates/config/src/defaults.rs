use crate::settings::*;
use vision::geometry::ColorRgb;

#[allow(clippy::derivable_impls)]
impl Default for PordaConfig {
    fn default() -> Self {
        Self {
            detection: DetectionConfig::default(),
            overlay: OverlayConfig::default(),
            hotkeys: HotkeyConfig::default(),
            performance: PerformanceConfig::default(),
            windows: WindowConfig::default(),
            startup: StartupConfig::default(),
            tracking: TrackingConfig::default(),
        }
    }
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            accuracy: 25,
            network_width: 544,
            network_height: 320,
            engine: "CPU Engine".to_string(),
            hardware_accelerated: true,
            is_detect_male: false,
            is_detect_female: true,
            active_timeout_ms: 65,
            sleep_timeout_ms: 500,
            keep_running_seconds: 10,
            nms_threshold: 0.1,
        }
    }
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            is_blur: true,
            is_bg_color: false,
            is_solid_color: false,
            rgb_color: ColorRgb::new(0, 0, 255),
        }
    }
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            toggle_key: "F2".to_string(),
            screenshot_key: "F1".to_string(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            is_priority_realtime: false,
            is_allow_max_cpu_limit: false,
            max_cpu_limit: 90,
            average_reading_interval: 30,
        }
    }
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            is_all_windows: false,
            is_include_window: true,
            is_exclude_window: false,
            include_windows: vec![
                "chrome.exe".to_string(),
                "msedge.exe".to_string(),
                "brave.exe".to_string(),
                "firefox.exe".to_string(),
                "opera.exe".to_string(),
                "PotPlayerMini64.exe".to_string(),
                "vlc.exe".to_string(),
            ],
            exclude_windows: vec![
                "explorer.exe".to_string(),
                "cmd.exe".to_string(),
                "winword.exe".to_string(),
                "pordaai.exe".to_string(),
            ],
            always_skip_windows: vec![
                ("explorer.exe".to_string(), "Progman".to_string()),
                ("explorer.exe".to_string(), "WorkerW".to_string()),
                ("explorer.exe".to_string(), "Shell_TrayWnd".to_string()),
                ("explorer.exe".to_string(), "LauncherTipWnd".to_string()),
                ("explorer.exe".to_string(), "SystemTray_Main".to_string()),
                (
                    "explorer.exe".to_string(),
                    "NotifyIconOverflowWindow".to_string(),
                ),
                (
                    "ShellExperienceHost.exe".to_string(),
                    "Shell_TrayWnd".to_string(),
                ),
                (
                    "ShellExperienceHost.exe".to_string(),
                    "Windows.UI.Core.CoreWindow".to_string(),
                ),
                (
                    "SearchApp.exe".to_string(),
                    "Windows.UI.Core.CoreWindow".to_string(),
                ),
            ],
        }
    }
}

impl Default for StartupConfig {
    fn default() -> Self {
        Self { auto_startup: true }
    }
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: "https://www.google-analytics.com/mp/collect".to_string(),
        }
    }
}

pub fn config_path() -> std::path::PathBuf {
    let app_dir = app_data_dir();
    app_dir.join("settings.json")
}

pub fn app_data_dir() -> std::path::PathBuf {
    if let Some(home) = dirs::home_dir() {
        home.join("PordaAi")
    } else {
        std::path::PathBuf::from("PordaAi")
    }
}

pub fn dataset_dir() -> std::path::PathBuf {
    app_data_dir().join("Dataset")
}

pub fn external_model_dir() -> std::path::PathBuf {
    app_data_dir().join("Extarnal-Model")
}

pub fn load_config() -> Result<PordaConfig, std::io::Error> {
    let path = config_path();
    if path.exists() {
        let data = std::fs::read_to_string(&path)?;
        let config: PordaConfig = serde_json::from_str(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(config)
    } else {
        Ok(PordaConfig::default())
    }
}

pub fn save_config(config: &PordaConfig) -> Result<(), std::io::Error> {
    let path = config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let data = serde_json::to_string_pretty(config)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, data)
}

pub fn ensure_directories() -> std::io::Result<()> {
    let dirs = [app_data_dir(), dataset_dir(), external_model_dir()];
    for dir in &dirs {
        std::fs::create_dir_all(dir)?;
    }
    Ok(())
}
