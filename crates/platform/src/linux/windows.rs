use vision::geometry::ScreenRect;

#[derive(Debug, Clone)]
pub struct WindowInfo {
    pub id: u64,
    pub title: String,
    pub app_id: String,
    pub geometry: ScreenRect,
    pub focused: bool,
    pub minimized: bool,
}

pub fn enumerate_windows() -> Vec<WindowInfo> {
    #[cfg(target_os = "linux")]
    {
        linux_impl::enumerate_windows_wayland()
    }
    #[cfg(not(target_os = "linux"))]
    {
        vec![]
    }
}

pub fn get_foreground_window() -> Option<WindowInfo> {
    enumerate_windows().into_iter().find(|w| w.focused)
}

pub fn should_skip_window(
    app_id: &str,
    include: &[String],
    exclude: &[String],
    always_skip: &[(String, String)],
) -> bool {
    let app_lower = app_id.to_lowercase();

    for (skip_app, _) in always_skip {
        if app_lower.contains(&skip_app.to_lowercase()) {
            return true;
        }
    }

    if !include.is_empty() {
        let included = include
            .iter()
            .any(|inc| app_lower.contains(&inc.to_lowercase()));
        if !included {
            return true;
        }
    }

    if !exclude.is_empty() {
        let excluded = exclude
            .iter()
            .any(|exc| app_lower.contains(&exc.to_lowercase()));
        if excluded {
            return true;
        }
    }

    false
}

#[cfg(target_os = "linux")]
mod linux_impl {
    use super::*;

    pub fn enumerate_windows_wayland() -> Vec<WindowInfo> {
        // Use ext-foreign-toplevel-list via D-Bus or compositor protocol
        // For now, try to get window list from KWin D-Bus interface
        let windows = Vec::new();

        // Try KWin's D-Bus interface
        if let Ok(output) = std::process::Command::new("qdbus")
            .args(["org.kde.KWin", "/KWin", "org.kde.KWin.getWindowList"])
            .output()
        {
            if let Ok(json) = String::from_utf8(output.stdout) {
                // Parse window list from KWin
                tracing::debug!("KWin window list: {}", json);
            }
        }

        // Fallback: use wlr-foreign-toplevel-management if available
        // This requires Wayland protocol interaction

        windows
    }
}
