use vision::detection::FrameData;
use vision::geometry::ScreenRect;

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("Platform not supported")]
    Unsupported,
    #[error("Capture failed: {0}")]
    Failed(String),
    #[error("No foreground window")]
    NoForegroundWindow,
    #[error("Window too small: {width}x{height}")]
    WindowTooSmall { width: u32, height: u32 },
}

pub trait ScreenCapturer: Send + Sync {
    fn capture_foreground(
        &self,
        include: &[String],
        exclude: &[String],
        always_skip: &[(String, String)],
    ) -> Result<CapturedFrame, CaptureError>;
}

pub struct CapturedFrame {
    pub frame: FrameData,
    pub window_rect: ScreenRect,
    pub hwnd: platform::WindowHandle,
    pub process_name: String,
    pub window_title: String,
}

pub struct PlatformCapturer;

impl PlatformCapturer {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PlatformCapturer {
    fn default() -> Self {
        Self::new()
    }
}

impl ScreenCapturer for PlatformCapturer {
    fn capture_foreground(
        &self,
        include: &[String],
        exclude: &[String],
        always_skip: &[(String, String)],
    ) -> Result<CapturedFrame, CaptureError> {
        let hwnd = platform::get_foreground_window().ok_or(CaptureError::NoForegroundWindow)?;

        let process_name = platform::get_window_process_name(hwnd).unwrap_or_default();

        let window_title = platform::get_window_title(hwnd).unwrap_or_default();

        if should_skip_window(&process_name, &window_title, include, exclude, always_skip) {
            return Err(CaptureError::Failed("Window filtered out".to_string()));
        }

        if !platform::is_window_visible(hwnd) {
            return Err(CaptureError::Failed("Window not visible".to_string()));
        }

        if platform::is_window_minimized(hwnd) {
            return Err(CaptureError::Failed("Window minimized".to_string()));
        }

        let client_rect = platform::get_client_rect(hwnd)
            .ok_or_else(|| CaptureError::Failed("Failed to get client rect".to_string()))?;

        if client_rect.width < 650 || client_rect.height < 400 {
            return Err(CaptureError::WindowTooSmall {
                width: client_rect.width,
                height: client_rect.height,
            });
        }

        let frame = platform::capture_window(hwnd)
            .ok_or_else(|| CaptureError::Failed("Failed to capture window".to_string()))?;

        Ok(CapturedFrame {
            frame,
            window_rect: client_rect,
            hwnd,
            process_name,
            window_title,
        })
    }
}

fn should_skip_window(
    process_name: &str,
    window_title: &str,
    include: &[String],
    exclude: &[String],
    always_skip: &[(String, String)],
) -> bool {
    let pn = process_name.to_lowercase();
    let wt = window_title.to_lowercase();

    for (skip_pn, skip_wt) in always_skip {
        if pn.contains(&skip_pn.to_lowercase()) && wt.contains(&skip_wt.to_lowercase()) {
            return true;
        }
    }

    if !include.is_empty() {
        let included = include.iter().any(|inc| {
            let inc_lower = inc.to_lowercase();
            pn.contains(&inc_lower)
        });
        if !included {
            return true;
        }
    }

    if !exclude.is_empty() {
        let excluded = exclude.iter().any(|exc| {
            let exc_lower = exc.to_lowercase();
            pn.contains(&exc_lower)
        });
        if excluded {
            return true;
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_skip_window() {
        let include = vec!["chrome.exe".to_string()];
        let exclude = vec!["explorer.exe".to_string()];
        let always_skip = vec![("explorer.exe".to_string(), "Shell_TrayWnd".to_string())];

        assert!(!should_skip_window(
            "chrome.exe",
            "Google",
            &include,
            &exclude,
            &always_skip
        ));
        assert!(should_skip_window(
            "explorer.exe",
            "Desktop",
            &include,
            &exclude,
            &always_skip
        ));
        assert!(should_skip_window(
            "firefox.exe",
            "Firefox",
            &include,
            &exclude,
            &always_skip
        ));
        assert!(should_skip_window(
            "explorer.exe",
            "Shell_TrayWnd",
            &include,
            &exclude,
            &always_skip
        ));
    }
}
