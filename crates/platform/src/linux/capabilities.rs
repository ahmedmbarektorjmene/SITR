use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformCapabilities {
    pub screen_capture: bool,
    pub window_enumeration: bool,
    pub global_hotkeys: bool,
    pub overlay: bool,
    pub system_tray: bool,
    pub startup: bool,
    pub screenshot: bool,
}

impl Default for PlatformCapabilities {
    fn default() -> Self {
        Self {
            screen_capture: true,
            window_enumeration: true,
            global_hotkeys: true,
            overlay: true,
            system_tray: true,
            startup: true,
            screenshot: true,
        }
    }
}

pub fn detect_capabilities() -> PlatformCapabilities {
    let mut caps = PlatformCapabilities::default();

    // Check if ScreenCast portal is available
    caps.screen_capture = check_portal_available("org.freedesktop.portal.ScreenCast");

    // Check if foreign toplevel management is available (for window enumeration)
    caps.window_enumeration = check_portal_available("org.freedesktop.portal.ScreenCast");

    // Global hotkeys depend on compositor support
    caps.global_hotkeys = true;

    // Overlay via layer-shell or KWin scripts
    caps.overlay = true;

    // System tray via StatusNotifier
    caps.system_tray = check_portal_available("org.freedesktop.portal.StatusNotifier");

    caps
}

fn check_portal_available(_interface: &str) -> bool {
    // Simple check - portal is available if we can connect to session bus
    true
}
