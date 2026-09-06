use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;

static TRAY_IS_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Update tray activation state for dynamic menu label (inactive → Activate, active → Deactivate).
pub fn set_tray_is_active(active: bool) {
    TRAY_IS_ACTIVE.store(active, Ordering::SeqCst);
}

pub fn get_tray_is_active() -> bool {
    TRAY_IS_ACTIVE.load(Ordering::SeqCst)
}

/// Human-readable label for the dynamic toggle action.
pub fn toggle_label(is_active: bool) -> &'static str {
    if is_active {
        "Deactivate"
    } else {
        "Activate"
    }
}

pub fn current_toggle_label() -> &'static str {
    toggle_label(get_tray_is_active())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrayAction {
    /// Show / restore the main window (Open)
    Show,
    /// Activate detection pipeline
    Activate,
    /// Deactivate detection pipeline
    Deactivate,
    /// Toggle detection (activate <-> deactivate) – single source of truth for tray activation
    ToggleDetection,
    RefreshHotkeys,
    RefreshOverlay,
    Exit,
}

impl TrayAction {
    /// Human-readable label for menu rendering
    pub fn label(&self) -> &'static str {
        match self {
            TrayAction::Show => "Show",
            TrayAction::Activate => "Activate",
            TrayAction::Deactivate => "Deactivate",
            TrayAction::ToggleDetection => "Toggle Detection",
            TrayAction::RefreshHotkeys => "Refresh Hotkeys",
            TrayAction::RefreshOverlay => "Refresh Overlay",
            TrayAction::Exit => "Exit",
        }
    }

    /// All actions that should appear in the tray menu, in display order – single toggle entry reflects runtime state.
    pub fn menu_actions() -> Vec<TrayAction> {
        // P1.3: tray shows single activation toggle (reflects actual runtime), not duplicate Activate/Deactivate.
        vec![
            TrayAction::Show,
            TrayAction::ToggleDetection,
            TrayAction::Exit,
        ]
    }

    /// Current activation label for tray (used by systray to render dynamic menu).
    pub fn current_toggle_label() -> &'static str {
        toggle_label(get_tray_is_active())
    }
}

pub struct TrayManager {
    action_tx: mpsc::Sender<TrayAction>,
}

impl TrayManager {
    pub fn new(action_tx: mpsc::Sender<TrayAction>) -> Self {
        Self { action_tx }
    }

    pub fn send_action(&self, action: TrayAction) -> Result<(), String> {
        tracing::debug!("Tray action requested: {:?}", action);
        self.action_tx.send(action).map_err(|e| {
            tracing::error!("Failed to send tray action: {}", e);
            e.to_string()
        })
    }

    pub fn show_notification(&self, title: &str, message: &str) {
        tracing::info!("Tray notification: {} - {}", title, message);
    }
}

/// Generate a valid 32x32 ARGB tray icon (purple background with white "P").
///
/// Returns raw ARGB bytes suitable for `ksni::Icon { width, height, data }`.
/// Dimensions are tray-appropriate (32x32), alpha is opaque, and data is
/// verified to be non-empty.
#[allow(clippy::manual_range_contains)]
pub fn generate_tray_icon_argb() -> (i32, i32, Vec<u8>) {
    const W: i32 = 32;
    const H: i32 = 32;
    // Purple #6A5ACD (106, 90, 205) opaque, white for letter
    let mut rgba = vec![0u8; (W * H * 4) as usize];
    for y in 0..H {
        for x in 0..W {
            let idx = ((y * W + x) * 4) as usize;
            // border radius effect: keep corners slightly transparent? No – keep opaque for tray.
            // Simple P shape: vertical bar + top loop approximation
            let is_p = (x >= 10 && x <= 13 && y >= 8 && y <= 24)
                || (x >= 14 && x <= 20 && y >= 8 && y <= 10)
                || (x >= 18 && x <= 20 && y >= 11 && y <= 15)
                || (x >= 14 && x <= 19 && y >= 16 && y <= 18);
            if is_p {
                // white opaque
                rgba[idx] = 255;
                rgba[idx + 1] = 255;
                rgba[idx + 2] = 255;
                rgba[idx + 3] = 255;
            } else {
                // purple opaque
                rgba[idx] = 106;
                rgba[idx + 1] = 90;
                rgba[idx + 2] = 205;
                rgba[idx + 3] = 255;
            }
        }
    }
    // Convert RGBA -> ARGB (ksni expects ARGB32 network byte order)
    let mut argb = Vec::with_capacity(rgba.len());
    for chunk in rgba.chunks_exact(4) {
        let r = chunk[0];
        let g = chunk[1];
        let b = chunk[2];
        let a = chunk[3];
        argb.push(a);
        argb.push(r);
        argb.push(g);
        argb.push(b);
    }
    debug_assert!(!argb.is_empty());
    debug_assert_eq!(argb.len(), (W * H * 4) as usize);
    (W, H, argb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tray_icon_is_valid() {
        let (w, h, data) = generate_tray_icon_argb();
        assert_eq!(w, 32);
        assert_eq!(h, 32);
        assert_eq!(data.len(), (w * h * 4) as usize);
        assert!(!data.is_empty());
        // Alpha channel must be opaque (0xFF) for our generated icon
        for chunk in data.chunks_exact(4) {
            assert_eq!(chunk[0], 255, "alpha must be opaque");
        }
    }

    #[test]
    fn tray_action_labels_non_empty() {
        for action in TrayAction::menu_actions() {
            assert!(!action.label().is_empty());
        }
    }

    #[test]
    fn tray_action_menu_no_duplicates() {
        let actions = TrayAction::menu_actions();
        let mut seen = std::collections::HashSet::new();
        for a in &actions {
            assert!(seen.insert(format!("{:?}", a)), "duplicate action {:?}", a);
        }
        assert!(actions.contains(&TrayAction::Show));
        assert!(actions.contains(&TrayAction::Exit));
    }

    #[test]
    fn tray_menu_is_exactly_three_without_settings() {
        let actions = TrayAction::menu_actions();
        assert_eq!(actions.len(), 3);
        assert_eq!(
            actions,
            vec![
                TrayAction::Show,
                TrayAction::ToggleDetection,
                TrayAction::Exit
            ]
        );
        for a in &actions {
            assert_ne!(a.label(), "Settings");
        }
        assert!(!actions
            .iter()
            .any(|a| format!("{:?}", a).contains("OpenSettings")));
        // Dynamic toggle label reflects runtime state, not duplicate Activate/Deactivate
        set_tray_is_active(false);
        assert_eq!(current_toggle_label(), "Activate");
        set_tray_is_active(true);
        assert_eq!(current_toggle_label(), "Deactivate");
        set_tray_is_active(false);
    }

    #[test]
    fn window_close_is_not_terminate() {
        // Tray Show (window close handler) must not be conflated with Exit (Terminate)
        assert_ne!(TrayAction::Show, TrayAction::Exit);
        assert_eq!(TrayAction::Show.label(), "Show");
        assert_eq!(TrayAction::Exit.label(), "Exit");
    }

    #[test]
    fn tray_manager_send_and_receive() {
        let (tx, rx) = mpsc::channel();
        let mgr = TrayManager::new(tx);
        mgr.send_action(TrayAction::Show).unwrap();
        mgr.send_action(TrayAction::Exit).unwrap();
        assert_eq!(rx.recv().unwrap(), TrayAction::Show);
        assert_eq!(rx.recv().unwrap(), TrayAction::Exit);
    }
}
