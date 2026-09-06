use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};

use platform::tray::{generate_tray_icon_argb, TrayAction, TrayManager};

static TRAY_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Inner ksni Tray implementation. Holds a channel sender to forward
/// menu activations back to the application's command handler.
#[derive(Debug)]
struct PordaTrayInner {
    action_tx: mpsc::Sender<TrayAction>,
}

impl ksni::Tray for PordaTrayInner {
    fn id(&self) -> String {
        "com.porda.ai".into()
    }

    fn title(&self) -> String {
        "Porda AI".into()
    }

    fn icon_pixmap(&self) -> Vec<ksni::Icon> {
        let (w, h, data) = generate_tray_icon_argb();
        tracing::debug!("tray icon created: {}x{} ARGB {} bytes", w, h, data.len());
        vec![ksni::Icon {
            width: w,
            height: h,
            data,
        }]
    }

    fn tool_tip(&self) -> ksni::ToolTip {
        ksni::ToolTip {
            icon_name: String::new(),
            icon_pixmap: vec![],
            title: "Porda AI".into(),
            description: "Offline on-device blur for modest viewing".into(),
        }
    }

    fn activate(&mut self, _x: i32, _y: i32) {
        tracing::info!("tray activate (left-click) -> Show");
        let _ = self.action_tx.send(TrayAction::Show);
    }

    fn menu(&self) -> Vec<ksni::MenuItem<Self>> {
        use ksni::menu::*;
        let tx_show = self.action_tx.clone();
        let tx_activate = self.action_tx.clone();
        let tx_deactivate = self.action_tx.clone();
        let tx_screenshot = self.action_tx.clone();
        let tx_exit = self.action_tx.clone();

        vec![
            StandardItem {
                label: TrayAction::Show.label().into(),
                activate: Box::new(move |_| {
                    tracing::info!("tray menu -> Show");
                    let _ = tx_show.send(TrayAction::Show);
                }),
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: TrayAction::Activate.label().into(),
                activate: Box::new(move |_| {
                    tracing::info!("tray menu -> Activate");
                    let _ = tx_activate.send(TrayAction::Activate);
                }),
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: TrayAction::Deactivate.label().into(),
                activate: Box::new(move |_| {
                    tracing::info!("tray menu -> Deactivate");
                    let _ = tx_deactivate.send(TrayAction::Deactivate);
                }),
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: TrayAction::TakeScreenshot.label().into(),
                activate: Box::new(move |_| {
                    tracing::info!("tray menu -> TakeScreenshot");
                    let _ = tx_screenshot.send(TrayAction::TakeScreenshot);
                }),
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            StandardItem {
                label: TrayAction::Exit.label().into(),
                icon_name: "application-exit".into(),
                activate: Box::new(move |_| {
                    tracing::info!("tray menu -> Exit");
                    let _ = tx_exit.send(TrayAction::Exit);
                }),
                ..Default::default()
            }
            .into(),
        ]
    }

    fn watcher_offline(&self, reason: ksni::OfflineReason) -> bool {
        tracing::warn!("StatusNotifierWatcher offline: {:?}", reason);
        // Keep the service alive; host may reappear (e.g., shell restart)
        true
    }

    fn watcher_online(&self) {
        tracing::info!("StatusNotifierWatcher online – tray should be visible");
    }
}

pub struct PordaTray {
    manager: TrayManager,
    // Retained handle keeps the DBus registration alive for the whole process.
    // Interior mutability allows `run(&self)` to keep the `let tray = ...` binding alive.
    handle: Arc<Mutex<Option<ksni::blocking::Handle<PordaTrayInner>>>>,
    action_tx: mpsc::Sender<TrayAction>,
}

impl PordaTray {
    pub fn new(action_tx: mpsc::Sender<TrayAction>) -> Self {
        let manager = TrayManager::new(action_tx.clone());
        Self {
            manager,
            handle: Arc::new(Mutex::new(None)),
            action_tx,
        }
    }

    /// Initialize and spawn the system tray (StatusNotifierItem).
    ///
    /// Traces each stage so failures are not swallowed:
    ///   startup -> backend selected -> icon created -> menu created -> event loop -> retained
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        tracing::info!("tray initialization requested");

        if TRAY_INITIALIZED.swap(true, Ordering::SeqCst) {
            tracing::warn!("tray already initialized – refusing duplicate initialization");
            return Err("tray already initialized (duplicate PordaTray::run call)".into());
        }

        tracing::info!("tray backend selected: ksni StatusNotifierItem (Wayland-native, DBus)");

        // Pre-validate icon before DBus registration to fail fast with clear error
        let (w, h, data) = generate_tray_icon_argb();
        if data.is_empty() || w <= 0 || h <= 0 {
            tracing::error!(
                "tray icon creation failed: invalid pixmap {}x{} len={}",
                w,
                h,
                data.len()
            );
            TRAY_INITIALIZED.store(false, Ordering::SeqCst);
            return Err(format!("invalid tray icon data {}x{} len={}", w, h, data.len()).into());
        }
        tracing::info!("tray icon created: {}x{} ARGB {} bytes", w, h, data.len());

        // Menu is created lazily by ksni via Tray::menu(), but we trace that it is defined
        tracing::info!("tray menu created: Show / Activate / Deactivate / Take Screenshot / Exit");

        // Diagnose desktop environment tray host
        diagnose_tray_host();

        let inner = PordaTrayInner {
            action_tx: self.action_tx.clone(),
        };

        tracing::info!("tray event processing starting (ksni blocking, dedicated thread)");
        let handle = match ksni::blocking::TrayMethods::spawn(inner) {
            Ok(h) => h,
            Err(e) => {
                // Detailed error handling – never silently discard
                match &e {
                    ksni::Error::Dbus(inner) => {
                        tracing::error!("tray DBus connection failed: {}", inner);
                    }
                    ksni::Error::Watcher(inner) => {
                        tracing::error!(
                            "tray StatusNotifierWatcher registration failed: {}",
                            inner
                        );
                    }
                    ksni::Error::WontShow => {
                        tracing::error!("tray WontShow: no StatusNotifierHost found – desktop has no tray host (KDE: need plasmashell, GNOME: need appindicator extension)");
                    }
                    _ => {
                        tracing::error!("tray unknown error: {:?}", e);
                    }
                }
                // Distinguish app bug vs missing host vs unsupported backend
                tracing::error!("tray initialization failed: {} – see https://www.freedesktop.org/wiki/Specifications/StatusNotifierItem/", e);
                TRAY_INITIALIZED.store(false, Ordering::SeqCst);
                return Err(Box::new(e));
            }
        };

        tracing::info!("tray event processing started – DBus service registered");

        // Retain handle for lifetime of application; dropping it would unregister the tray.
        {
            let mut guard = self.handle.lock().unwrap();
            *guard = Some(handle);
        }
        tracing::info!(
            "tray object retained for application lifetime – icon should remain visible"
        );
        tracing::info!("System tray initialized");
        Ok(())
    }

    /// Gracefully shutdown the tray (called during application termination)
    pub fn shutdown(&self) {
        if let Ok(mut guard) = self.handle.lock() {
            if let Some(handle) = guard.take() {
                tracing::info!("shutting down tray service");
                handle.shutdown().wait();
                tracing::info!("tray service shutdown complete – icon removed cleanly");
            }
        }
        TRAY_INITIALIZED.store(false, Ordering::SeqCst);
    }

    /// Returns true if tray service is still registered
    pub fn is_alive(&self) -> bool {
        if let Ok(guard) = self.handle.lock() {
            if let Some(handle) = guard.as_ref() {
                return !handle.is_closed();
            }
        }
        false
    }

    pub fn show_notification(&self, title: &str, message: &str) {
        self.manager.show_notification(title, message);
    }

    pub fn send_action(&self, action: TrayAction) -> Result<(), String> {
        self.manager.send_action(action)
    }
}

impl Drop for PordaTray {
    fn drop(&mut self) {
        // Ensure DBus unregistration on drop if not explicitly shutdown.
        // We do not block indefinitely – just take the handle and shutdown.
        if let Ok(mut guard) = self.handle.lock() {
            if let Some(handle) = guard.take() {
                tracing::debug!("PordaTray dropped – shutting down tray handle");
                handle.shutdown().wait();
            }
        }
        TRAY_INITIALIZED.store(false, Ordering::SeqCst);
    }
}

fn diagnose_tray_host() {
    let session_type = std::env::var("XDG_SESSION_TYPE").unwrap_or_else(|_| "unknown".into());
    let desktop = std::env::var("XDG_CURRENT_DESKTOP").unwrap_or_else(|_| "unknown".into());
    tracing::info!(
        "desktop environment: XDG_SESSION_TYPE={} XDG_CURRENT_DESKTOP={}",
        session_type,
        desktop
    );
    tracing::debug!("checking for StatusNotifierWatcher / StatusNotifierHost on session bus");
    // Non-fatal diagnostic: try to list well-known names via `busctl` or zbus would be async.
    // We log that KDE's plasmashell provides org.kde.StatusNotifierWatcher & Host; GNOME needs extension.
    if desktop.to_lowercase().contains("kde") {
        tracing::info!("KDE detected – plasmashell should provide StatusNotifierHost (org.kde.StatusNotifierWatcher)");
    } else if desktop.to_lowercase().contains("gnome") {
        tracing::warn!("GNOME detected – tray requires AppIndicator extension; without it icon will not be visible even if SNI item is created");
    } else if session_type.to_lowercase() == "wayland" {
        tracing::info!(
            "Wayland session – tray depends on StatusNotifier host, not Wayland core itself"
        );
    }
}

/// Map a TrayAction to the corresponding UiCommand for core.
///
/// Keeps tray logic thin – no inference/capture here – just command translation.
pub fn tray_action_to_ui_command(action: &TrayAction) -> Option<porda_core::commands::UiCommand> {
    use porda_core::commands::UiCommand;
    match action {
        TrayAction::Activate => Some(UiCommand::Activate),
        TrayAction::Deactivate => Some(UiCommand::Deactivate),
        TrayAction::ToggleDetection => Some(UiCommand::ToggleActivation),
        TrayAction::TakeScreenshot => Some(UiCommand::TakeScreenshot),
        TrayAction::RefreshHotkeys => Some(UiCommand::RefreshHotkeys),
        TrayAction::RefreshOverlay => Some(UiCommand::RefreshOverlay),
        // Show is UI concern handled separately in main.rs via window show
        TrayAction::Show | TrayAction::Exit => None,
    }
}

/// Reset the global init flag – only for tests.
#[cfg(test)]
pub fn reset_for_test() {
    TRAY_INITIALIZED.store(false, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;
    use porda_core::commands::UiCommand;
    use std::sync::Mutex;

    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn tray_icon_generated_valid() {
        let (w, h, data) = generate_tray_icon_argb();
        assert_eq!(w, 32);
        assert_eq!(h, 32);
        assert_eq!(data.len(), (w * h * 4) as usize);
        // At least one non-purple pixel (the P)
        let white_pixels = data
            .chunks_exact(4)
            .filter(|c| c[1] == 255 && c[2] == 255 && c[3] == 255)
            .count();
        assert!(white_pixels > 0, "icon should contain white P shape");
    }

    #[test]
    fn tray_action_to_command_mapping() {
        assert!(matches!(
            tray_action_to_ui_command(&TrayAction::Activate),
            Some(UiCommand::Activate)
        ));
        assert!(matches!(
            tray_action_to_ui_command(&TrayAction::Deactivate),
            Some(UiCommand::Deactivate)
        ));
        assert!(matches!(
            tray_action_to_ui_command(&TrayAction::ToggleDetection),
            Some(UiCommand::ToggleActivation)
        ));
        assert!(matches!(
            tray_action_to_ui_command(&TrayAction::TakeScreenshot),
            Some(UiCommand::TakeScreenshot)
        ));
        assert!(tray_action_to_ui_command(&TrayAction::Show).is_none());
        assert!(tray_action_to_ui_command(&TrayAction::Exit).is_none());
    }

    #[test]
    fn duplicate_init_protection() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset_for_test();
        let (tx, _rx) = mpsc::channel();
        let tray = PordaTray::new(tx);
        // We cannot actually spawn DBus in unit test without session bus; test the flag logic directly.
        // Simulate first run by setting flag
        TRAY_INITIALIZED.store(true, Ordering::SeqCst);
        let result = tray.run();
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("already initialized"));
        reset_for_test();
    }

    #[test]
    fn manager_sends_actions() {
        let (tx, rx) = mpsc::channel();
        let mgr = TrayManager::new(tx);
        mgr.send_action(TrayAction::Activate).unwrap();
        assert_eq!(rx.recv().unwrap(), TrayAction::Activate);
    }

    #[test]
    fn lifecycle_window_close_not_terminate() {
        // Window close must hide window, not send Terminate.
        // Tray Show is UI concern (None), Exit is termination path.
        assert!(tray_action_to_ui_command(&TrayAction::Show).is_none());
        // Show must NOT map to Terminate
        assert!(!matches!(
            tray_action_to_ui_command(&TrayAction::Show),
            Some(UiCommand::Terminate)
        ));
        // Exit also maps to None here, but main.rs handles Exit as Terminate + quit_event_loop
        assert!(tray_action_to_ui_command(&TrayAction::Exit).is_none());
    }

    #[test]
    fn lifecycle_tray_exit_is_terminate_only() {
        // Only Tray Exit should cause termination; window close (Show) must not.
        // Verify distinction: Show != Exit, both not mapping to core Activate, but Exit is the only
        // tray action that triggers UiCommand::Terminate in main.rs handler.
        let show = TrayAction::Show;
        let exit = TrayAction::Exit;
        assert_ne!(show, exit);
        assert_eq!(show.label(), "Show");
        assert_eq!(exit.label(), "Exit");
    }

    #[test]
    fn lifecycle_show_is_only_ui_opener() {
        // Show is the single tray action for displaying main UI.
        let actions = TrayAction::menu_actions();
        assert!(actions.contains(&TrayAction::Show));
        assert_eq!(
            actions.iter().filter(|a| **a == TrayAction::Show).count(),
            1
        );
        // No OpenSettings / Settings variant exists
        assert_eq!(actions.len(), 5);
        assert_eq!(
            actions,
            vec![
                TrayAction::Show,
                TrayAction::Activate,
                TrayAction::Deactivate,
                TrayAction::TakeScreenshot,
                TrayAction::Exit
            ]
        );
    }

    #[test]
    fn open_settings_removed() {
        // Compile-time guarantee: TrayAction::OpenSettings must not exist.
        // Runtime check: menu does not contain Settings label.
        let actions = TrayAction::menu_actions();
        for a in &actions {
            assert_ne!(a.label(), "Settings");
            assert_ne!(a.label(), "OpenSettings");
        }
        assert!(!actions
            .iter()
            .any(|a| format!("{:?}", a).contains("OpenSettings")));
    }
}
