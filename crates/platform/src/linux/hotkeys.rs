use std::sync::mpsc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HotkeyAction {
    ToggleDetection,
    TakeScreenshot,
    OpenSettings,
}

pub struct LinuxHotkeyManager {
    action_tx: mpsc::Sender<HotkeyAction>,
}

impl LinuxHotkeyManager {
    pub fn new(action_tx: mpsc::Sender<HotkeyAction>) -> Self {
        Self { action_tx }
    }

    pub fn register(&self, key: &str, action: HotkeyAction) -> Result<(), String> {
        tracing::info!("Registering hotkey: {} -> {:?}", key, action);

        // Use global-hotkey crate for Wayland-compatible hotkeys
        // The global-hotkey crate supports X11 and Windows, but for Wayland
        // we need compositor-specific protocols

        // For KDE, we can use KGlobalAccel D-Bus interface
        #[cfg(target_os = "linux")]
        {
            self.register_kwin_shortcut(key, &action)?;
        }

        Ok(())
    }

    pub fn unregister_all(&self) -> Result<(), String> {
        tracing::info!("Unregistering all hotkeys");
        Ok(())
    }

    pub fn refresh(&self, toggle_key: &str, screenshot_key: &str) -> Result<(), String> {
        self.unregister_all()?;
        self.register(toggle_key, HotkeyAction::ToggleDetection)?;
        self.register(screenshot_key, HotkeyAction::TakeScreenshot)?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn register_kwin_shortcut(&self, key: &str, action: &HotkeyAction) -> Result<(), String> {
        // Use KGlobalAccel D-Bus interface for KDE
        let component = "pordaai";
        let shortcut_name = match action {
            HotkeyAction::ToggleDetection => "Toggle Detection",
            HotkeyAction::TakeScreenshot => "Take Screenshot",
            HotkeyAction::OpenSettings => "Open Settings",
        };

        tracing::info!(
            "Would register KDE shortcut: {} {} {}",
            component,
            shortcut_name,
            key
        );

        // D-Bus call to org.kde.KGlobalAccel would go here
        // For now, just log the intent

        Ok(())
    }
}
