use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyAction {
    ToggleDetection,
    TakeScreenshot,
    OpenSettings,
}

pub struct HotkeyManager {
    hotkeys: Arc<Mutex<Vec<RegisteredHotkey>>>,
}

struct RegisteredHotkey {
    #[allow(dead_code)]
    key: String,
    #[allow(dead_code)]
    action: HotkeyAction,
}

impl HotkeyManager {
    pub fn new() -> Self {
        Self {
            hotkeys: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn register(&self, key: &str, action: HotkeyAction) -> Result<(), String> {
        let mut hotkeys = self.hotkeys.lock().map_err(|e| e.to_string())?;
        hotkeys.push(RegisteredHotkey {
            key: key.to_string(),
            action,
        });
        Ok(())
    }

    pub fn unregister_all(&self) -> Result<(), String> {
        let mut hotkeys = self.hotkeys.lock().map_err(|e| e.to_string())?;
        hotkeys.clear();
        Ok(())
    }

    pub fn refresh(&self, toggle_key: &str, screenshot_key: &str) -> Result<(), String> {
        self.unregister_all()?;
        self.register(toggle_key, HotkeyAction::ToggleDetection)?;
        self.register(screenshot_key, HotkeyAction::TakeScreenshot)?;
        Ok(())
    }
}

impl Default for HotkeyManager {
    fn default() -> Self {
        Self::new()
    }
}
