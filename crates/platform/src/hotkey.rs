use std::sync::{Arc, Mutex};

#[cfg(target_os = "linux")]
pub use crate::linux::hotkeys::{normalize_shortcut, parse_shortcut, HotkeyAction, HotkeyError};

#[cfg(not(target_os = "linux"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyAction {
    ToggleDetection,
}

#[cfg(not(target_os = "linux"))]
impl HotkeyAction {
    pub fn id(&self) -> &'static str {
        match self {
            Self::ToggleDetection => "porda_toggle_v2",
        }
    }
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "porda_toggle_v2" | "porda_toggle" | "toggle" => Some(Self::ToggleDetection),
            _ => None,
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn normalize_shortcut(input: &str) -> Result<String, String> {
    let t = input.trim();
    if t.is_empty() {
        return Err("shortcut is empty".to_string());
    }
    Ok(t.to_string())
}

#[cfg(not(target_os = "linux"))]
pub fn parse_shortcut(input: &str) -> Result<String, String> {
    normalize_shortcut(input)
}

/// Map a `HotkeyAction` to the existing `UiCommand` string for testing integration.
pub fn hotkey_action_to_command_name(action: HotkeyAction) -> &'static str {
    match action {
        HotkeyAction::ToggleDetection => "ToggleActivation",
    }
}

struct RegisteredHotkey {
    key: String,
    action: HotkeyAction,
}

/// Cross-platform in-memory manager for tests and non-Linux builds.
/// On Linux the real `LinuxHotkeyManager` (portal/KWin) should be used in production;
/// this manager provides the same `register`/`refresh` semantics for unit tests.
pub struct HotkeyManager {
    hotkeys: Arc<Mutex<Vec<RegisteredHotkey>>>,
}

impl HotkeyManager {
    pub fn new() -> Self {
        Self {
            hotkeys: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn register(&self, key: &str, action: HotkeyAction) -> Result<(), String> {
        #[cfg(target_os = "linux")]
        {
            let norm = crate::linux::hotkeys::normalize_shortcut(key).map_err(|e| e.to_string())?;
            let mut hotkeys = self.hotkeys.lock().map_err(|e| e.to_string())?;
            if hotkeys.iter().any(|h| h.key == norm && h.action == action) {
                return Ok(());
            }
            if hotkeys.iter().any(|h| h.key == norm) {
                return Err(format!("shortcut already registered: {}", norm));
            }
            hotkeys.push(RegisteredHotkey { key: norm, action });
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let norm = normalize_shortcut(key)?;
            let mut hotkeys = self.hotkeys.lock().map_err(|e| e.to_string())?;
            if hotkeys.iter().any(|h| h.key == norm && h.action == action) {
                return Ok(());
            }
            if hotkeys.iter().any(|h| h.key == norm) {
                return Err(format!("shortcut already registered: {}", norm));
            }
            hotkeys.push(RegisteredHotkey { key: norm, action });
            Ok(())
        }
    }

    pub fn unregister_all(&self) -> Result<(), String> {
        let mut hotkeys = self.hotkeys.lock().map_err(|e| e.to_string())?;
        hotkeys.clear();
        Ok(())
    }

    pub fn refresh(&self, toggle_key: &str) -> Result<(), String> {
        #[cfg(target_os = "linux")]
        {
            crate::linux::hotkeys::normalize_shortcut(toggle_key).map_err(|e| e.to_string())?;
        }
        #[cfg(not(target_os = "linux"))]
        {
            normalize_shortcut(toggle_key)?;
        }
        self.unregister_all()?;
        self.register(toggle_key, HotkeyAction::ToggleDetection)?;
        Ok(())
    }

    pub fn count(&self) -> usize {
        self.hotkeys.lock().map(|h| h.len()).unwrap_or(0)
    }

    pub fn is_registered(&self, key: &str) -> bool {
        #[cfg(target_os = "linux")]
        {
            if let Ok(n) = crate::linux::hotkeys::normalize_shortcut(key) {
                self.hotkeys
                    .lock()
                    .map(|h| h.iter().any(|e| e.key == n))
                    .unwrap_or(false)
            } else {
                false
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            if let Ok(n) = normalize_shortcut(key) {
                self.hotkeys
                    .lock()
                    .map(|h| h.iter().any(|e| e.key == n))
                    .unwrap_or(false)
            } else {
                false
            }
        }
    }
}

impl Default for HotkeyManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manager_register_and_refresh() {
        let m = HotkeyManager::new();
        m.register("Meta+Shift+P", HotkeyAction::ToggleDetection)
            .unwrap();
        assert_eq!(m.count(), 1);
        m.refresh("Meta+Shift+Q").unwrap();
        assert_eq!(m.count(), 1);
        assert!(m.is_registered("Meta+Shift+Q"));
        assert!(!m.is_registered("Meta+Shift+P"));
    }

    #[test]
    fn manager_duplicate_prevention() {
        let m = HotkeyManager::new();
        m.register("Meta+Shift+P", HotkeyAction::ToggleDetection)
            .unwrap();
        // same shortcut same action is idempotent, not an error
        assert!(m
            .register("Meta+Shift+P", HotkeyAction::ToggleDetection)
            .is_ok());
        assert_eq!(m.count(), 1);
        // duplicate prevention is per shortcut, not per action count
        assert!(m.is_registered("Meta+Shift+P"));
    }

    #[test]
    fn manager_refresh_idempotent() {
        let m = HotkeyManager::new();
        m.refresh("Meta+Shift+P").unwrap();
        m.refresh("Meta+Shift+P").unwrap();
        m.refresh("Meta+Shift+P").unwrap();
        assert_eq!(m.count(), 1);
    }

    #[test]
    fn hotkey_action_to_command_mapping() {
        assert_eq!(
            hotkey_action_to_command_name(HotkeyAction::ToggleDetection),
            "ToggleActivation"
        );
    }
}
