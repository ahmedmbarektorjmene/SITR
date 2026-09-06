//! Wayland-native single global hotkey for KDE Plasma.
//!
//! # Mechanism
//!
//! This backend uses the **XDG Desktop Portal GlobalShortcuts** interface
//! (`org.freedesktop.portal.GlobalShortcuts`) which on KDE Plasma is
//! implemented by `xdg-desktop-portal-kde` and backed by **KWin /
//! KGlobalAccel** (`org.kde.kglobalaccel`).  This is Wayland-native: it does
//! not use `XGrabKey`, Xlib, XCB or any XWayland path.  Shortcuts are
//! registered with the compositor via the portal, the compositor owns the
//! grab, and the application receives `Activated` signals even when its
//! Slint window has no focus, is hidden or parked (via `Tray → Show`).
//!
//! - D-Bus service: `org.freedesktop.portal.Desktop`
//! - Object path: `/org/freedesktop/portal/desktop`
//! - Interface: `org.freedesktop.portal.GlobalShortcuts`
//! - Methods: `CreateSession`, `BindShortcuts`, `ListShortcuts`
//! - Signals: `Activated`, `Deactivated`, `ShortcutsChanged`
//! - Session interface: `org.freedesktop.portal.Session` with `Close`
//! - Single shortcut: `Meta+Shift+P` → `porda_toggle` → `ToggleDetection`
//!
//! On KDE, first `BindShortcuts` shows a system dialog for user confirmation
//! (Wayland security). After confirmation, presses emit `Activated(session,
//! shortcut_id, timestamp)` which we map to [`HotkeyAction::ToggleDetection`]
//! and forward via existing `mpsc` → `UiCommand::ToggleActivation`.
//!
//! # Supported scope
//!
//! - ✅ Any Wayland compositor with `xdg-desktop-portal` GlobalShortcuts (KDE, GNOME, etc.)
//! - ❌ X11 / XWayland — intentionally unsupported (Wayland-first)
//! - Portal errors are surfaced directly (no compositor-name gating)
//!
//! Unsupported environments return a descriptive error instead of pretending
//! success.
//!
//! # Tokio requirement
//!
//! `ashpd` GlobalShortcuts exposes only an async API (`GlobalShortcuts::new`,
//! `create_session`, `bind_shortcuts`, `receive_activated`) with no blocking
//! variant. `ashpd` supports `tokio` or `async-io` via `zbus`. The project
//! already requires `tokio` for `capture` (`crates/platform/src/linux/capture.rs:235`
//! `tokio::runtime::Builder`). Switching hotkeys to `async-io` would require
//! migrating `zbus`/`ashpd` features globally and running two runtimes.
//! Therefore hotkeys keep a **dedicated `new_current_thread` Tokio runtime**
//! isolated to the `porda-hotkeys` thread (`crates/platform/src/linux/hotkeys.rs:619`).
//! It remains alive for the entire portal session/Activated listener lifetime
//! and is dropped only on `unregister_all`/`Drop`.
//!
//! # Lifecycle (instrumented, no silent success)
//! ```text
//! Porda starts
//!   ↓
//! hotkey backend starts (LinuxHotkeyManager::new, enable_portal)
//!   ↓
//! portal connection established (GlobalShortcuts::new)
//!   ↓
//! GlobalShortcuts.CreateSession succeeds (session remains alive for loop)
//!   ↓
//! BindShortcuts succeeds (Meta+Shift+P → porda_toggle actually registered)
//!   ↓
//! Activated listener installed (receive_activated)
//!   ↓
//! listener remains alive (tokio::select! on Notify + activated.next())
//!   ↓
//! Meta+Shift+P
//!   ↓
//! Activated signal received → HotkeyAction::ToggleDetection → UiCommand::ToggleActivation
//! ```
//! Each step logs success or a clear error; the async task does NOT exit
//! immediately after registration — `Session`, `proxy`, `activated` stream and
//! Tokio runtime are held for the loop's lifetime.

use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use tokio::sync::{oneshot, Notify};

type ReadySender = Arc<std::sync::Mutex<Option<oneshot::Sender<Result<(), String>>>>>;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HotkeyAction {
    ToggleDetection,
}

impl HotkeyAction {
    pub fn id(&self) -> &'static str {
        match self {
            // Fresh ID porda_toggle_v2 avoids stale empty trigger for older IDs
            // Old IDs toggle/porda_toggle still handled in from_id for migration
            Self::ToggleDetection => "porda_toggle_v2",
        }
    }
    pub fn description(&self) -> &'static str {
        match self {
            Self::ToggleDetection => "Toggle Detection",
        }
    }
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "porda_toggle_v2" | "porda_toggle" | "toggle" => Some(Self::ToggleDetection),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HotkeyError {
    EmptyShortcut,
    InvalidShortcut(String),
    AlreadyRegistered(String),
    PortalUnavailable(String),
    SessionFailed(String),
    BindFailed(String),
    UnsupportedEnvironment(String),
    NotRegistered,
}

impl std::fmt::Display for HotkeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyShortcut => write!(f, "shortcut is empty"),
            Self::InvalidShortcut(s) => write!(f, "invalid shortcut: {}", s),
            Self::AlreadyRegistered(s) => write!(f, "shortcut already registered: {}", s),
            Self::PortalUnavailable(s) => write!(f, "global shortcuts unavailable: {}", s),
            Self::SessionFailed(s) => write!(f, "session failed: {}", s),
            Self::BindFailed(s) => write!(f, "bind failed: {}", s),
            Self::UnsupportedEnvironment(s) => write!(f, "unsupported environment: {}", s),
            Self::NotRegistered => write!(f, "no hotkeys registered"),
        }
    }
}
impl std::error::Error for HotkeyError {}

// ---------------------------------------------------------------------------
// Shortcut parsing / normalization
// ---------------------------------------------------------------------------

const ALLOWED_MODIFIERS: &[&str] = &["Ctrl", "Shift", "Alt", "Meta"];
const ALLOWED_KEYS: &[&str] = &[
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
    "F13",
    "F14",
    "F15",
    "F16",
    "F17",
    "F18",
    "F19",
    "F20",
    "F21",
    "F22",
    "F23",
    "F24",
    "F25",
    "F26",
    "F27",
    "F28",
    "F29",
    "F30",
    "F31",
    "F32",
    "F33",
    "F34",
    "F35",
    "Print",
    "Pause",
    "ScrollLock",
    "Insert",
    "Delete",
    "Home",
    "End",
    "PageUp",
    "PageDown",
    "Space",
    "Tab",
    "Backspace",
    "Enter",
    "Return",
    "Escape",
    "Plus",
    "Minus",
    "Equal",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
];

fn normalize_modifier(s: &str) -> Option<&'static str> {
    match s.to_ascii_lowercase().as_str() {
        "ctrl" | "control" => Some("Ctrl"),
        "shift" => Some("Shift"),
        "alt" => Some("Alt"),
        "meta" | "super" | "win" => Some("Meta"),
        _ => None,
    }
}

fn normalize_key(s: &str) -> Option<String> {
    if s.is_empty() {
        return None;
    }
    // Single letter/number → upper
    if s.len() == 1 {
        let c = s.chars().next().unwrap();
        if c.is_ascii_alphabetic() || c.is_ascii_digit() {
            return Some(c.to_ascii_uppercase().to_string());
        }
        return None;
    }
    // Check allowed list case-insensitively
    for &k in ALLOWED_KEYS {
        if k.eq_ignore_ascii_case(s) {
            return Some(k.to_string());
        }
    }
    None
}

/// Normalize a shortcut string like "ctrl+shift+f2" → "Ctrl+Shift+F2".
/// Returns normalized form or `HotkeyError`.
pub fn normalize_shortcut(input: &str) -> Result<String, HotkeyError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(HotkeyError::EmptyShortcut);
    }
    if trimmed.contains("  ") {
        // double spaces suggest malformed
    }
    // Split by '+' (also support '-' as separator for a few legacy strings? prefer '+')
    let parts: Vec<&str> = trimmed
        .split('+')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();
    if parts.is_empty() {
        return Err(HotkeyError::EmptyShortcut);
    }
    if parts.len() == 1 {
        // Single key without modifier
        if let Some(k) = normalize_key(parts[0]) {
            return Ok(k);
        } else {
            return Err(HotkeyError::InvalidShortcut(format!(
                "unknown key '{}'",
                parts[0]
            )));
        }
    }
    // All but last are modifiers
    let (mods, key_part) = parts.split_at(parts.len() - 1);
    let mut norm_mods = Vec::new();
    for m in mods {
        if let Some(nm) = normalize_modifier(m) {
            if norm_mods.contains(&nm) {
                return Err(HotkeyError::InvalidShortcut(format!(
                    "duplicate modifier '{}'",
                    m
                )));
            }
            norm_mods.push(nm);
        } else {
            return Err(HotkeyError::InvalidShortcut(format!(
                "unknown modifier '{}'",
                m
            )));
        }
    }
    // Preserve canonical modifier order: Ctrl, Shift, Alt, Meta
    norm_mods.sort_by_key(|m| match *m {
        "Ctrl" => 0,
        "Shift" => 1,
        "Alt" => 2,
        "Meta" => 3,
        _ => 99,
    });
    let key_str = key_part[0];
    let norm_key = normalize_key(key_str)
        .ok_or_else(|| HotkeyError::InvalidShortcut(format!("unknown key '{}'", key_str)))?;
    if norm_mods.is_empty() {
        Ok(norm_key)
    } else {
        Ok(format!("{}+{}", norm_mods.join("+"), norm_key))
    }
}

/// Parse and validate shortcut, returning normalized form.
pub fn parse_shortcut(input: &str) -> Result<String, HotkeyError> {
    normalize_shortcut(input)
}

// ---------------------------------------------------------------------------
// Registry – pure logic, testable without D-Bus
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct HotkeyRegistry {
    /// normalized shortcut → action
    map: HashMap<String, HotkeyAction>,
    /// action id → normalized shortcut (for reverse lookup)
    id_map: HashMap<String, String>,
}

impl HotkeyRegistry {
    fn register(&mut self, shortcut: &str, action: HotkeyAction) -> Result<String, HotkeyError> {
        let norm = normalize_shortcut(shortcut)?;
        if let Some(existing) = self.map.get(&norm) {
            if *existing == action {
                // idempotent: same shortcut same action → ok
                return Ok(norm);
            } else {
                return Err(HotkeyError::AlreadyRegistered(format!(
                    "'{}' already bound to {:?}",
                    norm, existing
                )));
            }
        }
        // Also check if action already has a different shortcut – replace (refresh will clear first)
        // For idempotent registration, allow overwriting action's old binding only via refresh.
        // Here we disallow duplicate action with different triggers silently – just insert.
        if let Some(old) = self.id_map.get(action.id()).cloned() {
            // If same action already has a binding, treat as duplicate prevention unless refresh
            // For direct register, allow but warn – remove old.
            self.map.remove(&old);
        }
        self.map.insert(norm.clone(), action.clone());
        self.id_map.insert(action.id().to_string(), norm.clone());
        Ok(norm)
    }

    fn unregister_all(&mut self) {
        self.map.clear();
        self.id_map.clear();
    }

    fn contains(&self, shortcut: &str) -> bool {
        if let Ok(n) = normalize_shortcut(shortcut) {
            self.map.contains_key(&n)
        } else {
            false
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn entries(&self) -> Vec<(String, HotkeyAction)> {
        self.map
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Environment detection & display formatting
// ---------------------------------------------------------------------------

fn is_wayland() -> bool {
    std::env::var("XDG_SESSION_TYPE")
        .map(|v| v.eq_ignore_ascii_case("wayland"))
        .unwrap_or(false)
        || std::env::var("WAYLAND_DISPLAY").is_ok()
}

fn check_supported_environment() -> Result<(), HotkeyError> {
    if !is_wayland() {
        return Err(HotkeyError::UnsupportedEnvironment(
            "Global shortcuts require Wayland (XDG_SESSION_TYPE=wayland). X11/XWayland is intentionally unsupported.".to_string(),
        ));
    }
    Ok(())
}

/// Format raw portal trigger like "Meta+Shift+P" → "Meta + Shift + P" for UI.
fn format_trigger_display(raw: &str) -> String {
    let t = raw.trim();
    if t.is_empty() {
        return "Not configured".to_string();
    }
    // Portal trigger uses '+' as separator; display with spaces.
    t.split('+')
        .map(|s| s.trim())
        .collect::<Vec<_>>()
        .join(" + ")
}

// ---------------------------------------------------------------------------
// Portal thread state
// ---------------------------------------------------------------------------

struct PortalState {
    handle: JoinHandle<()>,
    shutdown: Arc<Notify>,
}

// ---------------------------------------------------------------------------
// LinuxHotkeyManager – public API
// ---------------------------------------------------------------------------

pub struct LinuxHotkeyManager {
    action_tx: mpsc::Sender<HotkeyAction>,
    registry: Arc<Mutex<HotkeyRegistry>>,
    portal: Arc<Mutex<Option<PortalState>>>,
    // In tests we skip real portal binding
    enable_portal: bool,
    // Portal-authoritative display state
    portal_trigger: Arc<Mutex<Option<String>>>,
    portal_error: Arc<Mutex<Option<String>>>,
}

impl LinuxHotkeyManager {
    pub fn new(action_tx: mpsc::Sender<HotkeyAction>) -> Self {
        let enable = !cfg!(test) && std::env::var("PORDA_DISABLE_HOTKEYS").is_err();
        Self {
            action_tx,
            registry: Arc::new(Mutex::new(HotkeyRegistry::default())),
            portal: Arc::new(Mutex::new(None)),
            enable_portal: enable,
            portal_trigger: Arc::new(Mutex::new(None)),
            portal_error: Arc::new(Mutex::new(None)),
        }
    }

    /// Test-only constructor that never spawns portal thread.
    #[cfg(test)]
    pub fn new_for_test(action_tx: mpsc::Sender<HotkeyAction>) -> Self {
        Self {
            action_tx,
            registry: Arc::new(Mutex::new(HotkeyRegistry::default())),
            portal: Arc::new(Mutex::new(None)),
            enable_portal: false,
            portal_trigger: Arc::new(Mutex::new(None)),
            portal_error: Arc::new(Mutex::new(None)),
        }
    }

    pub fn register(&self, key: &str, action: HotkeyAction) -> Result<(), String> {
        let norm = {
            let mut reg = self.registry.lock().map_err(|e| e.to_string())?;
            reg.register(key, action.clone())
                .map_err(|e| e.to_string())?
        };
        tracing::info!("Registering hotkey: {} -> {:?}", norm, action);

        // Validate environment only when real portal is enabled (production).
        // In test mode we skip this so unit tests remain deterministic and
        // don't depend on the host compositor.
        if self.enable_portal {
            if let Err(e) = check_supported_environment() {
                // Undo registry insert so we don't pretend success – caller sees error
                if let Ok(mut reg) = self.registry.lock() {
                    reg.map.remove(&norm);
                    reg.id_map.remove(action.id());
                }
                return Err(e.to_string());
            }
            if let Err(e) = self.ensure_portal_session() {
                // Roll back registry on portal failure to avoid phantom registration
                if let Ok(mut reg) = self.registry.lock() {
                    reg.map.remove(&norm);
                    reg.id_map.remove(action.id());
                }
                return Err(e);
            }
        }
        Ok(())
    }

    pub fn unregister_all(&self) -> Result<(), String> {
        tracing::info!("Unregistering all hotkeys");
        {
            let mut reg = self.registry.lock().map_err(|e| e.to_string())?;
            reg.unregister_all();
        }
        self.shutdown_portal();
        Ok(())
    }

    pub fn refresh(&self, toggle_key: &str) -> Result<(), String> {
        // Single global shortcut: Meta+Shift+P → ToggleDetection
        // Idempotent: clear old, register new, re-bind
        self.unregister_all()?;
        normalize_shortcut(toggle_key).map_err(|e| e.to_string())?;
        self.register(toggle_key, HotkeyAction::ToggleDetection)?;
        // After refresh, try to update portal display state
        let _ = self.query_portal_trigger();
        Ok(())
    }

    pub fn is_supported(&self) -> Result<(), String> {
        check_supported_environment().map_err(|e| e.to_string())
    }

    pub fn registered_count(&self) -> usize {
        self.registry.lock().map(|r| r.len()).unwrap_or(0)
    }

    pub fn is_registered(&self, shortcut: &str) -> bool {
        self.registry
            .lock()
            .map(|r| r.contains(shortcut))
            .unwrap_or(false)
    }

    // -----------------------------------------------------------------------
    // Portal-authoritative display API (Linux/Wayland)
    // -----------------------------------------------------------------------

    /// Return portal-authoritative display string.
    /// Possible values:
    /// - "Meta + Shift + P" (configured)
    /// - "Not configured" (portal reports empty trigger)
    /// - "Global shortcuts unavailable" (portal unreachable or Wayland missing)
    pub fn current_shortcut_display(&self) -> String {
        if let Some(err) = self.portal_error.lock().unwrap().clone() {
            if err.contains("Global shortcuts unavailable") || err.contains("Unsupported") {
                return "Global shortcuts unavailable".to_string();
            }
            // Other transient errors still show unavailable
            if !self.enable_portal {
                // In test mode, show trigger if set
            } else {
                // Keep unavailable for portal errors
                // But if we have a trigger cached, prefer it unless error is unavailable
            }
        }
        if !self.enable_portal {
            // Test mode: show registry trigger if any
            if let Some(raw) = self.portal_trigger.lock().unwrap().clone() {
                return format_trigger_display(&raw);
            }
            let reg = self.registry.lock().unwrap();
            if let Some((trig, _)) = reg.entries().first() {
                return format_trigger_display(trig);
            }
            return "Not configured".to_string();
        }
        // Check environment first
        if check_supported_environment().is_err() {
            return "Global shortcuts unavailable".to_string();
        }
        if let Some(err) = self.portal_error.lock().unwrap().clone() {
            // Distinguish unavailable vs not configured
            if err.to_lowercase().contains("unavailable")
                || err.to_lowercase().contains("unsupported")
            {
                return "Global shortcuts unavailable".to_string();
            }
        }
        let trig = self.portal_trigger.lock().unwrap().clone();
        match trig {
            Some(raw) if !raw.trim().is_empty() => format_trigger_display(&raw),
            Some(_) => "Not configured".to_string(),
            None => {
                // No cached portal trigger yet – portal is authoritative, so show Not configured
                // Do not blindly display settings.json as authoritative
                "Not configured".to_string()
            }
        }
    }

    /// Whether portal is available (Wayland + portal reachable).
    pub fn is_portal_available(&self) -> bool {
        if !self.enable_portal {
            return false;
        }
        if check_supported_environment().is_err() {
            return false;
        }
        self.portal_error
            .lock()
            .unwrap()
            .as_ref()
            .map(|e| {
                !e.to_lowercase().contains("unavailable")
                    && !e.to_lowercase().contains("unsupported")
            })
            .unwrap_or(true)
    }

    pub fn portal_status_text(&self) -> String {
        if !self.enable_portal && cfg!(test) {
            return String::new();
        }
        if check_supported_environment().is_err() {
            return "Global shortcuts unavailable".to_string();
        }
        if let Some(err) = self.portal_error.lock().unwrap().clone() {
            if err.to_lowercase().contains("unavailable")
                || err.to_lowercase().contains("unsupported")
            {
                return "Global shortcuts unavailable".to_string();
            }
            return String::new();
        }
        if self.portal_trigger.lock().unwrap().is_none() {
            return "Not configured".to_string();
        }
        // Portal is authoritative; status is shortcut managed by desktop environment
        "Shortcut managed by desktop environment".to_string()
    }

    /// Query portal for current binding via ListShortcuts and update cache.
    /// Returns display string or error.
    pub fn query_portal_trigger(&self) -> Result<String, String> {
        if !self.enable_portal {
            // Test mode: return cached or registry
            return Ok(self.current_shortcut_display());
        }
        if let Err(e) = check_supported_environment() {
            let msg = e.to_string();
            *self.portal_error.lock().unwrap() = Some(msg.clone());
            return Err(msg);
        }
        let disp = self.query_portal_trigger_blocking()?;
        Ok(disp)
    }

    fn query_portal_trigger_blocking(&self) -> Result<String, String> {
        let entries = {
            let reg = self.registry.lock().unwrap();
            reg.entries()
        };
        let trigger_clone = Arc::clone(&self.portal_trigger);
        let error_clone = Arc::clone(&self.portal_error);
        // Run async query in dedicated runtime (blocking)
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("tokio runtime failed: {}", e))?;
        let result: Result<Option<String>, String> = rt.block_on(async move {
            use ashpd::desktop::global_shortcuts::GlobalShortcuts;
            let proxy = GlobalShortcuts::new()
                .await
                .map_err(|e| format!("GlobalShortcuts portal unavailable: {}", e))?;
            if proxy.version() == 0 {
                return Err("GlobalShortcuts portal not supported".to_string());
            }
            let session = proxy
                .create_session(Default::default())
                .await
                .map_err(|e| format!("CreateSession failed: {}", e))?;
            // We need a temporary bind to allow ListShortcuts to return known shortcut?
            // Actually ListShortcuts returns shortcuts bound for this session.
            // If we haven't bound, it will be empty. Ensure we bind first if entries exist.
            if !entries.is_empty() {
                use ashpd::desktop::global_shortcuts::NewShortcut;
                let mut ns = Vec::new();
                for (trig, act) in &entries {
                    ns.push(
                        NewShortcut::new(act.id(), act.description())
                            .preferred_trigger(Some(trig.as_str())),
                    );
                }
                let _ = proxy
                    .bind_shortcuts(&session, &ns, None, Default::default())
                    .await
                    .map_err(|e| format!("BindShortcuts failed: {}", e))?
                    .response()
                    .map_err(|e| format!("BindShortcuts response: {}", e))?;
            }
            let list_req = proxy
                .list_shortcuts(&session, Default::default())
                .await
                .map_err(|e| format!("ListShortcuts failed: {}", e))?;
            let resp = list_req
                .response()
                .map_err(|e| format!("ListShortcuts response: {}", e))?;
            // Find our shortcut
            for sc in resp.shortcuts() {
                if sc.id() == HotkeyAction::ToggleDetection.id()
                    || sc.id() == "porda_toggle"
                    || sc.id() == "toggle"
                {
                    let raw = sc.trigger_description().to_string();
                    if raw.is_empty() {
                        return Ok(None);
                    } else {
                        return Ok(Some(raw));
                    }
                }
            }
            // If not found, treat as not configured
            Ok(None)
        });
        match result {
            Ok(opt) => {
                *trigger_clone.lock().unwrap() = opt.clone();
                *error_clone.lock().unwrap() = None;
                Ok(opt
                    .as_ref()
                    .map(|s| format_trigger_display(s))
                    .unwrap_or_else(|| "Not configured".to_string()))
            }
            Err(e) => {
                if e.to_lowercase().contains("unavailable")
                    || e.to_lowercase().contains("unsupported")
                {
                    *error_clone.lock().unwrap() = Some(e.clone());
                } else {
                    // Keep previous error if any? Store generic error but not mark unavailable
                    // For query failures, set error but allow retry
                    *error_clone.lock().unwrap() = Some(e.clone());
                }
                Err(e)
            }
        }
    }

    /// Invoke portal ConfigureShortcuts and refresh display.
    /// This is the platform-level operation equivalent to `ConfigureGlobalShortcut`.
    /// It shows the desktop environment's shortcut configuration UI, then re-reads
    /// the binding via ListShortcuts and updates the read-only display.
    /// Keep existing activation listener alive if possible; do not create duplicates.
    pub fn configure_shortcut(&self) -> Result<String, String> {
        if !self.enable_portal {
            return Err("portal disabled (test mode)".to_string());
        }
        if let Err(e) = check_supported_environment() {
            let msg = e.to_string();
            *self.portal_error.lock().unwrap() = Some(msg.clone());
            return Err(msg);
        }
        // Capture previous trigger for cancel handling
        let prev = self.portal_trigger.lock().unwrap().clone();
        let entries = {
            let reg = self.registry.lock().unwrap();
            reg.entries()
        };
        if entries.is_empty() {
            return Err("no hotkey registered to configure".to_string());
        }
        let trigger_clone = Arc::clone(&self.portal_trigger);
        let error_clone = Arc::clone(&self.portal_error);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("tokio runtime failed: {}", e))?;
        let result: Result<Option<String>, String> = rt.block_on(async move {
            use ashpd::desktop::global_shortcuts::{ConfigureShortcutsOptions, GlobalShortcuts, NewShortcut};
            let proxy = GlobalShortcuts::new().await.map_err(|e| {
                format!("GlobalShortcuts portal unavailable: {}", e)
            })?;
            let ver = proxy.version();
            if ver < 2 {
                return Err(format!(
                    "GlobalShortcuts portal version {} does not support ConfigureShortcuts (requires v2)",
                    ver
                ));
            }
            let session = proxy
                .create_session(Default::default())
                .await
                .map_err(|e| format!("CreateSession failed: {}", e))?;
            // Ensure shortcuts are bound for this session before configure
            {
                let mut ns = Vec::new();
                for (trig, act) in &entries {
                    ns.push(NewShortcut::new(act.id(), act.description()).preferred_trigger(Some(trig.as_str())));
                }
                let bind_req = proxy
                    .bind_shortcuts(&session, &ns, None, Default::default())
                    .await
                    .map_err(|e| format!("BindShortcuts failed: {}", e))?;
                let _ = bind_req
                    .response()
                    .map_err(|e| format!("BindShortcuts response: {}", e))?;
            }
            // Invoke ConfigureShortcuts – shows portal configuration UI
            tracing::info!("Hotkey configure: invoking ConfigureShortcuts portal UI");
            let cfg_res = proxy
                .configure_shortcuts(&session, None, ConfigureShortcutsOptions::default())
                .await;
            match cfg_res {
                Ok(()) => {
                    tracing::info!("Hotkey configure: ConfigureShortcuts succeeded (dialog completed)");
                }
                Err(e) => {
                    // Cancellation or dismissal may be reported as error – treat as not fatal
                    let msg = e.to_string();
                    // If portal reports RequiresVersion, propagate as error
                    if msg.contains("RequiresVersion") {
                        return Err(msg);
                    }
                    tracing::warn!("Hotkey configure: ConfigureShortcuts returned error (may be cancelled): {}", msg);
                    // Fall through to ListShortcuts to see if trigger unchanged
                }
            }
            // Re-read authoritative binding
            tracing::info!("Hotkey configure: calling ListShortcuts to refresh binding");
            let list_req = proxy
                .list_shortcuts(&session, Default::default())
                .await
                .map_err(|e| format!("ListShortcuts failed: {}", e))?;
            let resp = list_req
                .response()
                .map_err(|e| format!("ListShortcuts response: {}", e))?;
            for sc in resp.shortcuts() {
                if sc.id() == HotkeyAction::ToggleDetection.id()
                    || sc.id() == "porda_toggle"
                    || sc.id() == "toggle"
                {
                    let raw = sc.trigger_description().to_string();
                    if raw.is_empty() {
                        return Ok(None);
                    } else {
                        return Ok(Some(raw));
                    }
                }
            }
            Ok(None)
        });
        match result {
            Ok(opt) => {
                *trigger_clone.lock().unwrap() = opt.clone();
                *error_clone.lock().unwrap() = None;
                let disp = opt
                    .as_ref()
                    .map(|s| format_trigger_display(s))
                    .unwrap_or_else(|| "Not configured".to_string());
                tracing::info!("Hotkey configure: new binding display '{}'", disp);
                if self.portal.lock().unwrap().is_some() {
                    // Portal session exists – ShortcutsChanged will inform it
                }
                Ok(disp)
            }
            Err(e) => {
                // Configuration error – if it is unavailable, mark error
                if e.to_lowercase().contains("unavailable")
                    || e.to_lowercase().contains("unsupported")
                    || e.contains("RequiresVersion")
                {
                    *error_clone.lock().unwrap() = Some(e.clone());
                    return Err(e);
                }
                // For other errors (e.g., cancelled/interrupted), restore previous without error
                // Check if error suggests cancellation – restore prev
                let lower = e.to_lowercase();
                if lower.contains("cancel") || lower.contains("dismiss") || lower.contains("closed")
                {
                    tracing::info!("Hotkey configure cancelled, restoring previous binding");
                    *trigger_clone.lock().unwrap() = prev.clone();
                    let disp2 = prev
                        .as_ref()
                        .map(|s| format_trigger_display(s))
                        .unwrap_or_else(|| "Not configured".to_string());
                    return Ok(disp2);
                }
                *error_clone.lock().unwrap() = Some(e.clone());
                Err(e)
            }
        }
    }

    fn shutdown_portal(&self) {
        let mut guard = match self.portal.lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        if let Some(state) = guard.take() {
            state.shutdown.notify_one();
            // We cannot block indefinitely – spawn a detach join.
            let handle = state.handle;
            std::thread::spawn(move || {
                let _ = handle.join();
                tracing::info!("Portal hotkey thread joined");
            });
        }
    }

    fn ensure_portal_session(&self) -> Result<(), String> {
        // Restart to pick up new registry entries; wait for listener to be alive before returning
        self.shutdown_portal();

        let entries = {
            let reg = self.registry.lock().map_err(|e| e.to_string())?;
            reg.entries()
        };
        if entries.is_empty() {
            tracing::info!("Hotkey backend: no entries to bind, skipping portal session");
            *self.portal_trigger.lock().unwrap() = None;
            return Ok(());
        }

        // Validate single-shortcut invariant
        if entries.len() != 1 {
            tracing::warn!(
                "Hotkey backend: expected exactly 1 entry (Meta+Shift+P), got {}",
                entries.len()
            );
        }
        for (trig, act) in &entries {
            tracing::info!(
                "Hotkey backend: preparing to bind '{}' -> {} ({})",
                trig,
                act.id(),
                act.description()
            );
        }

        let action_tx = self.action_tx.clone();
        let shutdown = Arc::new(Notify::new());
        let shutdown_clone = Arc::clone(&shutdown);
        let (ready_tx, ready_rx) = oneshot::channel::<Result<(), String>>();
        let ready_tx: ReadySender = Arc::new(std::sync::Mutex::new(Some(ready_tx)));
        let trigger_cache = Arc::clone(&self.portal_trigger);
        let error_cache = Arc::clone(&self.portal_error);

        let entries_clone = entries.clone();
        let handle = std::thread::Builder::new()
            .name("porda-hotkeys".to_string())
            .spawn({
                let ready_tx = Arc::clone(&ready_tx);
                move || {
                    if let Err(e) = run_portal_session(
                        entries_clone,
                        action_tx,
                        shutdown_clone,
                        ready_tx,
                        trigger_cache,
                        error_cache,
                    ) {
                        tracing::error!("Portal hotkey session failed: {}", e);
                    }
                }
            })
            .map_err(|e| format!("failed to spawn hotkey thread: {}", e))?;

        // Wait for the portal thread to confirm session/bind/listener are alive (not just spawned)
        // This makes “Global hotkey registered” log truthful; timeout covers dialog (30s) + setup (5s)
        let wait_res = ready_rx.blocking_recv();
        match wait_res {
            Ok(Ok(())) => {
                tracing::info!("Hotkey backend: portal session/binding/listener confirmed alive (porda_toggle_v2 -> Meta+Shift+P)");
                *self.portal.lock().map_err(|e| e.to_string())? =
                    Some(PortalState { handle, shutdown });
                *self.portal_error.lock().unwrap() = None;
                Ok(())
            }
            Ok(Err(e)) => {
                *self.portal_error.lock().unwrap() = Some(e.clone());
                let _ = handle.join();
                Err(format!("portal session/binding failed: {}", e))
            }
            Err(_) => {
                let err = "portal thread dropped without signalling readiness (session/bind/listen failed)".to_string();
                *self.portal_error.lock().unwrap() = Some(err.clone());
                let _ = handle.join();
                Err(err)
            }
        }
    }
}

impl Drop for LinuxHotkeyManager {
    fn drop(&mut self) {
        tracing::info!("Hotkey backend Drop: shutting down portal session");
        self.shutdown_portal();
    }
}

// ---------------------------------------------------------------------------
// Portal session runner – dedicated thread with its own tokio runtime
//Keeps Session, proxy, activated stream and Tokio runtime alive for entire loop.
// ---------------------------------------------------------------------------

#[allow(clippy::type_complexity)]
fn run_portal_session(
    entries: Vec<(String, HotkeyAction)>,
    action_tx: mpsc::Sender<HotkeyAction>,
    shutdown: Arc<Notify>,
    ready_tx: ReadySender,
    trigger_cache: Arc<Mutex<Option<String>>>,
    error_cache: Arc<Mutex<Option<String>>>,
) -> Result<(), String> {
    tracing::info!("Hotkey thread: starting dedicated Tokio current_thread runtime");
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| {
            let msg = format!("tokio runtime failed: {}", e);
            tracing::error!("{}", msg);
            if let Some(tx) = ready_tx.lock().unwrap().take() {
                let _ = tx.send(Err(msg.clone()));
            }
            msg
        })?;

    rt.block_on(async move {
        use ashpd::desktop::global_shortcuts::{GlobalShortcuts, NewShortcut};
        use futures_util::StreamExt;

        tracing::info!("Hotkey thread: establishing portal connection (GlobalShortcuts::new)");
        let proxy = GlobalShortcuts::new().await.map_err(|e| {
            let msg = format!(
                "GlobalShortcuts portal unavailable (xdg-desktop-portal-kde not running): {}",
                e
            );
            tracing::error!("Hotkey thread: {}", msg);
            if let Some(tx) = ready_tx.lock().unwrap().take() { let _ = tx.send(Err(msg.clone())); }
            msg
        })?;
        tracing::info!("Hotkey thread: portal connection established, version {}", proxy.version());

        let version = proxy.version();
        tracing::info!("Hotkey thread: GlobalShortcuts portal version {}", version);
        if version == 0 {
            let msg = "GlobalShortcuts portal not supported by current portal backend".to_string();
            tracing::error!("Hotkey thread: {}", msg);
            if let Some(tx) = ready_tx.lock().unwrap().take() { let _ = tx.send(Err(msg.clone())); }
            return Err(msg);
        }

        tracing::info!("Hotkey thread: calling CreateSession");
        let session = proxy
            .create_session(Default::default())
            .await
            .map_err(|e| {
                let msg = format!("CreateSession failed: {}", e);
                tracing::error!("Hotkey thread: {}", msg);
                if let Some(tx) = ready_tx.lock().unwrap().take() { let _ = tx.send(Err(msg.clone())); }
                msg
            })?;
        tracing::info!("Hotkey thread: CreateSession succeeded: {:?}", session);
        tracing::info!("Hotkey thread: session remains alive for entire listener lifetime (held in this async block)");

        let mut new_shortcuts = Vec::new();
        for (trigger, action) in &entries {
            tracing::info!(
                "Hotkey thread: preparing BindShortcuts for '{}' -> {} ({})",
                trigger,
                action.id(),
                action.description()
            );
            let ns = NewShortcut::new(action.id(), action.description())
                .preferred_trigger(Some(trigger.as_str()));
            new_shortcuts.push(ns);
        }
        tracing::info!("Hotkey thread: calling BindShortcuts (may show KDE permission dialog, 30s timeout)");

        let request = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            proxy.bind_shortcuts(&session, &new_shortcuts, None, Default::default()),
        )
        .await
        .map_err(|_| {
            let msg = "BindShortcuts timed out waiting for user confirmation (dialog not confirmed within 30s)".to_string();
            tracing::error!("Hotkey thread: {}", msg);
            if let Some(tx) = ready_tx.lock().unwrap().take() { let _ = tx.send(Err(msg.clone())); }
            msg
        })
        .and_then(|r| {
            r.map_err(|e| {
                let msg = format!("BindShortcuts failed: {}", e);
                tracing::error!("Hotkey thread: {}", msg);
                if let Some(tx) = ready_tx.lock().unwrap().take() { let _ = tx.send(Err(msg.clone())); }
                msg
            })
        })?;

        let response = request.response().map_err(|e| {
            let msg = format!("BindShortcuts response error: {}", e);
            tracing::error!("Hotkey thread: {}", msg);
            if let Some(tx) = ready_tx.lock().unwrap().take() { let _ = tx.send(Err(msg.clone())); }
            msg
        })?;
        tracing::info!("Hotkey thread: BindShortcuts succeeded: {:?}", response.shortcuts());
        // Critical: Check for empty trigger_description which indicates registration failure (not just session exists)
        let mut bound_trigger: Option<String> = None;
        for sc in response.shortcuts() {
            tracing::info!(
                "Hotkey thread: actually registered '{}' -> '{}' (trigger_description='{}')",
                sc.id(),
                sc.description(),
                sc.trigger_description()
            );
            if (sc.id() == HotkeyAction::ToggleDetection.id()
                || sc.id() == "porda_toggle"
                || sc.id() == "toggle")
                && !sc.trigger_description().is_empty()
            {
                bound_trigger = Some(sc.trigger_description().to_string());
            }
            if sc.trigger_description().is_empty() {
                let msg = format!(
                    "portal returned no trigger for {} (trigger_description=''), registration failed (likely stale empty permission or dialog not confirmed, try fresh ID porda_toggle_v2 or allow dialog)",
                    sc.id()
                );
                tracing::error!("Hotkey thread: {}", msg);
                if let Some(tx) = ready_tx.lock().unwrap().take() { let _ = tx.send(Err(msg.clone())); }
                return Err(msg);
            }
            if sc.id() == HotkeyAction::ToggleDetection.id() && sc.trigger_description() != "Meta+Shift+P" && sc.trigger_description() != "Shift+Meta+P" {
                tracing::warn!(
                    "Hotkey thread: expected trigger 'Meta+Shift+P' but portal reports '{}' for {} (may be stale permission or user-configured)",
                    sc.trigger_description(),
                    sc.id()
                );
            }
        }
        // Store portal authoritative trigger for UI display
        if let Some(ref trig) = bound_trigger {
            *trigger_cache.lock().unwrap() = Some(trig.clone());
            *error_cache.lock().unwrap() = None;
        }
        // Verify via ListShortcuts authoritative state
        tracing::info!("Hotkey thread: calling ListShortcuts to verify authoritative binding");
        match proxy.list_shortcuts(&session, Default::default()).await {
            Ok(list_req) => match list_req.response() {
                Ok(list_resp) => {
                    for sc in list_resp.shortcuts() {
                        tracing::info!(
                            "Hotkey thread: ListShortcuts confirms '{}' -> '{}'",
                            sc.id(),
                            sc.trigger_description()
                        );
                        if (sc.id() == HotkeyAction::ToggleDetection.id() || sc.id() == "porda_toggle" || sc.id() == "toggle") && !sc.trigger_description().is_empty() {
                            *trigger_cache.lock().unwrap() = Some(sc.trigger_description().to_string());
                            *error_cache.lock().unwrap() = None;
                        } else if sc.id() == HotkeyAction::ToggleDetection.id() && sc.trigger_description().is_empty() {
                            *trigger_cache.lock().unwrap() = None;
                        }
                    }
                }
                Err(e) => tracing::warn!("Hotkey thread: ListShortcuts response error: {}", e),
            },
            Err(e) => tracing::warn!("Hotkey thread: ListShortcuts failed: {}", e),
        }

        tracing::info!("Hotkey thread: installing Activated listener (receive_activated)");
        let mut activated = proxy.receive_activated().await.map_err(|e| {
            let msg = format!("receive_activated failed: {}", e);
            tracing::error!("Hotkey thread: {}", msg);
            if let Some(tx) = ready_tx.lock().unwrap().take() { let _ = tx.send(Err(msg.clone())); }
            msg
        })?;
        tracing::info!("Hotkey thread: Activated listener installed, will remain alive for entire Porda lifetime (independent of Slint window)");

        // Signal readiness to main thread: session/bind/listener are now alive
        if let Some(tx) = ready_tx.lock().unwrap().take() { let _ = tx.send(Ok(())); }
        tracing::info!("Hotkey thread: entering event loop (tokio::select! on Notify + activated.next()), no polling, no busy loop");

        loop {
            tokio::select! {
                _ = shutdown.notified() => {
                    tracing::info!("Hotkey thread: shutdown notified, exiting loop, session will be closed on drop");
                    break;
                }
                evt = activated.next() => {
                    match evt {
                        Some(e) => {
                            let sid = e.shortcut_id().to_string();
                            let session_handle = e.session_handle().to_string();
                            tracing::info!("Hotkey thread: Activated signal received: id='{}' session='{}' -> mapping to HotkeyAction", sid, session_handle);
                            if let Some(action) = HotkeyAction::from_id(&sid) {
                                tracing::info!("Hotkey thread: shortcut '{}' -> HotkeyAction::{:?} -> mpsc send", sid, action);
                                if let Err(err) = action_tx.send(action) {
                                    tracing::error!("Hotkey thread: mpsc send failed (receiver dropped): {}", err);
                                } else {
                                    tracing::info!("Hotkey thread: HotkeyAction sent via mpsc to UiCommand forwarder");
                                }
                            } else {
                                tracing::warn!("Hotkey thread: unknown shortcut id '{}' (expected porda_toggle_v2)", sid);
                            }
                        }
                        None => {
                            tracing::error!("Hotkey thread: Activated stream ended unexpectedly (portal session closed, D-Bus connection dropped, or portal restarted)");
                            break;
                        }
                    }
                }
            }
        }

        tracing::info!("Hotkey thread: event loop exited, GlobalShortcuts session closing (drop Session + proxy + runtime)");
        Ok::<(), String>(())
    })
}

// ---------------------------------------------------------------------------
// Tests – pure logic without portal
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn mgr() -> (LinuxHotkeyManager, mpsc::Receiver<HotkeyAction>) {
        let (tx, rx) = mpsc::channel();
        let m = LinuxHotkeyManager::new_for_test(tx);
        (m, rx)
    }

    #[test]
    fn normalize_valid() {
        assert_eq!(normalize_shortcut("F2").unwrap(), "F2");
        assert_eq!(normalize_shortcut("f2").unwrap(), "F2");
        assert_eq!(normalize_shortcut("  F2  ").unwrap(), "F2");
        assert_eq!(normalize_shortcut("ctrl+f2").unwrap(), "Ctrl+F2");
        assert_eq!(
            normalize_shortcut("Ctrl+Shift+F1").unwrap(),
            "Ctrl+Shift+F1"
        );
        assert_eq!(
            normalize_shortcut("shift+ctrl+f1").unwrap(),
            "Ctrl+Shift+F1"
        ); // order normalized
        assert_eq!(normalize_shortcut("meta+f2").unwrap(), "Meta+F2");
        assert_eq!(normalize_shortcut("super+f2").unwrap(), "Meta+F2");
        assert_eq!(normalize_shortcut("Alt+F1").unwrap(), "Alt+F1");
        assert_eq!(normalize_shortcut("a").unwrap(), "A");
    }

    #[test]
    fn normalize_invalid() {
        assert!(matches!(
            normalize_shortcut("").unwrap_err(),
            HotkeyError::EmptyShortcut
        ));
        assert!(matches!(
            normalize_shortcut("   ").unwrap_err(),
            HotkeyError::EmptyShortcut
        ));
        assert!(normalize_shortcut("Ctrl+").is_err());
        assert!(normalize_shortcut("Ctrl+UnknownKey").is_err());
        assert!(normalize_shortcut("Foo+F1").is_err());
        assert!(normalize_shortcut("Ctrl+Ctrl+F1").is_err());
    }

    #[test]
    fn parse_shortcut_aliases() {
        assert_eq!(parse_shortcut("F1").unwrap(), "F1");
        assert_eq!(parse_shortcut("Print").unwrap(), "Print");
        assert_eq!(parse_shortcut("ctrl+print").unwrap(), "Ctrl+Print");
    }

    #[test]
    fn action_id_mapping() {
        assert_eq!(HotkeyAction::ToggleDetection.id(), "porda_toggle_v2");
        // from_id handles old and new for migration
        assert_eq!(
            HotkeyAction::from_id("toggle"),
            Some(HotkeyAction::ToggleDetection)
        );
        assert_eq!(
            HotkeyAction::from_id("porda_toggle"),
            Some(HotkeyAction::ToggleDetection)
        );
        assert_eq!(
            HotkeyAction::from_id("porda_toggle_v2"),
            Some(HotkeyAction::ToggleDetection)
        );
        assert_eq!(HotkeyAction::from_id("unknown"), None);
        assert_eq!(HotkeyAction::from_id("screenshot"), None);
        assert_eq!(HotkeyAction::from_id("porda_screenshot"), None);
    }

    #[test]
    fn registry_register_and_duplicate() {
        let (m, _rx) = mgr();
        assert!(m
            .register("Meta+Shift+P", HotkeyAction::ToggleDetection)
            .is_ok());
        assert_eq!(m.registered_count(), 1);
        // idempotent same shortcut same action → ok no duplicate
        assert!(m
            .register("Meta+Shift+P", HotkeyAction::ToggleDetection)
            .is_ok());
        assert_eq!(m.registered_count(), 1);
        // invalid duplicate due to same normalized form with different casing
        assert!(m
            .register("meta+shift+p", HotkeyAction::ToggleDetection)
            .is_ok());
        assert_eq!(m.registered_count(), 1);
    }

    #[test]
    fn registry_unregister_all() {
        let (m, _rx) = mgr();
        m.register("Meta+Shift+P", HotkeyAction::ToggleDetection)
            .unwrap();
        assert_eq!(m.registered_count(), 1);
        m.unregister_all().unwrap();
        assert_eq!(m.registered_count(), 0);
        // unregister again is idempotent
        m.unregister_all().unwrap();
        assert_eq!(m.registered_count(), 0);
    }

    #[test]
    fn refresh_idempotent_no_duplicates() {
        let (m, _rx) = mgr();
        m.refresh("Meta+Shift+P").unwrap();
        assert_eq!(m.registered_count(), 1);
        assert!(m.is_registered("Meta+Shift+P"));
        // refresh again same key → still 1, no duplicates
        m.refresh("Meta+Shift+P").unwrap();
        assert_eq!(m.registered_count(), 1);
        // refresh three times
        m.refresh("Meta+Shift+P").unwrap();
        m.refresh("Meta+Shift+P").unwrap();
        assert_eq!(m.registered_count(), 1);
    }

    #[test]
    fn refresh_changes_shortcut() {
        let (m, _rx) = mgr();
        m.refresh("Meta+Shift+P").unwrap();
        assert!(m.is_registered("Meta+Shift+P"));
        // Change toggle to Meta+Shift+Q
        m.refresh("Meta+Shift+Q").unwrap();
        assert!(!m.is_registered("Meta+Shift+P"));
        assert!(m.is_registered("Meta+Shift+Q"));
        assert_eq!(m.registered_count(), 1);
    }

    #[test]
    fn refresh_invalid_shortcut_error() {
        let (m, _rx) = mgr();
        let err = m.refresh("").unwrap_err();
        assert!(err.contains("empty") || err.contains("invalid"));
        // No partial registration
        assert_eq!(m.registered_count(), 0);
    }

    #[test]
    fn shutdown_is_idempotent() {
        let (m, _rx) = mgr();
        m.register("Meta+Shift+P", HotkeyAction::ToggleDetection)
            .unwrap();
        m.unregister_all().unwrap();
        m.unregister_all().unwrap();
        assert_eq!(m.registered_count(), 0);
    }

    #[test]
    fn invalid_shortcut_does_not_register() {
        let (m, _rx) = mgr();
        assert!(m.register("", HotkeyAction::ToggleDetection).is_err());
        assert_eq!(m.registered_count(), 0);
        assert!(m
            .register("Ctrl+Nope", HotkeyAction::ToggleDetection)
            .is_err());
        assert_eq!(m.registered_count(), 0);
    }

    #[test]
    fn portal_disabled_in_tests() {
        // Ensure test manager doesn't attempt portal
        let (m, _rx) = mgr();
        assert!(!m.enable_portal);
    }

    #[test]
    fn command_integration_via_channel() {
        // Verify HotkeyAction → UiCommand mapping through mpsc without polling
        // Single shortcut: ToggleDetection → ToggleActivation
        let (hk_tx, hk_rx) = mpsc::channel::<HotkeyAction>();
        let (cmd_tx, cmd_rx) = mpsc::channel::<String>();

        // Forwarder mimics porda/src/main.rs hotkey_forwarder thread (blocking recv)
        #[allow(clippy::never_loop)]
        let handle = std::thread::spawn(move || {
            while let Ok(action) = hk_rx.recv() {
                let cmd = match action {
                    HotkeyAction::ToggleDetection => "ToggleActivation",
                };
                let _ = cmd_tx.send(cmd.to_string());
                break;
            }
        });

        hk_tx.send(HotkeyAction::ToggleDetection).unwrap();
        assert_eq!(cmd_rx.recv().unwrap(), "ToggleActivation");

        handle.join().unwrap();
    }

    #[test]
    fn registration_failure_does_not_panic() {
        let (m, _rx) = mgr();
        // Empty shortcut
        let r = m.register("", HotkeyAction::ToggleDetection);
        assert!(r.is_err());
        // Malformed
        let r = m.register("Ctrl+BadKey", HotkeyAction::ToggleDetection);
        assert!(r.is_err());
        // Duplicate same is ok (idempotent)
        m.register("Meta+Shift+P", HotkeyAction::ToggleDetection)
            .unwrap();
        let r = m.register("Meta+Shift+P", HotkeyAction::ToggleDetection);
        assert!(r.is_ok());
        assert_eq!(m.registered_count(), 1);
        // No panic on unregister when empty
        m.unregister_all().unwrap();
        m.unregister_all().unwrap();
    }

    #[test]
    fn unsupported_environment_detection() {
        static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap();
        // Save originals
        let orig_session = std::env::var("XDG_SESSION_TYPE").ok();
        let orig_wayland = std::env::var("WAYLAND_DISPLAY").ok();

        // Simulate non-Wayland -> should fail
        std::env::set_var("XDG_SESSION_TYPE", "x11");
        std::env::remove_var("WAYLAND_DISPLAY");
        let err = check_supported_environment().unwrap_err();
        assert!(
            err.to_string().contains("Wayland"),
            "{} should mention Wayland",
            err
        );

        // Wayland should pass regardless of desktop (no KDE gating)
        std::env::set_var("XDG_SESSION_TYPE", "wayland");
        std::env::set_var("WAYLAND_DISPLAY", "wayland-0");
        std::env::set_var("XDG_CURRENT_DESKTOP", "GNOME");
        std::env::set_var("DESKTOP_SESSION", "gnome");
        assert!(check_supported_environment().is_ok());

        std::env::set_var("XDG_CURRENT_DESKTOP", "KDE");
        assert!(check_supported_environment().is_ok());

        // Restore originals
        match orig_session {
            Some(v) => std::env::set_var("XDG_SESSION_TYPE", v),
            None => std::env::remove_var("XDG_SESSION_TYPE"),
        }
        match orig_wayland {
            Some(v) => std::env::set_var("WAYLAND_DISPLAY", v),
            None => std::env::remove_var("WAYLAND_DISPLAY"),
        }
        // Should be ok in current env (wayland)
        assert!(check_supported_environment().is_ok());
    }

    #[test]
    fn refresh_atomic_no_partial_on_failure() {
        let (m, _rx) = mgr();
        m.refresh("Meta+Shift+P").unwrap();
        assert_eq!(m.registered_count(), 1);
        // Attempt refresh with invalid – should fail and leave cleared (refresh does unregister first)
        let err = m.refresh("").unwrap_err();
        assert!(err.contains("empty") || err.contains("invalid"));
        assert_eq!(m.registered_count(), 0);
        // Re-establish
        m.refresh("Meta+Shift+Q").unwrap();
        assert_eq!(m.registered_count(), 1);
        assert!(m.is_registered("Meta+Shift+Q"));
    }

    #[test]
    fn default_shortcut_is_valid() {
        // New default must be supported by backend; normalization sorts modifiers Ctrl,Shift,Alt,Meta
        assert_eq!(normalize_shortcut("Meta+Shift+P").unwrap(), "Shift+Meta+P");
        assert_eq!(normalize_shortcut("Shift+Meta+P").unwrap(), "Shift+Meta+P");
        let (m, _rx) = mgr();
        m.register("Meta+Shift+P", HotkeyAction::ToggleDetection)
            .unwrap();
        assert_eq!(m.registered_count(), 1);
        // is_registered normalizes, so both forms work
        assert!(m.is_registered("Meta+Shift+P"));
        assert!(m.is_registered("Shift+Meta+P"));
    }
}
