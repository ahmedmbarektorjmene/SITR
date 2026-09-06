use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex, OnceLock,
};

use crate::commands::UiCommandHandler;
use crate::state::SharedUiState;
use porda_core::commands::UiCommand;

slint::include_modules!();

static APP_WEAK: OnceLock<Mutex<Option<slint::Weak<AppWindow>>>> = OnceLock::new();
static UI_THREAD: OnceLock<Mutex<Option<std::thread::Thread>>> = OnceLock::new();
static QUIT: AtomicBool = AtomicBool::new(false);
static PARKED: AtomicBool = AtomicBool::new(false);

fn app_weak_lock() -> &'static Mutex<Option<slint::Weak<AppWindow>>> {
    APP_WEAK.get_or_init(|| Mutex::new(None))
}

fn ui_thread_lock() -> &'static Mutex<Option<std::thread::Thread>> {
    UI_THREAD.get_or_init(|| Mutex::new(None))
}

/// Request the main window to be shown. Callable from any thread (e.g., tray handler).
/// Returns true if the request was dispatched.
pub fn request_show_window() -> bool {
    let has_weak = app_weak_lock().lock().unwrap().is_some();
    let has_thread = ui_thread_lock().lock().unwrap().is_some();
    tracing::info!(
        "request_show_window ENTRY APP_WEAK={} UI_THREAD={} QUIT={} PARKED={}",
        has_weak,
        has_thread,
        QUIT.load(Ordering::SeqCst),
        PARKED.load(Ordering::SeqCst)
    );
    // If parked after hide, unpark is required – invoke alone won't wake parked thread
    if PARKED.load(Ordering::SeqCst) {
        if let Some(th) = ui_thread_lock().lock().unwrap().clone() {
            tracing::info!("Tray Show -> parked, unparking UI thread");
            th.unpark();
            return true;
        }
    }
    if let Some(weak) = app_weak_lock().lock().unwrap().clone() {
        let weak_clone = weak.clone();
        match slint::invoke_from_event_loop(move || {
            if let Some(app) = weak_clone.upgrade() {
                tracing::info!("Tray Show -> showing main window via event loop");
                let _ = app.show();
                app.window().request_redraw();
            } else {
                tracing::warn!("Show: weak upgrade failed");
            }
        }) {
            Ok(()) => {
                tracing::info!("invoke ok, returning");
                return true;
            }
            Err(e) => {
                tracing::warn!("invoke_from_event_loop failed (will unpark): {:?}", e);
            }
        }
    } else {
        tracing::warn!("APP_WEAK not set");
    }
    // Fallback unpark if invoke failed
    if let Some(th) = ui_thread_lock().lock().unwrap().clone() {
        tracing::info!("Tray Show -> unparking UI thread (fallback)");
        th.unpark();
        return true;
    }
    tracing::warn!("Show requested but AppWindow/UI thread not yet initialized");
    false
}

pub fn request_quit() {
    QUIT.store(true, Ordering::SeqCst);
    let _ = slint::quit_event_loop();
    if let Some(th) = ui_thread_lock().lock().unwrap().clone() {
        th.unpark();
    }
}

/// Update hotkey display from any thread (portal-authoritative).
pub fn request_hotkey_update(display: String, status: String, available: bool, configuring: bool) {
    if let Some(weak) = app_weak_lock().lock().unwrap().clone() {
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                app.set_hotkey_display(display.into());
                app.set_hotkey_status(status.into());
                app.set_hotkey_available(available);
                app.set_hotkey_configuring(configuring);
            }
        });
    }
}

pub fn request_hotkey_configuring(configuring: bool) {
    if let Some(weak) = app_weak_lock().lock().unwrap().clone() {
        let _ = slint::invoke_from_event_loop(move || {
            if let Some(app) = weak.upgrade() {
                app.set_hotkey_configuring(configuring);
            }
        });
    }
}

pub struct PordaApp {
    ui_state: SharedUiState,
    command_tx: std::sync::mpsc::Sender<UiCommand>,
}

impl PordaApp {
    pub fn new(ui_state: SharedUiState, command_tx: std::sync::mpsc::Sender<UiCommand>) -> Self {
        Self {
            ui_state,
            command_tx,
        }
    }

    pub fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let app = AppWindow::new()?;

        let handler = UiCommandHandler::new(Arc::clone(&self.ui_state), self.command_tx.clone());

        {
            let state = self.ui_state.lock().unwrap();
            app.set_is_active(state.is_active);
            app.set_is_blur(state.is_blur);
            app.set_is_bg_color(state.is_bg_color);
            app.set_is_solid_color(state.is_solid_color);
            app.set_accuracy(state.accuracy as i32);
            app.set_network_width(state.network_width as i32);
            app.set_network_height(state.network_height as i32);
            app.set_active_timeout(state.active_timeout_ms as i32);
            app.set_sleep_timeout(state.sleep_timeout_ms as i32);
            app.set_keep_running_seconds(state.keep_running_seconds as i32);
            app.set_is_detect_male(state.is_detect_male);
            app.set_is_detect_female(state.is_detect_female);
            app.set_is_all_windows(state.is_all_windows);
            app.set_is_include_window(state.is_include_window);
            app.set_is_exclude_window(state.is_exclude_window);
            app.set_auto_startup(state.auto_startup);
            app.set_is_priority_realtime(state.is_priority_realtime);
            app.set_is_allow_max_cpu_limit(state.is_allow_max_cpu_limit);
            app.set_max_cpu_limit(state.max_cpu_limit as i32);
            app.set_cpu_usage(state.cpu_usage);
            app.set_current_page(state.current_page as i32);
            app.set_toggle_key(state.toggle_key.clone().into());
            app.set_hotkey_display(state.hotkey_display.clone().into());
            app.set_hotkey_status(state.hotkey_status.clone().into());
            app.set_hotkey_available(state.hotkey_available);
            app.set_hotkey_configuring(state.hotkey_configuring);
            app.set_is_linux(cfg!(target_os = "linux"));
        }

        let weak = app.as_weak();

        {
            let h = handler.clone();
            app.on_activate(move || {
                h.activate();
            });
        }

        {
            let h = handler.clone();
            app.on_deactivate(move || {
                h.deactivate();
            });
        }

        {
            let h = handler.clone();
            app.on_toggle_activation(move || {
                h.toggle_activation();
            });
        }

        {
            let h = handler.clone();
            let ui_state = Arc::clone(&self.ui_state);
            let weak = weak.clone();
            app.on_save(move || {
                if let Some(app) = weak.upgrade() {
                    sync_ui_to_state(&ui_state, &app);
                }
                h.save_settings();
            });
        }

        {
            let h = handler.clone();
            let ui_state = Arc::clone(&self.ui_state);
            let weak = weak.clone();
            app.on_ok_and_close(move || {
                if let Some(app) = weak.upgrade() {
                    sync_ui_to_state(&ui_state, &app);
                    h.save_settings();
                    tracing::info!("OK -> hiding window (tray remains, app continues)");
                    let _ = app.hide();
                } else {
                    h.save_settings();
                }
                // Do NOT quit_event_loop — window hide keeps tray alive
            });
        }

        {
            let h = handler.clone();
            let ui_state = Arc::clone(&self.ui_state);
            let weak = weak.clone();
            app.on_apply(move || {
                if let Some(app) = weak.upgrade() {
                    sync_ui_to_state(&ui_state, &app);
                }
                h.apply_settings();
            });
        }

        {
            let h = handler.clone();
            let ui_state = Arc::clone(&self.ui_state);
            let weak = weak.clone();
            app.on_restore_defaults(move || {
                h.restore_defaults();
                if let Some(app) = weak.upgrade() {
                    let state = ui_state.lock().unwrap();
                    app.set_accuracy(state.accuracy as i32);
                    app.set_network_width(state.network_width as i32);
                    app.set_network_height(state.network_height as i32);
                    app.set_active_timeout(state.active_timeout_ms as i32);
                    app.set_sleep_timeout(state.sleep_timeout_ms as i32);
                    app.set_keep_running_seconds(state.keep_running_seconds as i32);
                    app.set_is_detect_male(state.is_detect_male);
                    app.set_is_detect_female(state.is_detect_female);
                    app.set_is_blur(state.is_blur);
                    app.set_is_bg_color(state.is_bg_color);
                    app.set_is_solid_color(state.is_solid_color);
                    app.set_auto_startup(state.auto_startup);
                    app.set_is_priority_realtime(state.is_priority_realtime);
                    app.set_is_allow_max_cpu_limit(state.is_allow_max_cpu_limit);
                    app.set_max_cpu_limit(state.max_cpu_limit as i32);
                    app.set_hotkey_display(state.hotkey_display.clone().into());
                    app.set_hotkey_status(state.hotkey_status.clone().into());
                    app.set_hotkey_available(state.hotkey_available);
                    app.set_hotkey_configuring(state.hotkey_configuring);
                }
            });
        }

        {
            let h = handler.clone();
            app.on_take_screenshot(move || {
                h.take_screenshot();
            });
        }

        {
            let h = handler.clone();
            app.on_refresh_hotkeys(move || {
                h.refresh_hotkeys();
            });
        }

        {
            let h = handler.clone();
            app.on_configure_hotkey(move || {
                h.configure_global_shortcut();
            });
        }

        {
            let h = handler.clone();
            app.on_terminate(move || {
                tracing::info!("Terminate button -> quitting event loop");
                QUIT.store(true, Ordering::SeqCst);
                h.terminate();
                slint::quit_event_loop().ok();
                // Unpark if parked waiting for Show
                if let Some(th) = ui_thread_lock().lock().unwrap().clone() {
                    th.unpark();
                }
            });
        }

        {
            let ui_state = Arc::clone(&self.ui_state);
            app.on_navigate(move |page| {
                let mut state = ui_state.lock().unwrap();
                state.current_page = page as usize;
            });
        }

        // Critical lifecycle: window close must NOT terminate app/tray.
        // Hide window instead of quitting event loop.
        // Use KeepWindowShown + manual hide to prevent winit from exiting event loop
        // when last window is hidden (HideWindow would destroy winit window and may exit loop).
        let weak_for_close = weak.clone();
        app.window().on_close_requested(move || {
            tracing::info!(
                "window close requested -> hiding window, tray remains alive, pipeline continues"
            );
            if let Some(a) = weak_for_close.upgrade() {
                let _ = a.hide();
            }
            slint::CloseRequestResponse::KeepWindowShown
        });

        // Store weak/thread for Tray Show to re-show window from any thread.
        // Resettable storage allows safe reinitialization if UI runtime is ever restarted
        // (e.g., in tests). The application currently guarantees exactly one UI lifetime
        // started from porda/src/main.rs, so this is a defensive measure.
        *app_weak_lock().lock().unwrap() = Some(weak.clone());
        *ui_thread_lock().lock().unwrap() = Some(std::thread::current());
        // Ensure QUIT is false at start (in case of previous run in same process for tests)
        QUIT.store(false, Ordering::SeqCst);

        // Keep event loop alive while tray-resident with no visible windows.
        // Verified: Slint's winit backend returns from `run_event_loop()` when the
        // last window is hidden (no active windows) – observed as
        // `window hidden, parking` ~8ms after `a.hide()`. The repeating timer
        // keeps the winit `ControlFlow` as `Wait` with a future deadline instead
        // of `Exit`, ensuring the loop stays alive for `Tray → Show` via
        // `invoke_from_event_loop`/`unpark` while the app remains in the tray.
        // Without this, the loop would exit on hide and require manual park.
        let _keep_alive = {
            let t = slint::Timer::default();
            t.start(
                slint::TimerMode::Repeated,
                std::time::Duration::from_secs(3600),
                || {},
            );
            t
        };

        // Main show/hide loop: window close hides, not quits. Keep thread alive for tray Show.
        loop {
            app.show()?;
            tracing::info!("UI window shown, event loop running (close will hide, not quit)");
            slint::run_event_loop()?;
            if QUIT.load(Ordering::SeqCst) {
                tracing::info!("UI event loop exited via Terminate/Exit");
                break;
            }
            tracing::info!("window hidden, parking UI thread, waiting for Tray Show (tray/pipeline remain alive)");
            PARKED.store(true, Ordering::SeqCst);
            std::thread::park();
            PARKED.store(false, Ordering::SeqCst);
            tracing::info!("UI thread unparked, will re-show window");
            if QUIT.load(Ordering::SeqCst) {
                break;
            }
        }

        Ok(())
    }
}

fn sync_ui_to_state(state: &SharedUiState, app: &AppWindow) {
    let mut s = state.lock().unwrap();
    s.is_blur = app.get_is_blur();
    s.is_bg_color = app.get_is_bg_color();
    s.is_solid_color = app.get_is_solid_color();
    s.accuracy = app.get_accuracy() as u8;
    s.network_width = app.get_network_width() as u32;
    s.network_height = app.get_network_height() as u32;
    s.active_timeout_ms = app.get_active_timeout() as u64;
    s.sleep_timeout_ms = app.get_sleep_timeout() as u64;
    s.keep_running_seconds = app.get_keep_running_seconds() as u64;
    s.is_detect_male = app.get_is_detect_male();
    s.is_detect_female = app.get_is_detect_female();
    s.is_all_windows = app.get_is_all_windows();
    s.is_include_window = app.get_is_include_window();
    s.is_exclude_window = app.get_is_exclude_window();
    s.auto_startup = app.get_auto_startup();
    s.is_priority_realtime = app.get_is_priority_realtime();
    s.is_allow_max_cpu_limit = app.get_is_allow_max_cpu_limit();
    s.max_cpu_limit = app.get_max_cpu_limit() as u8;
    s.toggle_key = app.get_toggle_key().to_string();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    fn reset_state() {
        *app_weak_lock().lock().unwrap() = None;
        *ui_thread_lock().lock().unwrap() = None;
        QUIT.store(false, Ordering::SeqCst);
        PARKED.store(false, Ordering::SeqCst);
    }

    #[test]
    fn parked_quit_initial_state() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset_state();
        assert!(!QUIT.load(Ordering::SeqCst));
        assert!(!PARKED.load(Ordering::SeqCst));
        QUIT.store(true, Ordering::SeqCst);
        assert!(QUIT.load(Ordering::SeqCst));
        PARKED.store(true, Ordering::SeqCst);
        assert!(PARKED.load(Ordering::SeqCst));
        reset_state();
        assert!(!QUIT.load(Ordering::SeqCst));
        assert!(!PARKED.load(Ordering::SeqCst));
    }

    #[test]
    fn app_weak_ui_thread_resettable() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset_state();
        assert!(app_weak_lock().lock().unwrap().is_none());
        assert!(ui_thread_lock().lock().unwrap().is_none());
        // Set UI thread to current thread and verify it can be cleared and set again
        *ui_thread_lock().lock().unwrap() = Some(std::thread::current());
        assert!(ui_thread_lock().lock().unwrap().is_some());
        *ui_thread_lock().lock().unwrap() = None;
        assert!(ui_thread_lock().lock().unwrap().is_none());
        *ui_thread_lock().lock().unwrap() = Some(std::thread::current());
        assert!(ui_thread_lock().lock().unwrap().is_some());
        // APP_WEAK resettable to None and back
        *app_weak_lock().lock().unwrap() = None;
        assert!(app_weak_lock().lock().unwrap().is_none());
        reset_state();
        assert!(app_weak_lock().lock().unwrap().is_none());
        assert!(ui_thread_lock().lock().unwrap().is_none());
    }

    #[test]
    fn request_show_window_handles_parked_state() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset_state();
        // Simulate UI thread parked after hide
        *ui_thread_lock().lock().unwrap() = Some(std::thread::current());
        PARKED.store(true, Ordering::SeqCst);
        // Spawn a thread that will request Show – should unpark this thread
        let handle = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            // This should see PARKED=true and unpark the waiting thread
            let ok = request_show_window();
            assert!(ok);
        });
        // Park with timeout – should be woken by request_show_window, not timeout
        std::thread::park_timeout(Duration::from_millis(500));
        // If we were woken by unpark, PARKED should still be true until UI loop clears it,
        // but request_show_window should have returned true
        handle.join().expect("request thread panicked");
        // Cleanup
        PARKED.store(false, Ordering::SeqCst);
        reset_state();
    }

    #[test]
    fn request_show_window_fallback_when_no_weak() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset_state();
        *ui_thread_lock().lock().unwrap() = Some(std::thread::current());
        PARKED.store(false, Ordering::SeqCst);
        // No APP_WEAK set, but UI_THREAD is set – invoke will fail (no weak) then fallback unpark
        // We test that it still returns true via fallback path without needing real Slint window.
        // To avoid actually parking, we test that it returns true.
        let ok = request_show_window();
        // Should return true via fallback unpark path (since APP_WEAK is None, it goes to fallback)
        assert!(ok);
        reset_state();
    }

    #[test]
    fn request_quit_sets_quit_and_unparks() {
        let _guard = TEST_MUTEX.lock().unwrap();
        reset_state();
        assert!(!QUIT.load(Ordering::SeqCst));
        // Set up a parked thread to verify unpark
        *ui_thread_lock().lock().unwrap() = Some(std::thread::current());
        PARKED.store(true, Ordering::SeqCst);
        let handle = std::thread::spawn(|| {
            std::thread::sleep(Duration::from_millis(50));
            request_quit();
            assert!(QUIT.load(Ordering::SeqCst));
        });
        std::thread::park_timeout(Duration::from_millis(500));
        handle.join().unwrap();
        assert!(QUIT.load(Ordering::SeqCst));
        reset_state();
    }
}
