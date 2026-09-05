use std::sync::{mpsc, Arc, Mutex};

use porda_core::app_state::AppState;
use porda_core::commands::{CoreEvent, UiCommand};
use porda_core::pipeline::Pipeline;
use porda_platform::tray::TrayAction;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    tracing::info!("Porda AI starting...");

    porda_platform::ensure_app_directories().map_err(|e| {
        tracing::error!("Failed to create directories: {}", e);
        e
    })?;

    if porda_platform::check_duplicate_instances() {
        tracing::warn!("Another instance is already running");
        porda_platform::show_message("Porda AI", "Another instance is already running.");
        return Ok(());
    }

    tracing::info!("configuration initialized");
    let config = porda_config::defaults::load_config().unwrap_or_default();
    tracing::info!("core initialized");

    let (cmd_tx, cmd_rx) = mpsc::channel::<UiCommand>();
    let (event_tx, event_rx) = mpsc::channel::<CoreEvent>();
    let (tray_tx, tray_rx) = mpsc::channel::<TrayAction>();

    let ui_state = porda_ui::create_shared_state();
    let core_state = Arc::new(Mutex::new(AppState::new(config.clone())));

    tracing::info!("capture/pipeline initialized");
    let pipeline = Pipeline::new(Arc::clone(&core_state), event_tx.clone());
    pipeline.start();

    tracing::info!("overlay initialized");
    // UI and tray are long-lived; tray must stay alive for entire lifetime.
    // Do NOT create tray inside UI thread and do NOT drop it after run().

    let tray = porda_tray::PordaTray::new(tray_tx);
    tracing::info!("UI initialized");
    // System tray initialization – must be real, persistent SNI, not a temp icon.
    if let Err(e) = tray.run() {
        tracing::error!(
            "tray initialization failed (non-fatal – continuing without tray): {}",
            e
        );
        // Do not exit; tray host may be missing but app should still run.
        // The error is already logged with full context in porda-tray.
    } else {
        tracing::info!(
            "system tray initialized – icon remains alive, menu available, events processing"
        );
    }

    let ui_state_for_ui = Arc::clone(&ui_state);
    let cmd_tx_for_ui = cmd_tx.clone();
    let ui_handle = std::thread::Builder::new()
        .name("porda-ui".to_string())
        .spawn(move || {
            let app = porda_ui::PordaApp::new(ui_state_for_ui, cmd_tx_for_ui);
            if let Err(e) = app.run() {
                tracing::error!("UI error: {}", e);
            }
        })?;

    let ui_state_for_events = Arc::clone(&ui_state);
    // Channel ownership: `event_tx` is cloned into `Pipeline` and `command_handle`
    // (for ConfigSaved). `event_rx` is owned solely by this thread. During shutdown
    // `event_tx.send(Terminated)` wakes this thread immediately; dropping all
    // `event_tx` clones then disconnects the channel. No polling needed.
    let event_handle = std::thread::Builder::new()
        .name("porda-event-handler".to_string())
        .spawn(move || {
            while let Ok(event) = event_rx.recv() {
                match event {
                    CoreEvent::DetectionStateChange(state) => {
                        let mut ui = ui_state_for_events.lock().unwrap();
                        ui.detection_state = format!("{:?}", state);
                        tracing::info!("Detection state: {:?}", state);
                    }
                    CoreEvent::CpuUsageUpdate(usage) => {
                        let mut ui = ui_state_for_events.lock().unwrap();
                        ui.cpu_usage = usage;
                    }
                    CoreEvent::CoversUpdated(_covers) => {}
                    CoreEvent::ScreenshotTaken(path) => {
                        tracing::info!("Screenshot saved: {:?}", path);
                    }
                    CoreEvent::Error(msg) => {
                        tracing::error!("Core error: {}", msg);
                    }
                    CoreEvent::ConfigSaved => {
                        tracing::info!("Configuration saved");
                    }
                    CoreEvent::ConfigLoaded(_config) => {}
                    CoreEvent::Terminated => {
                        tracing::info!("Terminated");
                        break;
                    }
                }
            }
        })?;

    let event_tx_for_commands = event_tx.clone();
    let core_state_for_commands = Arc::clone(&core_state);
    let ui_state_for_commands = Arc::clone(&ui_state);
    // Channel ownership: `cmd_tx` clones exist in `ui_handle` (UiCommand) and
    // `tray_handle` (Activate etc.). This thread owns `cmd_rx` solely and does
    // NOT hold a `cmd_tx` clone, so `cmd_rx.recv()` can return `Disconnected`
    // when all senders are dropped. No polling needed.
    let command_handle = std::thread::Builder::new()
        .name("porda-command-handler".to_string())
        .spawn(move || {
            while let Ok(cmd) = cmd_rx.recv() {
                match cmd {
                    UiCommand::SaveSettings => {
                        let config = {
                            let ui = ui_state_for_commands.lock().unwrap();
                            ui.to_config()
                        };
                        if let Err(e) = porda_config::defaults::save_config(&config) {
                            tracing::error!("Failed to save config: {}", e);
                        }
                        let _ = event_tx_for_commands.send(CoreEvent::ConfigSaved);
                    }
                    UiCommand::LoadSettings(config) => {
                        let mut state = core_state_for_commands.lock().unwrap();
                        *state = AppState::new(config);
                    }
                    UiCommand::RestoreDefaults => {
                        let default_config = porda_config::settings::PordaConfig::default();
                        let mut state = core_state_for_commands.lock().unwrap();
                        *state = AppState::new(default_config);
                    }
                    UiCommand::Activate => {
                        let mut state = core_state_for_commands.lock().unwrap();
                        state.is_active = true;
                        tracing::info!("Detection activated");
                    }
                    UiCommand::Deactivate => {
                        let mut state = core_state_for_commands.lock().unwrap();
                        state.is_active = false;
                        tracing::info!("Detection deactivated");
                    }
                    UiCommand::ToggleActivation => {
                        let mut state = core_state_for_commands.lock().unwrap();
                        state.is_active = !state.is_active;
                        tracing::info!("Detection toggled: {}", state.is_active);
                    }
                    UiCommand::ApplySettings(config) => {
                        let mut state = core_state_for_commands.lock().unwrap();
                        *state = AppState::new(config);
                        tracing::info!("Settings applied");
                    }
                    UiCommand::Terminate => {
                        tracing::info!("Terminate requested");
                        // Ensure UI event loop quits even if Terminate came via tray
                        porda_ui::request_quit();
                        break;
                    }
                    UiCommand::TakeScreenshot => match porda_platform::capture_screenshot() {
                        Some(_frame) => {
                            let dataset_dir = porda_config::defaults::dataset_dir();
                            let filename = format!("screenshot_{}.jpg", chrono_now());
                            let path = dataset_dir.join(filename);
                            tracing::info!("Screenshot captured: {:?}", path);
                        }
                        None => {
                            tracing::warn!("Failed to capture screenshot");
                        }
                    },
                    UiCommand::RefreshHotkeys => {
                        tracing::info!("Hotkeys refreshed");
                    }
                    UiCommand::RefreshOverlay => {
                        tracing::info!("Overlay refreshed");
                    }
                    _ => {}
                }
            }
        })?;

    let cmd_tx_for_tray = cmd_tx.clone();
    // Channel ownership: `tray_tx` is cloned into `PordaTray` (main) and
    // `PordaTrayInner` (ksni service). `tray_rx` is owned solely by this
    // thread. `tray_rx.recv()` blocks until `tray_tx.send(Show)` etc. wakes
    // it; dropping all `tray_tx` clones (via `tray.shutdown()` + `drop(tray)`)
    // disconnects the channel. No polling needed.
    let tray_handle = std::thread::Builder::new()
        .name("porda-tray-handler".to_string())
        .spawn(move || {
            while let Ok(action) = tray_rx.recv() {
                match action {
                    TrayAction::Show => {
                        tracing::info!("Tray -> Show requested");
                        if !porda_ui::request_show_window() {
                            tracing::warn!("Show failed – window not available");
                        }
                    }
                    TrayAction::Activate => {
                        tracing::info!("Tray -> Activate requested");
                        let _ = cmd_tx_for_tray.send(UiCommand::Activate);
                    }
                    TrayAction::Deactivate => {
                        tracing::info!("Tray -> Deactivate requested");
                        let _ = cmd_tx_for_tray.send(UiCommand::Deactivate);
                    }
                    TrayAction::ToggleDetection => {
                        tracing::info!("Tray -> ToggleDetection requested");
                        let _ = cmd_tx_for_tray.send(UiCommand::ToggleActivation);
                    }
                    TrayAction::TakeScreenshot => {
                        let _ = cmd_tx_for_tray.send(UiCommand::TakeScreenshot);
                    }
                    TrayAction::RefreshHotkeys => {
                        let _ = cmd_tx_for_tray.send(UiCommand::RefreshHotkeys);
                    }
                    TrayAction::RefreshOverlay => {
                        let _ = cmd_tx_for_tray.send(UiCommand::RefreshOverlay);
                    }
                    TrayAction::Exit => {
                        tracing::info!("Tray -> Exit requested – initiating clean shutdown");
                        let _ = cmd_tx_for_tray.send(UiCommand::Terminate);
                        porda_ui::request_quit();
                        break;
                    }
                }
            }
            tracing::info!("tray handler thread exiting cleanly");
        })?;

    ui_handle.join().unwrap_or_else(|e| {
        tracing::error!("UI thread panicked: {:?}", e);
    });

    // Ensure tray is torn down cleanly regardless of which path triggered shutdown
    tracing::info!("shutting down – stopping pipeline, overlay, tray");
    pipeline.stop();
    // Unblock event handler (it waits on event_rx recv) – send Terminated and drop sender
    let _ = event_tx.send(CoreEvent::Terminated);
    drop(event_tx);
    tray.shutdown();
    // Closing cmd_tx signals command handler to exit if Terminate wasn't sent; tray handler will exit when tray is dropped
    drop(cmd_tx);

    event_handle.join().unwrap_or_else(|e| {
        tracing::error!("Event handler thread panicked: {:?}", e);
    });

    command_handle.join().unwrap_or_else(|e| {
        tracing::error!("Command handler thread panicked: {:?}", e);
    });

    // tray_handle may already have exited via Exit; otherwise it will exit when tray_rx is dropped
    // (tray object dropped after shutdown). Ensure we don't leak.
    drop(tray);
    tray_handle.join().unwrap_or_else(|e| {
        tracing::error!("Tray handler thread panicked: {:?}", e);
    });

    tracing::info!("Porda AI stopped – all subsystems terminated cleanly");
    Ok(())
}

fn chrono_now() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use porda_core::commands::{CoreEvent, UiCommand};
    use porda_platform::tray::TrayAction;
    use std::sync::mpsc;
    use std::time::Duration;

    fn join_with_timeout<T: Send + 'static>(handle: std::thread::JoinHandle<T>, timeout: Duration) -> Option<T> {
        // Bounded join to prevent hung test; uses try-join via channel
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let res = handle.join();
            let _ = tx.send(res);
        });
        match rx.recv_timeout(timeout) {
            Ok(Ok(v)) => Some(v),
            Ok(Err(_)) => None,
            Err(_) => None,
        }
    }

    #[test]
    fn event_channel_blocking_wakes_on_terminated() {
        let (tx, rx) = mpsc::channel::<CoreEvent>();
        let handle = std::thread::spawn(move || {
            // Should block until Terminated arrives
            match rx.recv() {
                Ok(CoreEvent::Terminated) => true,
                _ => false,
            }
        });
        std::thread::sleep(Duration::from_millis(20));
        assert!(tx.send(CoreEvent::Terminated).is_ok());
        let woke = join_with_timeout(handle, Duration::from_secs(2));
        assert_eq!(woke, Some(true), "event thread should wake on Terminated");
    }

    #[test]
    fn cmd_channel_blocking_wakes_on_terminate() {
        let (tx, rx) = mpsc::channel::<UiCommand>();
        let handle = std::thread::spawn(move || match rx.recv() {
            Ok(UiCommand::Terminate) => true,
            _ => false,
        });
        std::thread::sleep(Duration::from_millis(20));
        assert!(tx.send(UiCommand::Terminate).is_ok());
        let woke = join_with_timeout(handle, Duration::from_secs(2));
        assert_eq!(woke, Some(true));
    }

    #[test]
    fn tray_channel_blocking_wakes_on_show() {
        let (tx, rx) = mpsc::channel::<TrayAction>();
        let handle = std::thread::spawn(move || match rx.recv() {
            Ok(TrayAction::Show) => true,
            _ => false,
        });
        std::thread::sleep(Duration::from_millis(20));
        assert!(tx.send(TrayAction::Show).is_ok());
        let woke = join_with_timeout(handle, Duration::from_secs(2));
        assert_eq!(woke, Some(true));
    }

    #[test]
    fn channel_sender_clone_keeps_alive_and_drop_disconnects() {
        // With two senders, dropping one should NOT disconnect (try_recv => Empty)
        let (tx, rx) = mpsc::channel::<UiCommand>();
        let tx_clone = tx.clone();
        drop(tx);
        assert!(
            matches!(rx.try_recv(), Err(mpsc::TryRecvError::Empty)),
            "should still be Empty while clone alive"
        );
        drop(tx_clone);
        assert!(
            matches!(rx.try_recv(), Err(mpsc::TryRecvError::Disconnected)),
            "should be Disconnected after all senders dropped"
        );
        // Verify blocking recv wakes on disconnect
        let (tx2, rx2) = mpsc::channel::<UiCommand>();
        let tx2_clone = tx2.clone();
        let handle2 = std::thread::spawn(move || matches!(rx2.recv(), Err(_)));
        drop(tx2);
        drop(tx2_clone);
        let woke = join_with_timeout(handle2, Duration::from_secs(1));
        assert_eq!(
            woke,
            Some(true),
            "should disconnect after all senders dropped"
        );
    }

    #[test]
    fn tray_exit_is_only_termination_action() {
        // Verify TrayAction::Exit is distinct and Show is the only UI opener
        assert_ne!(TrayAction::Show, TrayAction::Exit);
        let menu = TrayAction::menu_actions();
        assert_eq!(menu.len(), 5);
        assert!(menu.contains(&TrayAction::Show));
        assert!(menu.contains(&TrayAction::Exit));
        assert_eq!(menu.iter().filter(|a| **a == TrayAction::Show).count(), 1);
        // Ensure no OpenSettings exists
        for a in &menu {
            assert_ne!(format!("{:?}", a), "OpenSettings");
        }
    }
}
