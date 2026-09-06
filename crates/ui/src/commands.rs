use crate::state::SharedUiState;
use config::settings::PordaConfig;
use porda_core::commands::UiCommand;

#[derive(Clone)]
pub struct UiCommandHandler {
    state: SharedUiState,
    command_tx: std::sync::mpsc::Sender<UiCommand>,
}

impl UiCommandHandler {
    pub fn new(state: SharedUiState, command_tx: std::sync::mpsc::Sender<UiCommand>) -> Self {
        Self { state, command_tx }
    }

    pub fn save_settings(&self) {
        let config = {
            let state = self.state.lock().unwrap();
            state.to_config()
        };
        if let Err(e) = config::defaults::save_config(&config) {
            tracing::error!("Failed to save config: {}", e);
        }
        let _ = self.command_tx.send(UiCommand::SaveSettings);
    }

    pub fn restore_defaults(&self) {
        let default_config = PordaConfig::default();
        let mut state = self.state.lock().unwrap();
        *state = crate::state::UiState::from_config(&default_config);
        let _ = self.command_tx.send(UiCommand::RestoreDefaults);
    }

    pub fn apply_settings(&self) {
        let config = {
            let state = self.state.lock().unwrap();
            state.to_config()
        };
        let _ = self.command_tx.send(UiCommand::ApplySettings(config));
    }

    pub fn activate(&self) {
        // P1.3: Do NOT mutate UI state here – runtime (porda-core) is source of truth.
        // Send typed command through existing architecture; UI will be updated via request_active_update.
        let _ = self.command_tx.send(UiCommand::Activate);
    }

    pub fn deactivate(&self) {
        let _ = self.command_tx.send(UiCommand::Deactivate);
    }

    pub fn toggle_activation(&self) {
        // Single toggle path – core decides new state, UI derives from runtime feedback.
        let _ = self.command_tx.send(UiCommand::ToggleActivation);
    }

    pub fn refresh_hotkeys(&self) {
        let _ = self.command_tx.send(UiCommand::RefreshHotkeys);
    }

    pub fn configure_global_shortcut(&self) {
        let _ = self.command_tx.send(UiCommand::ConfigureGlobalShortcut);
    }

    pub fn terminate(&self) {
        let _ = self.command_tx.send(UiCommand::Terminate);
    }
}
