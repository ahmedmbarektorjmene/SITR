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
        {
            let mut state = self.state.lock().unwrap();
            state.is_active = true;
            state.detection_state = "Active".to_string();
        }
        let _ = self.command_tx.send(UiCommand::Activate);
    }

    pub fn deactivate(&self) {
        {
            let mut state = self.state.lock().unwrap();
            state.is_active = false;
            state.detection_state = "Sleep".to_string();
        }
        let _ = self.command_tx.send(UiCommand::Deactivate);
    }

    pub fn toggle_activation(&self) {
        let is_active = {
            let state = self.state.lock().unwrap();
            state.is_active
        };
        if is_active {
            self.deactivate();
        } else {
            self.activate();
        }
    }

    pub fn take_screenshot(&self) {
        let _ = self.command_tx.send(UiCommand::TakeScreenshot);
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
