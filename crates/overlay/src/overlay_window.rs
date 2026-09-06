use std::sync::mpsc;

use vision::detection::CoverRect;

#[derive(Debug, Clone)]
pub enum OverlayCommand {
    UpdateCovers(Vec<CoverRect>),
    Clear,
    Shutdown,
}

pub struct OverlayWindow {
    command_tx: mpsc::Sender<OverlayCommand>,
}

impl OverlayWindow {
    pub fn new(command_tx: mpsc::Sender<OverlayCommand>) -> Self {
        Self { command_tx }
    }

    pub fn send_command(&self, cmd: OverlayCommand) -> Result<(), String> {
        self.command_tx.send(cmd).map_err(|e| e.to_string())
    }

    pub fn update_covers(&self, covers: Vec<CoverRect>) -> Result<(), String> {
        self.send_command(OverlayCommand::UpdateCovers(covers))
    }

    pub fn clear(&self) -> Result<(), String> {
        self.send_command(OverlayCommand::Clear)
    }

    pub fn shutdown(&self) -> Result<(), String> {
        self.send_command(OverlayCommand::Shutdown)
    }
}
