use vision::detection::{CoverRect, FrameData};
use vision::geometry::{ColorRgb, ScreenRect};

#[derive(Debug, thiserror::Error)]
pub enum OverlayError {
    #[error("Overlay creation failed: {0}")]
    CreationFailed(String),
    #[error("Rendering failed: {0}")]
    RenderFailed(String),
    #[error("Unsupported: {0}")]
    Unsupported(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OverlayCapability {
    Supported,
    Unsupported(String),
}

impl OverlayCapability {
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported)
    }
}

pub trait OverlayRenderer: Send + Sync {
    fn capability(&self) -> OverlayCapability {
        OverlayCapability::Supported
    }
    fn initialize(&mut self, monitors: &[ScreenRect]) -> Result<(), OverlayError> {
        self.set_geometry(monitors)
    }
    fn update_covers(
        &mut self,
        covers: &[CoverRect],
        frame: &FrameData,
    ) -> Result<(), OverlayError>;
    fn clear(&mut self) -> Result<(), OverlayError>;
    fn set_geometry(&mut self, monitors: &[ScreenRect]) -> Result<(), OverlayError>;
    fn shutdown(&mut self) -> Result<(), OverlayError> {
        self.clear()
    }
}

pub struct CpuOverlayRenderer {
    covers: Vec<CoverRect>,
    #[allow(dead_code)]
    solid_color: ColorRgb,
}

impl CpuOverlayRenderer {
    pub fn new(solid_color: ColorRgb) -> Self {
        Self {
            covers: Vec::new(),
            solid_color,
        }
    }
}

impl OverlayRenderer for CpuOverlayRenderer {
    fn update_covers(
        &mut self,
        covers: &[CoverRect],
        _frame: &FrameData,
    ) -> Result<(), OverlayError> {
        self.covers = covers.to_vec();
        Ok(())
    }

    fn clear(&mut self) -> Result<(), OverlayError> {
        self.covers.clear();
        Ok(())
    }

    fn set_geometry(&mut self, _monitors: &[ScreenRect]) -> Result<(), OverlayError> {
        Ok(())
    }
}
