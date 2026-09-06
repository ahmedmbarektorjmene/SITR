#![allow(dead_code)]
#![allow(clippy::single_match)]
#![allow(clippy::identity_op)]
#![allow(clippy::items_after_test_module)]
pub mod compositor;
pub mod overlay_window;
#[cfg(target_os = "linux")]
pub mod wl_overlay;

pub use compositor::*;
pub use overlay_window::*;
#[cfg(target_os = "linux")]
pub use wl_overlay::{OverlayConfig, WaylandOverlay};
