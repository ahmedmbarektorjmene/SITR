pub mod capabilities;
pub mod capture;
pub mod hotkeys;
pub mod outputs;
pub mod startup;
pub mod system;
pub mod windows;

pub use capabilities::*;
pub use capture::*;
pub use hotkeys::*;
pub use outputs::*;
pub use startup::*;
pub use system::*;
pub use windows::*;

use std::sync::OnceLock;
use vision::detection::FrameData;
use vision::geometry::ScreenRect;

static CAPTURER: OnceLock<capture::LinuxScreenCapturer> = OnceLock::new();

fn get_capturer() -> &'static capture::LinuxScreenCapturer {
    CAPTURER.get_or_init(|| {
        tracing::info!("Initializing global PipeWire screen capturer");
        capture::LinuxScreenCapturer::new()
    })
}

pub fn capture_screen_frame() -> Option<(FrameData, ScreenRect)> {
    let capturer = get_capturer();
    match capturer.capture() {
        Ok((frame, info)) => {
            let rect = ScreenRect {
                x: 0,
                y: 0,
                width: info.width,
                height: info.height,
            };
            Some((frame, rect))
        }
        Err(capture::LinuxCaptureError::NoFrames) => None,
        Err(capture::LinuxCaptureError::FrameTimeout) => None,
        Err(e) => {
            tracing::debug!("PipeWire capture error: {}", e);
            None
        }
    }
}
