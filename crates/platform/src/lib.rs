#![allow(dead_code)]
#![allow(clippy::derivable_impls)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::new_without_default)]
#![allow(clippy::redundant_pattern_matching)]
#![allow(clippy::let_and_return)]
pub mod hotkey;
pub mod process;
pub mod tray;

#[cfg(target_os = "linux")]
pub mod linux;

pub use hotkey::*;
pub use process::*;
pub use tray::*;
