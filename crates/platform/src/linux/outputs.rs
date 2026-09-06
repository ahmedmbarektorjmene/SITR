use vision::geometry::ScreenRect;

#[derive(Debug, Clone)]
pub struct OutputInfo {
    pub id: u32,
    pub name: String,
    pub geometry: ScreenRect,
    pub scale_factor: f64,
    pub transform: OutputTransform,
    pub focused: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputTransform {
    Normal,
    Rotated90,
    Rotated180,
    Rotated270,
    Flipped,
    FlippedRotated90,
    FlippedRotated180,
    FlippedRotated270,
}

impl Default for OutputTransform {
    fn default() -> Self {
        Self::Normal
    }
}

pub fn get_outputs() -> Vec<OutputInfo> {
    #[cfg(target_os = "linux")]
    {
        linux_impl::get_outputs_wayland()
    }
    #[cfg(not(target_os = "linux"))]
    {
        vec![]
    }
}

pub fn get_primary_output() -> Option<OutputInfo> {
    get_outputs()
        .into_iter()
        .find(|o| o.focused)
        .or_else(|| get_outputs().into_iter().next())
}

#[cfg(target_os = "linux")]
mod linux_impl {
    use super::*;

    pub fn get_outputs_wayland() -> Vec<OutputInfo> {
        // Use wlr-randr or KWin D-Bus to get outputs
        // Fallback to environment detection
        let mut outputs = Vec::new();

        // Try to get output info from environment
        if let Ok(output) = std::process::Command::new("wlr-randr")
            .arg("--json")
            .output()
        {
            if let Ok(json) = String::from_utf8(output.stdout) {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&json) {
                    if let Some(arr) = parsed.as_array() {
                        for (i, item) in arr.iter().enumerate() {
                            let name = item["name"].as_str().unwrap_or("unknown").to_string();
                            let width = item["width"].as_u64().unwrap_or(1920) as u32;
                            let height = item["height"].as_u64().unwrap_or(1080) as u32;
                            let scale = item["scale"].as_f64().unwrap_or(1.0);
                            let x = item["x"].as_i64().unwrap_or(0) as i32;
                            let y = item["y"].as_i64().unwrap_or(0) as i32;

                            outputs.push(OutputInfo {
                                id: i as u32,
                                name,
                                geometry: ScreenRect::new(x, y, width, height),
                                scale_factor: scale,
                                transform: OutputTransform::Normal,
                                focused: i == 0,
                            });
                        }
                    }
                }
            }
        }

        // Fallback: single output
        if outputs.is_empty() {
            outputs.push(OutputInfo {
                id: 0,
                name: "DP-1".to_string(),
                geometry: ScreenRect::new(0, 0, 1920, 1080),
                scale_factor: 1.0,
                transform: OutputTransform::Normal,
                focused: true,
            });
        }

        outputs
    }
}
