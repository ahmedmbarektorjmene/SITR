use crate::WindowHandle;

pub fn get_cpu_usage() -> f32 {
    // Read /proc/stat for CPU usage
    if let Ok(stat) = std::fs::read_to_string("/proc/stat") {
        if let Some(line) = stat.lines().next() {
            if line.starts_with("cpu ") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    let user: u64 = parts[1].parse().unwrap_or(0);
                    let nice: u64 = parts[2].parse().unwrap_or(0);
                    let system: u64 = parts[3].parse().unwrap_or(0);
                    let idle: u64 = parts[4].parse().unwrap_or(0);
                    let total = user + nice + system + idle;
                    let busy = user + nice + system;

                    // Store previous values for delta calculation
                    static mut PREV_TOTAL: u64 = 0;
                    static mut PREV_BUSY: u64 = 0;

                    unsafe {
                        let delta_total = total.saturating_sub(PREV_TOTAL);
                        let delta_busy = busy.saturating_sub(PREV_BUSY);
                        PREV_TOTAL = total;
                        PREV_BUSY = busy;

                        if delta_total > 0 {
                            return (delta_busy as f32 / delta_total as f32) * 100.0;
                        }
                    }
                }
            }
        }
    }

    0.0
}

pub fn get_process_cpu_usage() -> f32 {
    let pid = std::process::id();
    let stat_path = format!("/proc/{}/stat", pid);

    if let Ok(stat) = std::fs::read_to_string(&stat_path) {
        let parts: Vec<&str> = stat.split_whitespace().collect();
        if parts.len() >= 14 {
            let utime: u64 = parts[13].parse().unwrap_or(0);
            let stime: u64 = parts[14].parse().unwrap_or(0);
            let total_time = utime + stime;

            // Get system uptime for percentage calculation
            if let Ok(uptime) = std::fs::read_to_string("/proc/uptime") {
                if let Some(uptime_str) = uptime.split_whitespace().next() {
                    if let Ok(uptime_secs) = uptime_str.parse::<f64>() {
                        let ticks_per_second = 100; // USER_HZ
                        let total_secs = total_time as f64 / ticks_per_second as f64;
                        return (total_secs / uptime_secs * 100.0) as f32;
                    }
                }
            }
        }
    }

    0.0
}

pub fn get_memory_usage() -> (u64, u64) {
    if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
        let mut total = 0u64;
        let mut available = 0u64;

        for line in meminfo.lines() {
            if line.starts_with("MemTotal:") {
                total = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            } else if line.starts_with("MemAvailable:") {
                available = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            }
        }

        return (total, available);
    }

    (0, 0)
}

pub fn get_process_id() -> u32 {
    std::process::id()
}

pub fn check_duplicate_instances() -> bool {
    let pid = std::process::id();
    let our_exe = match std::env::current_exe() {
        Ok(p) => p,
        Err(_) => return false,
    };

    if let Ok(entries) = std::fs::read_dir("/proc") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if let Some(pid_str) = name.to_str() {
                if let Ok(proc_pid) = pid_str.parse::<u32>() {
                    if proc_pid != pid {
                        if let Ok(other_exe) = std::fs::read_link(entry.path().join("exe")) {
                            if other_exe == our_exe {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }

    false
}

pub fn show_message(title: &str, message: &str) {
    tracing::info!("Message: {} - {}", title, message);

    // Try to use KDE notification
    #[cfg(target_os = "linux")]
    {
        let _ = std::process::Command::new("kdialog")
            .args(["--passivepopup", message, "5"])
            .spawn();
    }
}

pub fn set_graphics_preference() {
    // Not applicable on Linux
}

pub fn set_process_realtime_priority() {
    // Try to set nice value using /proc/self
    let nice_path = "/proc/self/oom_score_adj";
    if let Ok(_) = std::fs::write(nice_path, "-10") {
        tracing::info!("Set OOM score adjustment to -10");
    }
}

pub fn capture_portal_screenshot() -> Option<vision::detection::FrameData> {
    // Use GNOME/KDE screenshot portal
    tracing::info!("Capturing screenshot via portal");

    // Try using spectacle (KDE screenshot tool)
    if let Ok(output) = std::process::Command::new("spectacle")
        .args([
            "--background",
            "--nonotify",
            "--output",
            "/tmp/porda_screenshot.png",
        ])
        .output()
    {
        if output.status.success() {
            // Load the screenshot
            if let Ok(img) = image::open("/tmp/porda_screenshot.png") {
                let rgb = img.to_rgb8();
                let (width, height) = rgb.dimensions();
                return Some(vision::detection::FrameData::new_rgb(
                    width,
                    height,
                    rgb.into_raw(),
                ));
            }
        }
    }

    // Fallback: use grim for Wayland
    if let Ok(output) = std::process::Command::new("grim")
        .args(["-", "/tmp/porda_screenshot.png"])
        .output()
    {
        if output.status.success() {
            if let Ok(img) = image::open("/tmp/porda_screenshot.png") {
                let rgb = img.to_rgb8();
                let (width, height) = rgb.dimensions();
                return Some(vision::detection::FrameData::new_rgb(
                    width,
                    height,
                    rgb.into_raw(),
                ));
            }
        }
    }

    None
}

pub fn get_monitors() -> Vec<vision::geometry::ScreenRect> {
    super::outputs::get_outputs()
        .into_iter()
        .map(|o| o.geometry)
        .collect()
}

pub fn list_windows(
    include: &[String],
    exclude: &[String],
    always_skip: &[(String, String)],
) -> Vec<(WindowHandle, String, String, vision::geometry::ScreenRect)> {
    super::windows::enumerate_windows()
        .into_iter()
        .filter(|w| !super::windows::should_skip_window(&w.app_id, include, exclude, always_skip))
        .map(|w| (WindowHandle(w.id as usize), w.app_id, w.title, w.geometry))
        .collect()
}
