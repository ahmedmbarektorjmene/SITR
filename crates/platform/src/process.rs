use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct ProcessInfo {
    pub name: String,
    pub pid: u32,
}

pub fn get_foreground_window() -> Option<WindowHandle> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::get_foreground_window()
    }
    #[cfg(target_os = "linux")]
    {
        crate::linux::get_foreground_window().map(|w| WindowHandle(w.id as usize))
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        None
    }
}

pub fn get_window_rect(_hwnd: WindowHandle) -> Option<vision::geometry::ScreenRect> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::get_window_rect(_hwnd)
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

pub fn get_client_rect(_hwnd: WindowHandle) -> Option<vision::geometry::ScreenRect> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::get_client_rect(_hwnd)
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

pub fn get_window_process_name(_hwnd: WindowHandle) -> Option<String> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::get_window_process_name(_hwnd)
    }
    #[cfg(target_os = "linux")]
    {
        crate::linux::enumerate_windows()
            .into_iter()
            .find(|w| w.id as usize == _hwnd.0)
            .map(|w| w.app_id)
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        None
    }
}

pub fn get_window_title(_hwnd: WindowHandle) -> Option<String> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::get_window_title(_hwnd)
    }
    #[cfg(target_os = "linux")]
    {
        crate::linux::enumerate_windows()
            .into_iter()
            .find(|w| w.id as usize == _hwnd.0)
            .map(|w| w.title)
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        None
    }
}

pub fn is_window_visible(_hwnd: WindowHandle) -> bool {
    #[cfg(target_os = "windows")]
    {
        windows_impl::is_window_visible(_hwnd)
    }
    #[cfg(not(target_os = "windows"))]
    {
        true
    }
}

pub fn is_window_minimized(_hwnd: WindowHandle) -> bool {
    #[cfg(target_os = "windows")]
    {
        windows_impl::is_window_minimized(_hwnd)
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

pub fn capture_window(_hwnd: WindowHandle) -> Option<vision::detection::FrameData> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::capture_window(_hwnd)
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

pub fn capture_screenshot() -> Option<vision::detection::FrameData> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::capture_screenshot()
    }
    #[cfg(target_os = "linux")]
    {
        crate::linux::capture_portal_screenshot()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
pub fn linux_screen_capture() -> Option<(vision::detection::FrameData, vision::geometry::ScreenRect)>
{
    crate::linux::capture_screen_frame()
}

#[cfg(not(target_os = "linux"))]
pub fn linux_screen_capture() -> Option<(vision::detection::FrameData, vision::geometry::ScreenRect)>
{
    None
}

pub fn set_window_topmost(_hwnd: WindowHandle) {
    #[cfg(target_os = "windows")]
    {
        windows_impl::set_window_topmost(_hwnd)
    }
}

pub fn set_process_realtime_priority() {
    #[cfg(target_os = "windows")]
    {
        windows_impl::set_process_realtime_priority()
    }
    #[cfg(target_os = "linux")]
    {
        crate::linux::set_process_realtime_priority()
    }
}

pub fn add_startup_registry() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::add_startup_registry()
    }
    #[cfg(target_os = "linux")]
    {
        crate::linux::add_startup_entry()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        Err("Startup not supported on this platform".into())
    }
}

pub fn remove_startup_registry() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::remove_startup_registry()
    }
    #[cfg(target_os = "linux")]
    {
        crate::linux::remove_startup_entry()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        Err("Startup not supported on this platform".into())
    }
}

pub fn get_cpu_usage() -> f32 {
    #[cfg(target_os = "windows")]
    {
        windows_impl::get_cpu_usage()
    }
    #[cfg(target_os = "linux")]
    {
        crate::linux::get_cpu_usage()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        0.0
    }
}

pub fn show_message(title: &str, message: &str) {
    #[cfg(target_os = "windows")]
    {
        windows_impl::show_message(title, message)
    }
    #[cfg(target_os = "linux")]
    {
        crate::linux::show_message(title, message)
    }
}

pub fn set_graphics_preference() {
    #[cfg(target_os = "windows")]
    {
        windows_impl::set_graphics_preference()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WindowHandle(pub usize);

impl WindowHandle {
    pub fn invalid() -> Self {
        Self(0)
    }

    pub fn is_valid(self) -> bool {
        self.0 != 0
    }
}

pub struct MonitorInfo {
    pub bounds: vision::geometry::ScreenRect,
    pub work_area: vision::geometry::ScreenRect,
    pub is_primary: bool,
}

pub fn get_monitors() -> Vec<MonitorInfo> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::get_monitors()
    }
    #[cfg(target_os = "linux")]
    {
        crate::linux::get_monitors()
            .into_iter()
            .map(|b| MonitorInfo {
                bounds: b,
                work_area: b,
                is_primary: true,
            })
            .collect()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        vec![]
    }
}

pub fn list_windows(
    include: &[String],
    exclude: &[String],
    always_skip: &[(String, String)],
) -> Vec<(WindowHandle, String, String, vision::geometry::ScreenRect)> {
    #[cfg(target_os = "windows")]
    {
        windows_impl::list_windows(include, exclude, always_skip)
    }
    #[cfg(target_os = "linux")]
    {
        crate::linux::list_windows(include, exclude, always_skip)
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        vec![]
    }
}

pub fn check_duplicate_instances() -> bool {
    #[cfg(target_os = "windows")]
    {
        windows_impl::check_duplicate_instances()
    }
    #[cfg(target_os = "linux")]
    {
        crate::linux::check_duplicate_instances()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        false
    }
}

pub fn ensure_app_directories() -> std::io::Result<()> {
    config::defaults::ensure_directories()
}

pub fn onnx_model_path() -> PathBuf {
    let external = config::defaults::external_model_dir().join("porda.onnx");
    if external.exists() {
        return external;
    }
    // Workspace model dir
    let ws = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model/porda.onnx");
    if ws.exists() {
        return ws;
    }
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    let exe_model = exe_dir.join("model").join("porda.onnx");
    if exe_model.exists() {
        return exe_model;
    }
    // Dev fallback adjacent to crate
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model/porda.onnx");
    dev
}

pub fn model_paths() -> (PathBuf, PathBuf) {
    // Prefer ONNX path's directory for legacy cfg/weights if ONNX exists, else legacy
    let onnx = onnx_model_path();
    if onnx.exists() {
        let dir = onnx.parent().unwrap_or(std::path::Path::new("."));
        return (
            dir.join("pordav4x3.cfg"),
            dir.join("porda-19200-lr-0005-909.weights"),
        );
    }
    let external = config::defaults::external_model_dir();
    let cfg_files: Vec<_> = std::fs::read_dir(&external)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "cfg")
                .unwrap_or(false)
        })
        .collect();

    let weights_files: Vec<_> = std::fs::read_dir(&external)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "weights")
                .unwrap_or(false)
        })
        .collect();

    if cfg_files.len() == 1 && weights_files.len() == 1 {
        return (cfg_files[0].path(), weights_files[0].path());
    }

    // Fallback: check Python reference model location (for dev / audit)
    let python_model_dir = PathBuf::from("/home/torchi/Desktop/Porda-AI/Porda-AI/model");
    let py_cfg = python_model_dir.join("pordav4x3.cfg");
    let py_weights = python_model_dir.join("porda-19200-lr-0005-909.weights");
    if py_cfg.exists() && py_weights.exists() {
        tracing::info!("Using Python reference model at {:?}", python_model_dir);
        return (py_cfg, py_weights);
    }

    // Fallback: workspace-relative model dir
    let workspace_model = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../Porda-AI/model");
    let ws_cfg = workspace_model.join("pordav4x3.cfg");
    let ws_weights = workspace_model.join("porda-19200-lr-0005-909.weights");
    if ws_cfg.exists() && ws_weights.exists() {
        return (ws_cfg, ws_weights);
    }

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."));
    (
        exe_dir.join("model").join("pordav4x3.cfg"),
        exe_dir
            .join("model")
            .join("porda-19200-lr-0005-909.weights"),
    )
}

// Windows-specific implementations
#[cfg(target_os = "windows")]
mod windows_impl {
    use super::*;
    use vision::geometry::ScreenRect;

    pub fn get_foreground_window() -> Option<WindowHandle> {
        unsafe {
            let hwnd = windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow();
            if hwnd.0.is_null() {
                None
            } else {
                Some(WindowHandle(hwnd.0 as usize))
            }
        }
    }

    pub fn get_window_rect(hwnd: WindowHandle) -> Option<ScreenRect> {
        unsafe {
            use windows::Win32::UI::WindowsAndMessaging::GetWindowRect;
            let handle = windows::Win32::Foundation::HWND(hwnd.0 as *mut _);
            let mut rect = windows::Win32::Foundation::RECT::default();
            if GetWindowRect(handle, &mut rect).is_ok() {
                Some(ScreenRect {
                    x: rect.left,
                    y: rect.top,
                    width: (rect.right - rect.left) as u32,
                    height: (rect.bottom - rect.top) as u32,
                })
            } else {
                None
            }
        }
    }

    pub fn get_client_rect(hwnd: WindowHandle) -> Option<ScreenRect> {
        unsafe {
            use windows::Win32::Graphics::Gdi::ClientToScreen;
            use windows::Win32::UI::WindowsAndMessaging::GetClientRect;
            let handle = windows::Win32::Foundation::HWND(hwnd.0 as *mut _);
            let mut rect = windows::Win32::Foundation::RECT::default();
            if GetClientRect(handle, &mut rect).is_ok() {
                let mut point = windows::Win32::Foundation::POINT { x: 0, y: 0 };
                let _ = ClientToScreen(handle, &mut point);
                Some(ScreenRect {
                    x: point.x,
                    y: point.y,
                    width: (rect.right - rect.left) as u32,
                    height: (rect.bottom - rect.top) as u32,
                })
            } else {
                None
            }
        }
    }

    pub fn get_window_process_name(hwnd: WindowHandle) -> Option<String> {
        unsafe {
            use windows::Win32::Foundation::CloseHandle;
            use windows::Win32::System::Threading::OpenProcess;
            use windows::Win32::System::Threading::PROCESS_QUERY_LIMITED_INFORMATION;
            use windows::Win32::UI::WindowsAndMessaging::GetWindowThreadProcessId;

            let handle = windows::Win32::Foundation::HWND(hwnd.0 as *mut _);
            let mut pid = 0u32;
            GetWindowThreadProcessId(handle, Some(&mut pid));

            let process_handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
            let mut name = [0u16; 260];
            let mut size = 260u32;
            let result = windows::Win32::System::Threading::QueryFullProcessImageNameW(
                process_handle,
                windows::Win32::System::Threading::PROCESS_NAME_FORMAT(0),
                windows::core::PWSTR(name.as_mut_ptr()),
                &mut size,
            );
            let _ = CloseHandle(process_handle);

            if result.is_ok() {
                let name = String::from_utf16_lossy(&name[..size as usize]);
                std::path::Path::new(&name)
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
            } else {
                None
            }
        }
    }

    pub fn get_window_title(hwnd: WindowHandle) -> Option<String> {
        unsafe {
            use windows::Win32::UI::WindowsAndMessaging::GetWindowTextW;
            let handle = windows::Win32::Foundation::HWND(hwnd.0 as *mut _);
            let mut name = [0u16; 512];
            let len = GetWindowTextW(handle, &mut name);
            if len > 0 {
                Some(String::from_utf16_lossy(&name[..len as usize]))
            } else {
                None
            }
        }
    }

    pub fn is_window_visible(hwnd: WindowHandle) -> bool {
        unsafe {
            use windows::Win32::UI::WindowsAndMessaging::IsWindowVisible;
            let handle = windows::Win32::Foundation::HWND(hwnd.0 as *mut _);
            IsWindowVisible(handle).as_bool()
        }
    }

    pub fn is_window_minimized(hwnd: WindowHandle) -> bool {
        unsafe {
            use windows::Win32::UI::WindowsAndMessaging::IsIconic;
            let handle = windows::Win32::Foundation::HWND(hwnd.0 as *mut _);
            IsIconic(handle).as_bool()
        }
    }

    pub fn set_window_topmost(hwnd: WindowHandle) {
        unsafe {
            use windows::Win32::UI::WindowsAndMessaging::SetWindowPos;
            use windows::Win32::UI::WindowsAndMessaging::HWND_TOPMOST;
            use windows::Win32::UI::WindowsAndMessaging::SWP_NOMOVE;
            use windows::Win32::UI::WindowsAndMessaging::SWP_NOSIZE;
            let handle = windows::Win32::Foundation::HWND(hwnd.0 as *mut _);
            let _ = SetWindowPos(handle, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE);
        }
    }

    pub fn set_process_realtime_priority() {
        unsafe {
            use windows::Win32::System::Threading::GetCurrentProcess;
            use windows::Win32::System::Threading::SetPriorityClass;
            use windows::Win32::System::Threading::REALTIME_PRIORITY_CLASS;
            let handle = GetCurrentProcess();
            let _ = SetPriorityClass(handle, REALTIME_PRIORITY_CLASS);
        }
    }

    pub fn add_startup_registry() -> Result<(), Box<dyn std::error::Error>> {
        use windows::Win32::Foundation::ERROR_SUCCESS;
        use windows::Win32::System::Registry::*;
        unsafe {
            let exe_path = std::env::current_exe()?;
            let value = format!("\"{}\" --startup_by_windows", exe_path.display());

            let mut result_key = HKEY::default();
            let reg_path = windows::core::PCWSTR(
                windows::core::w!("Software\\Microsoft\\Windows\\CurrentVersion\\Run").as_ptr(),
            );

            let status = RegOpenKeyExW(
                HKEY_CURRENT_USER,
                reg_path,
                0,
                KEY_SET_VALUE,
                &mut result_key,
            );
            if status != ERROR_SUCCESS {
                return Err(format!("RegOpenKeyExW failed: {:?}", status).into());
            }

            let value_wide: Vec<u16> = value.encode_utf16().chain(std::iter::once(0)).collect();
            let data =
                std::slice::from_raw_parts(value_wide.as_ptr() as *const u8, value_wide.len() * 2);
            let status = RegSetValueExW(
                result_key,
                windows::core::PCWSTR(windows::core::w!("PordaAi").as_ptr()),
                0,
                REG_SZ,
                Some(data),
            );
            if status != ERROR_SUCCESS {
                let _ = RegCloseKey(result_key);
                return Err(format!("RegSetValueExW failed: {:?}", status).into());
            }

            let status = RegCloseKey(result_key);
            if status != ERROR_SUCCESS {
                return Err(format!("RegCloseKey failed: {:?}", status).into());
            }
            Ok(())
        }
    }

    pub fn remove_startup_registry() -> Result<(), Box<dyn std::error::Error>> {
        use windows::Win32::Foundation::ERROR_SUCCESS;
        use windows::Win32::System::Registry::*;
        unsafe {
            let mut result_key = HKEY::default();
            let reg_path = windows::core::PCWSTR(
                windows::core::w!("Software\\Microsoft\\Windows\\CurrentVersion\\Run").as_ptr(),
            );

            let status = RegOpenKeyExW(
                HKEY_CURRENT_USER,
                reg_path,
                0,
                KEY_SET_VALUE,
                &mut result_key,
            );
            if status != ERROR_SUCCESS {
                return Err(format!("RegOpenKeyExW failed: {:?}", status).into());
            }

            let status = RegDeleteValueW(
                result_key,
                windows::core::PCWSTR(windows::core::w!("PordaAi").as_ptr()),
            );
            if status != ERROR_SUCCESS {
                let _ = RegCloseKey(result_key);
                return Err(format!("RegDeleteValueW failed: {:?}", status).into());
            }

            let status = RegCloseKey(result_key);
            if status != ERROR_SUCCESS {
                return Err(format!("RegCloseKey failed: {:?}", status).into());
            }
            Ok(())
        }
    }

    pub fn get_cpu_usage() -> f32 {
        0.0
    }

    pub fn show_message(title: &str, message: &str) {
        unsafe {
            use windows::Win32::UI::WindowsAndMessaging::MessageBoxW;
            let title_wide: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
            let msg_wide: Vec<u16> = message.encode_utf16().chain(std::iter::once(0)).collect();
            MessageBoxW(
                None,
                windows::core::PCWSTR(msg_wide.as_ptr()),
                windows::core::PCWSTR(title_wide.as_ptr()),
                windows::Win32::UI::WindowsAndMessaging::MB_OK,
            );
        }
    }

    pub fn set_graphics_preference() {}

    pub fn capture_window(_hwnd: WindowHandle) -> Option<vision::detection::FrameData> {
        None
    }

    pub fn capture_screenshot() -> Option<vision::detection::FrameData> {
        None
    }

    pub fn get_monitors() -> Vec<MonitorInfo> {
        vec![MonitorInfo {
            bounds: ScreenRect::new(0, 0, 1920, 1080),
            work_area: ScreenRect::new(0, 0, 1920, 1080),
            is_primary: true,
        }]
    }

    pub fn list_windows(
        _include: &[String],
        _exclude: &[String],
        _always_skip: &[(String, String)],
    ) -> Vec<(WindowHandle, String, String, ScreenRect)> {
        vec![]
    }

    pub fn check_duplicate_instances() -> bool {
        false
    }
}
