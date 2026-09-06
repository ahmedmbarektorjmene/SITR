use std::path::PathBuf;

pub fn add_startup_entry() -> Result<(), Box<dyn std::error::Error>> {
    let autostart_dir = dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("~/.config"))
        .join("autostart");

    std::fs::create_dir_all(&autostart_dir)?;

    let exe_path = std::env::current_exe()?;
    let desktop_content = format!(
        r#"[Desktop Entry]
Type=Application
Name=Porda AI
Exec={} --startup
Icon=pordaai
Terminal=false
Categories=Utility;
Comment=Real-time desktop privacy overlay
X-GNOME-Autostart-enabled=true
"#,
        exe_path.display()
    );

    let desktop_path = autostart_dir.join("pordaai.desktop");
    std::fs::write(&desktop_path, desktop_content)?;

    tracing::info!("Startup entry created at {:?}", desktop_path);
    Ok(())
}

pub fn remove_startup_entry() -> Result<(), Box<dyn std::error::Error>> {
    let autostart_dir = dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("~/.config"))
        .join("autostart");

    let desktop_path = autostart_dir.join("pordaai.desktop");
    if desktop_path.exists() {
        std::fs::remove_file(&desktop_path)?;
        tracing::info!("Startup entry removed");
    }
    Ok(())
}

pub fn is_startup_enabled() -> bool {
    let autostart_dir = dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("~/.config"))
        .join("autostart");

    autostart_dir.join("pordaai.desktop").exists()
}
