fn main() {
    let ui_dir = std::path::Path::new("ui");
    slint_build::compile(ui_dir.join("app-window.slint")).unwrap();
}
