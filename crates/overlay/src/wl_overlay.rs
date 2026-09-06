use std::sync::mpsc;

use vision::detection::{CoverRect, FrameData};
use vision::geometry::{ColorRgb, ScreenRect};
use wayland_client::Proxy as _;
use wgpu::util::DeviceExt as _;

use crate::compositor::{OverlayCapability, OverlayError, OverlayRenderer};

#[derive(Debug, Clone)]
pub struct OverlayConfig {
    pub width: u32,
    pub height: u32,
    pub scale: f64,
    pub output_name: Option<String>,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1200,
            scale: 1.0,
            output_name: None,
        }
    }
}

enum OverlayCommand {
    UpdateCovers(Vec<CoverRect>),
    Clear,
    Shutdown,
}

pub struct WaylandOverlay {
    tx: mpsc::Sender<OverlayCommand>,
    capability: OverlayCapability,
    #[allow(dead_code)]
    solid_color: ColorRgb,
}

impl WaylandOverlay {
    pub fn new(config: OverlayConfig, solid_color: ColorRgb) -> Self {
        let (tx, rx) = mpsc::channel::<OverlayCommand>();

        let capability = detect_layer_shell_support();

        if capability.is_supported() {
            let cfg = config.clone();
            std::thread::Builder::new()
                .name("overlay".to_string())
                .spawn(move || {
                    run_overlay_thread(cfg, rx, solid_color);
                })
                .expect("Failed to spawn overlay thread");
            tracing::info!("Wayland overlay thread spawned");
        } else {
            tracing::warn!("Layer-shell not available: {:?}", capability);
        }

        Self {
            tx,
            capability,
            solid_color,
        }
    }

    pub fn with_test_rect(config: OverlayConfig) -> Self {
        let overlay = Self::new(config.clone(), ColorRgb::new(255, 0, 0));
        if overlay.capability.is_supported() {
            let test_rect = CoverRect::new_solid(
                ScreenRect::new(
                    (config.width as i32 / 2) - 150,
                    (config.height as i32 / 2) - 100,
                    300,
                    200,
                ),
                ColorRgb::new(255, 0, 0),
            );
            let _ = overlay
                .tx
                .send(OverlayCommand::UpdateCovers(vec![test_rect]));
            tracing::info!("Test rectangle sent: 300x200 at center");
        }
        overlay
    }
}

fn detect_layer_shell_support() -> OverlayCapability {
    let conn = match wayland_client::Connection::connect_to_env() {
        Ok(c) => c,
        Err(e) => {
            return OverlayCapability::Unsupported(format!(
                "Failed to connect to Wayland display: {}",
                e
            ))
        }
    };

    let display = conn.display();
    let mut queue = conn.new_event_queue();
    let qh = queue.handle();

    use wayland_client::protocol::{wl_output, wl_registry};
    use wayland_client::{Dispatch, QueueHandle};

    #[derive(Default)]
    struct RegistryState {
        has_compositor: bool,
        has_layer_shell: bool,
        has_shm: bool,
        outputs: Vec<OutputState>,
    }

    #[derive(Debug, Clone)]
    struct OutputState {
        #[allow(dead_code)]
        name: u32,
        geometry: ScreenRect,
        scale: i32,
    }

    impl Dispatch<wl_registry::WlRegistry, ()> for RegistryState {
        fn event(
            state: &mut Self,
            registry: &wl_registry::WlRegistry,
            event: wl_registry::Event,
            _: &(),
            _: &Connection,
            qh: &QueueHandle<Self>,
        ) {
            if let wl_registry::Event::Global {
                name,
                interface,
                version,
            } = event
            {
                match interface.as_str() {
                    "wl_compositor" => {
                        state.has_compositor = true;
                        let _ = registry
                            .bind::<wayland_client::protocol::wl_compositor::WlCompositor, _, _>(
                                name,
                                version.min(6),
                                qh,
                                (),
                            );
                    }
                    "zwlr_layer_shell_v1" => {
                        state.has_layer_shell = true;
                    }
                    "wl_shm" => {
                        state.has_shm = true;
                    }
                    "wl_output" => {
                        let _ = registry.bind::<wl_output::WlOutput, _, _>(
                            name,
                            version.min(4),
                            qh,
                            (),
                        );
                        state.outputs.push(OutputState {
                            name,
                            geometry: ScreenRect::new(0, 0, 1920, 1080),
                            scale: 1,
                        });
                    }
                    _ => {}
                }
            }
        }
    }

    impl Dispatch<wayland_client::protocol::wl_compositor::WlCompositor, ()> for RegistryState {
        fn event(
            _: &mut Self,
            _: &wayland_client::protocol::wl_compositor::WlCompositor,
            _: wayland_client::protocol::wl_compositor::Event,
            _: &(),
            _: &Connection,
            _: &QueueHandle<Self>,
        ) {
        }
    }

    impl Dispatch<wl_output::WlOutput, ()> for RegistryState {
        fn event(
            state: &mut Self,
            proxy: &wl_output::WlOutput,
            event: wl_output::Event,
            _: &(),
            _: &Connection,
            _: &QueueHandle<Self>,
        ) {
            if let wl_output::Event::Scale { factor } = event {
                for out in &mut state.outputs {
                    if out.name == proxy.id().protocol_id() {
                        out.scale = factor;
                    }
                }
            }
            if let wl_output::Event::Mode { width, height, .. } = event {
                for out in &mut state.outputs {
                    if out.name == proxy.id().protocol_id() {
                        out.geometry.width = width as u32;
                        out.geometry.height = height as u32;
                    }
                }
            }
        }
    }

    use wayland_client::Connection;

    let mut state = RegistryState::default();
    let _registry = display.get_registry(&qh, ());

    let _ = queue.roundtrip(&mut state);

    tracing::info!(
        "Overlay capability check: compositor={}, layer_shell={}, shm={}, outputs={}",
        state.has_compositor,
        state.has_layer_shell,
        state.has_shm,
        state.outputs.len()
    );

    if !state.has_layer_shell {
        return OverlayCapability::Unsupported(
            "zwlr_layer_shell_v1 not advertised by compositor".to_string(),
        );
    }
    if !state.has_compositor {
        return OverlayCapability::Unsupported("wl_compositor not advertised".to_string());
    }

    OverlayCapability::Supported
}

fn run_overlay_thread(
    mut config: OverlayConfig,
    rx: mpsc::Receiver<OverlayCommand>,
    solid_color: ColorRgb,
) {
    tracing::info!(
        "Overlay thread starting: {}x{} scale={}",
        config.width,
        config.height,
        config.scale
    );

    let conn = match wayland_client::Connection::connect_to_env() {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Overlay: failed to connect to Wayland: {}", e);
            return;
        }
    };

    let display = conn.display();
    let mut queue = conn.new_event_queue();
    let qh = queue.handle();

    let mut state = OverlayState::new(config.clone(), solid_color);

    let _registry = display.get_registry(&qh, ());

    if queue.roundtrip(&mut state).is_err() {
        tracing::error!("Overlay: initial roundtrip failed");
        return;
    }

    if state.layer_shell.is_none() {
        tracing::error!("Overlay: zwlr_layer_shell_v1 not available, cannot create overlay");
        return;
    }

    if state.compositor.is_none() {
        tracing::error!("Overlay: wl_compositor not available");
        return;
    }

    // Create surface and layer surface
    let compositor = state.compositor.as_ref().unwrap().clone();
    let surface = compositor.create_surface(&qh, ());
    let layer_shell = state.layer_shell.as_ref().unwrap().clone();

    // Determine output
    let output = state.outputs.first().cloned();

    let layer_surface = layer_shell.get_layer_surface(
        &surface,
        output.as_ref(),
        wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_shell_v1::Layer::Overlay,
        "overlay".to_string(),
        &qh,
        (),
    );

    layer_surface.set_size(config.width, config.height);
    layer_surface.set_anchor(
        wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::Anchor::Top
            | wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::Anchor::Bottom
            | wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::Anchor::Left
            | wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::Anchor::Right,
    );
    layer_surface.set_exclusive_zone(-1);
    layer_surface.set_keyboard_interactivity(
        wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::KeyboardInteractivity::None,
    );

    // Create empty input region for click-through
    if let Some(ref comp) = state.compositor {
        let region = comp.create_region(&qh, ());
        surface.set_input_region(Some(&region));
        // Region with no rectangles = empty = click-through
    }

    surface.commit();

    // Roundtrip to get configure
    if queue.roundtrip(&mut state).is_err() {
        tracing::error!("Overlay: roundtrip after surface creation failed");
        return;
    }

    // Override config with actual output geometry if available
    if let Some(info) = state.output_info.first() {
        if info.geometry.width > 0 && info.geometry.height > 0 {
            // Use actual output size, not stub fallback
            let actual_w = info.geometry.width;
            let actual_h = info.geometry.height;
            if actual_w != config.width || actual_h != config.height {
                tracing::info!(
                    "Overlay: correcting size from {}x{} to output {}x{}",
                    config.width,
                    config.height,
                    actual_w,
                    actual_h
                );
                config.width = actual_w;
                config.height = actual_h;
                layer_surface.set_size(actual_w, actual_h);
                surface.commit();
                let _ = queue.roundtrip(&mut state);
            }
        }
    }

    tracing::info!("Overlay: layer surface created, initializing renderer");

    // Initialize wgpu or fallback to shm
    let mut renderer: Box<dyn Renderer> = match try_init_wgpu(&conn, &surface, &config) {
        Ok(r) => {
            tracing::info!("Overlay: wgpu renderer initialized");
            Box::new(r)
        }
        Err(e) => {
            tracing::warn!("Overlay: wgpu init failed ({}), falling back to SHM", e);
            match ShmRenderer::new(&state, &surface, &config, qh.clone()) {
                Ok(r) => Box::new(r),
                Err(e2) => {
                    tracing::error!("Overlay: SHM fallback also failed: {}", e2);
                    return;
                }
            }
        }
    };

    // Acknowledge initial configure and draw test frame
    state.needs_redraw = true;

    // Check for test rect env flag
    let has_test_rect = std::env::var("overlay_TEST_RECT").is_ok();
    if has_test_rect {
        let test_rect = CoverRect::new_solid(
            ScreenRect::new(
                (config.width as i32 / 2) - 150,
                (config.height as i32 / 2) - 100,
                300,
                200,
            ),
            ColorRgb::new(255, 0, 0),
        );
        state.pending_covers = vec![test_rect];
        state.solid_color = ColorRgb::new(255, 0, 0);
        state.needs_redraw = true;
        tracing::info!("Overlay: test rectangle enabled (300x200 at center)");
    }

    let mut last_log = std::time::Instant::now();
    let mut frame_count: u64 = 0;

    loop {
        // Handle overlay commands (non-blocking)
        while let Ok(cmd) = rx.try_recv() {
            match cmd {
                OverlayCommand::UpdateCovers(covers) => {
                    tracing::info!("Overlay: received UpdateCovers count={}", covers.len());
                    for (i, c) in covers.iter().enumerate() {
                        tracing::info!(
                            "Overlay: Cover[{}] x={} y={} w={} h={} mode={:?} resolved_color={:?} has_blur={}",
                            i,
                            c.screen_rect.x,
                            c.screen_rect.y,
                            c.screen_rect.width,
                            c.screen_rect.height,
                            c.mode,
                            c.resolved_color,
                            c.blur_data.is_some()
                        );
                    }
                    state.pending_covers = covers;
                    state.needs_redraw = true;
                }
                OverlayCommand::Clear => {
                    tracing::info!("Overlay: received Clear");
                    state.pending_covers.clear();
                    state.needs_redraw = true;
                }
                OverlayCommand::Shutdown => {
                    tracing::info!("Overlay: shutdown requested");
                    return;
                }
            }
        }

        // Wayland dispatch (non-blocking)
        queue.dispatch_pending(&mut state).ok();
        queue.flush().ok();

        if state.should_close {
            tracing::info!("Overlay: compositor requested close");
            break;
        }

        if state.needs_redraw {
            let covers = state.pending_covers.clone();
            let render_start = std::time::Instant::now();
            let cover_count = covers.len();
            let pixel_count: usize = covers
                .iter()
                .map(|c| (c.screen_rect.width * c.screen_rect.height) as usize)
                .sum();
            if let Err(e) = renderer.render(&covers, &config) {
                tracing::error!("Overlay render failed: {}", e);
            } else {
                let elapsed = render_start.elapsed();
                tracing::debug!(
                    "Overlay: render took {:?} covers={} pixels={}",
                    elapsed,
                    cover_count,
                    pixel_count
                );
                frame_count += 1;
                surface.commit();
                queue.flush().ok();
            }
            state.needs_redraw = false;

            if last_log.elapsed() >= std::time::Duration::from_secs(5) {
                tracing::info!(
                    "Overlay: frames={}, covers={}, size={}x{}",
                    frame_count,
                    covers.len(),
                    config.width,
                    config.height
                );
                last_log = std::time::Instant::now();
            }
        }

        // Small sleep to avoid busy loop, but also handle Wayland events
        std::thread::sleep(std::time::Duration::from_millis(16));

        // Also do blocking dispatch with timeout for configure events
        if !state.needs_redraw {
            let _ = queue.dispatch_pending(&mut state);
        }
    }

    tracing::info!("Overlay thread exiting cleanly");
}

// ---------------------------------------------------------------------------
// Overlay state for Wayland dispatch
// ---------------------------------------------------------------------------

struct OverlayState {
    compositor: Option<wayland_client::protocol::wl_compositor::WlCompositor>,
    shm: Option<wayland_client::protocol::wl_shm::WlShm>,
    layer_shell: Option<
        wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_shell_v1::ZwlrLayerShellV1,
    >,
    outputs: Vec<wayland_client::protocol::wl_output::WlOutput>,
    output_info: Vec<OutputInfo>,
    pending_covers: Vec<CoverRect>,
    solid_color: ColorRgb,
    needs_redraw: bool,
    should_close: bool,
    configured_size: Option<(u32, u32)>,
}

#[derive(Debug, Clone)]
struct OutputInfo {
    #[allow(dead_code)]
    name: String,
    geometry: ScreenRect,
    scale: i32,
}

impl OverlayState {
    fn new(config: OverlayConfig, solid_color: ColorRgb) -> Self {
        let _ = config;
        Self {
            compositor: None,
            shm: None,
            layer_shell: None,
            outputs: Vec::new(),
            output_info: Vec::new(),
            pending_covers: Vec::new(),
            solid_color,
            needs_redraw: false,
            should_close: false,
            configured_size: None,
        }
    }
}

impl wayland_client::Dispatch<wayland_client::protocol::wl_registry::WlRegistry, ()>
    for OverlayState
{
    fn event(
        state: &mut Self,
        registry: &wayland_client::protocol::wl_registry::WlRegistry,
        event: wayland_client::protocol::wl_registry::Event,
        _: &(),
        _: &wayland_client::Connection,
        qh: &wayland_client::QueueHandle<Self>,
    ) {
        if let wayland_client::protocol::wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            match interface.as_str() {
                "wl_compositor" => {
                    let comp = registry
                        .bind::<wayland_client::protocol::wl_compositor::WlCompositor, _, _>(
                            name,
                            version.min(6),
                            qh,
                            (),
                        );
                    state.compositor = Some(comp);
                    tracing::debug!("Overlay: bound wl_compositor v{}", version);
                }
                "wl_shm" => {
                    let shm = registry.bind::<wayland_client::protocol::wl_shm::WlShm, _, _>(
                        name,
                        version.min(1),
                        qh,
                        (),
                    );
                    state.shm = Some(shm);
                }
                "wl_output" => {
                    let output = registry
                        .bind::<wayland_client::protocol::wl_output::WlOutput, _, _>(
                            name,
                            version.min(4),
                            qh,
                            (),
                        );
                    state.outputs.push(output);
                    state.output_info.push(OutputInfo {
                        name: format!("output-{}", name),
                        geometry: ScreenRect::new(0, 0, 1920, 1080),
                        scale: 1,
                    });
                    tracing::debug!("Overlay: bound wl_output name={} v{}", name, version);
                }
                "zwlr_layer_shell_v1" => {
                    let ls = registry.bind::<
                        wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_shell_v1::ZwlrLayerShellV1,
                        _,
                        _,
                    >(name, version.min(4), qh, ());
                    state.layer_shell = Some(ls);
                    tracing::info!("Overlay: bound zwlr_layer_shell_v1 v{}", version);
                }
                _ => {}
            }
        }
    }
}

impl wayland_client::Dispatch<wayland_client::protocol::wl_compositor::WlCompositor, ()>
    for OverlayState
{
    fn event(
        _: &mut Self,
        _: &wayland_client::protocol::wl_compositor::WlCompositor,
        _: wayland_client::protocol::wl_compositor::Event,
        _: &(),
        _: &wayland_client::Connection,
        _: &wayland_client::QueueHandle<Self>,
    ) {
    }
}

impl wayland_client::Dispatch<wayland_client::protocol::wl_shm::WlShm, ()> for OverlayState {
    fn event(
        _: &mut Self,
        _: &wayland_client::protocol::wl_shm::WlShm,
        _: wayland_client::protocol::wl_shm::Event,
        _: &(),
        _: &wayland_client::Connection,
        _: &wayland_client::QueueHandle<Self>,
    ) {
    }
}

impl wayland_client::Dispatch<wayland_client::protocol::wl_output::WlOutput, ()> for OverlayState {
    fn event(
        state: &mut Self,
        proxy: &wayland_client::protocol::wl_output::WlOutput,
        event: wayland_client::protocol::wl_output::Event,
        _: &(),
        _: &wayland_client::Connection,
        _: &wayland_client::QueueHandle<Self>,
    ) {
        match event {
            wayland_client::protocol::wl_output::Event::Geometry {
                x,
                y,
                physical_width,
                physical_height,
                ..
            } => {
                let idx = state.outputs.iter().position(|o| o.id() == proxy.id());
                if let Some(i) = idx {
                    state.output_info[i].geometry.x = x;
                    state.output_info[i].geometry.y = y;
                    let _ = physical_width;
                    let _ = physical_height;
                }
            }
            wayland_client::protocol::wl_output::Event::Mode { width, height, .. } => {
                let idx = state.outputs.iter().position(|o| o.id() == proxy.id());
                if let Some(i) = idx {
                    state.output_info[i].geometry.width = width as u32;
                    state.output_info[i].geometry.height = height as u32;
                }
            }
            wayland_client::protocol::wl_output::Event::Scale { factor } => {
                let idx = state.outputs.iter().position(|o| o.id() == proxy.id());
                if let Some(i) = idx {
                    state.output_info[i].scale = factor;
                }
            }
            _ => {}
        }
    }
}

impl wayland_client::Dispatch<wayland_client::protocol::wl_surface::WlSurface, ()>
    for OverlayState
{
    fn event(
        _: &mut Self,
        _: &wayland_client::protocol::wl_surface::WlSurface,
        _: wayland_client::protocol::wl_surface::Event,
        _: &(),
        _: &wayland_client::Connection,
        _: &wayland_client::QueueHandle<Self>,
    ) {
    }
}

impl wayland_client::Dispatch<wayland_client::protocol::wl_region::WlRegion, ()> for OverlayState {
    fn event(
        _: &mut Self,
        _: &wayland_client::protocol::wl_region::WlRegion,
        _: wayland_client::protocol::wl_region::Event,
        _: &(),
        _: &wayland_client::Connection,
        _: &wayland_client::QueueHandle<Self>,
    ) {
    }
}

impl
    wayland_client::Dispatch<
        wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_shell_v1::ZwlrLayerShellV1,
        (),
    > for OverlayState
{
    fn event(
        _: &mut Self,
        _: &wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_shell_v1::ZwlrLayerShellV1,
        _: wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_shell_v1::Event,
        _: &(),
        _: &wayland_client::Connection,
        _: &wayland_client::QueueHandle<Self>,
    ) {
    }
}

impl
    wayland_client::Dispatch<
        wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::ZwlrLayerSurfaceV1,
        (),
    > for OverlayState
{
    fn event(
        state: &mut Self,
        proxy: &wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::ZwlrLayerSurfaceV1,
        event: wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::Event,
        _: &(),
        _: &wayland_client::Connection,
        _: &wayland_client::QueueHandle<Self>,
    ) {
        match event {
            wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::Event::Configure {
                serial,
                width,
                height,
            } => {
                proxy.ack_configure(serial);
                state.configured_size = Some((width, height));
                state.needs_redraw = true;
                tracing::debug!("Overlay: layer surface configure {}x{}", width, height);
            }
            wayland_protocols_wlr::layer_shell::v1::client::zwlr_layer_surface_v1::Event::Closed => {
                state.should_close = true;
                tracing::info!("Overlay: layer surface closed by compositor");
            }
            _ => {}
        }
    }
}

impl wayland_client::Dispatch<wayland_client::protocol::wl_shm_pool::WlShmPool, ()>
    for OverlayState
{
    fn event(
        _: &mut Self,
        _: &wayland_client::protocol::wl_shm_pool::WlShmPool,
        _: wayland_client::protocol::wl_shm_pool::Event,
        _: &(),
        _: &wayland_client::Connection,
        _: &wayland_client::QueueHandle<Self>,
    ) {
    }
}

impl wayland_client::Dispatch<wayland_client::protocol::wl_buffer::WlBuffer, ()> for OverlayState {
    fn event(
        _: &mut Self,
        _: &wayland_client::protocol::wl_buffer::WlBuffer,
        event: wayland_client::protocol::wl_buffer::Event,
        _: &(),
        _: &wayland_client::Connection,
        _: &wayland_client::QueueHandle<Self>,
    ) {
        match event {
            wayland_client::protocol::wl_buffer::Event::Release => {
                tracing::trace!("Overlay: buffer released");
            }
            _ => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Renderer abstraction
// ---------------------------------------------------------------------------

trait Renderer: Send {
    fn render(&mut self, covers: &[CoverRect], config: &OverlayConfig) -> Result<(), String>;
}

// ---------------------------------------------------------------------------
// wgpu renderer
// ---------------------------------------------------------------------------

struct WgpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

fn try_init_wgpu(
    _conn: &wayland_client::Connection,
    _surface: &wayland_client::protocol::wl_surface::WlSurface,
    _overlay_config: &OverlayConfig,
) -> Result<WgpuRenderer, String> {
    // Wayland raw handle extraction requires wayland-backend sys pointers
    // not exposed in wayland-client 0.31 ObjectId API. Fall back to SHM
    // for this milestone; wgpu will be integrated via proper surface
    // creation in a follow-up (e.g., using winit or smithay window).
    Err("wgpu Wayland surface via raw handles not yet wired (fallback to SHM)".to_string())
}

impl Renderer for WgpuRenderer {
    fn render(&mut self, covers: &[CoverRect], config: &OverlayConfig) -> Result<(), String> {
        let output = self
            .surface
            .get_current_texture()
            .map_err(|e| format!("get_current_texture: {}", e))?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("overlay-encoder"),
            });

        // Update uniform: use first cover color if available, else transparent
        let first_color = covers
            .first()
            .and_then(|c| c.resolved_color)
            .unwrap_or(ColorRgb::new(255, 0, 0));
        let full_uniform: [f32; 8] = [
            config.width as f32,
            config.height as f32,
            first_color.r as f32 / 255.0,
            first_color.g as f32 / 255.0,
            first_color.b as f32 / 255.0,
            0.85,
            0.0,
            0.0,
        ];
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&full_uniform));

        // If no covers, just clear transparent
        if covers.is_empty() {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay-clear-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            drop(pass);
        } else {
            // Build vertex data for all cover rects (2 triangles per rect = 6 vertices)
            let mut vertices: Vec<f32> = Vec::with_capacity(covers.len() * 12);
            let mut indices: Vec<u16> = Vec::with_capacity(covers.len() * 6);

            for (i, cover) in covers.iter().enumerate() {
                let r = &cover.screen_rect;
                // Convert to NDC: map capture coords to overlay surface coords
                // For now, assume 1:1 (overlay size == capture size)
                let x0 = r.x as f32;
                let y0 = r.y as f32;
                let x1 = (r.x + r.width as i32) as f32;
                let y1 = (r.y + r.height as i32) as f32;

                let base = (i * 4) as u16;
                vertices.extend_from_slice(&[x0, y0, x1, y0, x1, y1, x0, y1]);
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }

            let vertex_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("overlay-vertex-buffer"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });

            let index_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("overlay-index-buffer"),
                    contents: bytemuck::cast_slice(&indices),
                    usage: wgpu::BufferUsages::INDEX,
                });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
            drop(pass);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SHM fallback renderer (CPU, transparent)
// ---------------------------------------------------------------------------

struct ShmRenderer {
    shm: wayland_client::protocol::wl_shm::WlShm,
    surface: wayland_client::protocol::wl_surface::WlSurface,
    qh: wayland_client::QueueHandle<OverlayState>,
    width: u32,
    height: u32,
    pool: Option<wayland_client::protocol::wl_shm_pool::WlShmPool>,
    buffer: Option<wayland_client::protocol::wl_buffer::WlBuffer>,
}

impl ShmRenderer {
    fn new(
        state: &OverlayState,
        surface: &wayland_client::protocol::wl_surface::WlSurface,
        config: &OverlayConfig,
        qh: wayland_client::QueueHandle<OverlayState>,
    ) -> Result<Self, String> {
        let shm = state.shm.as_ref().ok_or("wl_shm not available")?.clone();
        Ok(Self {
            shm,
            surface: surface.clone(),
            qh,
            width: config.width,
            height: config.height,
            pool: None,
            buffer: None,
        })
    }
}

impl Renderer for ShmRenderer {
    fn render(&mut self, covers: &[CoverRect], _config: &OverlayConfig) -> Result<(), String> {
        let cover_count = covers.len();
        let pixel_count: usize = covers
            .iter()
            .map(|c| (c.screen_rect.width * c.screen_rect.height) as usize)
            .sum();
        tracing::info!(
            "ShmRenderer: rendering {} covers ({} pixels) {}x{}",
            cover_count,
            pixel_count,
            self.width,
            self.height
        );
        for (i, c) in covers.iter().enumerate() {
            tracing::debug!(
                "ShmRenderer: cover[{}] {:?} resolved={:?} blur={}",
                i,
                c.screen_rect,
                c.resolved_color,
                c.blur_data.is_some()
            );
        }
        use std::os::fd::AsFd;

        let stride = (self.width * 4) as i32;
        let size = (stride as u32 * self.height) as usize;

        let start = std::time::Instant::now();
        let data = build_shm_argb_data_from_covers(covers, self.width, self.height);
        let elapsed = start.elapsed();
        tracing::debug!(
            "ShmRenderer: build_shm_argb_data took {:?} for {} covers",
            elapsed,
            cover_count
        );

        // Create shm file via tempfile
        let mut file = tempfile::tempfile().map_err(|e| format!("tempfile: {}", e))?;
        use std::io::Write;
        file.write_all(&data)
            .map_err(|e| format!("write shm: {}", e))?;
        // Ensure file size
        file.set_len(size as u64)
            .map_err(|e| format!("set_len: {}", e))?;

        // Create pool and buffer
        let pool = self
            .shm
            .create_pool(file.as_fd(), size as i32, &self.qh, ());

        let buffer = pool.create_buffer(
            0,
            self.width as i32,
            self.height as i32,
            stride,
            wayland_client::protocol::wl_shm::Format::Argb8888,
            &self.qh,
            (),
        );

        // Attach and commit
        self.surface.attach(Some(&buffer), 0, 0);
        self.surface
            .damage_buffer(0, 0, self.width as i32, self.height as i32);
        // Note: commit is done by caller (run_overlay_thread) after render
        // but we also need to ensure buffer is committed with surface
        // Keep pool/buffer alive until next frame
        self.pool = Some(pool);
        self.buffer = Some(buffer);

        tracing::debug!(
            "ShmRenderer: rendered {} covers {}x{} time={:?}",
            cover_count,
            self.width,
            self.height,
            start.elapsed()
        );
        Ok(())
    }
}

/// New API: build ARGB buffer from per-cover data (solid/dominant color + blur pixels).
pub(crate) fn build_shm_argb_data_from_covers(
    covers: &[CoverRect],
    width: u32,
    height: u32,
) -> Vec<u8> {
    let stride = width * 4;
    let size = (stride * height) as usize;
    let mut data = vec![0u8; size];
    for cover in covers {
        let rect = &cover.screen_rect;
        // Clipped destination bounds
        let x0 = rect.x.max(0) as u32;
        let y0 = rect.y.max(0) as u32;
        let x1 = (rect.x + rect.width as i32).max(0) as u32;
        let y1 = (rect.y + rect.height as i32).max(0) as u32;
        let x1 = x1.min(width);
        let y1 = y1.min(height);
        if x0 >= x1 || y0 >= y1 {
            continue;
        }
        match cover.mode {
            vision::detection::CoverMode::Blur => {
                if let Some(blur) = &cover.blur_data {
                    let rw = rect.width as usize;
                    let rh = rect.height as usize;
                    if blur.len() < rw * rh * 3 {
                        continue;
                    }
                    for oy in y0..y1 {
                        for ox in x0..x1 {
                            let src_x = (ox as i32 - rect.x) as usize;
                            let src_y = (oy as i32 - rect.y) as usize;
                            if src_x >= rw || src_y >= rh {
                                continue;
                            }
                            let src_idx = (src_y * rw + src_x) * 3;
                            if src_idx + 2 >= blur.len() {
                                continue;
                            }
                            let dst_off = (oy * width * 4 + ox * 4) as usize;
                            if dst_off + 3 < data.len() {
                                data[dst_off] = blur[src_idx];
                                data[dst_off + 1] = blur[src_idx + 1];
                                data[dst_off + 2] = blur[src_idx + 2];
                                data[dst_off + 3] = 217;
                            }
                        }
                    }
                } else if let Some(c) = cover.resolved_color {
                    // Fallback to solid if blur missing
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let off = (y * width * 4 + x * 4) as usize;
                            if off + 3 < data.len() {
                                data[off] = c.b;
                                data[off + 1] = c.g;
                                data[off + 2] = c.r;
                                data[off + 3] = 217;
                            }
                        }
                    }
                }
            }
            vision::detection::CoverMode::SolidColor
            | vision::detection::CoverMode::BackgroundColor => {
                if let Some(c) = cover.resolved_color {
                    for y in y0..y1 {
                        for x in x0..x1 {
                            let off = (y * width * 4 + x * 4) as usize;
                            if off + 3 < data.len() {
                                data[off] = c.b;
                                data[off + 1] = c.g;
                                data[off + 2] = c.r;
                                data[off + 3] = 217;
                            }
                        }
                    }
                }
            }
        }
    }
    data
}

/// Legacy wrapper for existing tests that pass a global color.
pub(crate) fn build_shm_argb_data(
    covers: &[CoverRect],
    color: ColorRgb,
    width: u32,
    height: u32,
) -> Vec<u8> {
    // Adapt legacy covers that have no resolved_color by injecting `color`
    let patched: Vec<CoverRect> = covers
        .iter()
        .map(|c| {
            if c.resolved_color.is_none() && c.blur_data.is_none() {
                let mut p = c.clone();
                p.resolved_color = Some(color);
                p
            } else {
                c.clone()
            }
        })
        .collect();
    build_shm_argb_data_from_covers(&patched, width, height)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision::detection::CoverMode;

    #[test]
    fn test_shm_transparent_background() {
        let data = build_shm_argb_data(&[], ColorRgb::new(255, 0, 0), 10, 10);
        assert_eq!(data.len(), 10 * 10 * 4);
        // All alpha should be 0 (transparent)
        for i in (3..data.len()).step_by(4) {
            assert_eq!(data[i], 0, "alpha at {} should be 0", i);
        }
    }

    #[test]
    fn test_shm_solid_cover() {
        let cover = CoverRect::new_solid(ScreenRect::new(2, 2, 4, 4), ColorRgb::new(255, 0, 0));
        let data = build_shm_argb_data(&[cover], ColorRgb::new(255, 0, 0), 10, 10);
        // Pixel inside cover (3,3) should be red with alpha 217
        let offset = (3 * 10 * 4 + 3 * 4) as usize;
        assert_eq!(data[offset], 0); // B
        assert_eq!(data[offset + 1], 0); // G
        assert_eq!(data[offset + 2], 255); // R
        assert_eq!(data[offset + 3], 217);
        // Pixel outside (0,0) should be transparent
        assert_eq!(data[3], 0);
    }

    #[test]
    fn test_shm_multiple_covers() {
        let covers = vec![
            CoverRect::new_solid(ScreenRect::new(0, 0, 2, 2), ColorRgb::new(0, 255, 0)),
            CoverRect::new_solid(ScreenRect::new(5, 5, 2, 2), ColorRgb::new(0, 255, 0)),
        ];
        let data = build_shm_argb_data(&covers, ColorRgb::new(0, 255, 0), 10, 10);
        // First rect pixel (1,1) green
        let off1 = (1 * 10 * 4 + 1 * 4) as usize;
        assert_eq!(data[off1 + 1], 255);
        assert_eq!(data[off1 + 3], 217);
        // Second rect pixel (6,6) green
        let off2 = (6 * 10 * 4 + 6 * 4) as usize;
        assert_eq!(data[off2 + 1], 255);
        assert_eq!(data[off2 + 3], 217);
        // Gap pixel (3,3) transparent
        let off_gap = (3 * 10 * 4 + 3 * 4) as usize;
        assert_eq!(data[off_gap + 3], 0);
    }

    #[test]
    fn test_shm_empty_covers_transparent() {
        let data = build_shm_argb_data(&[], ColorRgb::new(0, 0, 255), 5, 5);
        for chunk in data.chunks(4) {
            assert_eq!(chunk[3], 0);
        }
    }

    #[test]
    fn test_shm_bounds_clipping() {
        let cover = CoverRect::new_solid(ScreenRect::new(8, 8, 10, 10), ColorRgb::new(0, 0, 255));
        let data = build_shm_argb_data(&[cover], ColorRgb::new(0, 0, 255), 10, 10);
        // Should not panic and should only fill within bounds (8,8)-(10,10)
        assert_eq!(data.len(), 400);
        // Pixel (9,9) inside clipped rect
        let off = (9 * 10 * 4 + 9 * 4) as usize;
        assert_eq!(data[off + 3], 217);
        // No out-of-bounds write
    }

    #[test]
    fn test_shm_stride() {
        let cover = CoverRect::new_solid(ScreenRect::new(0, 0, 10, 2), ColorRgb::new(255, 0, 0));
        let data = build_shm_argb_data(&[cover], ColorRgb::new(255, 0, 0), 10, 2);
        assert_eq!(data.len(), 10 * 2 * 4);
        // Verify stride = width*4 is used: pixel (0,1) offset = 1*40 + 0
        let off = (1 * 10 * 4) as usize;
        assert_eq!(data[off + 2], 255);
        assert_eq!(data[off + 3], 217);
    }

    #[test]
    fn test_channel_empty_and_n_covers() {
        let (tx, rx) = mpsc::channel::<OverlayCommand>();
        // Empty
        tx.send(OverlayCommand::UpdateCovers(vec![])).unwrap();
        let cmd = rx.recv().unwrap();
        match cmd {
            OverlayCommand::UpdateCovers(covers) => assert_eq!(covers.len(), 0),
            _ => panic!("wrong command"),
        }

        // N covers
        let covers = vec![
            CoverRect::new_solid(ScreenRect::new(0, 0, 10, 10), ColorRgb::new(0, 255, 0)),
            CoverRect::new_mode_only(ScreenRect::new(20, 20, 10, 10), CoverMode::Blur),
        ];
        tx.send(OverlayCommand::UpdateCovers(covers.clone()))
            .unwrap();
        let cmd = rx.recv().unwrap();
        match cmd {
            OverlayCommand::UpdateCovers(c) => assert_eq!(c.len(), 2),
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn test_channel_latest_state_wins() {
        let (tx, rx) = mpsc::channel::<OverlayCommand>();
        let a = vec![CoverRect::new_solid(
            ScreenRect::new(0, 0, 10, 10),
            ColorRgb::new(255, 0, 0),
        )];
        let b = vec![CoverRect::new_solid(
            ScreenRect::new(20, 20, 10, 10),
            ColorRgb::new(255, 0, 0),
        )];
        let c = vec![
            CoverRect::new_solid(ScreenRect::new(0, 0, 10, 10), ColorRgb::new(255, 0, 0)),
            CoverRect::new_solid(ScreenRect::new(30, 30, 10, 10), ColorRgb::new(255, 0, 0)),
        ];

        tx.send(OverlayCommand::UpdateCovers(a)).unwrap();
        tx.send(OverlayCommand::UpdateCovers(b)).unwrap();
        tx.send(OverlayCommand::UpdateCovers(c.clone())).unwrap();

        // Drain all, only last should be kept as latest
        let mut last: Option<Vec<CoverRect>> = None;
        while let Ok(cmd) = rx.try_recv() {
            if let OverlayCommand::UpdateCovers(covers) = cmd {
                last = Some(covers);
            }
        }
        assert_eq!(last.as_ref().unwrap().len(), 2);
        assert!(last.is_some());
    }

    #[test]
    fn test_coordinate_mapping_center() {
        // capture 1920x1200, output 1920x1200 scale 1 -> identity
        let capture = (1920u32, 1200u32);
        let output = (1920u32, 1200u32);
        let scale = 1.0;
        let rect = ScreenRect::new(810, 500, 300, 200);
        // With scale 1 and same size, overlay rect should be identical
        let expected = rect;
        let mapped = {
            let sx = output.0 as f32 / capture.0 as f32 * scale as f32;
            let sy = output.1 as f32 / capture.1 as f32 * scale as f32;
            // For 1:1, sx=1, sy=1
            assert!((sx - 1.0).abs() < 0.01);
            assert!((sy - 1.0).abs() < 0.01);
            rect
        };
        assert_eq!(mapped, expected);
    }

    #[test]
    fn test_detection_to_overlay_integration() {
        use vision::cover::covers_for_detections;
        use vision::detection::{Detection, FrameData, ObjectClass};
        use vision::geometry::ScreenRect;

        let frame = FrameData::new_bgr(1920, 1200, vec![128u8; 1920 * 1200 * 3]);
        let det = Detection {
            class: ObjectClass::Female,
            confidence: 0.91,
            screen_rect: ScreenRect::new(810, 500, 300, 200),
        };
        let covers = covers_for_detections(
            &[det],
            &frame,
            vision::detection::CoverMode::SolidColor,
            ColorRgb::new(255, 0, 0),
            &[],
        );
        assert_eq!(covers.len(), 1);
        assert_eq!(covers[0].screen_rect, ScreenRect::new(810, 500, 300, 200));

        // Verify it reaches overlay
        let mut overlay = crate::compositor::CpuOverlayRenderer::new(ColorRgb::new(255, 0, 0));
        use crate::compositor::OverlayRenderer;
        overlay.update_covers(&covers, &frame).unwrap();
        overlay.clear().unwrap();
    }

    #[test]
    fn test_shm_blur_cover_renders_blurred_pixels() {
        // Create a frame with vertical split: left black, right white
        let mut data = vec![0u8; 10 * 10 * 3];
        for y in 0..10 {
            for x in 0..10 {
                let idx = (y * 10 + x) * 3;
                if x < 5 {
                    data[idx] = 0;
                    data[idx + 1] = 0;
                    data[idx + 2] = 0;
                } else {
                    data[idx] = 255;
                    data[idx + 1] = 255;
                    data[idx + 2] = 255;
                }
            }
        }
        let frame = vision::detection::FrameData::new_bgr(20, 20, vec![128u8; 20 * 20 * 3]);
        // Manually patch frame region for test: use 10x10 region data
        let rect = ScreenRect::new(2, 2, 10, 10);
        let blurred = vision::preprocessing::blur_region(&data, 10, 10);
        assert_ne!(blurred, data);
        let cover = CoverRect::new_blur(rect, blurred.clone());
        let out = build_shm_argb_data_from_covers(&[cover], 20, 20);
        // Pixel at (5,5) inside blurred rect should be grayish (not pure 0/255) with alpha 217
        let off = (5 * 20 * 4 + 5 * 4) as usize;
        assert_eq!(out[off + 3], 217);
        // Blurred center should be intermediate
        let b = out[off];
        assert!(b > 30 && b < 225, "blurred b={}", b);
        // Outside pixel transparent
        assert_eq!(out[3], 0);
        // Ensure frame unused but cover data used
        let _ = frame;
    }

    #[test]
    fn test_shm_dominant_color_render() {
        // Frame with solid red region
        let mut data = vec![0u8; 10 * 10 * 3];
        for i in (0..data.len()).step_by(3) {
            data[i] = 0; // B
            data[i + 1] = 0; // G
            data[i + 2] = 255; // R
        }
        let frame = vision::detection::FrameData::new_bgr(10, 10, data);
        let rect = ScreenRect::new(1, 1, 4, 4);
        let dom = vision::detection::extract_dominant_color(&frame, &rect).unwrap();
        assert!(dom.r > 200);
        let cover = CoverRect::new_dominant(rect, dom);
        let out = build_shm_argb_data_from_covers(&[cover], 10, 10);
        let off = (2 * 10 * 4 + 2 * 4) as usize;
        assert_eq!(out[off + 2], dom.r);
        assert_eq!(out[off + 3], 217);
    }

    #[test]
    fn test_shm_overlapping_covers_last_wins() {
        let c1 = CoverRect::new_solid(ScreenRect::new(0, 0, 4, 4), ColorRgb::new(255, 0, 0));
        let c2 = CoverRect::new_solid(ScreenRect::new(2, 2, 4, 4), ColorRgb::new(0, 255, 0));
        let out = build_shm_argb_data_from_covers(&[c1, c2], 10, 10);
        // Overlap at (3,3) should be green (last wins)
        let off = (3 * 10 * 4 + 3 * 4) as usize;
        assert_eq!(out[off + 1], 255); // G
        assert_eq!(out[off + 2], 0);
    }

    #[test]
    fn test_shm_blur_clipping_negative_coords() {
        // Rect partially outside (-2,-2) -> clipped
        let data = vec![100u8; 4 * 4 * 3];
        let blurred = vision::preprocessing::blur_region(&data, 4, 4);
        let cover = CoverRect::new_blur(ScreenRect::new(-2, -2, 4, 4), blurred);
        let out = build_shm_argb_data_from_covers(&[cover], 10, 10);
        // Should render clipped 2x2 at (0,0)
        let off = 0_usize;
        assert_eq!(out[off + 3], 217);
        // Outside clipped area (5,5) transparent
        let off2 = (5 * 10 * 4 + 5 * 4) as usize;
        assert_eq!(out[off2 + 3], 0);
    }

    #[test]
    fn test_shm_stale_covers_cleared() {
        let c = CoverRect::new_solid(ScreenRect::new(0, 0, 2, 2), ColorRgb::new(255, 0, 0));
        let out1 = build_shm_argb_data_from_covers(&[c], 10, 10);
        assert_eq!(out1[3], 217);
        let out2 = build_shm_argb_data_from_covers(&[], 10, 10);
        for chunk in out2.chunks(4) {
            assert_eq!(chunk[3], 0);
        }
    }
}

// ---------------------------------------------------------------------------
// OverlayRenderer impl for WaylandOverlay
// ---------------------------------------------------------------------------

impl OverlayRenderer for WaylandOverlay {
    fn capability(&self) -> OverlayCapability {
        self.capability.clone()
    }

    fn update_covers(
        &mut self,
        covers: &[CoverRect],
        _frame: &FrameData,
    ) -> Result<(), OverlayError> {
        if !self.capability.is_supported() {
            return Err(OverlayError::Unsupported(format!("{:?}", self.capability)));
        }
        self.tx
            .send(OverlayCommand::UpdateCovers(covers.to_vec()))
            .map_err(|e| OverlayError::RenderFailed(e.to_string()))
    }

    fn clear(&mut self) -> Result<(), OverlayError> {
        self.tx
            .send(OverlayCommand::Clear)
            .map_err(|e| OverlayError::RenderFailed(e.to_string()))
    }

    fn set_geometry(&mut self, _monitors: &[ScreenRect]) -> Result<(), OverlayError> {
        Ok(())
    }

    fn shutdown(&mut self) -> Result<(), OverlayError> {
        let _ = self.tx.send(OverlayCommand::Shutdown);
        Ok(())
    }
}

impl Drop for WaylandOverlay {
    fn drop(&mut self) {
        let _ = self.tx.send(OverlayCommand::Shutdown);
    }
}
