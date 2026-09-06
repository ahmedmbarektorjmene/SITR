use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use pipewire as pw;
use pw::properties::properties;
use pw::spa;
use pw::spa::pod::Pod;
use vision::detection::{FrameData, PixelFormat};

#[derive(Debug, thiserror::Error)]
pub enum LinuxCaptureError {
    #[error("Portal unavailable: {0}")]
    PortalUnavailable(String),
    #[error("Screen capture denied by user")]
    ScreenCaptureDenied,
    #[error("PipeWire unavailable: {0}")]
    PipeWireUnavailable(String),
    #[error("PipeWire connection failed: {0}")]
    PipeWireConnectionFailed(String),
    #[error("PipeWire stream failed: {0}")]
    PipeWireStreamFailed(String),
    #[error("Stream negotiation failed: {0}")]
    StreamNegotiationFailed(String),
    #[error("Unsupported pixel format: {0}")]
    UnsupportedPixelFormat(String),
    #[error("Capture disconnected")]
    CaptureDisconnected,
    #[error("No frames available")]
    NoFrames,
    #[error("Frame timeout")]
    FrameTimeout,
}

#[derive(Debug, Clone)]
pub struct StreamInfo {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub format: SpaVideoFormatTag,
    pub framerate_num: u32,
    pub framerate_denom: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpaVideoFormatTag {
    Bgrx,
    Rgbx,
    Bgra,
    Rgba,
    Bgr,
    Rgb,
    Xrgb,
    Xbgr,
    Argb,
    Abgr,
    Other(u32),
}

impl SpaVideoFormatTag {
    fn from_raw(raw: u32) -> Self {
        let fmt = spa::param::video::VideoFormat::from_raw(raw);
        match fmt {
            f if f == spa::param::video::VideoFormat::BGRx => Self::Bgrx,
            f if f == spa::param::video::VideoFormat::RGBx => Self::Rgbx,
            f if f == spa::param::video::VideoFormat::BGRA => Self::Bgra,
            f if f == spa::param::video::VideoFormat::RGBA => Self::Rgba,
            f if f == spa::param::video::VideoFormat::BGR => Self::Bgr,
            f if f == spa::param::video::VideoFormat::RGB => Self::Rgb,
            f if f == spa::param::video::VideoFormat::xRGB => Self::Xrgb,
            f if f == spa::param::video::VideoFormat::xBGR => Self::Xbgr,
            f if f == spa::param::video::VideoFormat::ARGB => Self::Argb,
            f if f == spa::param::video::VideoFormat::ABGR => Self::Abgr,
            _ => Self::Other(raw),
        }
    }

    fn bytes_per_pixel(self) -> u32 {
        match self {
            Self::Bgr | Self::Rgb => 3,
            Self::Bgrx
            | Self::Rgbx
            | Self::Bgra
            | Self::Rgba
            | Self::Xrgb
            | Self::Xbgr
            | Self::Argb
            | Self::Abgr => 4,
            Self::Other(_) => 0,
        }
    }

    fn is_supported(self) -> bool {
        self.bytes_per_pixel() > 0
    }
}

struct LatestFrame {
    frame: FrameData,
    stream_info: StreamInfo,
    #[allow(dead_code)]
    timestamp: Instant,
}

struct CaptureState {
    latest: Mutex<Option<LatestFrame>>,
    /// Recycled BGR pixel buffer (capacity reused across frames to avoid ~6–7 MiB allocs).
    recycle: Mutex<Vec<u8>>,
    condvar: Condvar,
    stream_info: Mutex<Option<StreamInfo>>,
    running: AtomicBool,
    frames_received: std::sync::atomic::AtomicU64,
    /// Last convert_to_bgr duration in nanoseconds (for PERF logging).
    last_convert_ns: std::sync::atomic::AtomicU64,
}

struct PipeWireUserData {
    state: Arc<CaptureState>,
    video_info: spa::param::video::VideoInfoRaw,
    format_negotiated: bool,
}

pub struct LinuxScreenCapturer {
    state: Arc<CaptureState>,
    thread_handle: Option<JoinHandle<()>>,
}

impl LinuxScreenCapturer {
    pub fn new() -> Self {
        let state = Arc::new(CaptureState {
            latest: Mutex::new(None),
            recycle: Mutex::new(Vec::new()),
            condvar: Condvar::new(),
            stream_info: Mutex::new(None),
            running: AtomicBool::new(true),
            frames_received: std::sync::atomic::AtomicU64::new(0),
            last_convert_ns: std::sync::atomic::AtomicU64::new(0),
        });

        let state_clone = Arc::clone(&state);
        let handle = thread::Builder::new()
            .name("porda-pw-capture".to_string())
            .spawn(move || {
                run_pipewire_capture_thread(state_clone);
            })
            .expect("Failed to spawn PipeWire capture thread");

        tracing::info!("PipeWire capture thread spawned");

        Self {
            state,
            thread_handle: Some(handle),
        }
    }

    pub fn capture(&self) -> Result<(FrameData, StreamInfo), LinuxCaptureError> {
        if !self.state.running.load(Ordering::Relaxed) {
            return Err(LinuxCaptureError::CaptureDisconnected);
        }

        let mut guard = self.state.latest.lock().unwrap();

        // Take ownership instead of cloning ~6–7 MiB BGR every poll.
        if let Some(latest) = guard.take() {
            return Ok((latest.frame, latest.stream_info));
        }

        let (mut guard, timeout) = self
            .state
            .condvar
            .wait_timeout(guard, Duration::from_millis(1000))
            .unwrap();

        if let Some(latest) = guard.take() {
            Ok((latest.frame, latest.stream_info))
        } else if timeout.timed_out() {
            Err(LinuxCaptureError::FrameTimeout)
        } else {
            Err(LinuxCaptureError::NoFrames)
        }
    }

    /// Last PipeWire buffer copy time in milliseconds (raw BGRx memcpy; no BGR convert).
    pub fn last_convert_ms(&self) -> f64 {
        self.state.last_convert_ns.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    pub fn is_running(&self) -> bool {
        self.state.running.load(Ordering::Relaxed)
    }

    pub fn frames_received(&self) -> u64 {
        self.state.frames_received.load(Ordering::Relaxed)
    }

    pub fn stream_info(&self) -> Option<StreamInfo> {
        self.state.stream_info.lock().unwrap().clone()
    }
}

impl Drop for LinuxScreenCapturer {
    fn drop(&mut self) {
        tracing::info!("Shutting down PipeWire capture");
        self.state.running.store(false, Ordering::Relaxed);
        self.state.condvar.notify_all();

        if let Some(handle) = self.thread_handle.take() {
            let _ = handle.join();
        }
        tracing::info!("PipeWire capture thread joined");
    }
}

fn run_pipewire_capture_thread(state: Arc<CaptureState>) {
    tracing::info!("PipeWire capture thread starting");

    let portal_result = match request_portal_screen_cast() {
        Ok(result) => result,
        Err(e) => {
            tracing::error!("Portal screen cast failed: {}", e);
            state.running.store(false, Ordering::Relaxed);
            state.condvar.notify_all();
            return;
        }
    };

    tracing::info!(
        "Portal session established: node_id={}",
        portal_result.node_id
    );

    if let Err(e) = connect_pipewire_and_run(state.clone(), portal_result) {
        tracing::error!("PipeWire capture failed: {}", e);
        state.running.store(false, Ordering::Relaxed);
        state.condvar.notify_all();
    }
}

struct PortalResult {
    node_id: u32,
    fd: std::os::fd::OwnedFd,
}

fn request_portal_screen_cast() -> Result<PortalResult, LinuxCaptureError> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| LinuxCaptureError::PortalUnavailable(e.to_string()))?;

    rt.block_on(async {
        use ashpd::desktop::screencast::{
            CursorMode, Screencast, SelectSourcesOptions, SourceType,
        };
        use ashpd::desktop::PersistMode;

        let proxy = Screencast::new()
            .await
            .map_err(|e| LinuxCaptureError::PortalUnavailable(e.to_string()))?;

        tracing::info!("Portal Screencast proxy connected");

        let session = proxy
            .create_session(Default::default())
            .await
            .map_err(|e| {
                if is_cancelled(&e) {
                    LinuxCaptureError::ScreenCaptureDenied
                } else {
                    LinuxCaptureError::PortalUnavailable(e.to_string())
                }
            })?;

        let source_types = ashpd::enumflags2::BitFlags::from_flag(SourceType::Monitor);
        proxy
            .select_sources(
                &session,
                SelectSourcesOptions::default()
                    .set_cursor_mode(CursorMode::Metadata)
                    .set_sources(Some(source_types))
                    .set_multiple(false)
                    .set_persist_mode(PersistMode::DoNot),
            )
            .await
            .map_err(|e| {
                if is_cancelled(&e) {
                    LinuxCaptureError::ScreenCaptureDenied
                } else {
                    LinuxCaptureError::PortalUnavailable(e.to_string())
                }
            })?;

        tracing::info!("Source selection submitted, waiting for user response");

        let response = proxy
            .start(&session, None, Default::default())
            .await
            .map_err(|e| {
                if is_cancelled(&e) {
                    LinuxCaptureError::ScreenCaptureDenied
                } else {
                    LinuxCaptureError::PortalUnavailable(e.to_string())
                }
            })?
            .response()
            .map_err(|e| {
                if is_cancelled(&e) {
                    LinuxCaptureError::ScreenCaptureDenied
                } else {
                    LinuxCaptureError::PortalUnavailable(e.to_string())
                }
            })?;

        let streams = response.streams();
        if streams.is_empty() {
            return Err(LinuxCaptureError::StreamNegotiationFailed(
                "No streams returned by portal".to_string(),
            ));
        }

        let first_stream = &streams[0];
        let node_id = first_stream.pipe_wire_node_id();

        tracing::info!(
            "Portal stream: node_id={}, position={:?}, size={:?}, source_type={:?}",
            node_id,
            first_stream.position(),
            first_stream.size(),
            first_stream.source_type()
        );

        let fd = proxy
            .open_pipe_wire_remote(&session, Default::default())
            .await
            .map_err(|e| LinuxCaptureError::PipeWireConnectionFailed(e.to_string()))?;

        Ok(PortalResult { node_id, fd })
    })
}

fn is_cancelled(e: &ashpd::Error) -> bool {
    let msg = e.to_string();
    msg.contains("ancelled") || msg.contains("Canceled")
}

fn connect_pipewire_and_run(
    state: Arc<CaptureState>,
    portal: PortalResult,
) -> Result<(), LinuxCaptureError> {
    pw::init();

    let mainloop = pw::main_loop::MainLoopRc::new(None).map_err(|e| {
        LinuxCaptureError::PipeWireUnavailable(format!("MainLoop creation failed: {}", e))
    })?;

    let context = pw::context::ContextRc::new(&mainloop, None).map_err(|e| {
        LinuxCaptureError::PipeWireUnavailable(format!("Context creation failed: {}", e))
    })?;

    let core = context.connect_fd_rc(portal.fd, None).map_err(|e| {
        LinuxCaptureError::PipeWireConnectionFailed(format!("connect_fd failed: {}", e))
    })?;

    tracing::info!("PipeWire core connected via portal fd");

    let mut user_data = PipeWireUserData {
        state: Arc::clone(&state),
        video_info: spa::param::video::VideoInfoRaw::new(),
        format_negotiated: false,
    };

    let stream = pw::stream::StreamRc::new(
        core,
        "capture",
        properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Screen",
        },
    )
    .map_err(|e| {
        LinuxCaptureError::PipeWireStreamFailed(format!("Stream creation failed: {}", e))
    })?;

    let _listener = stream
        .add_local_listener_with_user_data(&mut user_data)
        .state_changed(|_stream, user_data, old, new| {
            tracing::debug!("PipeWire stream state: {:?} -> {:?}", old, new);
            match new {
                pw::stream::StreamState::Error(msg) => {
                    tracing::error!("PipeWire stream error: {}", msg);
                    user_data.state.running.store(false, Ordering::Relaxed);
                    user_data.state.condvar.notify_all();
                }
                pw::stream::StreamState::Streaming => {
                    tracing::info!("PipeWire stream is now streaming");
                }
                _ => {}
            }
        })
        .param_changed(|_stream, user_data, id, param| {
            if id != spa::param::ParamType::Format.as_raw() {
                return;
            }
            let Some(param) = param else {
                return;
            };

            let (media_type, media_subtype) = match spa::param::format_utils::parse_format(param) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!("Failed to parse format: {:?}", e);
                    return;
                }
            };

            if media_type != spa::param::format::MediaType::Video
                || media_subtype != spa::param::format::MediaSubtype::Raw
            {
                tracing::warn!(
                    "Unexpected media type/subtype: {:?}/{:?}",
                    media_type,
                    media_subtype
                );
                return;
            }

            if let Err(e) = user_data.video_info.parse(param) {
                tracing::warn!("Failed to parse video format: {:?}", e);
                return;
            }

            let format_tag = SpaVideoFormatTag::from_raw(user_data.video_info.format().as_raw());
            let size = user_data.video_info.size();
            let framerate = user_data.video_info.framerate();

            if !format_tag.is_supported() {
                tracing::error!(
                    "Unsupported pixel format: {:?} (raw={})",
                    format_tag,
                    user_data.video_info.format().as_raw()
                );
                return;
            }

            let bpp = format_tag.bytes_per_pixel();
            let tight_stride = size.width * bpp;

            let stream_info = StreamInfo {
                width: size.width,
                height: size.height,
                stride: tight_stride,
                format: format_tag,
                framerate_num: framerate.num,
                framerate_denom: framerate.denom,
            };

            tracing::info!(
                "PipeWire format negotiated: {}x{}, format={:?}, bpp={}, stride={}, framerate={}/{}",
                stream_info.width,
                stream_info.height,
                stream_info.format,
                bpp,
                stream_info.stride,
                stream_info.framerate_num,
                stream_info.framerate_denom
            );

            *user_data.state.stream_info.lock().unwrap() = Some(stream_info);
            user_data.format_negotiated = true;
        })
        .process(|stream, user_data| {
            if !user_data.format_negotiated {
                return;
            }

            let Some(mut buffer) = stream.dequeue_buffer() else {
                return;
            };

            let datas = buffer.datas_mut();
            if datas.is_empty() {
                return;
            }

            let data = &mut datas[0];

            let chunk_size = data.chunk().size() as usize;
            let chunk_stride = data.chunk().stride();

            if chunk_size == 0 {
                return;
            }

            let Some(src_slice) = data.data() else {
                return;
            };

            let si = match user_data.state.stream_info.lock().unwrap().clone() {
                Some(si) => si,
                None => return,
            };

            let bpp = si.format.bytes_per_pixel();
            if bpp == 0 {
                return;
            }

            let actual_stride = if chunk_stride > 0 {
                chunk_stride as u32
            } else {
                si.width * bpp
            };

            let required_size = (actual_stride * si.height) as usize;
            if chunk_size < required_size {
                tracing::warn!(
                    "Buffer too small: chunk_size={}, required={}, stride={}, h={}",
                    chunk_size,
                    required_size,
                    actual_stride,
                    si.height
                );
                return;
            }

            // Raw copy of PipeWire-mapped pixels (BGRx/etc). No full-frame BGR convert —
            // detector downsamples BGRX→NCHW directly. Memcpy of ~9MiB ≪ 73ms pixel shuffle.
            let mut raw_buf = {
                let mut latest = user_data.state.latest.lock().unwrap();
                if let Some(old) = latest.take() {
                    old.frame.data
                } else {
                    drop(latest);
                    let mut recycle = user_data.state.recycle.lock().unwrap();
                    std::mem::take(&mut *recycle)
                }
            };

            let t_copy = Instant::now();
            if raw_buf.len() != required_size {
                raw_buf.resize(required_size, 0);
            }
            raw_buf[..required_size].copy_from_slice(&src_slice[..required_size]);
            let copy_ns = t_copy.elapsed().as_nanos() as u64;
            user_data
                .state
                .last_convert_ns
                .store(copy_ns, Ordering::Relaxed);

            let pixel_format = spa_to_pixel_format(si.format);
            let frame = FrameData::new_with_stride(
                si.width,
                si.height,
                actual_stride,
                raw_buf,
                pixel_format,
            );

            let new_frame = LatestFrame {
                frame,
                stream_info: si,
                timestamp: Instant::now(),
            };

            {
                let mut latest = user_data.state.latest.lock().unwrap();
                if let Some(old) = latest.replace(new_frame) {
                    let mut recycle = user_data.state.recycle.lock().unwrap();
                    *recycle = old.frame.data;
                }
            }
            user_data.state.frames_received.fetch_add(1, Ordering::Relaxed);
            user_data.state.condvar.notify_one();
        })
        .register()
        .map_err(|e| LinuxCaptureError::PipeWireStreamFailed(format!("Listener registration failed: {}", e)))?;

    let obj = pw::spa::pod::object!(
        spa::utils::SpaTypes::ObjectParamFormat,
        spa::param::ParamType::EnumFormat,
        spa::pod::property!(
            spa::param::format::FormatProperties::MediaType,
            Id,
            spa::param::format::MediaType::Video
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::MediaSubtype,
            Id,
            spa::param::format::MediaSubtype::Raw
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::VideoFormat,
            Choice,
            Enum,
            Id,
            spa::param::video::VideoFormat::BGRx,
            spa::param::video::VideoFormat::BGRx,
            spa::param::video::VideoFormat::RGBx,
            spa::param::video::VideoFormat::BGRA,
            spa::param::video::VideoFormat::RGBA,
            spa::param::video::VideoFormat::BGR,
            spa::param::video::VideoFormat::RGB,
            spa::param::video::VideoFormat::xRGB,
            spa::param::video::VideoFormat::xBGR
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::VideoSize,
            Choice,
            Range,
            Rectangle,
            spa::utils::Rectangle {
                width: 1920,
                height: 1080
            },
            spa::utils::Rectangle {
                width: 1,
                height: 1
            },
            spa::utils::Rectangle {
                width: 8192,
                height: 8192
            }
        ),
        spa::pod::property!(
            spa::param::format::FormatProperties::VideoFramerate,
            Choice,
            Range,
            Fraction,
            spa::utils::Fraction { num: 30, denom: 1 },
            spa::utils::Fraction { num: 0, denom: 1 },
            spa::utils::Fraction { num: 120, denom: 1 }
        ),
    );

    let values: Vec<u8> = spa::pod::serialize::PodSerializer::serialize(
        std::io::Cursor::new(Vec::new()),
        &spa::pod::Value::Object(obj),
    )
    .map_err(|e| {
        LinuxCaptureError::PipeWireStreamFailed(format!("Pod serialization failed: {}", e))
    })?
    .0
    .into_inner();

    let mut params = [Pod::from_bytes(&values).ok_or_else(|| {
        LinuxCaptureError::PipeWireStreamFailed("Failed to create format pod".to_string())
    })?];

    stream
        .connect(
            spa::utils::Direction::Input,
            Some(portal.node_id),
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
            &mut params,
        )
        .map_err(|e| {
            LinuxCaptureError::PipeWireStreamFailed(format!("Stream connect failed: {}", e))
        })?;

    tracing::info!(
        "PipeWire stream connected to node {}, waiting for frames...",
        portal.node_id
    );

    let mut last_log = Instant::now();
    let mut total_frames: u64 = 0;

    while state.running.load(Ordering::Relaxed) {
        mainloop.loop_().iterate(Duration::from_millis(100));

        let count = state.frames_received.load(Ordering::Relaxed);
        if count > total_frames {
            total_frames = count;
        }

        if last_log.elapsed() >= Duration::from_secs(5) {
            if let Some(ref si) = *state.stream_info.lock().unwrap() {
                tracing::info!(
                    "PipeWire capture: frames_total={}, resolution={}x{}, format={:?}, stride={}",
                    total_frames,
                    si.width,
                    si.height,
                    si.format,
                    si.stride
                );
            }
            last_log = Instant::now();
        }
    }

    tracing::info!("PipeWire capture thread exiting cleanly");
    Ok(())
}

fn spa_to_pixel_format(fmt: SpaVideoFormatTag) -> PixelFormat {
    match fmt {
        SpaVideoFormatTag::Bgrx
        | SpaVideoFormatTag::Bgra
        | SpaVideoFormatTag::Xbgr
        | SpaVideoFormatTag::Abgr => PixelFormat::Bgra,
        SpaVideoFormatTag::Rgbx
        | SpaVideoFormatTag::Rgba
        | SpaVideoFormatTag::Xrgb
        | SpaVideoFormatTag::Argb => PixelFormat::Rgba,
        SpaVideoFormatTag::Bgr => PixelFormat::Bgr,
        SpaVideoFormatTag::Rgb => PixelFormat::Rgb,
        SpaVideoFormatTag::Other(_) => PixelFormat::Bgra,
    }
}

/// Fallback path for exotic layouts that need channel reordering into packed BGR.
#[allow(dead_code)]
fn convert_to_bgr_into(src: &[u8], si: &StreamInfo, actual_stride: u32, bgr: &mut Vec<u8>) {
    let needed = (si.width * si.height * 3) as usize;
    if bgr.len() != needed {
        bgr.resize(needed, 0);
    }

    let bpp = si.format.bytes_per_pixel() as usize;
    let dst_stride = (si.width * 3) as usize;

    for y in 0..si.height as usize {
        let src_row_start = y * actual_stride as usize;
        let dst_row_start = y * dst_stride;

        for x in 0..si.width as usize {
            let src_px = src_row_start + x * bpp;
            let dst_px = dst_row_start + x * 3;

            if src_px + 3 >= src.len() || dst_px + 2 >= bgr.len() {
                continue;
            }

            match si.format {
                SpaVideoFormatTag::Bgrx | SpaVideoFormatTag::Bgra => {
                    bgr[dst_px] = src[src_px];
                    bgr[dst_px + 1] = src[src_px + 1];
                    bgr[dst_px + 2] = src[src_px + 2];
                }
                SpaVideoFormatTag::Rgbx | SpaVideoFormatTag::Rgba => {
                    bgr[dst_px] = src[src_px + 2];
                    bgr[dst_px + 1] = src[src_px + 1];
                    bgr[dst_px + 2] = src[src_px];
                }
                SpaVideoFormatTag::Bgr => {
                    bgr[dst_px] = src[src_px];
                    bgr[dst_px + 1] = src[src_px + 1];
                    bgr[dst_px + 2] = src[src_px + 2];
                }
                SpaVideoFormatTag::Rgb => {
                    bgr[dst_px] = src[src_px + 2];
                    bgr[dst_px + 1] = src[src_px + 1];
                    bgr[dst_px + 2] = src[src_px];
                }
                SpaVideoFormatTag::Xrgb | SpaVideoFormatTag::Argb => {
                    bgr[dst_px] = src[src_px + 3];
                    bgr[dst_px + 1] = src[src_px + 2];
                    bgr[dst_px + 2] = src[src_px + 1];
                }
                SpaVideoFormatTag::Xbgr | SpaVideoFormatTag::Abgr => {
                    bgr[dst_px] = src[src_px + 1];
                    bgr[dst_px + 1] = src[src_px + 2];
                    bgr[dst_px + 2] = src[src_px + 3];
                }
                SpaVideoFormatTag::Other(_) => {
                    bgr[dst_px] = src[src_px];
                    bgr[dst_px + 1] = src[src_px + 1];
                    bgr[dst_px + 2] = src[src_px + 2];
                }
            }
        }
    }
}
