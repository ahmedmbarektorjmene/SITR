use ort::ep;
use ort::inputs;
use ort::logging::LogLevel;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::ops::Deref;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};
use vision::detection::{Detection, FrameData, ObjectClass, PixelFormat};
use vision::geometry::ScreenRect;

use crate::yolo::{decode_heads, filter_and_nms, YoloCandidate};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of output tensors/heads supported without a heap allocation
/// in the per-frame inference path.
///
/// This is NOT the number of heads in the model.
/// The actual model output count is checked at runtime.
const MAX_OUTPUTS: usize = 16;

/// Reasonable initial capacity for decoded YOLO candidates.
///
/// This is only an initial allocation during detector construction.
/// The Vec is reused across frames.
const INITIAL_CANDIDATE_CAPACITY: usize = 1024;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Inference failed: {0}")]
    Failed(String),

    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),

    #[error("Input shape mismatch: got {got} elements, expected {expected}")]
    ShapeMismatch { got: usize, expected: usize },

    #[error("Model has {count} outputs, but maximum supported is {max}")]
    TooManyOutputs { count: usize, max: usize },
}

// ---------------------------------------------------------------------------
// Per-frame timing breakdown
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default)]
pub struct DetectTimings {
    /// Fused native-format → NCHW f32
    /// (downsample + channel swap + /255)
    pub preprocess_ms: f64,

    /// Tensor/view/binding setup.
    pub tensor_view_ms: f64,

    /// `session.run` / `run_binding`.
    pub session_run_ms: f64,

    /// Output tensor view extraction.
    pub output_extract_ms: f64,

    /// YOLO decode.
    pub decode_ms: f64,

    /// Sorting step inside NMS.
    pub sort_ms: f64,

    /// NMS suppression.
    pub nms_ms: f64,

    /// Candidate → Detection conversion.
    pub final_conv_ms: f64,

    /// Aggregate post-processing.
    pub postprocess_ms: f64,

    /// Entire detect call.
    pub total_ms: f64,
}

impl DetectTimings {
    #[inline]
    fn ms(d: std::time::Duration) -> f64 {
        d.as_secs_f64() * 1000.0
    }
}

fn profile_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

    *CACHED.get_or_init(|| match std::env::var("PORDA_PROFILE") {
        Ok(v) => v != "0" && !v.eq_ignore_ascii_case("false"),
        Err(_) => true,
    })
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Head<'a> = (&'a [f32], u32, u32);

// ---------------------------------------------------------------------------
// ORT environment
// ---------------------------------------------------------------------------

/// Initialize ORT environment exactly once.
fn ensure_ort_env() {
    static INIT: std::sync::Once = std::sync::Once::new();

    INIT.call_once(|| {
        let verbose = std::env::var("PORDA_ORT_VERBOSE")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true);

        let committed = ort::init()
            .with_name("porda")
            .with_telemetry(false)
            .with_logger(Arc::new(
                move |level: LogLevel,
                      category: &str,
                      id: &str,
                      code_location: &str,
                      message: &str| {
                    match level {
                        LogLevel::Verbose if verbose => {
                            tracing::debug!(
                                target: "ort",
                                category,
                                id,
                                code_location,
                                "{message}"
                            );
                        }

                        LogLevel::Info => {
                            tracing::info!(
                                target: "ort",
                                category,
                                id,
                                code_location,
                                "{message}"
                            );
                        }

                        LogLevel::Warning => {
                            tracing::warn!(
                                target: "ort",
                                category,
                                id,
                                code_location,
                                "{message}"
                            );
                        }

                        LogLevel::Error | LogLevel::Fatal => {
                            tracing::error!(
                                target: "ort",
                                category,
                                id,
                                code_location,
                                "{message}"
                            );
                        }

                        LogLevel::Verbose => {}
                    }
                },
            ))
            .commit();

        if committed {
            info!(
                "OrtDetector: ORT environment committed \
                 (verbose WebGPU registration logs enabled)"
            );
        } else {
            warn!(
                "OrtDetector: ORT environment already committed elsewhere; \
                 logger may not apply"
            );
        }
    });
}

// ---------------------------------------------------------------------------
// Detector trait
// ---------------------------------------------------------------------------

pub trait Detector: Send {
    /// Detect objects and write them into `detections`.
    ///
    /// `detections` is caller-owned and should be reused between frames.
    /// The implementation clears it but does not free its capacity.
    fn detect(
        &mut self,
        frame: &FrameData,
        confidence_threshold: f32,
        nms_threshold: f32,
        target_classes: &[i32],
        network_width: u32,
        network_height: u32,
        screen_rect: &ScreenRect,
        detections: &mut Vec<Detection>,
    ) -> Result<(), InferenceError>;

    fn backend_name(&self) -> &str;

    fn last_timings(&self) -> DetectTimings {
        DetectTimings::default()
    }
}

// ---------------------------------------------------------------------------
// Mock detector
// ---------------------------------------------------------------------------

pub struct MockDetector;

impl Detector for MockDetector {
    fn detect(
        &mut self,
        frame: &FrameData,
        confidence_threshold: f32,
        _nms_threshold: f32,
        target_classes: &[i32],
        _network_width: u32,
        _network_height: u32,
        _screen_rect: &ScreenRect,
        detections: &mut Vec<Detection>,
    ) -> Result<(), InferenceError> {
        detections.clear();

        tracing::info!(
            "MockDetector: frame {}x{} target_classes={:?} conf_thresh={:.2}",
            frame.width,
            frame.height,
            target_classes,
            confidence_threshold
        );

        if std::env::var("PORDA_MOCK_DETECTIONS").is_ok()
            && target_classes.contains(&1)
            && confidence_threshold <= 0.9
        {
            let w = (frame.width / 4).min(300);
            let h = (frame.height / 4).min(200);

            let x = (frame.width as i32 / 2) - (w as i32 / 2);
            let y = (frame.height as i32 / 2) - (h as i32 / 2);

            detections.push(Detection {
                class: ObjectClass::Female,
                confidence: 0.91,
                screen_rect: ScreenRect::new(x, y, w, h),
            });
        }

        Ok(())
    }

    fn backend_name(&self) -> &str {
        "mock"
    }
}

// ---------------------------------------------------------------------------
// Execution provider
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    WebGpu,
    Cpu,
}

// ---------------------------------------------------------------------------
// OrtDetector
// ---------------------------------------------------------------------------

pub struct OrtDetector {
    session: Session,

    exec_provider: ExecutionProvider,

    input_name: String,
    output_names: Vec<String>,

    network_width: u32,
    network_height: u32,

    /// Reusable CPU input buffer.
    preprocess_buffer: Vec<f32>,

    last_timings: DetectTimings,

    /// Reusable YOLO candidate buffer.
    candidate_buf: Vec<YoloCandidate>,

    /// Reusable NMS mask.
    keep_buf: Vec<bool>,
}

impl std::fmt::Debug for OrtDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtDetector")
            .field("exec_provider", &self.exec_provider)
            .field("input_name", &self.input_name)
            .field("output_names", &self.output_names)
            .field("network_width", &self.network_width)
            .field("network_height", &self.network_height)
            .field("backend", &self.exec_provider)
            .finish()
    }
}

impl OrtDetector {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    pub fn new(onnx_path: &std::path::Path) -> Result<Self, InferenceError> {
        ensure_ort_env();

        let (session, exec_provider) =
            Self::build_session_with_fallback(onnx_path).map_err(|e| {
                InferenceError::BackendNotAvailable(format!("all EP builds failed: {e}"))
            })?;

        let input_name = Self::first_input_name(&session);
        let output_names = Self::output_names(&session);
        let input_shape = Self::extract_input_dims(&session);

        let (net_w, net_h) = {
            let mut w = 544u32;
            let mut h = 320u32;

            if let Some((c, ch, cw)) = input_shape {
                if c == 3 {
                    w = cw as u32;
                    h = ch as u32;
                }
            }

            (w, h)
        };

        let input_capacity = 3usize * net_w as usize * net_h as usize;

        // Do NOT use input pixel count as candidate capacity.
        //
        // Candidate count is determined by model output grids, not by
        // input resolution.
        let candidate_buf = Vec::with_capacity(INITIAL_CANDIDATE_CAPACITY);

        let keep_buf = Vec::with_capacity(INITIAL_CANDIDATE_CAPACITY);

        info!(
            "OrtDetector: loaded EP={:?} input={} \
             outputs={:?} static_input_shape={:?}",
            exec_provider, input_name, output_names, input_shape
        );

        Ok(Self {
            session,
            exec_provider,
            input_name,
            output_names,
            network_width: net_w,
            network_height: net_h,

            preprocess_buffer: vec![0.0f32; input_capacity],

            last_timings: DetectTimings::default(),

            candidate_buf,
            keep_buf,
        })
    }

    // -----------------------------------------------------------------------
    // Session construction
    // -----------------------------------------------------------------------

    fn build_session_with_fallback(
        onnx_path: &std::path::Path,
    ) -> Result<(Session, ExecutionProvider), String> {
        match Self::try_webgpu_session(onnx_path) {
            Ok(session) => {
                info!("OrtDetector: WebGPU Execution Provider ACTIVE");
                Ok((session, ExecutionProvider::WebGpu))
            }

            Err(e) => {
                warn!("OrtDetector: WebGPU unavailable ({e}), falling back to CPU");

                Self::try_cpu_session(onnx_path).map(|session| (session, ExecutionProvider::Cpu))
            }
        }
    }

    fn commit_session(
        onnx_path: &std::path::Path,
        providers: &[ep::ExecutionProviderDispatch],
        intra: usize,
        inter: usize,
        parallel: bool,
    ) -> Result<Session, String> {
        let builder =
            Session::builder().map_err(|e| format!("session builder init failed: {e}"))?;

        let builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("set graph optimisation level failed: {e}"))?;

        let builder = builder
            .with_intra_threads(intra)
            .map_err(|e| format!("set intra threads failed: {e}"))?;

        let builder = builder
            .with_inter_threads(inter)
            .map_err(|e| format!("set inter threads failed: {e}"))?;

        let builder = builder
            .with_parallel_execution(parallel)
            .map_err(|e| format!("set parallel execution failed: {e}"))?;

        let mut builder = builder
            .with_execution_providers(providers)
            .map_err(|e| format!("set execution providers failed: {e}"))?;

        let model_bytes =
            std::fs::read(onnx_path).map_err(|e| format!("read model bytes failed: {e}"))?;

        builder
            .commit_from_memory(&model_bytes)
            .map_err(|e| format!("commit session failed: {e}"))
    }

    fn try_webgpu_session(onnx_path: &std::path::Path) -> Result<Session, String> {
        let webgpu = ep::WebGPU::default()
            .with_device_id(0)
            .with_enable_graph_capture(true)
            .build()
            .error_on_failure();

        Self::commit_session(onnx_path, &[webgpu], 2, 1, false)
    }

    fn try_cpu_session(onnx_path: &std::path::Path) -> Result<Session, String> {
        let cpu = ep::CPU::default().with_arena_allocator(true).build();

        Self::commit_session(onnx_path, &[cpu], 2, 1, false)
    }

    // -----------------------------------------------------------------------
    // Cached I/O names
    // -----------------------------------------------------------------------

    fn first_input_name(session: &Session) -> String {
        let inputs = session.inputs();

        if inputs.is_empty() {
            return "input".to_string();
        }

        inputs
            .first()
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| "input".to_string())
    }

    fn output_names(session: &Session) -> Vec<String> {
        session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect()
    }

    fn extract_input_dims(session: &Session) -> Option<(usize, usize, usize)> {
        let inputs = session.inputs();
        let first = inputs.first()?;

        let shape = first.dtype().tensor_shape()?;
        let dims = shape.deref();

        if dims.len() >= 4 && dims[0] == 1i64 {
            Some((dims[1] as usize, dims[2] as usize, dims[3] as usize))
        } else {
            None
        }
    }

    // -----------------------------------------------------------------------
    // Preprocessing
    // -----------------------------------------------------------------------

    /// Fused nearest-neighbour downsample + channel reorder + /255
    /// directly into NCHW.
    ///
    /// No intermediate image buffer is created.
    #[inline]
    fn preprocess_frame_to_nchw(
        frame: &FrameData,
        out: &mut [f32],
        net_w: usize,
        net_h: usize,
    ) -> Result<(), InferenceError> {
        let expected = net_w * net_h * 3;

        if out.len() < expected {
            return Err(InferenceError::ShapeMismatch {
                got: out.len(),
                expected,
            });
        }

        let src = &frame.data;

        let bpp = frame.format.bytes_per_pixel() as usize;

        let fw = frame.width as usize;

        let fh = frame.height as usize;

        let stride = if frame.stride as usize > 0 {
            frame.stride as usize
        } else {
            fw * bpp
        };

        let (ri, gi, bi) = match frame.format {
            PixelFormat::Bgr | PixelFormat::Bgra => (2usize, 1usize, 0usize),

            PixelFormat::Rgb | PixelFormat::Rgba => (0usize, 1usize, 2usize),
        };

        let plane = net_w * net_h;

        let inv_255 = 1.0f32 / 255.0;

        let last = src.len().saturating_sub(1);

        for y in 0..net_h {
            let sy = (y * fh) / net_h;

            let row = sy * stride;

            let dst_row = y * net_w;

            for x in 0..net_w {
                let sx = (x * fw) / net_w;

                let ssi = row + sx * bpp;

                if ssi + bi.max(ri).max(gi) > last {
                    continue;
                }

                let r = src[ssi + ri] as f32 * inv_255;

                let g = src[ssi + gi] as f32 * inv_255;

                let b = src[ssi + bi] as f32 * inv_255;

                let i = dst_row + x;

                out[i] = r;

                out[plane + i] = g;

                out[2 * plane + i] = b;
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Coordinate mapping
    // -----------------------------------------------------------------------

    #[inline]
    fn network_rect_to_screen(
        r: &ScreenRect,
        screen: &ScreenRect,
        net_w: u32,
        net_h: u32,
    ) -> ScreenRect {
        let sx = screen.width as f32 / net_w as f32;

        let sy = screen.height as f32 / net_h as f32;

        ScreenRect::new(
            (screen.x as f32 + r.x as f32 * sx) as i32,
            (screen.y as f32 + r.y as f32 * sy) as i32,
            (r.width as f32 * sx) as u32,
            (r.height as f32 * sy) as u32,
        )
    }

    #[inline]
    fn candidate_to_detection(
        c: &YoloCandidate,
        screen: &ScreenRect,
        net_w: u32,
        net_h: u32,
    ) -> Detection {
        Detection {
            class: ObjectClass::from_id(c.class_id).unwrap_or(ObjectClass::Female),

            confidence: c.confidence,

            screen_rect: Self::network_rect_to_screen(&c.rect_network, screen, net_w, net_h),
        }
    }

    // -----------------------------------------------------------------------
    // Post-processing
    // -----------------------------------------------------------------------

    #[inline]
    fn postprocess(
        candidate_buf: &mut Vec<YoloCandidate>,
        keep_buf: &mut Vec<bool>,
        heads: &[Head<'_>],
        cfg_w: u32,
        cfg_h: u32,
        target_classes: &[i32],
        confidence_threshold: f32,
        nms_threshold: f32,
        screen_rect: &ScreenRect,
        detections: &mut Vec<Detection>,
        timings: &mut DetectTimings,
    ) {
        // ---------------------------------------------------------------
        // Decode
        // ---------------------------------------------------------------

        candidate_buf.clear();

        let t_decode = Instant::now();

        decode_heads(
            heads,
            cfg_w,
            cfg_h,
            target_classes,
            confidence_threshold,
            candidate_buf,
        );

        timings.decode_ms = DetectTimings::ms(t_decode.elapsed());

        // ---------------------------------------------------------------
        // NMS
        // ---------------------------------------------------------------

        keep_buf.clear();

        let t_nms = Instant::now();

        filter_and_nms(candidate_buf, nms_threshold, keep_buf);

        timings.nms_ms = DetectTimings::ms(t_nms.elapsed());

        // ---------------------------------------------------------------
        // Candidate -> Detection
        // ---------------------------------------------------------------

        let t_convert = Instant::now();

        detections.clear();

        // This only grows the caller-owned buffer if necessary.
        // Once it has reached the required capacity, this is allocation-free.
        if detections.capacity() < candidate_buf.len() {
            detections.reserve(candidate_buf.len() - detections.capacity());
        }

        for candidate in candidate_buf.iter() {
            detections.push(OrtDetector::candidate_to_detection(
                candidate,
                screen_rect,
                cfg_w,
                cfg_h,
            ));
        }

        timings.final_conv_ms = DetectTimings::ms(t_convert.elapsed());

        timings.postprocess_ms = timings.decode_ms + timings.nms_ms + timings.final_conv_ms;
    }
}

// ---------------------------------------------------------------------------
// Detector implementation
// ---------------------------------------------------------------------------

impl Detector for OrtDetector {
    fn detect(
        &mut self,
        frame: &FrameData,
        confidence_threshold: f32,
        nms_threshold: f32,
        target_classes: &[i32],
        network_width: u32,
        network_height: u32,
        screen_rect: &ScreenRect,
        detections: &mut Vec<Detection>,
    ) -> Result<(), InferenceError> {
        let t_total = Instant::now();
        let mut timings = DetectTimings::default();

        let cfg_w = if network_width > 0 {
            network_width
        } else {
            self.network_width
        };

        let cfg_h = if network_height > 0 {
            network_height
        } else {
            self.network_height
        };

        let net_w = cfg_w as usize;
        let net_h = cfg_h as usize;
        let needed = 3 * net_w * net_h;

        let n_outputs = self.output_names.len();

        if n_outputs > MAX_OUTPUTS {
            return Err(InferenceError::TooManyOutputs {
                count: n_outputs,
                max: MAX_OUTPUTS,
            });
        }

        // IMPORTANT:
        //
        // Do not write:
        //
        //     let backend = self.backend_name();
        //
        // because that creates an immutable borrow of `self` which can
        // conflict with the mutable borrows below.
        //
        // These are all &'static str literals.
        let backend: &'static str = match self.exec_provider {
            ExecutionProvider::WebGpu => "ort-webgpu",
            ExecutionProvider::Cpu => "ort-cpu",
        };

        // ===================================================================
        // CPU / pageable CUDA path
        // ===================================================================
        {
            // ---------------------------------------------------------------
            // Preprocess
            // ---------------------------------------------------------------

            let t0 = Instant::now();

            // Resize while preserving existing capacity where possible.
            if self.preprocess_buffer.len() != needed {
                self.preprocess_buffer.resize(needed, 0.0f32);
            }

            Self::preprocess_frame_to_nchw(frame, &mut self.preprocess_buffer, net_w, net_h)?;

            timings.preprocess_ms = DetectTimings::ms(t0.elapsed());

            // ---------------------------------------------------------------
            // Tensor view
            // ---------------------------------------------------------------

            let t1 = Instant::now();

            let input = TensorRef::from_array_view((
                [1usize, 3usize, net_h, net_w],
                &*self.preprocess_buffer,
            ))
            .map_err(|e| InferenceError::Failed(format!("build input tensor view: {e}")))?;

            timings.tensor_view_ms = DetectTimings::ms(t1.elapsed());

            // ---------------------------------------------------------------
            // Inference + output processing
            // ---------------------------------------------------------------
            //
            // `outputs` borrows `self.session`.
            // Keep all work that needs those output slices inside this scope.
            //
            // `postprocess()` only mutably borrows candidate_buf/keep_buf,
            // not the whole detector.
            {
                let t2 = Instant::now();

                let outputs = self
                    .session
                    .run(inputs![input])
                    .map_err(|e| InferenceError::Failed(format!("inference failed: {e}")))?;

                timings.session_run_ms = DetectTimings::ms(t2.elapsed());

                // -----------------------------------------------------------
                // Extract output tensor views
                // -----------------------------------------------------------

                let t3 = Instant::now();

                let mut heads: [Head<'_>; MAX_OUTPUTS] = [(&[], 0, 0); MAX_OUTPUTS];

                for idx in 0..n_outputs {
                    let out = &outputs[idx];

                    let (shape, data) = out.try_extract_tensor::<f32>().map_err(|e| {
                        InferenceError::Failed(format!("extract output {idx}: {e}"))
                    })?;

                    let dims = shape.deref();

                    if dims.len() < 4 {
                        return Err(InferenceError::Failed(format!(
                            "output {idx} has {} dimensions; \
                                 expected at least 4",
                            dims.len()
                        )));
                    }

                    heads[idx] = (data, dims[2] as u32, dims[3] as u32);
                }

                timings.output_extract_ms = DetectTimings::ms(t3.elapsed());

                // -----------------------------------------------------------
                // Decode + NMS + conversion
                // -----------------------------------------------------------

                Self::postprocess(
                    &mut self.candidate_buf,
                    &mut self.keep_buf,
                    &heads[..n_outputs],
                    cfg_w,
                    cfg_h,
                    target_classes,
                    confidence_threshold,
                    nms_threshold,
                    screen_rect,
                    detections,
                    &mut timings,
                );

                // `outputs` drops here.
            }
        }

        // ===================================================================
        // Final timing
        // ===================================================================

        timings.total_ms = DetectTimings::ms(t_total.elapsed());

        self.last_timings = timings;

        if profile_enabled() {
            info!(
                "PERF detect: total={:.2}ms | \
                 preprocess={:.2}ms \
                 tensor={:.2}ms \
                 session.run={:.2}ms \
                 extract={:.2}ms \
                 decode={:.2}ms \
                 nms={:.2}ms \
                 final={:.2}ms \
                 post={:.2}ms | \
                 dets={} \
                 backend={} \
                 net={}x{} \
                 frame={}x{} \
                 fmt={:?}",
                timings.total_ms,
                timings.preprocess_ms,
                timings.tensor_view_ms,
                timings.session_run_ms,
                timings.output_extract_ms,
                timings.decode_ms,
                timings.nms_ms,
                timings.final_conv_ms,
                timings.postprocess_ms,
                detections.len(),
                backend,
                cfg_w,
                cfg_h,
                frame.width,
                frame.height,
                frame.format,
            );
        }

        Ok(())
    }

    fn backend_name(&self) -> &str {
        match self.exec_provider {
            ExecutionProvider::WebGpu => "ort-webgpu",
            ExecutionProvider::Cpu => "ort-cpu",
        }
    }

    fn last_timings(&self) -> DetectTimings {
        self.last_timings
    }
}
