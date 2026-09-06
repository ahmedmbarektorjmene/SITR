#[cfg(feature = "opencv")]
use opencv::prelude::*;
use vision::detection::{Detection, FrameData};
use vision::geometry::ScreenRect;

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("Model not loaded")]
    ModelNotLoaded,
    #[error("Inference failed: {0}")]
    Failed(String),
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),
}

pub trait Detector: Send + Sync {
    fn detect(
        &self,
        frame: &FrameData,
        confidence_threshold: f32,
        nms_threshold: f32,
        target_classes: &[i32],
        network_width: u32,
        network_height: u32,
        screen_rect: &ScreenRect,
    ) -> Result<Vec<Detection>, InferenceError>;

    fn backend_name(&self) -> &str;
}

pub struct MockDetector;

impl Detector for MockDetector {
    fn detect(
        &self,
        frame: &FrameData,
        confidence_threshold: f32,
        _nms_threshold: f32,
        target_classes: &[i32],
        _network_width: u32,
        _network_height: u32,
        screen_rect: &ScreenRect,
    ) -> Result<Vec<Detection>, InferenceError> {
        tracing::info!(
            "MockDetector: frame {}x{} target_classes={:?} conf_thresh={:.2}",
            frame.width,
            frame.height,
            target_classes,
            confidence_threshold
        );
        if std::env::var("PORDA_MOCK_DETECTIONS").is_ok() {
            tracing::info!(
                "MockDetector: PORDA_MOCK_DETECTIONS set, generating synthetic detection"
            );
            if target_classes.contains(&1) && confidence_threshold <= 0.9 {
                let w = (frame.width / 4).min(300);
                let h = (frame.height / 4).min(200);
                let x = (frame.width as i32 / 2) - (w as i32 / 2);
                let y = (frame.height as i32 / 2) - (h as i32 / 2);
                tracing::info!(
                    "MockDetector: returning 1 detection at ({},{},{},{})",
                    x,
                    y,
                    w,
                    h
                );
                return Ok(vec![Detection {
                    class: vision::detection::ObjectClass::Female,
                    confidence: 0.91,
                    screen_rect: ScreenRect::new(x, y, w, h),
                }]);
            }
        }
        let _ = screen_rect;
        Ok(vec![])
    }

    fn backend_name(&self) -> &str {
        "mock"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceDevice {
    Auto,
    Cpu,
    Gpu,
}

impl std::str::FromStr for InferenceDevice {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "gpu" => Ok(Self::Gpu),
            _ => Err(format!("unknown device {s}, expected auto|cpu|gpu")),
        }
    }
}

impl std::fmt::Display for InferenceDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Cpu => write!(f, "cpu"),
            Self::Gpu => write!(f, "gpu"),
        }
    }
}

#[cfg(feature = "opencv")]
pub struct OpenCvDetector {
    onnx_path: std::path::PathBuf,
    device: InferenceDevice,
    net: std::sync::Arc<std::sync::Mutex<Option<opencv::dnn::Net>>>,
}

#[cfg(not(feature = "opencv"))]
pub struct OpenCvDetector {
    onnx_path: std::path::PathBuf,
    device: InferenceDevice,
}

#[cfg(feature = "opencv")]
impl OpenCvDetector {
    pub fn new(onnx_path: std::path::PathBuf, device: InferenceDevice) -> Self {
        Self::new_with_device(onnx_path, device)
    }

    /// Backward compat: old signature took cfg+weights, now expects onnx. If called with cfg, try to find sibling porda.onnx.
    pub fn new_from_darknet_legacy(_cfg: std::path::PathBuf, _weights: std::path::PathBuf) -> Self {
        let candidate = std::path::PathBuf::from("model/porda.onnx");
        Self::new(candidate, InferenceDevice::Cpu)
    }

    pub fn new_with_device(onnx_path: std::path::PathBuf, device: InferenceDevice) -> Self {
        tracing::info!(
            "OpenCvDetector (ONNX): device={:?} onnx={:?}",
            device,
            onnx_path
        );
        if !onnx_path.exists() {
            tracing::warn!("OpenCvDetector: onnx not found at {:?}", onnx_path);
            return Self {
                onnx_path,
                device,
                net: std::sync::Arc::new(std::sync::Mutex::new(None)),
            };
        }
        let net = match opencv::dnn::read_net_from_onnx_def(onnx_path.to_str().unwrap_or("")) {
            Ok(mut n) => {
                // configure backend
                let configured = Self::configure_backend(&mut n, device);
                if configured {
                    tracing::info!(
                        "OpenCvDetector: ONNX model loaded successfully ({:?})",
                        onnx_path
                    );
                    Some(n)
                } else {
                    tracing::warn!(
                        "OpenCvDetector: backend configuration failed, still keeping net"
                    );
                    Some(n)
                }
            }
            Err(e) => {
                tracing::error!("OpenCvDetector: failed to load ONNX {:?}: {}", onnx_path, e);
                None
            }
        };
        Self {
            onnx_path,
            device,
            net: std::sync::Arc::new(std::sync::Mutex::new(net)),
        }
    }

    fn configure_backend(net: &mut opencv::dnn::Net, device: InferenceDevice) -> bool {
        use opencv::dnn::{DNN_BACKEND_OPENCV, DNN_TARGET_CPU, DNN_TARGET_OPENCL};
        fn try_cpu(net: &mut opencv::dnn::Net) -> bool {
            net.set_preferable_backend(DNN_BACKEND_OPENCV).is_ok()
                && net.set_preferable_target(DNN_TARGET_CPU).is_ok()
        }
        fn try_gpu(net: &mut opencv::dnn::Net) -> bool {
            let has_cl = opencv::core::have_opencl().unwrap_or(false);
            if !has_cl {
                tracing::info!("OpenCvDetector: OpenCL not available");
                return false;
            }
            if let Err(e) = opencv::core::set_use_opencl(true) {
                tracing::info!("OpenCvDetector: setUseOpenCL failed: {}", e);
                return false;
            }
            let ok = net.set_preferable_backend(DNN_BACKEND_OPENCV).is_ok()
                && net.set_preferable_target(DNN_TARGET_OPENCL).is_ok();
            if ok {
                tracing::info!("OpenCvDetector: configured GPU (OpenCL)");
            } else {
                tracing::info!("OpenCvDetector: GPU target not available");
            }
            ok
        }
        match device {
            InferenceDevice::Cpu => try_cpu(net),
            InferenceDevice::Gpu => {
                if try_gpu(net) {
                    true
                } else {
                    tracing::warn!(
                        "OpenCvDetector: GPU requested but not available, falling back to CPU"
                    );
                    try_cpu(net)
                }
            }
            InferenceDevice::Auto => {
                if try_gpu(net) {
                    true
                } else {
                    try_cpu(net)
                }
            }
        }
    }

    pub fn onnx_path(&self) -> &std::path::Path {
        &self.onnx_path
    }
    pub fn device(&self) -> InferenceDevice {
        self.device
    }
}

#[cfg(not(feature = "opencv"))]
impl OpenCvDetector {
    pub fn new(onnx_path: std::path::PathBuf, device: InferenceDevice) -> Self {
        if !onnx_path.exists() {
            tracing::warn!("OpenCvDetector: onnx not found at {:?}", onnx_path);
        }
        Self { onnx_path, device }
    }
    pub fn new_with_device(onnx_path: std::path::PathBuf, device: InferenceDevice) -> Self {
        Self::new(onnx_path, device)
    }
    pub fn onnx_path(&self) -> &std::path::Path {
        &self.onnx_path
    }
    pub fn device(&self) -> InferenceDevice {
        self.device
    }
}

// Keep legacy constructor signature for pipeline compatibility during migration
impl OpenCvDetector {
    /// Legacy Darknet constructor (cfg, weights) – now redirects to ONNX if available.
    /// This keeps `Pipeline` code that calls `OpenCvDetector::new(cfg, weights)` compiling
    /// while migration is in progress. Prefers `model/porda.onnx` if present.
    pub fn new_legacy_darknet(
        config_path: std::path::PathBuf,
        weights_path: std::path::PathBuf,
    ) -> Self {
        let onnx_candidate = config_path
            .parent()
            .map(|p| p.join("porda.onnx"))
            .unwrap_or_else(|| std::path::PathBuf::from("model/porda.onnx"));
        if onnx_candidate.exists() {
            tracing::info!(
                "OpenCvDetector: legacy Darknet paths {:?}/{:?} redirected to ONNX {:?}",
                config_path,
                weights_path,
                onnx_candidate
            );
            return Self::new(onnx_candidate, InferenceDevice::Auto);
        }
        // Fallback: treat first arg as onnx if it ends with .onnx, else use onnx_candidate
        if config_path
            .extension()
            .map(|e| e == "onnx")
            .unwrap_or(false)
        {
            return Self::new(config_path, InferenceDevice::Auto);
        }
        // If ONNX not found, still create with missing path to surface ModelNotLoaded
        Self::new(onnx_candidate, InferenceDevice::Auto)
    }
}

#[cfg(feature = "opencv")]
impl Detector for OpenCvDetector {
    fn detect(
        &self,
        frame: &FrameData,
        confidence_threshold: f32,
        nms_threshold: f32,
        target_classes: &[i32],
        network_width: u32,
        network_height: u32,
        screen_rect: &ScreenRect,
    ) -> Result<Vec<Detection>, InferenceError> {
        use opencv::core::{Mat, Scalar, Size, Vector, CV_32F};
        use opencv::prelude::*;

        let mut guard = self
            .net
            .lock()
            .map_err(|e| InferenceError::Failed(format!("Model mutex poisoned: {}", e)))?;
        let net = guard.as_mut().ok_or(InferenceError::ModelNotLoaded)?;

        let (padded_data, x_ratio, y_ratio) = vision::preprocessing::resize_and_pad(
            &frame.data,
            frame.width,
            frame.height,
            network_width,
            network_height,
        );

        let (padded_w, padded_h) = if (x_ratio - 1.0).abs() < 0.01 && (y_ratio - 1.0).abs() < 0.01 {
            (frame.width, frame.height)
        } else {
            (network_width, network_height)
        };

        tracing::info!(
            "OpenCvDetector(ONNX): {}x{} -> padded {}x{} ratios {:.3},{:.3} network {}x{} device {:?}",
            frame.width,
            frame.height,
            padded_w,
            padded_h,
            x_ratio,
            y_ratio,
            network_width,
            network_height,
            self.device
        );

        let vec3b_slice: &[opencv::core::Vec3b] = unsafe {
            std::slice::from_raw_parts(
                padded_data.as_ptr() as *const opencv::core::Vec3b,
                (padded_w * padded_h) as usize,
            )
        };
        let mat = Mat::new_rows_cols_with_data(padded_h as i32, padded_w as i32, vec3b_slice)
            .map_err(|e| InferenceError::Failed(format!("Mat creation failed: {}", e)))?;

        // Create blob 1/255, swapRB, size network
        let blob = opencv::dnn::blob_from_image(
            &mat,
            1.0 / 255.0,
            Size::new(network_width as i32, network_height as i32),
            Scalar::default(),
            true,
            false,
            CV_32F,
        )
        .map_err(|e| InferenceError::Failed(format!("blob_from_image failed: {}", e)))?;

        net.set_input(&blob, "", 1.0, Scalar::default())
            .map_err(|e| InferenceError::Failed(format!("set_input failed: {}", e)))?;

        // Get output names
        let out_names: Vector<String> = net.get_unconnected_out_layers_names().map_err(|e| {
            InferenceError::Failed(format!("get_unconnected_out_layers_names failed: {}", e))
        })?;
        if out_names.len() != 2 {
            tracing::warn!(
                "OpenCvDetector: expected 2 outputs, got {}",
                out_names.len()
            );
        }
        let mut outs: Vector<Mat> = Vector::new();
        net.forward(&mut outs, &out_names)
            .map_err(|e| InferenceError::Failed(format!("forward failed: {}", e)))?;

        if outs.len() != 2 {
            return Err(InferenceError::Failed(format!(
                "expected 2 outputs, got {}",
                outs.len()
            )));
        }
        let mat0 = outs
            .get(0)
            .map_err(|e| InferenceError::Failed(format!("outs.get(0) failed: {}", e)))?;
        let mat1 = outs
            .get(1)
            .map_err(|e| InferenceError::Failed(format!("outs.get(1) failed: {}", e)))?;

        // Extract data. Mat is 4D NCHW. We use data_typed if available, else fallback to manual.
        let (data0, h0, w0) = mat_to_vec(&mat0)?;
        let (data1, h1, w1) = mat_to_vec(&mat1)?;

        // Determine which head is small grid
        // h0=10 small, h1=20 large; order may be as produced. Use size to pick mask.
        let heads = if h0 < h1 {
            vec![(data0, h0, w0), (data1, h1, w1)]
        } else {
            vec![(data1, h1, w1), (data0, h0, w0)]
        };

        let candidates = crate::yolo::decode_heads(&heads, network_width, network_height);
        let filtered = crate::yolo::filter_and_nms(
            candidates,
            confidence_threshold,
            nms_threshold,
            target_classes,
        );

        // Convert network coords to screen coords
        let mut detections = Vec::new();
        let padded_w_f = padded_w as f32;
        let padded_h_f = padded_h as f32;
        let net_w_f = network_width as f32;
        let net_h_f = network_height as f32;

        for cand in filtered {
            // rect_network is in network 544x320 coords
            let rn = cand.rect_network;
            // network -> padded
            let x_padded = rn.x as f32 * (padded_w_f / net_w_f);
            let y_padded = rn.y as f32 * (padded_h_f / net_h_f);
            let w_padded = rn.width as f32 * (padded_w_f / net_w_f);
            let h_padded = rn.height as f32 * (padded_h_f / net_h_f);
            // padded -> original via x_ratio/y_ratio
            let x_orig = (x_padded * x_ratio) as i32 + screen_rect.x;
            let y_orig = (y_padded * y_ratio) as i32 + screen_rect.y;
            let w_orig = (w_padded * x_ratio) as u32;
            let h_orig = (h_padded * y_ratio) as u32;
            let class = vision::detection::ObjectClass::from_id(cand.class_id)
                .unwrap_or(vision::detection::ObjectClass::Female);
            detections.push(Detection {
                class,
                confidence: cand.confidence,
                screen_rect: ScreenRect::new(x_orig, y_orig, w_orig, h_orig),
            });
        }

        tracing::info!(
            "OpenCvDetector(ONNX): {} detections after NMS",
            detections.len()
        );
        Ok(detections)
    }

    fn backend_name(&self) -> &str {
        match self.device {
            InferenceDevice::Cpu => "opencv-dnn-cpu",
            InferenceDevice::Gpu => "opencv-dnn-gpu",
            InferenceDevice::Auto => "opencv-dnn-auto",
        }
    }
}

#[cfg(feature = "opencv")]
fn mat_to_vec(mat: &opencv::core::Mat) -> Result<(Vec<f32>, u32, u32), InferenceError> {
    use opencv::prelude::MatTraitConstManual;
    let dims = mat.dims();
    if dims == 4 {
        let ms = mat.mat_size();
        let n = ms
            .get(0)
            .map_err(|e| InferenceError::Failed(format!("mat_size 0 {}", e)))?;
        let c = ms
            .get(1)
            .map_err(|e| InferenceError::Failed(format!("mat_size 1 {}", e)))?;
        let h = ms
            .get(2)
            .map_err(|e| InferenceError::Failed(format!("mat_size 2 {}", e)))?;
        let w = ms
            .get(3)
            .map_err(|e| InferenceError::Failed(format!("mat_size 3 {}", e)))?;
        if n != 1 || c != 21 {
            return Err(InferenceError::Failed(format!(
                "unexpected ONNX output shape dims {} n={} c={} h={} w={}",
                dims, n, c, h, w
            )));
        }
        let total = (c * h * w) as usize;
        let slice = mat
            .data_typed::<f32>()
            .map_err(|e| InferenceError::Failed(format!("data_typed failed: {}", e)))?;
        if slice.len() < total {
            return Err(InferenceError::Failed(format!(
                "slice len {} < total {}",
                slice.len(),
                total
            )));
        }
        Ok((slice[..total].to_vec(), h as u32, w as u32))
    } else if dims == 3 {
        let ms = mat.mat_size();
        let c = ms
            .get(0)
            .map_err(|e| InferenceError::Failed(format!("mat_size 0 {}", e)))?;
        let h = ms
            .get(1)
            .map_err(|e| InferenceError::Failed(format!("mat_size 1 {}", e)))?;
        let w = ms
            .get(2)
            .map_err(|e| InferenceError::Failed(format!("mat_size 2 {}", e)))?;
        let total = (c * h * w) as usize;
        let slice = mat
            .data_typed::<f32>()
            .map_err(|e| InferenceError::Failed(format!("data_typed 3d failed: {}", e)))?;
        Ok((slice[..total].to_vec(), h as u32, w as u32))
    } else {
        Err(InferenceError::Failed(format!(
            "unsupported mat dims {}",
            dims
        )))
    }
}

#[cfg(not(feature = "opencv"))]
impl Detector for OpenCvDetector {
    fn detect(
        &self,
        frame: &FrameData,
        _confidence_threshold: f32,
        _nms_threshold: f32,
        _target_classes: &[i32],
        network_width: u32,
        network_height: u32,
        screen_rect: &ScreenRect,
    ) -> Result<Vec<Detection>, InferenceError> {
        if !self.onnx_path.exists() {
            return Err(InferenceError::ModelNotLoaded);
        }
        tracing::warn!(
            "OpenCvDetector: onnx found at {:?} ({}x{}), but inference requires `opencv` feature. Frame {}x{} -> {:?}. Returning BackendNotAvailable.",
            self.onnx_path,
            network_width,
            network_height,
            frame.width,
            frame.height,
            screen_rect
        );
        Err(InferenceError::BackendNotAvailable(
            "opencv feature not enabled (need `cargo build -p inference --features opencv` with opencv5)"
                .to_string(),
        ))
    }

    fn backend_name(&self) -> &str {
        "opencv-dnn"
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    use vision::detection::FrameData;
    use vision::geometry::ScreenRect;

    fn onnx_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../model/porda.onnx")
    }

    #[test]
    fn test_onnx_file_exists() {
        let p = onnx_path();
        assert!(p.exists(), "porda.onnx not found at {:?}", p);
        let meta = std::fs::metadata(&p).unwrap();
        assert!(meta.len() > 1_000_000, "onnx too small: {}", meta.len());
    }

    #[test]
    #[cfg(feature = "opencv")]
    fn test_onnx_loads_successfully() {
        let p = onnx_path();
        let det = OpenCvDetector::new(p.clone(), InferenceDevice::Cpu);
        // check that net is Some by trying detect on small frame (should not be ModelNotLoaded)
        let frame = FrameData::new_bgr(544, 320, vec![128u8; 544 * 320 * 3]);
        let res = det.detect(
            &frame,
            0.25,
            0.1,
            &[1],
            544,
            320,
            &ScreenRect::new(0, 0, 544, 320),
        );
        assert!(
            res.is_ok(),
            "detect should succeed on CPU, got {:?}",
            res.err()
        );
    }

    #[test]
    #[cfg(feature = "opencv")]
    fn test_cpu_inference_deterministic() {
        let p = onnx_path();
        let det = OpenCvDetector::new(p, InferenceDevice::Cpu);
        // 800x600 needs padding (426x320)
        let frame = FrameData::new_bgr(800, 600, vec![64u8; 800 * 600 * 3]);
        let r1 = det
            .detect(
                &frame,
                0.25,
                0.1,
                &[1],
                544,
                320,
                &ScreenRect::new(0, 0, 800, 600),
            )
            .unwrap();
        let r2 = det
            .detect(
                &frame,
                0.25,
                0.1,
                &[1],
                544,
                320,
                &ScreenRect::new(0, 0, 800, 600),
            )
            .unwrap();
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_eq!(a.class, b.class);
            assert!((a.confidence - b.confidence).abs() < 1e-6);
            assert_eq!(a.screen_rect, b.screen_rect);
        }
    }

    #[test]
    #[cfg(feature = "opencv")]
    fn test_early_return_1920x1200() {
        let p = onnx_path();
        let det = OpenCvDetector::new(p, InferenceDevice::Cpu);
        // 1920x1200 triggers early return (right 32 bottom 0)
        let frame = FrameData::new_bgr(1920, 1200, vec![0u8; 1920 * 1200 * 3]);
        let res = det.detect(
            &frame,
            0.25,
            0.1,
            &[1],
            544,
            320,
            &ScreenRect::new(0, 0, 1920, 1200),
        );
        assert!(res.is_ok());
        // With blank image should be 0 detections
        assert_eq!(res.unwrap().len(), 0);
    }

    #[test]
    #[cfg(feature = "opencv")]
    fn test_gpu_fallback_auto() {
        let p = onnx_path();
        let det = OpenCvDetector::new(p, InferenceDevice::Auto);
        let frame = FrameData::new_bgr(544, 320, vec![128u8; 544 * 320 * 3]);
        let res = det.detect(
            &frame,
            0.25,
            0.1,
            &[1],
            544,
            320,
            &ScreenRect::new(0, 0, 544, 320),
        );
        assert!(
            res.is_ok(),
            "Auto device should fallback to CPU if GPU not available"
        );
    }

    #[test]
    #[cfg(feature = "opencv")]
    fn test_target_class_filtering() {
        let p = onnx_path();
        let det = OpenCvDetector::new(p, InferenceDevice::Cpu);
        let frame = FrameData::new_bgr(544, 320, vec![200u8; 544 * 320 * 3]);
        let r_female = det
            .detect(
                &frame,
                0.25,
                0.1,
                &[1],
                544,
                320,
                &ScreenRect::new(0, 0, 544, 320),
            )
            .unwrap();
        let r_male = det
            .detect(
                &frame,
                0.25,
                0.1,
                &[0],
                544,
                320,
                &ScreenRect::new(0, 0, 544, 320),
            )
            .unwrap();
        let r_both = det
            .detect(
                &frame,
                0.25,
                0.1,
                &[0, 1],
                544,
                320,
                &ScreenRect::new(0, 0, 544, 320),
            )
            .unwrap();
        // female + male should >= each individually
        assert!(r_both.len() >= r_female.len());
        assert!(r_both.len() >= r_male.len());
        for d in &r_female {
            assert_eq!(d.class as i32, 1);
        }
        for d in &r_male {
            assert_eq!(d.class as i32, 0);
        }
    }
}
