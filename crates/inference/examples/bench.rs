use inference::detector::{Detector, OrtDetector};
use std::time::Instant;
use vision::detection::{Detection, FrameData};
use vision::geometry::ScreenRect;

const WARMUPS: usize = 5;
const ITERATIONS: usize = 100;

fn main() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_target(false)
        .try_init();

    let onnx = std::path::PathBuf::from("model/porda_mobilenetv4_small.onnx");

    if !onnx.exists() {
        anyhow::bail!("model not found at {:?}", onnx);
    }

    // ---------------------------------------------------------
    // Load
    // ---------------------------------------------------------

    let load_start = Instant::now();

    let mut detector = OrtDetector::new(&onnx).expect("OrtDetector should build");

    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let backend = detector.backend_name().to_owned();

    // ---------------------------------------------------------
    // Input
    // ---------------------------------------------------------

    let frame = FrameData::new_bgr(1920, 1200, vec![128u8; 1920 * 1200 * 3]);

    let rect = ScreenRect::new(0, 0, 544, 320);

    // Reused for every inference.
    let mut detections = Vec::<Detection>::with_capacity(32);

    // ---------------------------------------------------------
    // Warmup
    // ---------------------------------------------------------

    for _ in 0..WARMUPS {
        detector.detect(&frame, 0.25, 0.1, &[1], 544, 320, &rect, &mut detections)?;
    }

    // ---------------------------------------------------------
    // Benchmark
    // ---------------------------------------------------------

    let benchmark_start = Instant::now();

    for _ in 0..ITERATIONS {
        detector.detect(&frame, 0.25, 0.1, &[1], 544, 320, &rect, &mut detections)?;
    }

    let total_elapsed_ms = benchmark_start.elapsed().as_secs_f64() * 1000.0;

    let avg_ms = total_elapsed_ms / ITERATIONS as f64;
    let fps = 1000.0 / avg_ms;

    let timings = detector.last_timings();

    // ---------------------------------------------------------
    // Final summary
    // ---------------------------------------------------------

    println!();
    println!("# Benchmark Summary: ONNX via ORT");
    println!();
    println!("## Configuration");
    println!("- **Framework**: ONNX Runtime 1.28");
    println!("- **Model**: `porda_mobilenetv4_small.onnx`");
    println!("- **Backend**: {}", backend);
    println!("- **Network Input**: 544×320 (3 channels)");
    println!("- **Frame Size**: 1920×1200");
    println!("- **Format**: BGR");
    println!();

    println!("## Performance Results");
    println!();
    println!("| Metric | Value |");
    println!("|--------|-------|");
    println!("| **Load Time** | {:.2} ms |", load_ms);
    println!("| **Avg Inference** | **{:.2} ms** |", avg_ms);
    println!("| **Avg FPS** | **{:.1}** |", fps);
    println!("| **Iterations** | {} |", ITERATIONS);
    println!();

    println!("### Timing Breakdown (avg)");
    println!("- Preprocess: {:.2} ms", timings.preprocess_ms);
    println!("- Tensor view: {:.2} ms", timings.tensor_view_ms);
    println!("- **session.run**: **{:.2} ms**", timings.session_run_ms);
    println!("- Output extract: {:.2} ms", timings.output_extract_ms);
    println!("- Postprocess: {:.2} ms", timings.postprocess_ms);
    println!();

    println!("## Status");
    println!("| Component | Status |");
    println!("|-----------|--------|");

    if backend.contains("cuda") {
        println!("| CUDA | ✅ Active |");
        println!("| CPU | ℹ️ Available as fallback |");
    } else {
        println!("| CUDA | ❌ Unavailable |");
        println!("| CPU | ✅ Active |");
    }

    println!();

    if !backend.contains("cuda") {
        println!("> **Note:** CUDA EP is unavailable; inference is running entirely on CPU.");
        println!();
    }

    println!("## Benchmark Details");
    println!("- Warmup iterations: {}", WARMUPS);
    println!("- Timed iterations: {}", ITERATIONS);
    println!("- Detection buffer: reused");
    println!("- Detection capacity: {}", detections.capacity());

    Ok(())
}
