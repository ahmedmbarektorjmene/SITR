use inference::detector::{Detector, InferenceDevice, OpenCvDetector};
use std::time::Instant;
use vision::detection::FrameData;
use vision::geometry::ScreenRect;

fn main() -> anyhow::Result<()> {
    let onnx = std::path::PathBuf::from("model/porda.onnx");
    if !onnx.exists() {
        eprintln!("model not found at {:?}", onnx);
        std::process::exit(1);
    }
    println!("Benchmark: ONNX OpenCV5 CPU");
    println!("Model: {:?}", onnx);
    let t0 = Instant::now();
    let det = OpenCvDetector::new(onnx.clone(), InferenceDevice::Cpu);
    let load_ms = t0.elapsed().as_millis();
    println!("Load time: {} ms", load_ms);

    let frame = FrameData::new_bgr(544, 320, vec![128u8; 544 * 320 * 3]);
    let rect = ScreenRect::new(0, 0, 544, 320);

    // warmup
    for _ in 0..5 {
        let _ = det.detect(&frame, 0.25, 0.1, &[1], 544, 320, &rect)?;
    }
    let iters = 100;
    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = det.detect(&frame, 0.25, 0.1, &[1], 544, 320, &rect)?;
    }
    let elapsed = t1.elapsed();
    let avg_ms = elapsed.as_millis() as f64 / iters as f64;
    let fps = 1000.0 / avg_ms;
    println!(
        "Avg inference: {:.2} ms ({:.1} FPS) over {} iters",
        avg_ms, fps, iters
    );

    // single
    let t2 = Instant::now();
    let _ = det.detect(&frame, 0.25, 0.1, &[1], 544, 320, &rect)?;
    println!(
        "Single inference: {:.2} ms",
        t2.elapsed().as_secs_f64() * 1000.0
    );

    // Test GPU if available
    if std::env::var("BENCH_GPU").is_ok() {
        println!("\nBenchmark GPU (Auto):");
        let t0 = Instant::now();
        let det_gpu = OpenCvDetector::new(onnx, InferenceDevice::Auto);
        println!("GPU load: {} ms", t0.elapsed().as_millis());
        let t1 = Instant::now();
        for _ in 0..50 {
            let _ = det_gpu.detect(&frame, 0.25, 0.1, &[1], 544, 320, &rect)?;
        }
        let avg = t1.elapsed().as_millis() as f64 / 50.0;
        println!("GPU avg: {:.2} ms ({:.1} FPS)", avg, 1000.0 / avg);
    }
    Ok(())
}
