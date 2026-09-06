use inference::detector::{Detector, InferenceDevice, OpenCvDetector};
use opencv::prelude::*;
use vision::detection::FrameData;
use vision::geometry::ScreenRect;

fn main() -> anyhow::Result<()> {
    let mat = opencv::imgcodecs::imread("/tmp/lena.jpg", opencv::imgcodecs::IMREAD_COLOR)?;
    let w = mat.cols() as u32;
    let h = mat.rows() as u32;
    println!("lena {}x{}", w, h);
    // mat is BGR, get data
    let bgr = mat.data_bytes()?.to_vec();
    let frame = FrameData::new_bgr(w, h, bgr);
    let onnx = std::path::PathBuf::from("model/porda.onnx");
    let det = OpenCvDetector::new(onnx, InferenceDevice::Cpu);
    let rect = ScreenRect::new(0, 0, w as u32, h as u32);
    let dets = det.detect(&frame, 0.25, 0.1, &[1], 544, 320, &rect)?;
    println!("detections {}", dets.len());
    for d in &dets {
        println!(
            "{:?} conf {:.3} bbox {:?}",
            d.class, d.confidence, d.screen_rect
        );
    }
    // also test with both classes
    let dets2 = det.detect(&frame, 0.25, 0.1, &[0, 1], 544, 320, &rect)?;
    println!("both classes {}", dets2.len());
    for d in &dets2 {
        println!(
            " both {:?} conf {:.3} bbox {:?}",
            d.class, d.confidence, d.screen_rect
        );
    }
    Ok(())
}
