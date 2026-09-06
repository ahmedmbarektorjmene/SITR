# Porda AI — Rust Rewrite (SITR) — ONNX + OpenCV 5

Offline on-device blur for modest viewing. Rust workspace, Slint UI, Wayland overlay, XDG Portal + PipeWire capture.

## Stack
- **Inference:** OpenCV 5 DNN `readNetFromONNX` → `porda.onnx` (544×320) → explicit YOLO decode
- **No Darknet:** `readNetFromDarknet` / `DetectionModel` removed. No `onnxruntime` dependency.
- **Preprocessing:** exact `resize_and_pad` from Python `main.py:782-801` (INTER_LINEAR, bottom/right pad, early-return `bottom<55 && right==0` or `right<70 && bottom==0`, ratios `w/new_w`, `h/new_h`).
- **Model:** `model/porda.onnx` converted from `pordav4x3.cfg` + `porda-19200-lr-0005-909.weights` (2 classes Male=0 Female=1, anchors 10,14 23,27 37,58 81,82 135,169 344,319, masks 3,4,5 and 0,1,2, `scale_x_y=1.05`).

## Requirements
- Rust 1.77+
- OpenCV 5 (`/usr/include/opencv5`, `/usr/lib/libopencv_*.so.500`, `pkg-config opencv5 --modversion` 5.0.0)
- `clang` + `libclang` (`LIBCLANG_PATH=/usr/lib`)
- Wayland compositor with `zwlr_layer_shell_v1` and `shm` (fallback to CPU stub otherwise)
- Arch/CachyOS: `opencv` (5.0) from extra; Ubuntu: build OpenCV 5 or use `opencv5.pc`

## Build & Run
```bash
cargo run
cargo run -p porda
cargo build
```
Default Cargo features now include `opencv` so `cargo run` uses OpenCV 5 without `--features opencv`. Previous `OPENCV_PKGCONFIG_NAME=opencv5` via `.cargo/config.toml`.

```bash
cargo check --workspace
cargo test --workspace
cargo test -p inference   # includes ONNX load + CPU inference
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

## Model Layout
```
model/
├── porda.onnx                              # ONNX, 23.5 MB, runtime required
├── pordav4x3.cfg                           # preserved for reproducibility
└── porda-19200-lr-0005-909.weights          # preserved
```
Conversion: `python scripts/convert_darknet_to_onnx.py` (parses cfg, loads weights, emits `Conv`+`BatchNormalization`+`LeakyReLU`, `Split` for `groups=2`, `Concat`, `MaxPool`, `Resize`, outputs `conv29_out [1,21,10,17]` and `conv36_out [1,21,20,34]`, opset 11).

## Inference Pipeline
```
FrameData (BGR)
  → resize_and_pad → Mat (padded 544×320 or original if early-return)
  → blobFromImage(1/255, 544×320, swapRB, INTER_LINEAR)
  → Net::forward → [1,21,10,17] + [1,21,20,34]
  → YOLO decode (sigmoid*1.05, exp*anchor, obj*cls)
  → confidence ≥0.25, class in target (Female=1)
  → NMS 0.10
  → network→padded (*padded/net) → original (*x_ratio + screen offset)
  → Detection { class, confidence, screen_rect }
  → covers_for_detections → WaylandOverlay (SHM or wgpu)
```

Thresholds: `confidence 0.25`, `nms 0.10`, target `Female` by default (`is_detect_female=true`).

## Tests
- `vision` 18: `resize_and_pad` exact dimensions, early-return 1920×1200→1,1, `nms`, `cover`, `geometry`.
- `inference` 9 (default opencv): `porda.onnx` exists, loads, CPU deterministic, early-return, `Auto` GPU fallback, class filtering.
- `cargo test --workspace` 33 tests.
- Python harness `scripts/equivalence_test.py` compares Darknet (OpenCV 4 `readNetFromDarknet` raw `conv_29/36`) vs ONNX (OpenCV 5 raw) mean abs diff 0.00034 max 0.003, decoded boxes within 2 px.

## Benchmark
`cargo run -p inference --example bench` (100 iters, 544×320, CPU):
- Load 43 ms, avg 69 ms (14.5 FPS). Darknet (OpenCV 4) 69.7 ms on same blob.

`cargo run -p inference --example lena` on 512×512 Lena → 1 Female 78,155,308,345 conf 0.758 (Darknet letterbox 80,155,308,345 diff 2 px).

## Device
`InferenceDevice::Auto` → GPU if `have_opencl()` else CPU. `Cpu`/`Gpu` explicit, `Gpu` falls back to CPU if unavailable. `inference_DEVICE=cpu|gpu|auto`.

## Wayland Overlay
`overlay::WaylandOverlay` if compositor advertises `zwlr_layer_shell_v1`; else `CpuOverlayRenderer` stub. With `PORDA_MOCK_DETECTIONS=1 PORDA_FORCE_ACTIVE=1` generates synthetic 300×200 center cover and `ShmRenderer` renders.

## License
AGPL-3.0. See `LICENSE`.

## Migration Notes
- `.cargo/config.toml` now `OPENCV_PKGCONFIG_NAME="opencv5"`, no `ld-wrapper.sh`.
- `ldd target/debug/porda` shows `libopencv_*.so.500`, no `libopencv_*.so.414`, no `libprotobuf`/`abseil`.
- Docs: `docs/migration-report.md`, `docs/experiment-opencv4-darknet.md`.
