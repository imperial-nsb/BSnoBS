# BSnoBS — BubSize no BS

Microbubble sizing from optical microscopy using a tiny distilled YOLO student, running on ONNX Runtime. Rust + `egui`. No Python.

## Build & run

```bash
cargo run --release
# or
cargo build --release
./target/release/bsnobs
```

The student weights (`assets/student.onnx`, 10 MB) are bundled at build time.

## Use

1. Window opens → bundled model auto-loads on a background thread (modal overlay shows progress).
2. **Select folder…** → pick a directory of `.tif` / `.png` / `.jpg` images.
3. Tune **Detection** (confidence, NMS IoU, max detections), **Physics** (μm/px, μL/frame, min/max diameter), optionally **Static dirt rejection** (cluster persistent detections across frames and reject them).
4. **Run inference** — non-blocking; status bar shows per-frame progress.
5. Scroll the image viewer with arrow keys / slider; circles overlay each detection (green = valid, red = rejected by size, dashed yellow = static dirt).
6. **Export results…** → writes `<sample>_bubble_data_full.csv`, `<sample>_summary.json`, `<sample>_static_rejects.csv` (if any), `<sample>_run_metadata.json` to a directory of your choice.

## Repo layout

```
Cargo.toml         eframe/egui 0.32, ort 2.0.0-rc.10 (CoreML + CPU), image, ndarray, rfd
assets/
  student.onnx     bundled YOLO student exported from PyTorch (opset 17, 640×640)
src/
  main.rs          eframe entry
  app.rs           AppState — side panel + viewer + worker channels
  inference.rs     letterbox preprocess → ort Session → decode → NMS
  static_filter.rs cross-frame persistent-detection rejection
  exporter.rs      CSV + JSON writers (Python-compatible field names)
  types.rs         BubbleData / FrameResult / AnalysisResults / AnalysisParameters
```
