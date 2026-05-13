# AGENTS.md — BSnoBS

Working notes for future agents iterating on this codebase. Keep this file updated when you change something load-bearing.

## What this is

A single-binary Rust GUI (eframe/egui + ONNX Runtime via `ort`) for microbubble sizing from optical-microscopy images. The deep-learning model is a tiny YOLO student exported to ONNX, embedded in the binary at compile time. There is no Python in this repo — that was deliberately removed; do not bring it back.

## Build / run

```bash
cargo run --release          # dev
cargo build --release        # ./target/release/bsnobs (~38 MB; ONNX baked in)
```

`bsnobs` is the only binary. On macOS, ort will use CoreML; elsewhere CPU. The user picks `Auto / CPU / CoreML` in the Model collapsing-header (default Auto).

## Repo layout

```
Cargo.toml           eframe/egui 0.32, ort 2.0.0-rc.10, image, ndarray, rfd, chrono
assets/student.onnx  bundled via include_bytes! (do not move/rename without
                     updating inference.rs::BUNDLED_WEIGHTS)
src/
  main.rs            eframe entry; window size; AppState::new
  app.rs             ★ The whole UI. Big file (~1.1k LOC) — read this first.
  inference.rs       letterbox → ort Session → decode → class-agnostic NMS
  static_filter.rs   cross-frame persistent-detection rejection
  exporter.rs        CSV + JSON writers, PNG overlay + histogram rasterizers (imageproc + ab_glyph)
  types.rs           BubbleData / FrameResult / AnalysisResults / AnalysisParameters
```

## Architecture in one paragraph

`AppState` (in `src/app.rs`) owns everything: model (`StudentAnalyzer`), settings, `workspace: Vec<WorkspaceEntry>`, `results_list: Vec<ResultEntry>`, viewer state, and the worker-thread `mpsc::Receiver`. The eframe `update()` drains the worker channel, draws four panels (left split into Settings + Workspace, right = Viewer + Results, center = image + slider, bottom = status), and re-requests repaint while work is in flight. Inference runs on `std::thread::spawn`'d workers — the analyzer is **moved onto the thread and sent back via the `Done` message** so we never reload it.

## UI layout (left → right)

- **Left side panel** (`SidePanel::left("left-side")`)
  - Top: scrollable Settings — Model (collapsed by default), Detection, Physics, Static-dirt rejection.
  - Bottom: `TopBottomPanel::bottom("workspace-panel")` showing the Workspace list (add folder via `+`, remove per-row via `×`) and a big centered green **RUN** button.
- **Right side panel** (`SidePanel::right("results-panel")`)
  - **Viewer** section: folder ComboBox (focus picker), `B&W mode`, `Show overlays`.
  - **Results** section: per-folder stats with a visibility checkbox (export inclusion only — no longer drives focus).
  - Centered **Export…** button at the bottom.
- **Center panel**: zoom controls row, image viewer (fills available space), frame slider.
- **Bottom**: status bar + progress bar during runs.

## Data flow / lifecycle

1. **Model load** — `start_model_load` spawns a thread, sends back `Result<(StudentAnalyzer, Device), String>`. The bundled path uses `load_from_bytes(BUNDLED_WEIGHTS, device)`; a user-picked `.onnx` uses `load(path, device)`. While loading, a foreground-layer overlay dims the app and shows a spinner + elapsed ms.
2. **Workspace** — user clicks `+` to add a folder (`rfd::FileDialog::pick_folder` → `list_images` populates `image_files`). Each entry has its own `run_enabled` checkbox.
3. **Run** — `start_run` takes the analyzer out of `AppState`, sends it to a worker that iterates every checked workspace folder sequentially. For each folder the worker emits `Progress { folder_idx, folder_total, frame_idx, frame_total, ... }`, then `FolderDone(ResultEntry)`. After the last folder it sends `AllDone(analyzer)` so the main thread can put the analyzer back. Re-running a folder replaces its existing `ResultEntry` (matched by `source_path`).
4. **Settings invalidation** — at the top of every `update()`, `maybe_invalidate_results` snapshots all results-affecting settings (conf, iou, max_det, scale, μL/frame, min/max diameter, static-filter config, reject_static). If the snapshot differs from last frame's, `results_list` is wiped. After a run completes, the baseline snapshot is refreshed so float-drift doesn't immediately invalidate the just-produced results.
5. **Viewer** — `ensure_frame_texture` lazily loads + uploads the image for the focused result's current frame. Cache key is `(focus_idx, frame_idx, bw_mode)`. B&W is applied at upload time via Rec. 601 luma. Overlays are drawn on top in `draw_overlays` (green = valid, red = size-rejected, dashed yellow = static dirt).
6. **Export** — `export_visible_results` writes every `ResultEntry` with `visible == true`. The user picks the destination folder via `save_file` (default name `bsnobs_run_…` or `<sample>_count_…`). One subdir per sample inside the chosen dir. Each sample dir always gets `*_run_metadata.json`; the three checkboxes below the Export button add: `csv` → `*_bubble_data.csv` (accepted bubbles only, no rejects file), `png` → `overlays/<frame>_overlay.png` rasterized via `imageproc` (respects the viewer's accepted/rejected/static toggles), `hist` → `*_histogram.png` per sample plus a `combined_histogram.png` at the root when there are 2+ samples. Histograms use the active `HistMode` and shared global x/y axes.

## Design decisions (and why)

- **Embedded model**: `include_bytes!("../assets/student.onnx")` + `Session::commit_from_memory`. No fs lookup, no path discovery, no "model missing" failure mode. Trade-off: 10 MB added to binary; rebuild on weight change. Easy to add a model picker later (custom path already works via `pick_weights`).
- **Single binary, no Python**: previous attempt had a parallel Python CLI/GUI — explicitly deleted. If you find yourself reaching for Python, port the logic to Rust or extend `exporter.rs` instead.
- **Per-folder run + multi-result panel**: the original UI was one folder at a time; users wanted comparative analysis. Workspace = inputs, Results = outputs, kept separate.
- **Settings invalidate stored results**: user explicitly requested this. Avoids the "stale results from old conf threshold" confusion. The trade-off is that a slider tweak after a long run nukes the output — but the worker can be re-run instantly because the model stays loaded.
- **Inference worker hands the analyzer back**: ort `Session` is `Send` but expensive to build; moving it onto the worker via the `Done` message avoids the reload-after-run bug that bit us once already.
- **Progress is a modal overlay, not a fake progress bar**: ort gives no progress signal for model load, so we show an indeterminate spinner + elapsed-ms instead of inventing a bogus bar. Same principle for the run progress — the bar represents per-frame `frame_idx / frame_total * folder_total`, which is real.
- **Auto-device disclosure**: ort 2.0-rc.10 has no API to query the active execution provider. We resolve `Device::Auto` at compile time (macOS → CoreML, else CPU) and surface "auto (coreml)" in the status — accurate as long as CoreML init doesn't silently fail.
- **Scroll/pinch zoom around cursor**: `i.smooth_scroll_delta.y` for wheel + `i.zoom_delta()` for trackpad pinch. The pan-vs-zoom pivot math keeps the image pixel under the cursor fixed: `new_pan = v * (1 - r) + pan * r` where `r = new_zoom / old_zoom` and `v = pivot - rect_center`. Range clamped 0.2× – 8× to keep things sensible. Double-click resets.
- **Texture cache key includes B&W flag**: toggling B&W rebuilds the texture once (per-frame), then caches.

## Known gotchas

- **Borrow checker in `drain_worker`**: drain messages into a local `Vec` first; you can't hold a borrow of `worker_rx` while mutating it. Same pattern in any future per-frame message loop.
- **Float field changes invalidate results immediately**: a single drag of a `DragValue` produces many ticks. We accept this — `results_list` is empty after the first tick, so subsequent ticks are no-ops.
- **`AnalysisParameters::sample_volume_per_frame_uL`** is serialized with `_uL` to stay compatible with the Python tool's JSON outputs, but the Rust field is snake_case (`sample_volume_per_frame_ul`). Don't rename without grep'ing both forms.
- **`BubbleData.frame_name` is the source of truth for the per-bubble frame label** — `FrameResult` no longer carries one. Don't reintroduce a duplicate.
- **Comparison plots are not implemented in Rust** yet. The Python `compare.py` rendered cross-sample histograms with matplotlib. Porting it (e.g. with `plotters`) is the obvious next step if multi-folder export becomes load-bearing.

## Things deliberately left out (don't add without asking)

- Workspace persistence across launches (no settings file yet).
- Pan clamping / rubber-banding when zoomed beyond image edge.
- Comments explaining what code does (write comments only when the *why* is non-obvious — the user is terse and dislikes obvious comments).
- Backwards-compat shims, feature flags, and "just in case" abstractions.
