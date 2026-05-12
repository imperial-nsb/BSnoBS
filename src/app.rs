use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use anyhow::Result;
use eframe::egui::{
    self, Color32, ColorImage, Layout, Pos2, Rect, Sense, Slider, Stroke, TextureHandle, Vec2,
};
use image::ImageReader;

use crate::exporter;
use crate::inference::{list_images, Device, StudentAnalyzer, INPUT_SIZE};
use crate::static_filter::{self, StaticFilterConfig};
use crate::types::{AnalysisParameters, AnalysisResults, FrameResult};

pub struct AppState {
    // Folder / weights
    image_dir: Option<PathBuf>,
    image_files: Vec<PathBuf>,
    weights_path: PathBuf,
    using_bundled: bool,
    device: Device,

    // Cached model
    analyzer: Option<StudentAnalyzer>,
    model_status: String,
    model_load_time_ms: Option<u128>,
    model_loading: bool,
    model_load_start: Option<Instant>,
    model_load_rx: Option<mpsc::Receiver<Result<(StudentAnalyzer, Device), String>>>,
    resolved_device: Option<Device>,

    // Settings
    params: AnalysisParameters,
    conf: f32,
    imgsz: u32,
    iou: f32,
    max_det: usize,
    reject_static: bool,
    static_cfg: StaticFilterConfig,

    // Results + viewer state
    results: Option<AnalysisResults>,
    current_frame: usize,
    texture: Option<TextureHandle>,
    texture_for_frame: Option<usize>,
    zoom: f32,

    // Worker plumbing
    worker_rx: Option<mpsc::Receiver<WorkerMsg>>,
    in_progress: bool,
    progress: (usize, usize),
    status: String,
    start_time: Instant,
}

enum WorkerMsg {
    Progress(usize, usize, String),
    Done(AnalysisResults, StudentAnalyzer),
    Failed(String, StudentAnalyzer),
}

/// Resolve `Device::Auto` to the concrete device ONNX Runtime will actually use.
/// On macOS, ort prefers CoreML (registered first in our provider list) and falls
/// back to CPU if CoreML init fails — here we assume CoreML succeeds on Apple
/// Silicon/Intel macOS builds.
fn resolve_device(d: Device) -> Device {
    match d {
        Device::Auto => {
            if cfg!(target_os = "macos") {
                Device::CoreML
            } else {
                Device::Cpu
            }
        }
        other => other,
    }
}

fn bundled_weights() -> PathBuf {
    // Bundled relative to the running binary at debug/release time.
    // First try CARGO_MANIFEST_DIR (dev), then exe-relative.
    if let Some(d) = option_env!("CARGO_MANIFEST_DIR") {
        let p = Path::new(d).join("assets/student.onnx");
        if p.exists() {
            return p;
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for cand in [
                dir.join("../assets/student.onnx"),
                dir.join("assets/student.onnx"),
                dir.join("../share/bsnobs-rs/student.onnx"),
            ] {
                if cand.exists() {
                    return cand;
                }
            }
        }
    }
    PathBuf::from("assets/student.onnx")
}

impl AppState {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let weights = bundled_weights();
        let mut s = Self {
            image_dir: None,
            image_files: Vec::new(),
            using_bundled: true,
            weights_path: weights,
            device: Device::Auto,

            analyzer: None,
            model_status: "Not loaded.".into(),
            model_load_time_ms: None,
            model_loading: false,
            model_load_start: None,
            model_load_rx: None,
            resolved_device: None,

            params: AnalysisParameters::default(),
            conf: 0.55,
            imgsz: INPUT_SIZE,
            iou: 0.45,
            max_det: 1000,
            reject_static: false,
            static_cfg: StaticFilterConfig::default(),

            results: None,
            current_frame: 0,
            texture: None,
            texture_for_frame: None,
            zoom: 1.0,

            worker_rx: None,
            in_progress: false,
            progress: (0, 0),
            status: "Pick a folder of microscopy images to begin.".into(),
            start_time: Instant::now(),
        };
        // Auto-load the bundled model on startup.
        s.start_model_load();
        s
    }

    fn start_model_load(&mut self) {
        if self.model_loading { return; }
        self.analyzer = None;
        self.model_load_time_ms = None;
        self.resolved_device = None;
        self.model_status = "Loading model…".into();
        self.model_loading = true;
        self.model_load_start = Some(Instant::now());

        let (tx, rx) = mpsc::channel();
        self.model_load_rx = Some(rx);
        let weights = self.weights_path.clone();
        let device = self.device;
        let resolved = resolve_device(device);
        thread::spawn(move || {
            let result = StudentAnalyzer::load(&weights, device)
                .map(|a| (a, resolved))
                .map_err(|e| e.to_string());
            let _ = tx.send(result);
        });
    }

    fn show_loading_overlay(&self, ctx: &egui::Context) {
        let screen = ctx.screen_rect();
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("loading-dim"),
        ));
        painter.rect_filled(screen, 0.0, Color32::from_black_alpha(160));

        let elapsed_ms = self
            .model_load_start
            .map(|t| t.elapsed().as_millis())
            .unwrap_or(0);

        egui::Area::new(egui::Id::new("loading-overlay"))
            .order(egui::Order::Foreground)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .interactable(false)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style())
                    .inner_margin(egui::Margin::same(24))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.add(egui::Spinner::new().size(28.0));
                            ui.vertical(|ui| {
                                ui.label(
                                    egui::RichText::new("Loading model…")
                                        .heading()
                                        .strong(),
                                );
                                ui.label(
                                    egui::RichText::new(format!("{} ms", elapsed_ms))
                                        .monospace()
                                        .weak(),
                                );
                            });
                        });
                    });
            });
    }

    fn drain_model_load(&mut self, _ctx: &egui::Context) {
        if self.model_load_rx.is_none() { return; }
        let maybe = self.model_load_rx.as_ref().unwrap().try_recv();
        match maybe {
            Ok(Ok((analyzer, resolved))) => {
                let ms = self.model_load_start
                    .map(|t| t.elapsed().as_millis()).unwrap_or(0);
                self.analyzer = Some(analyzer);
                self.model_load_time_ms = Some(ms);
                self.resolved_device = Some(resolved);
                let dev_show = if self.device == Device::Auto {
                    format!("{} ({})", self.device.label(), resolved.label())
                } else {
                    self.device.label().to_string()
                };
                self.model_status = format!(
                    "Loaded.\nDevice: {}\nLoad time: {} ms",
                    dev_show, ms
                );
                self.status = format!("Model loaded in {ms} ms (device: {dev_show}).");
                self.model_loading = false;
                self.model_load_rx = None;
            }
            Ok(Err(e)) => {
                self.model_status = format!("Load failed: {e}");
                self.status = format!("Model load failed: {e}");
                self.model_loading = false;
                self.model_load_rx = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
            Err(mpsc::TryRecvError::Disconnected) => {
                self.model_status = "Load failed: worker died.".into();
                self.model_loading = false;
                self.model_load_rx = None;
            }
        }
    }

    fn pick_folder(&mut self) {
        if let Some(p) = rfd::FileDialog::new().pick_folder() {
            let files = list_images(&p);
            if files.is_empty() {
                self.status = format!("No images in {}", p.display());
                return;
            }
            self.status = format!("Folder: {} ({} images)", p.display(), files.len());
            self.image_files = files;
            self.image_dir = Some(p);
        }
    }

    fn pick_weights(&mut self) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("ONNX model", &["onnx"])
            .pick_file()
        {
            self.weights_path = p;
            self.using_bundled = false;
            self.analyzer = None;
            self.model_status = "Not loaded.".into();
            self.model_load_time_ms = None;
        }
    }

    fn reset_weights(&mut self) {
        self.weights_path = bundled_weights();
        self.using_bundled = true;
        self.analyzer = None;
        self.model_status = "Not loaded.".into();
        self.model_load_time_ms = None;
    }

    fn start_run(&mut self) {
        let Some(image_dir) = self.image_dir.clone() else { return };
        let Some(_) = self.analyzer.as_ref() else { return };
        let files = self.image_files.clone();
        // Move analyzer onto worker thread; we'll restore (a new) one when done.
        let mut analyzer = self.analyzer.take().expect("checked above");
        let params = self.params.clone();
        let conf = self.conf;
        let iou = self.iou;
        let max_det = self.max_det;
        let reject_static = self.reject_static;
        let static_cfg = self.static_cfg;
        let sample_name = image_dir
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("sample")
            .to_string();

        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.in_progress = true;
        self.start_time = Instant::now();
        self.progress = (0, files.len());
        self.status = "Running inference…".into();

        thread::spawn(move || {
            let n = files.len();
            let mut results = AnalysisResults {
                sample_name,
                parameters: params.clone(),
                frames: Vec::with_capacity(n),
            };
            for (i, path) in files.iter().enumerate() {
                let _ = tx.send(WorkerMsg::Progress(
                    i,
                    n,
                    path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string(),
                ));
                match analyzer.predict_image(path, &params, conf, iou, max_det) {
                    Ok(frame) => results.frames.push(frame),
                    Err(e) => {
                        let _ = tx.send(WorkerMsg::Failed(
                            format!("{}: {}", path.display(), e),
                            analyzer,
                        ));
                        return;
                    }
                }
            }
            let _ = tx.send(WorkerMsg::Progress(n, n, "done".into()));
            if reject_static {
                let s = static_filter::apply(&mut results, &static_cfg);
                let _ = tx.send(WorkerMsg::Progress(
                    n,
                    n,
                    format!(
                        "static filter: {} detections across {} persistent locations",
                        s.n_rejected, s.n_clusters
                    ),
                ));
            }
            let _ = tx.send(WorkerMsg::Done(results, analyzer));
        });
    }

    fn drain_worker(&mut self, ctx: &egui::Context) {
        if self.worker_rx.is_none() { return; }
        let mut drained: Vec<WorkerMsg> = Vec::new();
        {
            let rx = self.worker_rx.as_ref().unwrap();
            while let Ok(msg) = rx.try_recv() {
                drained.push(msg);
            }
        }
        for msg in drained {
            match msg {
                WorkerMsg::Progress(i, n, name) => {
                    self.progress = (i, n);
                    self.status = format!("[{i}/{n}] {name}");
                }
                WorkerMsg::Done(results, analyzer) => {
                    let elapsed = self.start_time.elapsed().as_millis();
                    self.status = format!(
                        "Done — {} bubbles across {} frames in {} ms",
                        results.total_bubbles(),
                        results.frames.len(),
                        elapsed
                    );
                    self.results = Some(results);
                    self.current_frame = 0;
                    self.texture = None;
                    self.texture_for_frame = None;
                    self.in_progress = false;
                    self.worker_rx = None;
                    self.analyzer = Some(analyzer);
                }
                WorkerMsg::Failed(err, analyzer) => {
                    self.status = format!("Inference failed: {err}");
                    self.in_progress = false;
                    self.worker_rx = None;
                    self.analyzer = Some(analyzer);
                }
            }
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
    }

    fn ensure_frame_texture(&mut self, ctx: &egui::Context) {
        let Some(res) = &self.results else { return };
        if res.frames.is_empty() {
            return;
        }
        if self.texture_for_frame == Some(self.current_frame) && self.texture.is_some() {
            return;
        }
        let frame = &res.frames[self.current_frame];
        let path = &frame.image_path;
        let img = match ImageReader::open(path).and_then(|r| r.with_guessed_format()) {
            Ok(r) => match r.decode() {
                Ok(d) => d,
                Err(_) => return,
            },
            Err(_) => return,
        };
        let rgba = img.to_rgba8();
        let (w, h) = rgba.dimensions();
        let pixels: Vec<Color32> = rgba
            .pixels()
            .map(|p| Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]))
            .collect();
        let color_img = ColorImage {
            size: [w as usize, h as usize],
            pixels,
            source_size: Vec2::new(w as f32, h as f32),
        };
        let tex = ctx.load_texture(format!("frame-{}", self.current_frame), color_img, Default::default());
        self.texture = Some(tex);
        self.texture_for_frame = Some(self.current_frame);
    }

    fn export_results(&mut self) {
        let Some(results) = &self.results else { return };
        let Some(image_dir) = &self.image_dir else { return };
        let ts = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
        let conf_tag = format!("conf{:02}", (self.conf * 100.0).round() as u32);
        let suffix = if self.reject_static { "_static_rs" } else { "_rs" };
        let default_name = format!(
            "{}_count_{}{}_{}",
            image_dir.file_name().and_then(|s| s.to_str()).unwrap_or("sample"),
            conf_tag,
            suffix,
            ts
        );
        let default_dir = image_dir.parent().unwrap_or(image_dir).join(&default_name);

        let chosen = rfd::FileDialog::new()
            .set_directory(image_dir.parent().unwrap_or(image_dir))
            .set_title(&format!("Choose parent dir; will create {}", default_name))
            .pick_folder();
        let target = match chosen {
            Some(parent) => parent.join(&default_name),
            None => default_dir,
        };
        if let Err(e) = std::fs::create_dir_all(&target) {
            self.status = format!("Could not create {}: {}", target.display(), e);
            return;
        }
        match exporter::export(results, &target) {
            Ok(()) => {
                self.status = format!("Exported to {}", target.display());
                if let Err(e) = write_metadata(self, results, &target) {
                    self.status = format!("Exported, but metadata failed: {e}");
                }
            }
            Err(e) => self.status = format!("Export failed: {e}"),
        }
    }
}

fn write_metadata(
    app: &AppState,
    results: &AnalysisResults,
    target: &Path,
) -> Result<()> {
    let n_static = results
        .frames
        .iter()
        .flat_map(|f| f.bubbles.iter())
        .filter(|b| b.is_static)
        .count();
    let meta = serde_json::json!({
        "command": "gui-rs",
        "timestamp": chrono::Local::now().to_rfc3339(),
        "image_dir": app.image_dir.as_ref().map(|p| p.display().to_string()),
        "output_dir": target.display().to_string(),
        "weights": app.weights_path.display().to_string(),
        "weights_bundled": app.using_bundled,
        "device": app.device.label(),
        "yolo": {
            "conf": app.conf,
            "iou": app.iou,
            "imgsz": app.imgsz,
            "max_det": app.max_det,
        },
        "physics": {
            "scale_um_per_pixel": app.params.scale_um_per_pixel,
            "sample_volume_per_frame_uL": app.params.sample_volume_per_frame_ul,
            "min_diameter_um": app.params.min_diameter_um,
            "max_diameter_um": app.params.max_diameter_um,
        },
        "static_filter": {
            "enabled": app.reject_static,
            "min_frame_frac": app.static_cfg.min_frame_frac,
            "tol_px": app.static_cfg.tol_px,
            "diameter_tol_frac": app.static_cfg.diameter_tol_frac,
        },
        "n_frames": results.frames.len(),
        "n_bubbles_accepted": results.total_bubbles(),
        "n_bubbles_rejected": results.total_rejected(),
        "n_static_rejected": n_static,
    });
    let sample = &results.sample_name;
    std::fs::write(
        target.join(format!("{sample}_run_metadata.json")),
        serde_json::to_string_pretty(&meta)?,
    )?;
    Ok(())
}

impl eframe::App for AppState {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_worker(ctx);
        self.drain_model_load(ctx);
        if self.model_loading {
            // Repaint frequently so the spinner animates and the elapsed counter ticks.
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }

        egui::SidePanel::left("settings")
            .resizable(true)
            .default_width(330.0)
            .show(ctx, |ui| {
                self.left_panel(ui);
            });

        egui::TopBottomPanel::bottom("statusbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
                ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                    let (i, n) = self.progress;
                    if n > 0 {
                        ui.label(format!("{i}/{n}"));
                        let pct = if n > 0 { i as f32 / n as f32 } else { 0.0 };
                        ui.add(egui::ProgressBar::new(pct).desired_width(180.0));
                    }
                });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.center_panel(ui, ctx);
        });

        if self.model_loading {
            self.show_loading_overlay(ctx);
        }
    }
}

impl AppState {
    fn left_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("BSnoBS — Rust");

        if let Some(d) = &self.image_dir {
            ui.label(format!("Folder: {}", d.display()));
            ui.label(format!("{} images", self.image_files.len()));
        } else {
            ui.label(egui::RichText::new("no folder selected").italics());
        }
        if ui.button("Select folder…").clicked() {
            self.pick_folder();
        }
        ui.separator();

        // ---- Model panel
        egui::CollapsingHeader::new("Model")
            .default_open(true)
            .show(ui, |ui| {
                let tag = if self.using_bundled { "bundled (default)" } else { "custom" };
                ui.label(format!("{tag}: {}", self.weights_path.display()));
                ui.horizontal(|ui| {
                    if ui.button("Pick .onnx…").clicked() { self.pick_weights(); }
                    if ui.button("Use bundled").clicked() { self.reset_weights(); }
                });
                ui.horizontal(|ui| {
                    ui.label("Device:");
                    egui::ComboBox::from_id_salt("device")
                        .selected_text(self.device.label())
                        .show_ui(ui, |ui| {
                            for d in [Device::Auto, Device::Cpu, Device::CoreML] {
                                if ui.selectable_label(self.device == d, d.label()).clicked() {
                                    if self.device != d {
                                        self.device = d;
                                        self.analyzer = None;
                                        self.model_status = "Not loaded.".into();
                                        self.model_load_time_ms = None;
                                    }
                                }
                            }
                        });
                });
                let btn_text = if self.analyzer.is_some() { "Reload model" } else { "Load model" };
                let load_enabled = !self.model_loading;
                if ui.add_enabled(load_enabled, egui::Button::new(btn_text)).clicked() {
                    self.start_model_load();
                }
                if !self.model_loading {
                    ui.label(&self.model_status);
                }
            });
        ui.separator();

        // ---- Detection panel
        egui::CollapsingHeader::new("Detection")
            .default_open(true)
            .show(ui, |ui| {
                ui.add(Slider::new(&mut self.conf, 0.01..=1.0).text("Confidence"));
                ui.add(Slider::new(&mut self.iou, 0.05..=0.95).text("NMS IoU"));
                ui.horizontal(|ui| {
                    ui.label("Max detections");
                    ui.add(egui::DragValue::new(&mut self.max_det).speed(10).range(10..=10_000));
                });
                ui.label(format!("Inference resolution: {}×{} (fixed)", self.imgsz, self.imgsz));
            });
        ui.separator();

        // ---- Physics panel
        egui::CollapsingHeader::new("Physics")
            .default_open(true)
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Scale (μm/px)");
                    ui.add(egui::DragValue::new(&mut self.params.scale_um_per_pixel).speed(0.001).max_decimals(4));
                });
                ui.horizontal(|ui| {
                    ui.label("Volume (μL/frame)");
                    ui.add(egui::DragValue::new(&mut self.params.sample_volume_per_frame_ul).speed(0.00001).max_decimals(6));
                });
                ui.horizontal(|ui| {
                    ui.label("Min diam (μm)");
                    ui.add(egui::DragValue::new(&mut self.params.min_diameter_um).speed(0.1));
                });
                ui.horizontal(|ui| {
                    ui.label("Max diam (μm)");
                    ui.add(egui::DragValue::new(&mut self.params.max_diameter_um).speed(0.5));
                });
            });
        ui.separator();

        // ---- Static dirt panel
        egui::CollapsingHeader::new("Static dirt rejection")
            .default_open(false)
            .show(ui, |ui| {
                ui.checkbox(&mut self.reject_static, "Reject persistent locations");
                ui.horizontal(|ui| {
                    ui.label("Min frame frac");
                    ui.add(egui::DragValue::new(&mut self.static_cfg.min_frame_frac).speed(0.01).range(0.05..=1.0));
                });
                ui.horizontal(|ui| {
                    ui.label("Centroid tol (px)");
                    ui.add(egui::DragValue::new(&mut self.static_cfg.tol_px).speed(0.5));
                });
                ui.horizontal(|ui| {
                    ui.label("Diam tol");
                    ui.add(egui::DragValue::new(&mut self.static_cfg.diameter_tol_frac).speed(0.05));
                });
            });
        ui.separator();

        let can_run = self.image_dir.is_some() && self.analyzer.is_some() && !self.in_progress;
        let run_text = if self.in_progress { "Running…" } else { "Run inference" };
        if ui.add_enabled(can_run, egui::Button::new(run_text)).clicked() {
            self.start_run();
        }

        let can_export = self.results.is_some();
        if ui.add_enabled(can_export, egui::Button::new("Export results…")).clicked() {
            self.export_results();
        }

        if let Some(r) = &self.results {
            ui.separator();
            ui.monospace(format!(
                "Frames        : {}\n\
                 Accepted      : {}\n\
                 Rejected      : {}",
                r.frames.len(),
                r.total_bubbles(),
                r.total_rejected(),
            ));
            let diams = r.diameters_valid();
            if !diams.is_empty() {
                let mean: f32 = diams.iter().sum::<f32>() / diams.len() as f32;
                let mut sorted = diams.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = sorted[sorted.len() / 2];
                ui.monospace(format!(
                    "Mean diameter : {:.2} μm\n\
                     Median diam   : {:.2} μm",
                    mean, median
                ));
            }
        }
    }

    fn center_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let n_frames = self.results.as_ref().map(|r| r.frames.len()).unwrap_or(0);
        ui.horizontal(|ui| {
            if ui.button("◀").clicked() && self.current_frame > 0 {
                self.current_frame -= 1;
            }
            let mut idx = self.current_frame as i64;
            if n_frames > 0 {
                ui.add(Slider::new(&mut idx, 0..=(n_frames as i64 - 1)).show_value(true));
                self.current_frame = idx.clamp(0, n_frames as i64 - 1) as usize;
            } else {
                ui.label("(no frames)");
            }
            if ui.button("▶").clicked() && n_frames > 0 && self.current_frame + 1 < n_frames {
                self.current_frame += 1;
            }
            ui.label(if n_frames > 0 {
                format!("{}/{}", self.current_frame + 1, n_frames)
            } else {
                "—/—".into()
            });
        });

        // Keyboard navigation
        if n_frames > 0 {
            let (next, prev) = ui.input(|i| (
                i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::Space),
                i.key_pressed(egui::Key::ArrowLeft),
            ));
            if next && self.current_frame + 1 < n_frames { self.current_frame += 1; }
            if prev && self.current_frame > 0 { self.current_frame -= 1; }
        }

        self.ensure_frame_texture(ctx);
        ui.separator();

        if let Some(res) = self.results.as_ref() {
            if let Some(tex) = self.texture.clone() {
                let frame = &res.frames[self.current_frame];
                let avail = ui.available_size();
                let img_size = tex.size_vec2();
                let scale = (avail.x / img_size.x)
                    .min(avail.y / img_size.y)
                    .min(self.zoom);
                let disp = img_size * scale;
                let (rect, resp) = ui.allocate_exact_size(disp, Sense::hover());
                let painter = ui.painter_at(rect);
                // image
                painter.image(
                    tex.id(),
                    rect,
                    Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
                draw_overlays(&painter, rect, scale, frame, res.parameters.scale_um_per_pixel);

                // Per-frame stats
                ui.with_layout(Layout::left_to_right(egui::Align::Center), |ui| {
                    let n_static = frame.bubbles.iter().filter(|b| b.is_static).count();
                    ui.monospace(format!(
                        "frame: {}   valid: {}   rejected: {}   static: {}",
                        frame.frame_name, frame.num_valid, frame.num_rejected, n_static,
                    ));
                });
                let _ = resp;
            } else {
                ui.label("Loading image…");
            }
        } else {
            ui.centered_and_justified(|ui| {
                ui.heading("Pick a folder, load the model, then click Run.");
            });
        }
    }
}

fn draw_overlays(
    painter: &egui::Painter,
    rect: Rect,
    scale: f32,
    frame: &FrameResult,
    um_per_pixel: f32,
) {
    let origin = rect.min;
    for b in &frame.bubbles {
        let cx = origin.x + b.centroid_x * scale;
        let cy = origin.y + b.centroid_y * scale;
        let r_px = (b.diameter_um / 2.0) / um_per_pixel;
        let r = r_px * scale;
        let (color, dashed) = if b.is_static {
            (Color32::from_rgb(255, 234, 0), true)
        } else if b.is_valid {
            (Color32::from_rgb(0, 230, 118), false)
        } else {
            (Color32::from_rgb(255, 23, 68), false)
        };
        let stroke = Stroke::new(1.5, color);
        if dashed {
            // crude dash: draw shorter arcs
            let n = 24;
            for k in 0..n {
                if k % 2 == 1 {
                    continue;
                }
                let a0 = (k as f32 / n as f32) * std::f32::consts::TAU;
                let a1 = ((k + 1) as f32 / n as f32) * std::f32::consts::TAU;
                let p0 = Pos2::new(cx + r * a0.cos(), cy + r * a0.sin());
                let p1 = Pos2::new(cx + r * a1.cos(), cy + r * a1.sin());
                painter.line_segment([p0, p1], stroke);
            }
        } else {
            painter.circle_stroke(Pos2::new(cx, cy), r, stroke);
        }
        if b.is_valid {
            painter.text(
                Pos2::new(cx + r + 2.0, cy - r - 2.0),
                egui::Align2::LEFT_BOTTOM,
                format!("{:.1}", b.diameter_um),
                egui::FontId::monospace(9.0),
                color,
            );
        }
    }
}

