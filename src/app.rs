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
use crate::inference::{Device, StudentAnalyzer, INPUT_SIZE};
use crate::static_filter::{self, StaticFilterConfig};
use crate::types::{AnalysisParameters, AnalysisResults, FrameResult};

// -------------------------------------------------------------------
// Data
// -------------------------------------------------------------------

pub struct WorkspaceEntry {
    pub path: PathBuf,
    pub name: String,
    pub image_files: Vec<PathBuf>,
    pub run_enabled: bool,
    pub children: Vec<WorkspaceEntry>,
}

impl WorkspaceEntry {
    pub fn set_run_enabled_recursive(&mut self, enabled: bool) {
        self.run_enabled = enabled;
        for c in &mut self.children {
            c.set_run_enabled_recursive(enabled);
        }
    }

    pub fn total_images(&self) -> usize {
        self.image_files.len() + self.children.iter().map(|c| c.total_images()).sum::<usize>()
    }
}

pub struct ResultEntry {
    pub name: String,
    pub source_path: PathBuf,
    pub results: AnalysisResults,
    pub visible: bool, // include in export / shown in viewer dropdown
    pub static_note: Option<String>,
}

/// Hash-able snapshot of every setting that affects results.
/// When this changes between frames, stored results are invalidated.
#[derive(Clone, PartialEq)]
struct SettingsSnapshot {
    conf: u32,
    iou: u32,
    max_det: usize,
    scale: u32,
    vol: u32,
    min_d: u32,
    max_d: u32,
    reject_static: bool,
    min_frame_frac: u32,
    tol_px: u32,
    diam_tol: u32,
}

pub struct AppState {
    // Model
    weights_path: PathBuf,
    using_bundled: bool,
    device: Device,
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
    last_settings: Option<SettingsSnapshot>,

    // Workspace + results
    workspace: Vec<WorkspaceEntry>,
    results_list: Vec<ResultEntry>,
    focused_result: Option<usize>,

    // Viewer state
    logo_texture: Option<TextureHandle>,
    current_frame: usize,
    texture: Option<TextureHandle>,
    texture_for: Option<(usize, usize, bool)>, // (result idx, frame idx, bw)
    zoom: f32,
    pan: Vec2,
    bw_mode: bool,
    show_overlays: bool,
    show_bubbles: bool,
    show_rejected: bool,
    show_static: bool,
    edit_mode: bool,
    hovered_bubble: Option<usize>,
    suppress_hover_bubble: Option<usize>,

    // Worker
    worker_rx: Option<mpsc::Receiver<WorkerMsg>>,
    in_progress: bool,
    progress: RunProgress,
    status: String,
    start_time: Instant,
}

#[derive(Default, Clone, Copy)]
struct RunProgress {
    folder_idx: usize,
    folder_total: usize,
    frame_idx: usize,
    frame_total: usize,
}

enum WorkerMsg {
    Progress {
        folder_idx: usize,
        folder_total: usize,
        folder_name: String,
        frame_idx: usize,
        frame_total: usize,
        frame_name: String,
    },
    FolderDone(ResultEntry),
    AllDone(StudentAnalyzer),
    Failed(String, StudentAnalyzer),
}

// -------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------

fn resolve_device(d: Device) -> Device {
    match d {
        Device::Auto => {
            if cfg!(target_os = "macos") { Device::CoreML } else { Device::Cpu }
        }
        other => other,
    }
}

fn bundled_weights() -> PathBuf {
    PathBuf::from("<embedded>")
}

fn fbits(x: f32) -> u32 { x.to_bits() }

// -------------------------------------------------------------------
// AppState impl
// -------------------------------------------------------------------

impl AppState {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut fonts = egui::FontDefinitions::default();
        fonts.font_data.insert(
            "Inter".to_owned(),
            std::sync::Arc::new(egui::FontData::from_static(include_bytes!("../assets/Inter-Regular.ttf"))),
        );
        fonts
            .families
            .entry(egui::FontFamily::Proportional)
            .or_default()
            .insert(0, "Inter".to_owned());
        cc.egui_ctx.set_fonts(fonts);
        egui_extras::install_image_loaders(&cc.egui_ctx);

        let weights = bundled_weights();
        let mut s = Self {
            weights_path: weights,
            using_bundled: true,
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
            reject_static: true,
            static_cfg: StaticFilterConfig::default(),
            last_settings: None,

            workspace: Vec::new(),
            results_list: Vec::new(),
            focused_result: None,

            logo_texture: None,
            current_frame: 0,
            texture: None,
            texture_for: None,
            zoom: 1.0,
            pan: Vec2::ZERO,
            bw_mode: true,
            show_overlays: true,
            show_bubbles: true,
            show_rejected: true,
            show_static: true,
            edit_mode: false,
            hovered_bubble: None,
            suppress_hover_bubble: None,

            worker_rx: None,
            in_progress: false,
            progress: RunProgress::default(),
            status: "Add folders to your workspace to begin.".into(),
            start_time: Instant::now(),
        };
        s.last_settings = Some(s.snapshot_settings());

        let logo_bytes = include_bytes!("../assets/logo.png");
        if let Ok(img) = image::load_from_memory(logo_bytes) {
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            let color_image = egui::ColorImage::from_rgba_unmultiplied(
                [w as usize, h as usize],
                rgba.as_raw(),
            );
            s.logo_texture = Some(cc.egui_ctx.load_texture(
                "logo",
                color_image,
                egui::TextureOptions::LINEAR,
            ));
        }

        s.start_model_load();
        s
    }

    fn snapshot_settings(&self) -> SettingsSnapshot {
        SettingsSnapshot {
            conf: fbits(self.conf),
            iou: fbits(self.iou),
            max_det: self.max_det,
            scale: fbits(self.params.scale_um_per_pixel),
            vol: fbits(self.params.sample_volume_per_frame_ul),
            min_d: fbits(self.params.min_diameter_um),
            max_d: fbits(self.params.max_diameter_um),
            reject_static: self.reject_static,
            min_frame_frac: fbits(self.static_cfg.min_frame_frac),
            tol_px: fbits(self.static_cfg.tol_px),
            diam_tol: fbits(self.static_cfg.diameter_tol_frac),
        }
    }

    fn maybe_invalidate_results(&mut self) {
        if self.in_progress { return; }
        let now = self.snapshot_settings();
        let changed = match &self.last_settings {
            Some(prev) => prev != &now,
            None => true,
        };
        self.last_settings = Some(now);
        if changed && !self.results_list.is_empty() {
            self.results_list.clear();
            self.focused_result = None;
            self.texture = None;
            self.texture_for = None;
            self.current_frame = 0;
            self.status = "Settings changed — previous results cleared.".into();
        }
    }

    // ------------- Model loading -------------

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
        let use_bundled = self.using_bundled;
        thread::spawn(move || {
            let loaded = if use_bundled {
                StudentAnalyzer::load_from_bytes(crate::inference::BUNDLED_WEIGHTS, device)
            } else {
                StudentAnalyzer::load(&weights, device)
            };
            let result = loaded.map(|a| (a, resolved)).map_err(|e| e.to_string());
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
                                ui.label(egui::RichText::new("Loading model…").heading().strong());
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
                let ms = self.model_load_start.map(|t| t.elapsed().as_millis()).unwrap_or(0);
                self.analyzer = Some(analyzer);
                self.model_load_time_ms = Some(ms);
                self.resolved_device = Some(resolved);
                let dev_show = if self.device == Device::Auto {
                    format!("{} ({})", self.device.label(), resolved.label())
                } else {
                    self.device.label().to_string()
                };
                self.model_status = format!("Loaded.\nDevice: {}\nLoad time: {} ms", dev_show, ms);
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

    // ------------- Workspace -------------

    fn build_workspace_entry(dir: &Path, depth: usize) -> WorkspaceEntry {
        let name = dir.file_name().and_then(|s| s.to_str()).unwrap_or("folder").to_string();
        let mut image_files = Vec::new();
        let mut children = Vec::new();

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                        let l = ext.to_lowercase();
                        if l == "tif" || l == "tiff" || l == "png" || l == "jpg" || l == "jpeg" {
                            image_files.push(path);
                        }
                    }
                } else if path.is_dir() && depth < 5 {
                    let child_entry = Self::build_workspace_entry(&path, depth + 1);
                    if !child_entry.image_files.is_empty() || !child_entry.children.is_empty() {
                        children.push(child_entry);
                    }
                }
            }
        }

        image_files.sort();
        children.sort_by(|a, b| a.name.cmp(&b.name));

        WorkspaceEntry {
            path: dir.to_path_buf(),
            name,
            image_files,
            run_enabled: true,
            children,
        }
    }

    fn add_folder(&mut self) {
        if let Some(p) = rfd::FileDialog::new().pick_folder() {
            if self.workspace.iter().any(|w| w.path == p) {
                self.status = format!("Already in workspace: {}", p.display());
                return;
            }
            let entry = Self::build_workspace_entry(&p, 0);
            if entry.image_files.is_empty() && entry.children.is_empty() {
                self.status = format!("No images in {}", p.display());
                return;
            }
            self.status = format!("Added {} ({} images)", entry.name, entry.total_images());
            self.workspace.push(entry);
        }
    }

    // ------------- Run -------------

    fn start_run(&mut self) {
        if self.analyzer.is_none() { return; }
        if self.in_progress { return; }
        let mut flat_folders = Vec::new();
        fn collect_folders(entry: &WorkspaceEntry, folders: &mut Vec<(PathBuf, String, Vec<PathBuf>)>) {
            if entry.run_enabled && !entry.image_files.is_empty() {
                folders.push((entry.path.clone(), entry.name.clone(), entry.image_files.clone()));
            }
            for c in &entry.children {
                collect_folders(c, folders);
            }
        }
        for w in &self.workspace {
            collect_folders(w, &mut flat_folders);
        }

        let folders: Vec<(usize, PathBuf, String, Vec<PathBuf>)> = flat_folders
            .into_iter()
            .enumerate()
            .map(|(i, (p, n, f))| (i, p, n, f))
            .collect();
        if folders.is_empty() {
            self.status = "Nothing to run — tick at least one folder in the workspace.".into();
            return;
        }

        let mut analyzer = self.analyzer.take().expect("checked above");
        let params = self.params.clone();
        let conf = self.conf;
        let iou = self.iou;
        let max_det = self.max_det;
        let reject_static = self.reject_static;
        let static_cfg = self.static_cfg;

        // Drop any existing results for paths we're about to re-run.
        let rerun_paths: std::collections::HashSet<_> =
            folders.iter().map(|(_, p, _, _)| p.clone()).collect();
        self.results_list.retain(|r| !rerun_paths.contains(&r.source_path));

        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.in_progress = true;
        self.start_time = Instant::now();
        let folder_total = folders.len();
        self.progress = RunProgress {
            folder_idx: 0,
            folder_total,
            frame_idx: 0,
            frame_total: 0,
        };
        self.status = format!("Running inference on {folder_total} folder(s)…");

        thread::spawn(move || {
            for (fi, (_, src_path, name, files)) in folders.into_iter().enumerate() {
                let n = files.len();
                let mut results = AnalysisResults {
                    sample_name: name.clone(),
                    parameters: params.clone(),
                    frames: Vec::with_capacity(n),
                };
                for (i, path) in files.iter().enumerate() {
                    let _ = tx.send(WorkerMsg::Progress {
                        folder_idx: fi,
                        folder_total,
                        folder_name: name.clone(),
                        frame_idx: i,
                        frame_total: n,
                        frame_name: path.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string(),
                    });
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
                let mut static_note: Option<String> = None;
                if reject_static {
                    let s = static_filter::apply(&mut results, &static_cfg);
                    if s.n_rejected > 0 {
                        static_note = Some(format!(
                            "{}: static filter rejected {} detection(s) across {} cluster(s)",
                            name, s.n_rejected, s.n_clusters
                        ));
                    }
                }
                let _ = tx.send(WorkerMsg::FolderDone(ResultEntry {
                    name: name.clone(),
                    source_path: src_path,
                    results,
                    visible: false,
                    static_note,
                }));
            }
            let _ = tx.send(WorkerMsg::AllDone(analyzer));
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
                WorkerMsg::Progress {
                    folder_idx, folder_total, folder_name, frame_idx, frame_total, frame_name,
                } => {
                    self.progress = RunProgress { folder_idx, folder_total, frame_idx, frame_total };
                    self.status = format!(
                        "[{}/{}] {}  —  frame [{}/{}] {}",
                        folder_idx + 1, folder_total, folder_name,
                        frame_idx + 1, frame_total, frame_name
                    );
                }
                WorkerMsg::FolderDone(entry) => {
                    if let Some(note) = &entry.static_note {
                        self.status = note.clone();
                    }
                    self.results_list.push(entry);
                    self.focused_result = Some(self.results_list.len() - 1);
                    self.current_frame = 0;
                    self.texture = None;
                    self.texture_for = None;
                }
                WorkerMsg::AllDone(analyzer) => {
                    let elapsed = self.start_time.elapsed().as_millis();
                    let n_samples = self.progress.folder_total;
                    let total_bubbles: usize =
                        self.results_list.iter().map(|r| r.results.total_bubbles()).sum();
                    self.status = format!(
                        "Done — {} sample(s), {} bubbles total in {} ms",
                        n_samples, total_bubbles, elapsed
                    );
                    self.in_progress = false;
                    self.worker_rx = None;
                    self.analyzer = Some(analyzer);
                    // After a run completes, fix the settings baseline so post-run
                    // tweaks compare against the values that produced these results.
                    self.last_settings = Some(self.snapshot_settings());
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

    // ------------- Viewer -------------

    fn focused_results(&self) -> Option<&ResultEntry> {
        self.focused_result.and_then(|i| self.results_list.get(i))
    }

    fn ensure_frame_texture(&mut self, ctx: &egui::Context) {
        let Some(focus_idx) = self.focused_result else { return };
        let Some(res) = self.results_list.get(focus_idx) else { return };
        if res.results.frames.is_empty() { return; }
        let key = (focus_idx, self.current_frame, self.bw_mode);
        if self.texture_for == Some(key) && self.texture.is_some() {
            return;
        }
        let frame = &res.results.frames[self.current_frame];
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
        let bw = self.bw_mode;
        let pixels: Vec<Color32> = rgba
            .pixels()
            .map(|p| {
                if bw {
                    // Rec. 601 luma
                    let l = (0.299 * p[0] as f32
                        + 0.587 * p[1] as f32
                        + 0.114 * p[2] as f32)
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    Color32::from_rgba_unmultiplied(l, l, l, p[3])
                } else {
                    Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3])
                }
            })
            .collect();
        let color_img = ColorImage {
            size: [w as usize, h as usize],
            pixels,
            source_size: Vec2::new(w as f32, h as f32),
        };
        let tex = ctx.load_texture(
            format!("frame-{}-{}-{}", focus_idx, self.current_frame, bw as u8),
            color_img,
            Default::default(),
        );
        self.texture = Some(tex);
        self.texture_for = Some(key);
    }

    // ------------- Export -------------

    fn export_visible_results(&mut self) {
        let visible: Vec<&ResultEntry> = self.results_list.iter().filter(|r| r.visible).collect();
        if visible.is_empty() {
            self.status = "Nothing visible to export.".into();
            return;
        }
        let ts = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
        let conf_tag = format!("conf{:02}", (self.conf * 100.0).round() as u32);
        let suffix = if self.reject_static { "_static_rs" } else { "_rs" };
        let default_name = if visible.len() == 1 {
            format!("{}_count_{}{}_{}", visible[0].name, conf_tag, suffix, ts)
        } else {
            format!("bsnobs_run_{}{}_{}", conf_tag, suffix, ts)
        };
        let parent_hint = visible[0]
            .source_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));

        let Some(parent) = rfd::FileDialog::new()
            .set_directory(&parent_hint)
            .set_title(&format!("Choose parent dir; will create {default_name}"))
            .pick_folder()
        else { return };
        let target = parent.join(&default_name);
        if let Err(e) = std::fs::create_dir_all(&target) {
            self.status = format!("Could not create {}: {}", target.display(), e);
            return;
        }

        for r in &visible {
            let sub = if visible.len() == 1 { target.clone() } else { target.join(&r.name) };
            if let Err(e) = std::fs::create_dir_all(&sub) {
                self.status = format!("Could not create {}: {}", sub.display(), e);
                return;
            }
            if let Err(e) = exporter::export(&r.results, &sub) {
                self.status = format!("Export failed for {}: {}", r.name, e);
                return;
            }
            if let Err(e) = write_metadata(self, &r.results, &r.source_path, &sub) {
                self.status = format!("Metadata write failed for {}: {}", r.name, e);
                return;
            }
        }
        self.status = format!("Exported {} sample(s) to {}", visible.len(), target.display());
    }
}

// -------------------------------------------------------------------
// Metadata write (per-sample)
// -------------------------------------------------------------------

fn write_metadata(
    app: &AppState,
    results: &AnalysisResults,
    source_path: &Path,
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
        "image_dir": source_path.display().to_string(),
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

// -------------------------------------------------------------------
// eframe::App
// -------------------------------------------------------------------

impl eframe::App for AppState {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_worker(ctx);
        self.drain_model_load(ctx);
        if self.model_loading {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        }
        // Snap settings invalidation BEFORE drawing so widget changes from
        // last frame are caught.
        self.maybe_invalidate_results();

        let side_frame = egui::Frame::side_top_panel(&ctx.style())
            .inner_margin(egui::Margin::symmetric(12, 10));

        // Left side: workspace at top, vfill, settings at bottom.
        egui::SidePanel::left("left-side")
            .resizable(true)
            .default_width(340.0)
            .width_range(220.0..=480.0)
            .frame(side_frame)
            .show(ctx, |ui| {
                egui::TopBottomPanel::top("logo-panel")
                    .resizable(false)
                    .frame(egui::Frame::side_top_panel(&ctx.style())
                        .inner_margin(egui::Margin::symmetric(4, 6)))
                    .show_inside(ui, |ui| {
                        if let Some(tex) = &self.logo_texture {
                            let logo_h = 34.0;
                            let logo_w = logo_h * (tex.size()[0] as f32 / tex.size()[1] as f32);
                            ui.vertical_centered(|ui| {
                                ui.add(egui::Image::from_texture(
                                    egui::load::SizedTexture::new(tex.id(), Vec2::new(logo_w, logo_h)),
                                ));
                            });
                        }
                    });

                let max_settings_h = (ui.available_height() - 250.0).max(100.0);
                egui::TopBottomPanel::bottom("settings-panel")
                    .resizable(false)
                    .height_range(0.0..=max_settings_h)
                    .frame(egui::Frame::side_top_panel(&ctx.style())
                        .inner_margin(egui::Margin::symmetric(4, 8)))
                    .show_inside(ui, |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            self.settings_panel(ui);
                        });
                    });

                egui::CentralPanel::default()
                    .frame(egui::Frame::side_top_panel(&ctx.style())
                        .inner_margin(egui::Margin::symmetric(4, 8)))
                    .show_inside(ui, |ui| {
                        self.workspace_panel(ui);
                    });
            });

        // Right side: results + export.
        egui::SidePanel::right("results-panel")
            .resizable(true)
            .default_width(290.0)
            .frame(side_frame)
            .show(ctx, |ui| {
                self.results_panel(ui);
            });

        // Status bar
        egui::TopBottomPanel::bottom("statusbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(&self.status);
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

// -------------------------------------------------------------------
// Panels
// -------------------------------------------------------------------

impl AppState {
    fn settings_panel(&mut self, ui: &mut egui::Ui) {
        egui::CollapsingHeader::new("Model")
            .default_open(false)
            .show(ui, |ui| {
                if self.using_bundled {
                    let kb = crate::inference::BUNDLED_WEIGHTS.len() as f64 / 1024.0;
                    ui.label(format!("bundled (embedded in binary, {:.0} KB)", kb));
                } else {
                    ui.label(format!("custom: {}", self.weights_path.display()));
                }
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

        egui::CollapsingHeader::new("Static dirt rejection")
            .default_open(true)
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
    }

    fn render_workspace_node(ui: &mut egui::Ui, w: &mut WorkspaceEntry, remove_idx: &mut Option<usize>, my_idx: Option<usize>) {
        ui.horizontal(|ui| {
            let mut checked = w.run_enabled;
            if ui.checkbox(&mut checked, "").changed() {
                w.set_run_enabled_recursive(checked);
            }

            ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                if let Some(i) = my_idx {
                    if ui.small_button("×").on_hover_text("Remove").clicked() {
                        *remove_idx = Some(i);
                    }
                }
                
                ui.with_layout(Layout::left_to_right(egui::Align::Center), |ui| {
                    if w.children.is_empty() {
                        ui.label(format!("{}  ({})", w.name, w.image_files.len()));
                    } else {
                        let label = if w.image_files.is_empty() {
                            format!("{}  ({} total)", w.name, w.total_images())
                        } else {
                            format!("{}  ({} here, {} total)", w.name, w.image_files.len(), w.total_images())
                        };
                        egui::CollapsingHeader::new(label)
                            .id_salt(&w.path)
                            .default_open(true)
                            .show(ui, |ui| {
                                for child in &mut w.children {
                                    Self::render_workspace_node(ui, child, remove_idx, None);
                                }
                            });
                    }
                });
            });
        });
    }

    fn workspace_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.heading("Workspace");
            ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("+").on_hover_text("Add folder…").clicked() {
                    self.add_folder();
                }
            });
        });
        ui.separator();

        let list_h = (ui.available_height() - 48.0).max(60.0);
        let mut remove_idx: Option<usize> = None;
        egui::ScrollArea::both()
            .id_salt("workspace-list")
            .max_height(list_h)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if self.workspace.is_empty() {
                    ui.label(egui::RichText::new("(no folders — click +)").italics().weak());
                }
                for (i, w) in self.workspace.iter_mut().enumerate() {
                    Self::render_workspace_node(ui, w, &mut remove_idx, Some(i));
                }
            });
        if let Some(i) = remove_idx {
            self.workspace.remove(i);
        }

        ui.add_space(10.0);
        fn any_enabled(entry: &WorkspaceEntry) -> bool {
            (entry.run_enabled && !entry.image_files.is_empty()) || entry.children.iter().any(any_enabled)
        }
        let can_run = self.analyzer.is_some()
            && !self.in_progress
            && self.workspace.iter().any(any_enabled);
        let run_text = if self.in_progress { "Running…" } else { "RUN" };
        ui.vertical_centered(|ui| {
            ui.scope(|ui| {
                let visuals = &mut ui.style_mut().visuals;
                visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(46, 160, 67);
                visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(60, 180, 80);
                visuals.widgets.active.weak_bg_fill = Color32::from_rgb(40, 140, 60);
                let btn_w = (ui.available_width() * 0.7).clamp(140.0, 240.0);
                let button = egui::Button::new(
                    egui::RichText::new(run_text).strong().color(Color32::WHITE).size(16.0),
                )
                .min_size(Vec2::new(btn_w, 38.0));
                if ui.add_enabled(can_run, button).clicked() {
                    self.start_run();
                }
            });
        });
    }

    fn results_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.heading("Results");
        ui.separator();

        // Viewer checkboxes
        ui.checkbox(&mut self.bw_mode, "B&W mode");
        ui.checkbox(&mut self.show_overlays, "Show overlays");
        ui.add_enabled_ui(self.show_overlays, |ui| {
            ui.indent("overlay-sub", |ui| {
                ui.checkbox(&mut self.show_bubbles, "Accepted");
                ui.checkbox(&mut self.show_rejected, "Rejected");
                ui.checkbox(&mut self.show_static, "Static");
            });
        });
        ui.separator();

        // Result blocks
        let mut focus_change: Option<usize> = None;
        let results_h = (ui.available_height() - 54.0).max(80.0);
        let focused = self.focused_result;
        let dim_border = ui.visuals().widgets.noninteractive.bg_stroke.color;
        let blue_border = Color32::from_rgb(70, 140, 220);

        egui::ScrollArea::vertical()
            .id_salt("results-list")
            .max_height(results_h)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                if self.results_list.is_empty() {
                    ui.label(egui::RichText::new("(no results yet)").italics().weak());
                }
                for (i, r) in self.results_list.iter_mut().enumerate() {
                    let is_focused = focused == Some(i);
                    let (stroke_w, stroke_col) = if is_focused {
                        (2.0, blue_border)
                    } else {
                        (1.0, dim_border)
                    };

                    ui.horizontal(|ui| {
                        ui.checkbox(&mut r.visible, "")
                            .on_hover_text("Include in export");

                        let block_w = ui.available_width();
                        let pad = 8.0_f32;
                        let frame = egui::Frame::new()
                            .stroke(Stroke::new(stroke_w, stroke_col))
                            .inner_margin(egui::Margin::symmetric(pad as i8, 6))
                            .corner_radius(egui::CornerRadius::same(4));

                        let resp = frame.show(ui, |ui| {
                            ui.vertical(|ui| {
                                ui.set_min_width(block_w - 2.0 * pad - 2.0 * stroke_w);
                                ui.spacing_mut().item_spacing.y = 2.0;

                                ui.add(
                                    egui::Label::new(egui::RichText::new(&r.name).strong())
                                        .truncate(),
                                );

                                let n_static = r.results.frames.iter()
                                    .flat_map(|f| f.bubbles.iter())
                                    .filter(|b| b.is_static)
                                    .count();
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(format!(
                                            "frames: {}  bubbles: {}  rejected: {}  static: {}",
                                            r.results.frames.len(),
                                            r.results.total_bubbles(),
                                            r.results.total_rejected(),
                                            n_static,
                                        ))
                                        .small()
                                        .monospace(),
                                    )
                                    .truncate(),
                                );
                                let diams = r.results.diameters_valid();
                                if !diams.is_empty() {
                                    let mean: f32 = diams.iter().sum::<f32>() / diams.len() as f32;
                                    let mut sorted = diams.clone();
                                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                                    let median = sorted[sorted.len() / 2];
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(format!(
                                                "mean: {:.2} μm  median: {:.2} μm", mean, median,
                                            ))
                                            .small()
                                            .monospace(),
                                        )
                                        .truncate(),
                                    );
                                }
                            });
                        });

                        let block_resp = ui.interact(
                            resp.response.rect,
                            ui.id().with(("result-block", i)),
                            Sense::click(),
                        );
                        if block_resp.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                        }
                        if block_resp.clicked() {
                            focus_change = Some(i);
                        }
                    });
                    ui.add_space(4.0);
                }
            });

        if let Some(i) = focus_change {
            self.focused_result = Some(i);
            self.current_frame = 0;
            self.texture = None;
            self.texture_for = None;
            self.zoom = 1.0;
            self.pan = Vec2::ZERO;
        }

        ui.add_space(8.0);
        let can_export = self.results_list.iter().any(|r| r.visible) && !self.in_progress;
        ui.vertical_centered(|ui| {
            let btn_w = (ui.available_width() * 0.7).clamp(140.0, 220.0);
            let btn = egui::Button::new(egui::RichText::new("Export…").strong().size(15.0))
                .min_size(Vec2::new(btn_w, 32.0));
            if ui.add_enabled(can_export, btn).clicked() {
                self.export_visible_results();
            }
        });
        ui.add_space(6.0);
    }


    fn center_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let focus_idx = self.focused_result;
        let n_frames = focus_idx
            .and_then(|i| self.results_list.get(i))
            .map(|r| r.results.frames.len())
            .unwrap_or(0);

        // Top row: zoom controls
        ui.horizontal(|ui| {
            ui.label("Zoom");
            ui.add(
                egui::DragValue::new(&mut self.zoom)
                    .speed(0.05)
                    .range(1.0..=8.0)
                    .max_decimals(2),
            );
            if ui.button("−").clicked() { self.zoom = (self.zoom * 0.8).max(1.0); }
            if ui.button("+").clicked() { self.zoom = (self.zoom * 1.25).min(8.0); }
            if ui.button("Fit").clicked() { self.zoom = 1.0; self.pan = Vec2::ZERO; }
            ui.separator();
            if let Some(r) = self.focused_results() {
                ui.label(format!("{}  ({} frames)", r.name, r.results.frames.len()));
            }
            ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                let has_results = self.focused_result
                    .and_then(|i| self.results_list.get(i))
                    .map(|r| !r.results.frames.is_empty())
                    .unwrap_or(false);
                ui.add_enabled(has_results, egui::Checkbox::new(&mut self.edit_mode, "✏ Edit"));
            });
        });
        ui.separator();

        // Reserve bottom strip for the slider so the image fills the rest.
        let slider_h = 32.0;
        let total = ui.available_size();
        let image_size = Vec2::new(total.x, (total.y - slider_h - 4.0).max(50.0));
        let sense = if self.edit_mode {
            Sense::click_and_drag() | Sense::hover()
        } else {
            Sense::click_and_drag()
        };
        let (image_rect, image_resp) =
            ui.allocate_exact_size(image_size, sense);

        self.ensure_frame_texture(ctx);

        // ---- Interaction: scroll-zoom around cursor, drag to pan,
        //      double-click to reset ----
        if image_resp.hovered() {
            let (raw_scroll, zoom_delta, modifiers, pointer) = ui.input(|i| (
                i.smooth_scroll_delta,
                i.zoom_delta(),
                i.modifiers,
                i.pointer.hover_pos(),
            ));
            let scroll_factor = if raw_scroll.y.abs() > 0.0 {
                let step = if modifiers.shift_only() { 0.002 } else { 0.005 };
                (raw_scroll.y * step).exp()
            } else {
                1.0
            };
            let factor = scroll_factor * zoom_delta;
            if (factor - 1.0).abs() > 1e-4 {
                let pivot = pointer.unwrap_or(image_rect.center());
                let new_zoom = (self.zoom * factor).clamp(1.0, 8.0);
                let r = new_zoom / self.zoom;
                let v = pivot - image_rect.center();
                self.pan = v * (1.0 - r) + self.pan * r;
                self.zoom = new_zoom;
            }
        }
        if !self.edit_mode || self.hovered_bubble.is_none() {
            if image_resp.dragged() {
                self.pan += image_resp.drag_delta();
            }
            let dbl = ui.input(|i| {
                i.pointer.button_double_clicked(egui::PointerButton::Primary)
                    && i.pointer.interact_pos()
                        .map(|p| image_rect.contains(p))
                        .unwrap_or(false)
            });
            if dbl {
                self.zoom = 1.0;
                self.pan = Vec2::ZERO;
            }
        }
        if image_resp.secondary_clicked() {
            let has_results = self.focused_result
                .and_then(|i| self.results_list.get(i))
                .map(|r| !r.results.frames.is_empty())
                .unwrap_or(false);
            if has_results {
                self.edit_mode = !self.edit_mode;
            }
        }

        let painter = ui.painter_at(image_rect);
        painter.rect_filled(image_rect, 0.0, Color32::from_gray(20));





        if let (Some(focus_idx), Some(tex)) = (focus_idx, self.texture.clone()) {
            if let Some(res) = self.results_list.get(focus_idx) {
                if !res.results.frames.is_empty() {
                    let frame = &res.results.frames[self.current_frame];
                    let img_size = tex.size_vec2();
                    let fit_scale =
                        (image_rect.width() / img_size.x).min(image_rect.height() / img_size.y);
                    let scale = fit_scale * self.zoom;
                    let disp = img_size * scale;
                    let center = image_rect.center() + self.pan;
                    let target_rect =
                        Rect::from_center_size(Pos2::new(center.x, center.y), disp);
                    painter.image(
                        tex.id(),
                        target_rect,
                        Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0)),
                        Color32::WHITE,
                    );
                    if self.edit_mode {
                        let border_rect = target_rect.intersect(image_rect);
                        if border_rect.is_positive() {
                            painter.rect_stroke(
                                border_rect, 0.0,
                                Stroke::new(2.0, Color32::from_rgb(255, 165, 0)),
                                egui::StrokeKind::Inside,
                            );
                        }
                    }
                    if self.show_overlays {
                        draw_overlays(
                            &painter,
                            target_rect,
                            scale,
                            frame,
                            res.results.parameters.scale_um_per_pixel,
                            self.edit_mode,
                            self.hovered_bubble,
                            self.show_bubbles,
                            self.show_rejected,
                            self.show_static,
                        );
                    }

                    // ---- Edit mode: hit-test + click-to-reject/accept ----
                    if self.edit_mode {
                        let um_per_pixel = res.results.parameters.scale_um_per_pixel;
                        let pointer = ui.input(|i| i.pointer.hover_pos());
                        let mut closest: Option<(usize, f32)> = None;
                        if let Some(mp) = pointer {
                            if image_rect.contains(mp) {
                                let origin = target_rect.min;
                                for (bi, b) in frame.bubbles.iter().enumerate() {
                                    let cx = origin.x + b.centroid_x * scale;
                                    let cy = origin.y + b.centroid_y * scale;
                                    let r_px = (b.diameter_um / 2.0) / um_per_pixel;
                                    let r_screen = r_px * scale;
                                    let dist = ((mp.x - cx).powi(2) + (mp.y - cy).powi(2)).sqrt();
                                    if dist <= r_screen.max(8.0) {
                                        if closest.is_none() || dist < closest.unwrap().1 {
                                            closest = Some((bi, dist));
                                        }
                                    }
                                }
                            }
                        }
                        self.hovered_bubble = closest.map(|(i, _)| i);

                        // Suppress hover on the bubble we just clicked until cursor leaves it
                        if let Some(suppressed) = self.suppress_hover_bubble {
                            if self.hovered_bubble == Some(suppressed) {
                                self.hovered_bubble = None;
                            } else {
                                self.suppress_hover_bubble = None;
                            }
                        }

                        if image_resp.clicked() {
                            if let Some(bi) = self.hovered_bubble {
                                if let Some(res) = self.results_list.get_mut(focus_idx) {
                                    let frame = &mut res.results.frames[self.current_frame];
                                    if let Some(bubble) = frame.bubbles.get_mut(bi) {
                                        if bubble.is_valid {
                                            bubble.is_valid = false;
                                        } else {
                                            bubble.is_valid = true;
                                            bubble.is_static = false;
                                        }
                                    }
                                    frame.num_valid = frame.bubbles.iter().filter(|b| b.is_valid).count();
                                    frame.num_rejected = frame.bubbles.iter().filter(|b| !b.is_valid).count();
                                }
                                self.suppress_hover_bubble = Some(bi);
                                self.hovered_bubble = None;
                            }
                        }

                        // Change cursor when hovering a bubble
                        if self.hovered_bubble.is_some() {
                            ctx.set_cursor_icon(egui::CursorIcon::PointingHand);
                        } else if image_rect.contains(ui.input(|i| i.pointer.hover_pos().unwrap_or_default())) {
                            ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
                        }
                    } else {
                        self.hovered_bubble = None;
                    }
                }
            }
        } else if focus_idx.is_none() {
            painter.text(
                image_rect.center(),
                egui::Align2::CENTER_CENTER,
                if self.results_list.is_empty() {
                    "Add folders to the workspace, then click RUN."
                } else {
                    "Click a result block on the right to view it."
                },
                egui::FontId::proportional(16.0),
                Color32::from_gray(180),
            );
        }

        // Bottom row: slider
        ui.add_space(8.0);
        ui.with_layout(
            egui::Layout::left_to_right(egui::Align::Center).with_main_align(egui::Align::Center),
            |ui| {
                let available = ui.available_width();
                // ◀ button, ▶ button, and label take up roughly 120px together.
                let desired_slider_width = (available - 160.0).max(100.0);
                ui.spacing_mut().slider_width = desired_slider_width;

                if ui.button("◀").clicked() && self.current_frame > 0 {
                    self.current_frame -= 1;
                }
                if n_frames > 0 {
                    let mut idx = self.current_frame as i64;
                    let max = (n_frames as i64 - 1).max(0);
                    ui.scope(|ui| {
                        let v = &mut ui.style_mut().visuals;
                        v.selection.bg_fill = Color32::from_gray(110);
                        v.widgets.inactive.fg_stroke.color = Color32::from_gray(160);
                        v.widgets.hovered.fg_stroke.color  = Color32::from_gray(200);
                        v.widgets.active.fg_stroke.color   = Color32::from_gray(220);
                        ui.add(
                            Slider::new(&mut idx, 0..=max)
                                .show_value(true)
                                .clamping(egui::SliderClamping::Always)
                                .trailing_fill(true),
                        );
                    });
                    self.current_frame = idx.clamp(0, max) as usize;
                } else {
                    ui.add_enabled(false, Slider::new(&mut 0i64, 0..=0).trailing_fill(true));
                }
                if ui.button("▶").clicked() && n_frames > 0 && self.current_frame + 1 < n_frames {
                    self.current_frame += 1;
                }
                if n_frames > 0 {
                    ui.label(format!("{}/{}", self.current_frame + 1, n_frames));
                }
            },
        );

        // Keyboard navigation
        if n_frames > 0 {
            let (next, prev) = ui.input(|i| (
                i.key_pressed(egui::Key::ArrowRight) || i.key_pressed(egui::Key::Space),
                i.key_pressed(egui::Key::ArrowLeft),
            ));
            if next && self.current_frame + 1 < n_frames { self.current_frame += 1; }
            if prev && self.current_frame > 0 { self.current_frame -= 1; }
        }
        if ui.input(|i| i.key_pressed(egui::Key::S)) {
            self.show_overlays = !self.show_overlays;
        }
        if ui.input(|i| i.key_pressed(egui::Key::E)) {
            let has_results = self.focused_result
                .and_then(|i| self.results_list.get(i))
                .map(|r| !r.results.frames.is_empty())
                .unwrap_or(false);
            if has_results {
                self.edit_mode = !self.edit_mode;
            }
        }

        // Inference progress overlay
        if self.in_progress && self.progress.folder_total > 0 {
            let p = self.progress;
            let frac_folder = if p.frame_total > 0 {
                p.frame_idx as f32 / p.frame_total as f32
            } else { 0.0 };
            let frac_total =
                (p.folder_idx as f32 + frac_folder) / p.folder_total as f32;

            egui::Area::new(egui::Id::new("inference-progress"))
                .order(egui::Order::Foreground)
                .pivot(egui::Align2::CENTER_CENTER)
                .current_pos(image_rect.center())
                .interactable(false)
                .show(ctx, |ui| {
                    egui::Frame::popup(ui.style())
                        .inner_margin(egui::Margin::same(24))
                        .show(ui, |ui| {
                            ui.vertical_centered(|ui| {
                                ui.label(egui::RichText::new("Running Inference").heading().strong());
                                ui.add_space(8.0);
                                ui.label(format!(
                                    "{}/{} folder · {}/{} frame",
                                    p.folder_idx + 1, p.folder_total,
                                    p.frame_idx + 1, p.frame_total,
                                ));
                                ui.add(egui::ProgressBar::new(frac_total).desired_width(240.0));
                            });
                        });
                });
        }
    }
}

// -------------------------------------------------------------------
// Overlay drawing
// -------------------------------------------------------------------

fn draw_overlays(
    painter: &egui::Painter,
    rect: Rect,
    scale: f32,
    frame: &FrameResult,
    um_per_pixel: f32,
    edit_mode: bool,
    hovered_bubble: Option<usize>,
    show_bubbles: bool,
    show_rejected: bool,
    show_static: bool,
) {
    let origin = rect.min;
    for (bi, b) in frame.bubbles.iter().enumerate() {
        let is_hovered = edit_mode && hovered_bubble == Some(bi);

        // Skip based on sub-toggles (but always draw hovered bubble)
        if !is_hovered {
            if b.is_static && !show_static { continue; }
            if !b.is_static && b.is_valid && !show_bubbles { continue; }
            if !b.is_static && !b.is_valid && !show_rejected { continue; }
        }

        let cx = origin.x + b.centroid_x * scale;
        let cy = origin.y + b.centroid_y * scale;
        let r_px = (b.diameter_um / 2.0) / um_per_pixel;
        let r = r_px * scale;

        // Hover: red+× for valid (reject), green+✓ for rejected/static (accept)
        let hover_is_accept = is_hovered && !b.is_valid;

        let (color, dashed) = if is_hovered && hover_is_accept {
            (Color32::from_rgb(0, 200, 0), false)
        } else if is_hovered {
            (Color32::from_rgb(255, 40, 40), false)
        } else if b.is_static {
            (Color32::from_rgb(255, 234, 0), true)
        } else if b.is_valid {
            (Color32::from_rgb(0, 200, 0), false)
        } else {
            (Color32::from_rgb(255, 23, 68), false)
        };

        let stroke_w = if is_hovered { 3.0 } else { 2.0 };
        let stroke = Stroke::new(stroke_w, color);
        if dashed {
            let n = 24;
            for k in 0..n {
                if k % 2 == 1 { continue; }
                let a0 = (k as f32 / n as f32) * std::f32::consts::TAU;
                let a1 = ((k + 1) as f32 / n as f32) * std::f32::consts::TAU;
                let p0 = Pos2::new(cx + r * a0.cos(), cy + r * a0.sin());
                let p1 = Pos2::new(cx + r * a1.cos(), cy + r * a1.sin());
                painter.line_segment([p0, p1], stroke);
            }
        } else {
            painter.circle_stroke(Pos2::new(cx, cy), r, stroke);
        }

        if is_hovered {
            let fill_color = if hover_is_accept {
                Color32::from_rgba_unmultiplied(0, 200, 0, 50)
            } else {
                Color32::from_rgba_unmultiplied(255, 40, 40, 50)
            };
            painter.circle_filled(Pos2::new(cx, cy), r, fill_color);

            let mark_size = r.min(12.0).max(4.0);
            let mark_stroke = Stroke::new(2.0, color);
            if hover_is_accept {
                // Draw ✓
                painter.line_segment(
                    [Pos2::new(cx - mark_size * 0.5, cy), Pos2::new(cx - mark_size * 0.1, cy + mark_size * 0.5)],
                    mark_stroke,
                );
                painter.line_segment(
                    [Pos2::new(cx - mark_size * 0.1, cy + mark_size * 0.5), Pos2::new(cx + mark_size * 0.5, cy - mark_size * 0.4)],
                    mark_stroke,
                );
            } else {
                // Draw ×
                painter.line_segment(
                    [Pos2::new(cx - mark_size, cy - mark_size), Pos2::new(cx + mark_size, cy + mark_size)],
                    mark_stroke,
                );
                painter.line_segment(
                    [Pos2::new(cx + mark_size, cy - mark_size), Pos2::new(cx - mark_size, cy + mark_size)],
                    mark_stroke,
                );
            }
        } else if b.is_valid {
            painter.text(
                Pos2::new(cx, cy),
                egui::Align2::CENTER_CENTER,
                format!("{:.1}", b.diameter_um),
                egui::FontId::monospace(9.0),
                color,
            );
        }
    }
}
