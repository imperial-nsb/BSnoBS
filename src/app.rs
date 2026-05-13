use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use anyhow::Result;
use eframe::egui::{
    self, Color32, ColorImage, Layout, Pos2, Rect, Sense, Slider, Stroke, TextureHandle, Vec2,
};
use image::ImageReader;

use crate::exporter::{
    self, ExportOpts, HistAxisMode,
};
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
    hist_mode: HistMode,
    edit_mode: bool,
    hovered_bubble: Option<usize>,
    suppress_hover_bubble: Option<usize>,
    undo_stack: Vec<UndoAction>,

    // Export options
    export_csv: bool,
    export_png: bool,
    export_hist: bool,

    // Worker
    worker_rx: Option<mpsc::Receiver<WorkerMsg>>,
    in_progress: bool,
    progress: RunProgress,
    status: String,
    start_time: Instant,
}

enum UndoAction {
    ToggleBubble { result_idx: usize, frame_idx: usize, bubble_idx: usize, was_valid: bool, was_static: bool },
    DiscardFrame  { result_idx: usize, frame_idx: usize, frame: crate::types::FrameResult },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum HistMode {
    AvgPerImage,
    Counts,
    Percent,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum SettingsSection {
    Model,
    Detection,
    Physics,
    Static,
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

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct SavedDefaults {
    device: Device,
    conf: f32,
    iou: f32,
    max_det: usize,
    reject_static: bool,
    params: AnalysisParameters,
    static_cfg: StaticFilterConfig,
}

impl Default for SavedDefaults {
    fn default() -> Self {
        Self {
            device: Device::Auto,
            conf: 0.55,
            iou: 0.45,
            max_det: 1000,
            reject_static: true,
            params: AnalysisParameters::default(),
            static_cfg: StaticFilterConfig::default(),
        }
    }
}

fn defaults_path() -> Option<PathBuf> {
    let base = if cfg!(target_os = "macos") {
        std::env::var_os("HOME").map(|h| PathBuf::from(h).join("Library/Application Support/bsnobs"))
    } else if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(|p| PathBuf::from(p).join("bsnobs"))
    } else {
        std::env::var_os("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .map(|p| p.join("bsnobs"))
    };
    base.map(|p| p.join("defaults.json"))
}

fn load_saved_defaults() -> Option<SavedDefaults> {
    let path = defaults_path()?;
    let txt = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&txt).ok()
}

fn save_defaults(d: &SavedDefaults) -> Result<()> {
    let path = defaults_path().ok_or_else(|| anyhow::anyhow!("no config dir"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(d)?)?;
    Ok(())
}

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
        let defaults = load_saved_defaults().unwrap_or_default();
        let mut s = Self {
            weights_path: weights,
            using_bundled: true,
            device: defaults.device,

            analyzer: None,
            model_status: "Not loaded.".into(),
            model_load_time_ms: None,
            model_loading: false,
            model_load_start: None,
            model_load_rx: None,
            resolved_device: None,

            params: defaults.params,
            conf: defaults.conf,
            imgsz: INPUT_SIZE,
            iou: defaults.iou,
            max_det: defaults.max_det,
            reject_static: defaults.reject_static,
            static_cfg: defaults.static_cfg,
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
            show_rejected: false,
            show_static: false,
            hist_mode: HistMode::AvgPerImage,
            edit_mode: false,
            hovered_bubble: None,
            suppress_hover_bubble: None,
            undo_stack: Vec::new(),

            export_csv: true,
            export_png: false,
            export_hist: true,

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
        let Some(paths) = rfd::FileDialog::new().pick_folders() else { return };
        let mut added = 0usize;
        let mut total_images = 0usize;
        let mut skipped: Vec<String> = Vec::new();
        for p in paths {
            if self.workspace.iter().any(|w| w.path == p) {
                skipped.push(format!("already in workspace: {}", p.display()));
                continue;
            }
            let entry = Self::build_workspace_entry(&p, 0);
            if entry.image_files.is_empty() && entry.children.is_empty() {
                skipped.push(format!("no images in {}", p.display()));
                continue;
            }
            total_images += entry.total_images();
            added += 1;
            self.workspace.push(entry);
        }
        self.status = match (added, skipped.is_empty()) {
            (0, false) => skipped.join("; "),
            (0, true) => return,
            (n, true) => format!("Added {n} folder(s), {total_images} images"),
            (n, false) => format!("Added {n} folder(s), {total_images} images ({})", skipped.join("; ")),
        };
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
        let visible_idx: Vec<usize> = self
            .results_list
            .iter()
            .enumerate()
            .filter(|(_, r)| r.visible)
            .map(|(i, _)| i)
            .collect();
        if visible_idx.is_empty() {
            self.status = "Nothing visible to export.".into();
            return;
        }
        let ts = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
        let conf_tag = format!("conf{:02}", (self.conf * 100.0).round() as u32);
        let default_name = if visible_idx.len() == 1 {
            format!(
                "{}_count_{}_{}",
                self.results_list[visible_idx[0]].name, conf_tag, ts
            )
        } else {
            format!("bsnobs_run_{}_{}", conf_tag, ts)
        };
        let parent_hint = self.results_list[visible_idx[0]]
            .source_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));

        let Some(parent) = rfd::FileDialog::new()
            .set_directory(&parent_hint)
            .set_file_name(&default_name)
            .set_title("Choose export folder name")
            .save_file()
        else { return };
        let target = parent;
        if let Err(e) = std::fs::create_dir_all(&target) {
            self.status = format!("Could not create {}: {}", target.display(), e);
            return;
        }

        let opts = ExportOpts {
            csv: self.export_csv,
            png: self.export_png,
            hist: self.export_hist,
            show_bubbles: self.show_bubbles,
            show_rejected: self.show_rejected,
            show_static: self.show_static,
        };
        let hist_mode = match self.hist_mode {
            HistMode::Counts => HistAxisMode::Counts,
            HistMode::AvgPerImage => HistAxisMode::AvgPerImage,
            HistMode::Percent => HistAxisMode::Percent,
        };

        let visible_refs: Vec<&AnalysisResults> = visible_idx
            .iter()
            .map(|&i| &self.results_list[i].results)
            .collect();
        let (x_min, x_max) = exporter::global_diameter_range(&visible_refs);
        let y_max = exporter::global_y_max(&visible_refs, x_min, x_max, hist_mode);

        let total_frames: usize = visible_idx
            .iter()
            .filter(|_| opts.png)
            .map(|&i| self.results_list[i].results.frames.len())
            .sum();
        let mut done_frames = 0usize;

        for &i in &visible_idx {
            let r = &self.results_list[i];
            let sub = target.join(&r.name);
            if let Err(e) = std::fs::create_dir_all(&sub) {
                self.status = format!("Could not create {}: {}", sub.display(), e);
                return;
            }
            if let Err(e) = write_metadata(self, &r.results, &r.source_path, &sub) {
                self.status = format!("Metadata write failed for {}: {}", r.name, e);
                return;
            }
            if opts.csv {
                if let Err(e) = exporter::export_csv_accepted(&r.results, &sub) {
                    self.status = format!("CSV export failed for {}: {}", r.name, e);
                    return;
                }
            }
            if opts.hist {
                let hist_path = sub.join(format!("{}_histogram.png", r.name));
                if let Err(e) = exporter::render_sample_histogram(
                    &r.results, hist_mode, x_min, x_max, y_max, &hist_path,
                ) {
                    self.status = format!("Histogram export failed for {}: {}", r.name, e);
                    return;
                }
            }
            if opts.png {
                let overlays_dir = sub.join("overlays");
                if let Err(e) = std::fs::create_dir_all(&overlays_dir) {
                    self.status = format!("Could not create {}: {}", overlays_dir.display(), e);
                    return;
                }
                for frame in &r.results.frames {
                    let stem = frame
                        .image_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("frame");
                    let out = overlays_dir.join(format!("{stem}_overlay.png"));
                    if let Err(e) = exporter::render_overlay_png(
                        frame,
                        r.results.parameters.scale_um_per_pixel,
                        opts,
                        &out,
                    ) {
                        self.status = format!("PNG overlay failed for {}: {}", r.name, e);
                        return;
                    }
                    done_frames += 1;
                    if total_frames > 0 && done_frames % 10 == 0 {
                        self.status =
                            format!("Exporting overlays… {done_frames}/{total_frames}");
                    }
                }
            }
        }

        if opts.hist && visible_idx.len() > 1 {
            let combined: Vec<(&str, &AnalysisResults)> = visible_idx
                .iter()
                .map(|&i| {
                    (
                        self.results_list[i].name.as_str(),
                        &self.results_list[i].results,
                    )
                })
                .collect();
            let combined_path = target.join("combined_histogram.png");
            if let Err(e) = exporter::render_combined_histogram(
                &combined, hist_mode, x_min, x_max, y_max, &combined_path,
            ) {
                self.status = format!("Combined histogram failed: {}", e);
                return;
            }
        }

        self.status = format!(
            "Exported {} sample(s) to {}",
            visible_idx.len(),
            target.display()
        );
        open_in_file_manager(&target);
    }
}

fn open_in_file_manager(path: &Path) {
    #[cfg(target_os = "macos")]
    let cmd = std::process::Command::new("open").arg(path).spawn();
    #[cfg(target_os = "windows")]
    let cmd = std::process::Command::new("explorer").arg(path).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let cmd = std::process::Command::new("xdg-open").arg(path).spawn();
    let _ = cmd;
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

                egui::TopBottomPanel::bottom("controls-panel")
                    .resizable(false)
                    .frame(egui::Frame::side_top_panel(&ctx.style())
                        .inner_margin(egui::Margin::symmetric(4, 6)))
                    .show_inside(ui, |ui| {
                        self.controls_bar(ui);
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
    fn model_section(&mut self, ui: &mut egui::Ui) {
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
    }

    fn detection_section(&mut self, ui: &mut egui::Ui) {
        ui.add(Slider::new(&mut self.conf, 0.01..=1.0).text("Confidence"));
        ui.add(Slider::new(&mut self.iou, 0.05..=0.95).text("NMS IoU"));
        ui.horizontal(|ui| {
            ui.label("Max detections");
            ui.add(egui::DragValue::new(&mut self.max_det).speed(10).range(10..=10_000));
        });
        ui.label(format!("Inference resolution: {}×{} (fixed)", self.imgsz, self.imgsz));
    }

    fn physics_section(&mut self, ui: &mut egui::Ui) {
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
    }

    fn static_section(&mut self, ui: &mut egui::Ui) {
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
    }

    fn run_button(&mut self, ui: &mut egui::Ui) {
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

    /// Bottom strip of the left side panel: a stack of standard
    /// collapsing-header sections above the Run button. Because the
    /// panel is bottom-anchored, expanding one section grows the
    /// panel upward, squeezing the workspace tree above rather than
    /// pushing the Run button down.
    fn controls_bar(&mut self, ui: &mut egui::Ui) {
        let sections = [
            (SettingsSection::Model, "Model", false),
            (SettingsSection::Detection, "Detection", true),
            (SettingsSection::Physics, "Physics", true),
            (SettingsSection::Static, "Static dirt rejection", false),
        ];
        for (section, label, default_open) in sections {
            egui::CollapsingHeader::new(label)
                .id_salt(("section", section))
                .default_open(default_open)
                .show(ui, |ui| match section {
                    SettingsSection::Model => self.model_section(ui),
                    SettingsSection::Detection => self.detection_section(ui),
                    SettingsSection::Physics => self.physics_section(ui),
                    SettingsSection::Static => self.static_section(ui),
                });
        }
        ui.add_space(6.0);
        ui.vertical_centered(|ui| {
            if ui
                .button("💾 Save as default")
                .on_hover_text("Persist current settings as the app's startup defaults")
                .clicked()
            {
                let d = SavedDefaults {
                    device: self.device,
                    conf: self.conf,
                    iou: self.iou,
                    max_det: self.max_det,
                    reject_static: self.reject_static,
                    params: self.params.clone(),
                    static_cfg: self.static_cfg,
                };
                match save_defaults(&d) {
                    Ok(_) => {
                        self.status = match defaults_path() {
                            Some(p) => format!("Saved defaults to {}", p.display()),
                            None => "Saved defaults.".into(),
                        };
                    }
                    Err(e) => {
                        self.status = format!("Could not save defaults: {}", e);
                    }
                }
            }
        });
        ui.add_space(6.0);
        self.run_button(ui);
        ui.add_space(4.0);
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

        let mut remove_idx: Option<usize> = None;
        egui::ScrollArea::both()
            .id_salt("workspace-list")
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
    }

    /// Draw a compact bar histogram of bubble diameters into `rect`.
    /// `axis_min`/`axis_max` define the X-axis range and `y_max` the
    /// shared Y-axis ceiling, so multiple histograms can be visually
    /// compared. Bars are scaled in the chosen `mode` (raw counts or
    /// average bubbles per image).
    fn draw_mini_histogram(
        ui: &egui::Ui,
        rect: Rect,
        diams: &[f32],
        axis_min: f32,
        axis_max: f32,
        n_frames: usize,
        mode: HistMode,
        y_max: f32,
    ) {
        let painter = ui.painter();
        let bg = ui.visuals().extreme_bg_color;
        let bar_col = ui.visuals().widgets.inactive.fg_stroke.color;
        let label_col = ui.visuals().weak_text_color();

        painter.rect_filled(rect, 2.0, bg);

        if axis_max <= axis_min || y_max <= 0.0 {
            return;
        }

        let n_bins = 24;
        let bin_w = (axis_max - axis_min) / n_bins as f32;
        let mut counts = vec![0u32; n_bins];
        for &d in diams {
            if d < axis_min || d > axis_max { continue; }
            let idx = (((d - axis_min) / bin_w) as usize).min(n_bins - 1);
            counts[idx] += 1;
        }

        let total = diams.len() as f32;
        let denom = match mode {
            HistMode::Counts => 1.0,
            HistMode::AvgPerImage => n_frames.max(1) as f32,
            HistMode::Percent => total / 100.0,
        };

        // Reserve a strip at the top of the rect for the Y-max label so
        // tall bars in the leftmost bins don't overlap it.
        let plot_top = rect.top() + 13.0;
        let plot_bot = rect.bottom() - 4.0;
        let plot_h = plot_bot - plot_top;
        let plot_left = rect.left() + 4.0;
        let plot_right = rect.right() - 4.0;
        let plot_w = plot_right - plot_left;
        let bar_px = plot_w / n_bins as f32;

        for (i, &c) in counts.iter().enumerate() {
            if c == 0 { continue; }
            let v = c as f32 / denom;
            let h = (v / y_max).min(1.0) * plot_h;
            let x0 = plot_left + i as f32 * bar_px + 0.5;
            let x1 = x0 + (bar_px - 1.0).max(1.0);
            let y0 = plot_bot - h;
            painter.rect_filled(
                Rect::from_min_max(Pos2::new(x0, y0), Pos2::new(x1, plot_bot)),
                0.0,
                bar_col,
            );
        }

        let y_label = match mode {
            HistMode::Counts => format!("{:.0}", y_max),
            HistMode::AvgPerImage => {
                if y_max >= 10.0 { format!("{:.0}", y_max) }
                else if y_max >= 1.0 { format!("{:.1}", y_max) }
                else { format!("{:.2}", y_max) }
            }
            HistMode::Percent => format!("{:.0}%", y_max),
        };
        painter.text(
            Pos2::new(plot_left, rect.top() + 2.0),
            egui::Align2::LEFT_TOP,
            y_label,
            egui::FontId::monospace(9.0),
            label_col,
        );
    }

    fn results_panel(&mut self, ui: &mut egui::Ui) {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.heading("Results");
            ui.with_layout(Layout::right_to_left(egui::Align::Center), |ui| {
                let has_any = !self.results_list.is_empty();
                if ui
                    .add_enabled(has_any, egui::Button::new("×"))
                    .on_hover_text("Clear all results")
                    .clicked()
                {
                    self.results_list.clear();
                    self.focused_result = None;
                    self.texture = None;
                    self.texture_for = None;
                    self.current_frame = 0;
                    self.zoom = 1.0;
                    self.pan = Vec2::ZERO;
                    self.undo_stack.clear();
                    self.status = "Results cleared.".into();
                }
            });
        });
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
        let results_h = (ui.available_height() - 150.0).max(80.0);
        let focused = self.focused_result;
        let dim_border = ui.visuals().widgets.noninteractive.bg_stroke.color;
        let blue_border = Color32::from_rgb(70, 140, 220);

        // Histogram Y-axis mode toggle (kept above the result cards so it
        // sits next to the stacks it controls).
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("Histogram:").small());
            ui.selectable_value(
                &mut self.hist_mode,
                HistMode::AvgPerImage,
                "avg bub/img",
            );
            ui.selectable_value(&mut self.hist_mode, HistMode::Counts, "counts");
            ui.selectable_value(&mut self.hist_mode, HistMode::Percent, "%");
        });
        ui.add_space(2.0);

        // Global diameter range, used to keep histogram axes consistent across cards.
        let (global_min, global_max) = {
            let mut mn = f32::INFINITY;
            let mut mx = f32::NEG_INFINITY;
            for r in &self.results_list {
                for d in r.results.diameters_valid() {
                    if d < mn { mn = d; }
                    if d > mx { mx = d; }
                }
            }
            if mn.is_finite() && mx.is_finite() && mx > mn {
                (mn, mx)
            } else {
                (0.0, 1.0)
            }
        };

        // Global Y-axis ceiling across all stacks, in the active mode.
        let hist_mode = self.hist_mode;
        let global_y_max: f32 = if hist_mode == HistMode::Percent {
            100.0
        } else {
            let n_bins = 24;
            let bin_w = (global_max - global_min) / n_bins as f32;
            let mut y_max: f32 = 0.0;
            if bin_w > 0.0 {
                for r in &self.results_list {
                    let mut counts = vec![0u32; n_bins];
                    for d in r.results.diameters_valid() {
                        if d < global_min || d > global_max { continue; }
                        let idx = (((d - global_min) / bin_w) as usize).min(n_bins - 1);
                        counts[idx] += 1;
                    }
                    let max_c = *counts.iter().max().unwrap_or(&0) as f32;
                    let v = match hist_mode {
                        HistMode::Counts => max_c,
                        HistMode::AvgPerImage => {
                            max_c / r.results.frames.len().max(1) as f32
                        }
                        HistMode::Percent => unreachable!(),
                    };
                    if v > y_max { y_max = v; }
                }
            }
            if y_max > 0.0 { y_max } else { 1.0 }
        };

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
                        let accent = Color32::from_rgb(0xE3, 0xCB, 0x83);
                        let accent_hover = Color32::from_rgb(0xEC, 0xD8, 0x9F);
                        let accent_active = Color32::from_rgb(0xC9, 0xB0, 0x66);
                        ui.scope(|ui| {
                            let v = &mut ui.style_mut().visuals;
                            v.widgets.inactive.bg_stroke = Stroke::new(1.5, accent);
                            v.widgets.hovered.bg_stroke = Stroke::new(1.5, accent_hover);
                            v.widgets.active.bg_stroke = Stroke::new(1.5, accent_active);
                            ui.checkbox(&mut r.visible, "")
                                .on_hover_text("Include in export");
                        });

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

                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(format!(
                                            "frames: {}  bubbles: {}",
                                            r.results.frames.len(),
                                            r.results.total_bubbles(),
                                        ))
                                        .small()
                                        .monospace(),
                                    )
                                    .truncate(),
                                );

                                let diams = r.results.diameters_valid();
                                let mut sorted: Vec<f32> = Vec::new();
                                if !diams.is_empty() {
                                    let mean: f32 = diams.iter().sum::<f32>() / diams.len() as f32;
                                    sorted = diams.clone();
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

                                if !sorted.is_empty() {
                                    let min_d = sorted[0];
                                    let max_d = sorted[sorted.len() - 1];
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(format!(
                                                "min:  {:.2} μm  max:    {:.2} μm",
                                                min_d, max_d,
                                            ))
                                            .small()
                                            .monospace(),
                                        )
                                        .truncate(),
                                    );
                                }

                                if !sorted.is_empty() {
                                    ui.add_space(4.0);
                                    let hist_w = ui.available_width();
                                    let hist_h = 52.0_f32;
                                    let (rect, _resp) = ui.allocate_exact_size(
                                        Vec2::new(hist_w, hist_h),
                                        Sense::hover(),
                                    );
                                    Self::draw_mini_histogram(
                                        ui,
                                        rect,
                                        &sorted,
                                        global_min,
                                        global_max,
                                        r.results.frames.len(),
                                        hist_mode,
                                        global_y_max,
                                    );
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::Label::new(
                                                egui::RichText::new(format!("{:.1}", global_min))
                                                    .small()
                                                    .monospace()
                                                    .weak(),
                                            )
                                            .truncate(),
                                        );
                                        ui.with_layout(
                                            Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                ui.add(
                                                    egui::Label::new(
                                                        egui::RichText::new(format!(
                                                            "{:.1} μm",
                                                            global_max,
                                                        ))
                                                        .small()
                                                        .monospace()
                                                        .weak(),
                                                    )
                                                    .truncate(),
                                                );
                                            },
                                        );
                                    });
                                }

                                if is_focused {
                                    let n_static = r.results.frames.iter()
                                        .flat_map(|f| f.bubbles.iter())
                                        .filter(|b| b.is_static)
                                        .count();
                                    ui.add_space(2.0);
                                    ui.add(
                                        egui::Label::new(
                                            egui::RichText::new(format!(
                                                "rejected: {}  static: {}",
                                                r.results.total_rejected(),
                                                n_static,
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
        let any_visible = self.results_list.iter().any(|r| r.visible);
        let can_export = any_visible
            && !self.in_progress
            && (self.export_csv || self.export_png || self.export_hist);
        let accent = Color32::from_rgb(0xE3, 0xCB, 0x83);
        let accent_hover = Color32::from_rgb(0xEC, 0xD8, 0x9F);
        let accent_active = Color32::from_rgb(0xC9, 0xB0, 0x66);

        ui.vertical_centered(|ui| {
            let txt = if any_visible { " " } else { "Select stacks above for export…" };
            ui.label(egui::RichText::new(txt).italics().weak().small());
        });
        ui.add_space(4.0);

        ui.vertical_centered(|ui| {
            let btn_w = (ui.available_width() * 0.7).clamp(140.0, 240.0);
            ui.scope(|ui| {
                let visuals = &mut ui.style_mut().visuals;
                visuals.widgets.inactive.weak_bg_fill = accent;
                visuals.widgets.hovered.weak_bg_fill = accent_hover;
                visuals.widgets.active.weak_bg_fill = accent_active;
                let btn = egui::Button::new(
                    egui::RichText::new("Export…")
                        .strong()
                        .color(Color32::from_rgb(40, 30, 10))
                        .size(16.0),
                )
                .min_size(Vec2::new(btn_w, 38.0));
                if ui.add_enabled(can_export, btn).clicked() {
                    self.export_visible_results();
                }
            });
            ui.add_space(6.0);
        });
        ui.horizontal(|ui| {
            let approx_row_w = 170.0;
            let pad = ((ui.available_width() - approx_row_w) / 2.0).max(0.0);
            ui.add_space(pad);
            ui.scope(|ui| {
                let visuals = &mut ui.style_mut().visuals;
                visuals.widgets.inactive.bg_stroke = Stroke::new(1.5, accent);
                visuals.widgets.hovered.bg_stroke = Stroke::new(1.5, accent_hover);
                visuals.widgets.active.bg_stroke = Stroke::new(1.5, accent_active);
                ui.checkbox(&mut self.export_csv, "csv");
                ui.checkbox(&mut self.export_png, "png");
                ui.checkbox(&mut self.export_hist, "hist");
            });
        });
        ui.add_space(16.0);
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
                let can_undo = !self.undo_stack.is_empty();
                if ui
                    .add_enabled(can_undo, egui::Button::new("↩ Undo"))
                    .on_hover_text("Undo last edit")
                    .clicked()
                {
                    if let Some(action) = self.undo_stack.pop() {
                        match action {
                            UndoAction::ToggleBubble { result_idx, frame_idx, bubble_idx, was_valid, was_static } => {
                                if let Some(res) = self.results_list.get_mut(result_idx) {
                                    if let Some(frame) = res.results.frames.get_mut(frame_idx) {
                                        if let Some(bubble) = frame.bubbles.get_mut(bubble_idx) {
                                            bubble.is_valid  = was_valid;
                                            bubble.is_static = was_static;
                                        }
                                        frame.num_valid    = frame.bubbles.iter().filter(|b| b.is_valid).count();
                                        frame.num_rejected = frame.bubbles.iter().filter(|b| !b.is_valid).count();
                                    }
                                }
                            }
                            UndoAction::DiscardFrame { result_idx, frame_idx, frame } => {
                                if let Some(res) = self.results_list.get_mut(result_idx) {
                                    let idx = frame_idx.min(res.results.frames.len());
                                    res.results.frames.insert(idx, frame);
                                    if self.focused_result == Some(result_idx) {
                                        self.current_frame = frame_idx;
                                        self.texture = None;
                                        self.texture_for = None;
                                    }
                                }
                            }
                        }
                    }
                }
                if ui
                    .add_enabled(has_results, egui::Button::new("🗑 Discard"))
                    .on_hover_text("Remove this image from the results")
                    .clicked()
                {
                    if let Some(focus_idx) = self.focused_result {
                        if let Some(res) = self.results_list.get_mut(focus_idx) {
                            let fi = self.current_frame;
                            if fi < res.results.frames.len() {
                                let removed = res.results.frames.remove(fi);
                                self.undo_stack.push(UndoAction::DiscardFrame {
                                    result_idx: focus_idx,
                                    frame_idx: fi,
                                    frame: removed,
                                });
                                let new_len = res.results.frames.len();
                                if new_len == 0 {
                                    self.current_frame = 0;
                                } else {
                                    self.current_frame = fi.min(new_len - 1);
                                }
                                self.texture = None;
                                self.texture_for = None;
                            }
                        }
                    }
                }
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
        painter.rect_filled(image_rect, 0.0, ui.visuals().extreme_bg_color);





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
                                    let fi = self.current_frame;
                                    let frame = &mut res.results.frames[fi];
                                    if let Some(bubble) = frame.bubbles.get_mut(bi) {
                                        let prev_valid  = bubble.is_valid;
                                        let prev_static = bubble.is_static;
                                        if bubble.is_valid {
                                            bubble.is_valid = false;
                                        } else {
                                            bubble.is_valid = true;
                                            bubble.is_static = false;
                                        }
                                        self.undo_stack.push(UndoAction::ToggleBubble {
                                            result_idx: focus_idx,
                                            frame_idx:  fi,
                                            bubble_idx: bi,
                                            was_valid:  prev_valid,
                                            was_static: prev_static,
                                        });
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
                ui.visuals().weak_text_color(),
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
                    ui.add_enabled(
                        false,
                        Slider::new(&mut 0i64, 0..=1).show_value(false).trailing_fill(true),
                    );
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
        if self.edit_mode && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.edit_mode = false;
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
                                    "{}/{} folders · {}/{} frames",
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
