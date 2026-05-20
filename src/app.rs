mod export_panel;
mod model;
mod results;
mod run;
mod settings;
mod updater_ui;
mod viewer;
mod workspace;

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

use anyhow::Result;
use eframe::egui::{self, TextureHandle, Vec2};

use crate::inference::{Device, StudentAnalyzer, INPUT_SIZE};
use crate::static_filter::StaticFilterConfig;
use crate::types::AnalysisParameters;
use crate::updater::UpdateUi;

use results::{HistMode, ResultEntry};
use run::{RunProgress, WorkerMsg};
use settings::SettingsSnapshot;
use viewer::UndoAction;
use workspace::WorkspaceEntry;

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
    next_selection_rank: u32,

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

    // Updater
    updater: UpdateUi,
    updater_auto_checked: bool,
    show_update_modal: bool,
    show_restart_modal: bool,
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

        let weights = model::bundled_weights();
        let defaults = settings::load_saved_defaults().unwrap_or_default();
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
            next_selection_rank: 0,

            export_csv: true,
            export_png: false,
            export_hist: true,

            worker_rx: None,
            in_progress: false,
            progress: RunProgress::default(),
            status: "Add folders to your workspace to begin.".into(),
            start_time: Instant::now(),

            updater: UpdateUi::default(),
            updater_auto_checked: false,
            show_update_modal: false,
            show_restart_modal: false,
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
}

// -------------------------------------------------------------------
// eframe::App
// -------------------------------------------------------------------

impl eframe::App for AppState {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain_worker(ctx);
        self.drain_model_load(ctx);
        self.drain_updater(ctx);
        if !self.updater_auto_checked {
            self.updater_auto_checked = true;
            self.updater.check();
        }
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
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    self.update_status_widget(ui);
                });
            });
        });

        self.update_modals(ctx);

        egui::CentralPanel::default().show(ctx, |ui| {
            self.center_panel(ui, ctx);
        });

        if self.model_loading {
            self.show_loading_overlay(ctx);
        }
    }
}
