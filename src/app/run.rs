use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use eframe::egui::{self, Color32, Vec2};

use crate::inference::StudentAnalyzer;
use crate::static_filter;
use crate::types::AnalysisResults;

use super::workspace::WorkspaceEntry;
use super::{AppState, ResultEntry};

#[derive(Default, Clone, Copy)]
pub(super) struct RunProgress {
    pub folder_idx: usize,
    pub folder_total: usize,
    pub frame_idx: usize,
    pub frame_total: usize,
}

pub(super) enum WorkerMsg {
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

impl AppState {
    pub(super) fn start_run(&mut self) {
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
                    selection_order: None,
                    static_note,
                }));
            }
            let _ = tx.send(WorkerMsg::AllDone(analyzer));
        });
    }

    pub(super) fn drain_worker(&mut self, ctx: &egui::Context) {
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

    pub(super) fn run_button(&mut self, ui: &mut egui::Ui) {
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
}
