use std::path::PathBuf;

use anyhow::Result;
use eframe::egui::{self, Slider};

use crate::inference::Device;
use crate::static_filter::StaticFilterConfig;
use crate::types::AnalysisParameters;

use super::AppState;

/// Hash-able snapshot of every setting that affects results.
/// When this changes between frames, stored results are invalidated.
#[derive(Clone, PartialEq)]
pub(super) struct SettingsSnapshot {
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

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum SettingsSection {
    Model,
    Detection,
    Physics,
    Static,
}

fn fbits(x: f32) -> u32 { x.to_bits() }

#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub(super) struct SavedDefaults {
    pub device: Device,
    pub conf: f32,
    pub iou: f32,
    pub max_det: usize,
    pub reject_static: bool,
    pub params: AnalysisParameters,
    pub static_cfg: StaticFilterConfig,
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

pub(super) fn load_saved_defaults() -> Option<SavedDefaults> {
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

impl AppState {
    pub(super) fn snapshot_settings(&self) -> SettingsSnapshot {
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
        ui.horizontal(|ui| {
            ui.label("Injection vol (μL)")
                .on_hover_text("Total injected sample volume — used for gas-dose calc in summary.json");
            ui.add(egui::DragValue::new(&mut self.params.injection_volume_ul).speed(1.0).max_decimals(2));
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

    /// Bottom strip of the left side panel: a stack of standard
    /// collapsing-header sections above the Run button. Because the
    /// panel is bottom-anchored, expanding one section grows the
    /// panel upward, squeezing the workspace tree above rather than
    /// pushing the Run button down.
    pub(super) fn controls_bar(&mut self, ui: &mut egui::Ui) {
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
}
