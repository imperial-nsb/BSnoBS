use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use eframe::egui::{self, Color32};

use crate::inference::{Device, StudentAnalyzer};

use super::AppState;

pub(super) fn resolve_device(d: Device) -> Device {
    match d {
        Device::Auto => {
            if cfg!(target_os = "macos") { Device::CoreML } else { Device::Cpu }
        }
        other => other,
    }
}

pub(super) fn bundled_weights() -> PathBuf {
    PathBuf::from("<embedded>")
}

impl AppState {
    pub(super) fn start_model_load(&mut self) {
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

    pub(super) fn show_loading_overlay(&self, ctx: &egui::Context) {
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

    pub(super) fn drain_model_load(&mut self, _ctx: &egui::Context) {
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

    pub(super) fn pick_weights(&mut self) {
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

    pub(super) fn reset_weights(&mut self) {
        self.weights_path = bundled_weights();
        self.using_bundled = true;
        self.analyzer = None;
        self.model_status = "Not loaded.".into();
        self.model_load_time_ms = None;
    }
}
