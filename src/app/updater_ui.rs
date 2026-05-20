use eframe::egui;

use crate::updater::{UpdateMessage, UpdateUi};

use super::AppState;

impl AppState {
    pub(super) fn drain_updater(&mut self, ctx: &egui::Context) {
        if self.updater.busy {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
        }
        if self.updater.poll() {
            match self.updater.last.clone() {
                Some(UpdateMessage::Available { .. }) => {
                    self.show_update_modal = true;
                }
                Some(UpdateMessage::Installed { .. }) => {
                    self.show_restart_modal = true;
                }
                Some(UpdateMessage::Error(e)) => {
                    log::warn!("update check failed: {e}");
                }
                _ => {}
            }
        }
    }

    pub(super) fn update_status_widget(&mut self, ui: &mut egui::Ui) {
        let version = UpdateUi::current_version();
        if self.updater.busy {
            ui.spinner();
            ui.label("Checking for updates…");
            return;
        }
        if self.updater.available.is_some() {
            if ui.button("Update available — install").clicked() {
                self.show_update_modal = true;
            }
            ui.separator();
        }
        if ui.small_button("Check for updates").clicked() {
            self.updater.check();
        }
        ui.label(format!("v{version}"));
        if matches!(self.updater.last, Some(UpdateMessage::UpToDate)) {
            ui.separator();
            ui.label("Up to date.");
        }
    }

    pub(super) fn update_modals(&mut self, ctx: &egui::Context) {
        if self.show_update_modal {
            let mut open = true;
            let mut start_install = false;
            egui::Window::new("Update available")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.label(format!(
                        "A newer version of BSnoBS is available.\nYou are running v{}.",
                        UpdateUi::current_version()
                    ));
                    ui.add_space(8.0);
                    ui.label("Click Install to download and replace the running binary. You will need to restart the app afterwards.");
                    ui.add_space(12.0);
                    ui.horizontal(|ui| {
                        if ui.button("Install").clicked() {
                            start_install = true;
                        }
                        if ui.button("Later").clicked() {
                            self.show_update_modal = false;
                        }
                    });
                });
            if !open {
                self.show_update_modal = false;
            }
            if start_install {
                self.show_update_modal = false;
                self.updater.install();
            }
        }

        if self.show_restart_modal {
            let new_version = match &self.updater.last {
                Some(UpdateMessage::Installed { new_version }) => new_version.clone(),
                _ => "the new version".to_string(),
            };
            let mut open = true;
            egui::Window::new("Update installed")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.label(format!("BSnoBS v{new_version} has been installed."));
                    ui.add_space(8.0);
                    ui.label("Quit and relaunch BSnoBS to start using the new version.");
                    ui.add_space(12.0);
                    if ui.button("Quit now").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            if !open {
                self.show_restart_modal = false;
            }
        }
    }
}
