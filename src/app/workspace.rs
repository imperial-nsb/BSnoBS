use std::path::{Path, PathBuf};

use eframe::egui::{self, Layout};

use super::AppState;

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

pub(super) fn build_workspace_entry(dir: &Path, depth: usize) -> WorkspaceEntry {
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
                let child_entry = build_workspace_entry(&path, depth + 1);
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

impl AppState {
    pub(super) fn add_folder(&mut self) {
        let Some(paths) = rfd::FileDialog::new().pick_folders() else { return };
        let mut added = 0usize;
        let mut total_images = 0usize;
        let mut skipped: Vec<String> = Vec::new();
        for p in paths {
            if self.workspace.iter().any(|w| w.path == p) {
                skipped.push(format!("already in workspace: {}", p.display()));
                continue;
            }
            let entry = build_workspace_entry(&p, 0);
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

    pub(super) fn workspace_panel(&mut self, ui: &mut egui::Ui) {
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
}
