use std::path::PathBuf;

use eframe::egui::{self, Color32, Layout, Pos2, Rect, Sense, Stroke, Vec2};

use crate::types::AnalysisResults;

use super::AppState;

pub struct ResultEntry {
    pub name: String,
    pub source_path: PathBuf,
    pub results: AnalysisResults,
    pub visible: bool, // include in export / shown in viewer dropdown
    pub selection_order: Option<u32>, // monotonic tick rank; None = never ticked
    pub static_note: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum HistMode {
    AvgPerImage,
    Counts,
    Percent,
}

impl AppState {
    pub(super) fn focused_results(&self) -> Option<&ResultEntry> {
        self.focused_result.and_then(|i| self.results_list.get(i))
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

    pub(super) fn results_panel(&mut self, ui: &mut egui::Ui) {
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
        let mut newly_selected: Vec<usize> = Vec::new();
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

        // Select-all toggle: ticks/unticks every result's visibility checkbox.
        {
            let n_total = self.results_list.len();
            let n_visible = self.results_list.iter().filter(|r| r.visible).count();
            let mut all_checked = n_total > 0 && n_visible == n_total;
            let indeterminate = n_visible > 0 && n_visible < n_total;
            let label = if n_total == 0 {
                "Select all".to_string()
            } else {
                format!("Select all ({}/{})", n_visible, n_total)
            };
            let accent = Color32::from_rgb(0xE3, 0xCB, 0x83);
            let accent_hover = Color32::from_rgb(0xEC, 0xD8, 0x9F);
            let accent_active = Color32::from_rgb(0xC9, 0xB0, 0x66);
            let resp = ui.scope(|ui| {
                let v = &mut ui.style_mut().visuals;
                v.widgets.inactive.bg_stroke = Stroke::new(1.5, accent);
                v.widgets.hovered.bg_stroke = Stroke::new(1.5, accent_hover);
                v.widgets.active.bg_stroke = Stroke::new(1.5, accent_active);
                let cb = egui::Checkbox::new(&mut all_checked, label).indeterminate(indeterminate);
                ui.add_enabled(n_total > 0, cb)
            }).inner;
            if resp.changed() {
                if all_checked {
                    for r in self.results_list.iter_mut() {
                        if !r.visible {
                            r.visible = true;
                            r.selection_order = Some(self.next_selection_rank);
                            self.next_selection_rank = self.next_selection_rank.wrapping_add(1);
                        }
                    }
                } else {
                    for r in self.results_list.iter_mut() {
                        if r.visible {
                            r.visible = false;
                            r.selection_order = None;
                        }
                    }
                }
            }
        }

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
                            let was = r.visible;
                            let resp = ui.checkbox(&mut r.visible, "")
                                .on_hover_text("Include in export");
                            if resp.changed() {
                                if !was && r.visible {
                                    newly_selected.push(i);
                                } else if was && !r.visible {
                                    r.selection_order = None;
                                }
                            }
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

        for i in newly_selected {
            if let Some(r) = self.results_list.get_mut(i) {
                r.selection_order = Some(self.next_selection_rank);
                self.next_selection_rank = self.next_selection_rank.wrapping_add(1);
            }
        }

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

        ui.horizontal(|ui| {
            let cog_w = 32.0_f32;
            let gap = 6.0_f32;
            let row_w = ui.available_width();
            let btn_w = (row_w * 0.7).clamp(140.0, 240.0);
            let pad = ((row_w - btn_w - cog_w - gap) / 2.0).max(0.0);
            ui.add_space(pad);
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
            ui.add_space(gap);
            ui.scope(|ui| {
                let visuals = &mut ui.style_mut().visuals;
                visuals.widgets.inactive.bg_stroke = Stroke::new(1.5, accent);
                visuals.widgets.hovered.bg_stroke = Stroke::new(1.5, accent_hover);
                visuals.widgets.active.bg_stroke = Stroke::new(1.5, accent_active);
                let cog = egui::Button::new(egui::RichText::new("⚙").size(16.0))
                    .min_size(Vec2::new(cog_w, 38.0));
                let (resp, _) = egui::containers::menu::MenuButton::from_button(cog)
                    .ui(ui, |ui| {
                        ui.set_min_width(140.0);
                        ui.label(egui::RichText::new("Export contents").small().weak());
                        ui.separator();
                        ui.checkbox(&mut self.export_csv, "csv");
                        ui.checkbox(&mut self.export_png, "png");
                        ui.checkbox(&mut self.export_hist, "hist");
                    });
                resp.on_hover_text("Export settings");
            });
        });
        ui.add_space(16.0);
    }
}
