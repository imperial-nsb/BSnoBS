use eframe::egui::{
    self, Color32, ColorImage, Layout, Pos2, Rect, Sense, Slider, Stroke, Vec2,
};
use image::ImageReader;

use crate::types::FrameResult;

use super::AppState;

pub(super) enum UndoAction {
    ToggleBubble {
        result_idx: usize,
        frame_idx: usize,
        bubble_idx: usize,
        was_valid: bool,
        was_static: bool,
    },
    DiscardFrame {
        result_idx: usize,
        frame_idx: usize,
        frame: FrameResult,
    },
}

impl AppState {
    pub(super) fn ensure_frame_texture(&mut self, ctx: &egui::Context) {
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

    pub(super) fn center_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
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
