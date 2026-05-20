use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::exporter::{self, ExportOpts, HistAxisMode};
use crate::types::AnalysisResults;

use super::results::HistMode;
use super::AppState;

fn open_in_file_manager(path: &Path) {
    #[cfg(target_os = "macos")]
    let cmd = std::process::Command::new("open").arg(path).spawn();
    #[cfg(target_os = "windows")]
    let cmd = std::process::Command::new("explorer").arg(path).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let cmd = std::process::Command::new("xdg-open").arg(path).spawn();
    let _ = cmd;
}

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

impl AppState {
    pub(super) fn export_visible_results(&mut self) {
        let mut visible_idx: Vec<usize> = self
            .results_list
            .iter()
            .enumerate()
            .filter(|(_, r)| r.visible)
            .map(|(i, _)| i)
            .collect();
        visible_idx.sort_by_key(|&i| {
            (
                self.results_list[i].selection_order.unwrap_or(u32::MAX),
                i,
            )
        });
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
            if let Err(e) = exporter::export_summary_json(&r.results, &sub) {
                self.status = format!("Summary write failed for {}: {}", r.name, e);
                return;
            }
            if opts.csv {
                if let Err(e) = exporter::export_csv_accepted(&r.results, &sub) {
                    self.status = format!("CSV export failed for {}: {}", r.name, e);
                    return;
                }
                if let Err(e) = exporter::export_histogram_csv(&r.results, &sub) {
                    self.status = format!("Histogram CSV failed for {}: {}", r.name, e);
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

        let summaries: Vec<exporter::SampleSummary> = visible_idx
            .iter()
            .map(|&i| exporter::compute_sample_summary(&self.results_list[i].results))
            .collect();
        let combined_summary_path = target.join("combined_summary.csv");
        if let Err(e) = exporter::export_combined_summary_csv(&summaries, &combined_summary_path) {
            self.status = format!("Combined summary CSV failed: {}", e);
            return;
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
