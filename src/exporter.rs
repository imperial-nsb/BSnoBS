//! CSV + JSON exporter (v1 — no plots or xlsx).

use std::path::Path;

use anyhow::Result;
use serde::Serialize;

use crate::types::AnalysisResults;

#[derive(Serialize)]
struct BubbleRow<'a> {
    frame: &'a str,
    bubble_id: usize,
    diameter_um: f32,
    volume_um3: f32,
    centroid_x_px: f32,
    centroid_y_px: f32,
}

#[derive(Serialize)]
struct StaticRow<'a> {
    frame: &'a str,
    bubble_id: usize,
    diameter_um: f32,
    centroid_x_px: f32,
    centroid_y_px: f32,
}

#[derive(Serialize)]
struct Summary {
    sample: String,
    n_frames: usize,
    n_bubbles_accepted: usize,
    n_bubbles_rejected: usize,
    n_static_rejected: usize,
    mean_diameter_um: f32,
    median_diameter_um: f32,
    std_diameter_um: f32,
    min_diameter_um: f32,
    max_diameter_um: f32,
    #[serde(rename = "concentration_per_uL")]
    concentration_per_ul: f32,
    #[serde(rename = "total_sample_volume_uL")]
    total_sample_volume_ul: f32,
}

pub fn export(results: &AnalysisResults, out_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let sample = &results.sample_name;

    // bubble_data_full.csv (valid only — matches Python exporter)
    let mut wtr = csv::Writer::from_path(out_dir.join(format!("{sample}_bubble_data_full.csv")))?;
    for f in &results.frames {
        for b in &f.bubbles {
            if !b.is_valid {
                continue;
            }
            wtr.serialize(BubbleRow {
                frame: &b.frame_name,
                bubble_id: b.bubble_id,
                diameter_um: b.diameter_um,
                volume_um3: b.volume_um3,
                centroid_x_px: b.centroid_x,
                centroid_y_px: b.centroid_y,
            })?;
        }
    }
    wtr.flush()?;

    // static_rejects.csv (if any)
    let static_rows: Vec<_> = results
        .frames
        .iter()
        .flat_map(|f| f.bubbles.iter())
        .filter(|b| b.is_static)
        .collect();
    if !static_rows.is_empty() {
        let mut wtr =
            csv::Writer::from_path(out_dir.join(format!("{sample}_static_rejects.csv")))?;
        for b in static_rows {
            wtr.serialize(StaticRow {
                frame: &b.frame_name,
                bubble_id: b.bubble_id,
                diameter_um: b.diameter_um,
                centroid_x_px: b.centroid_x,
                centroid_y_px: b.centroid_y,
            })?;
        }
        wtr.flush()?;
    }

    // summary.json
    let diams = results.diameters_valid();
    let n_valid = diams.len();
    let n_static = results
        .frames
        .iter()
        .flat_map(|f| f.bubbles.iter())
        .filter(|b| b.is_static)
        .count();

    let total_vol_ul = results.parameters.sample_volume_per_frame_ul
        * results.frames.len() as f32;

    let (mean, std, median, min, max) = if n_valid == 0 {
        (0.0, 0.0, 0.0, 0.0, 0.0)
    } else {
        let mean: f32 = diams.iter().sum::<f32>() / n_valid as f32;
        let var: f32 =
            diams.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / n_valid as f32;
        let std = var.sqrt();
        let mut sorted = diams.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[n_valid / 2];
        let min = sorted[0];
        let max = sorted[n_valid - 1];
        (mean, std, median, min, max)
    };

    let summary = Summary {
        sample: sample.clone(),
        n_frames: results.frames.len(),
        n_bubbles_accepted: results.total_bubbles(),
        n_bubbles_rejected: results.total_rejected(),
        n_static_rejected: n_static,
        mean_diameter_um: mean,
        median_diameter_um: median,
        std_diameter_um: std,
        min_diameter_um: min,
        max_diameter_um: max,
        concentration_per_ul: if total_vol_ul > 0.0 {
            results.total_bubbles() as f32 / total_vol_ul
        } else {
            0.0
        },
        total_sample_volume_ul: total_vol_ul,
    };
    std::fs::write(
        out_dir.join(format!("{sample}_summary.json")),
        serde_json::to_string_pretty(&summary)?,
    )?;

    Ok(())
}
