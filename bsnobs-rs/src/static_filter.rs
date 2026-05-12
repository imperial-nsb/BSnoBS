//! Port of `bsnobs.static_filter` — reject detections that persist across many frames.

use crate::types::AnalysisResults;

#[derive(Clone, Copy, Debug)]
pub struct StaticFilterConfig {
    pub min_frame_frac: f32,
    pub tol_px: f32,
    pub diameter_tol_frac: f32,
}

impl Default for StaticFilterConfig {
    fn default() -> Self {
        Self {
            min_frame_frac: 0.4,
            tol_px: 6.0,
            diameter_tol_frac: 0.5,
        }
    }
}

pub struct FilterSummary {
    pub n_rejected: usize,
    pub n_clusters: usize,
}

pub fn apply(results: &mut AnalysisResults, cfg: &StaticFilterConfig) -> FilterSummary {
    let n_frames = results.frames.len();
    if n_frames == 0 {
        return FilterSummary { n_rejected: 0, n_clusters: 0 };
    }

    // Collect (frame_idx, bubble_idx) for currently-valid detections.
    let mut items: Vec<(usize, usize, f32, f32, f32)> = Vec::new(); // (fi, bi, x, y, d)
    for (fi, frame) in results.frames.iter().enumerate() {
        for (bi, b) in frame.bubbles.iter().enumerate() {
            if b.is_valid {
                items.push((fi, bi, b.centroid_x, b.centroid_y, b.diameter_um));
            }
        }
    }
    if items.is_empty() {
        return FilterSummary { n_rejected: 0, n_clusters: 0 };
    }

    let threshold = (cfg.min_frame_frac * n_frames as f32).ceil() as usize;
    let tol_sq = cfg.tol_px * cfg.tol_px;

    let mut rejected = vec![false; items.len()];
    for i in 0..items.len() {
        let (_, _, xi, yi, di) = items[i];
        let mut frames_seen = std::collections::HashSet::<usize>::new();
        for j in 0..items.len() {
            let (fj, _, xj, yj, dj) = items[j];
            let dx = xi - xj;
            let dy = yi - yj;
            if dx * dx + dy * dy <= tol_sq {
                if cfg.diameter_tol_frac > 0.0 && di > 0.0 {
                    if (di - dj).abs() / di > cfg.diameter_tol_frac {
                        continue;
                    }
                }
                frames_seen.insert(fj);
            }
        }
        if frames_seen.len() >= threshold {
            rejected[i] = true;
        }
    }

    // Cluster count (distinct rejected locations) — greedy.
    let mut seen = vec![false; items.len()];
    let mut clusters = 0usize;
    for i in 0..items.len() {
        if !rejected[i] || seen[i] {
            continue;
        }
        let (_, _, xi, yi, _) = items[i];
        for j in 0..items.len() {
            if !rejected[j] {
                continue;
            }
            let (_, _, xj, yj, _) = items[j];
            let dx = xi - xj;
            let dy = yi - yj;
            if dx * dx + dy * dy <= tol_sq {
                seen[j] = true;
            }
        }
        clusters += 1;
    }

    let mut n_rejected = 0usize;
    for (idx, &is_static) in rejected.iter().enumerate() {
        if !is_static {
            continue;
        }
        let (fi, bi, ..) = items[idx];
        let b = &mut results.frames[fi].bubbles[bi];
        b.is_valid = false;
        b.is_static = true;
        results.frames[fi].num_valid = results.frames[fi].num_valid.saturating_sub(1);
        results.frames[fi].num_rejected += 1;
        n_rejected += 1;
    }

    FilterSummary { n_rejected, n_clusters: clusters }
}
