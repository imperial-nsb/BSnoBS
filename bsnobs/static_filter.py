"""Reject persistent static detections (dirt, debris) across a batch of frames.

A dirt spot appears at (nearly) the same pixel coordinates in many frames,
while real bubbles drift. We cluster detections across frames by centroid
proximity and mark any cluster present in more than `min_frame_frac` of frames
as static; those detections get `is_valid=False, is_static=True`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .data import AnalysisResults, BubbleData


@dataclass
class StaticFilterConfig:
    min_frame_frac: float = 0.4   # cluster must appear in >= this fraction of frames
    tol_px: float = 6.0           # centroid match tolerance in pixels
    diameter_tol_frac: float = 0.5  # |Δd|/d must be <= this within a cluster (size consistency)


def apply_static_filter(
    results: AnalysisResults,
    cfg: StaticFilterConfig,
) -> dict:
    """Mark static detections in-place on `results`. Returns a summary dict.

    Algorithm: single-pass greedy clustering across all frames. For each
    detection we compute the set of frames in which a "nearby" detection
    (centroid within `tol_px` AND diameter within `diameter_tol_frac`)
    exists, then reject the detection if that frame-coverage exceeds the
    threshold. This avoids needing a canonical cluster centre and naturally
    handles small jitter.
    """
    frames = results.frames
    n_frames = len(frames)
    if n_frames == 0:
        return {"n_static_rejected": 0, "n_clusters": 0}

    # Gather all currently-valid detections (don't re-filter shape-rejects).
    detections: List[tuple[int, BubbleData]] = []
    for fi, frame in enumerate(frames):
        for b in frame.bubbles:
            if b.is_valid:
                detections.append((fi, b))

    if not detections:
        return {"n_static_rejected": 0, "n_clusters": 0}

    xs = np.array([b.centroid_x for _, b in detections])
    ys = np.array([b.centroid_y for _, b in detections])
    ds = np.array([b.diameter_um for _, b in detections])
    fis = np.array([fi for fi, _ in detections])

    threshold = cfg.min_frame_frac * n_frames
    tol_sq = cfg.tol_px ** 2

    rejected_mask = np.zeros(len(detections), dtype=bool)
    # For each detection, count distinct frames within tolerance.
    for i in range(len(detections)):
        dxs = xs - xs[i]
        dys = ys - ys[i]
        near = (dxs * dxs + dys * dys) <= tol_sq
        if cfg.diameter_tol_frac > 0 and ds[i] > 0:
            near &= np.abs(ds - ds[i]) / ds[i] <= cfg.diameter_tol_frac
        frames_hit = np.unique(fis[near])
        if len(frames_hit) >= threshold:
            rejected_mask[i] = True

    n_clusters_seen = 0
    seen = np.zeros(len(detections), dtype=bool)
    for i in range(len(detections)):
        if not rejected_mask[i] or seen[i]:
            continue
        dxs = xs - xs[i]
        dys = ys - ys[i]
        near = ((dxs * dxs + dys * dys) <= tol_sq) & rejected_mask
        seen |= near
        n_clusters_seen += 1

    n_rejected = 0
    for is_static, (fi, b) in zip(rejected_mask, detections):
        if is_static:
            b.is_valid = False
            b.is_static = True
            frames[fi].num_rejected += 1
            frames[fi].num_bubbles = max(0, frames[fi].num_bubbles - 1)
            n_rejected += 1

    return {"n_static_rejected": int(n_rejected), "n_clusters": int(n_clusters_seen)}
