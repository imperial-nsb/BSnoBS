"""Comparative plots across multiple sample runs.

Given a list of `AnalysisResults`, render histograms / cumulative curves on
shared axes so distributions can be eyeballed against each other.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import AnalysisResults

# Categorical palette (Okabe–Ito-ish, colour-blind friendly).
_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#CC79A7",  # magenta
    "#56B4E9",  # sky
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
]


def _color(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def render_comparison(results_list: List[AnalysisResults], output_dir: Path) -> dict:
    """Write comparison plots + a combined CSV. Returns a summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not results_list:
        return {"n_samples": 0}

    # Shared bin edges — assume same physical params; fall back to first sample's.
    p = results_list[0].parameters
    edges = np.arange(
        p.min_diameter_um, p.max_diameter_um + p.bin_size_um, p.bin_size_um
    )
    centers = (edges[:-1] + edges[1:]) / 2

    # Pre-compute per-sample binned counts and volumes (reused below).
    binned = []
    for r in results_list:
        d = r.diameters
        v = r.volumes
        if len(d) == 0:
            binned.append((np.zeros_like(centers), np.zeros_like(centers), 0.0))
            continue
        counts, _ = np.histogram(d, bins=edges)
        vol_per_bin = np.array([
            v[(d >= edges[j]) & (d < edges[j + 1])].sum()
            for j in range(len(centers))
        ])
        total_vol_uL = r.parameters.sample_volume_per_frame_uL * r.num_frames
        binned.append((counts, vol_per_bin, total_vol_uL))

    def _plot_series(series_fn, ylabel, title, fname, label_fn=None):
        fig, ax = plt.subplots(figsize=(9, 6))
        for i, r in enumerate(results_list):
            y = series_fn(i, r)
            if y is None:
                continue
            label = (label_fn(r) if label_fn else r.sample_name)
            ax.step(centers, y, where="mid", color=_color(i), linewidth=2, label=label)
        ax.set_xlabel("Bubble Diameter (μm)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(p.min_diameter_um, p.max_diameter_um)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)

    # ---- 1a. Number — percent ---------------------------------------------
    def _num_pct(i, r):
        counts = binned[i][0]
        s = counts.sum()
        return counts / s * 100 if s else None
    _plot_series(
        _num_pct, "Number (%)", "Number Size Distribution (%) — comparison",
        "comparison_number_pct.png",
        label_fn=lambda r: f"{r.sample_name} (n={r.total_bubbles})",
    )

    # ---- 1b. Number — raw counts ------------------------------------------
    def _num_raw(i, r):
        counts = binned[i][0]
        return counts if counts.sum() else None
    _plot_series(
        _num_raw, "Count", "Number Size Distribution (raw counts) — comparison",
        "comparison_number_counts.png",
        label_fn=lambda r: f"{r.sample_name} (n={r.total_bubbles}, frames={r.num_frames})",
    )

    # ---- 1b'. Number — per image (counts / num_frames) --------------------
    def _num_per_image(i, r):
        counts = binned[i][0]
        if not r.num_frames or not counts.sum():
            return None
        return counts / r.num_frames
    _plot_series(
        _num_per_image, "Bubbles per image",
        "Number Size Distribution (per image) — comparison",
        "comparison_number_per_image.png",
        label_fn=lambda r: f"{r.sample_name} (n={r.total_bubbles}, frames={r.num_frames})",
    )

    # ---- 1c. Number — concentration (bubbles per μL) ----------------------
    def _num_conc(i, r):
        counts, _vol, total_vol_uL = binned[i]
        if not total_vol_uL or not counts.sum():
            return None
        return counts / total_vol_uL
    _plot_series(
        _num_conc, "Concentration (bubbles/μL)",
        "Number Concentration — comparison",
        "comparison_concentration.png",
    )

    # ---- 2a. Volume — percent ---------------------------------------------
    def _vol_pct(i, r):
        vol = binned[i][1]
        s = vol.sum()
        return vol / s * 100 if s else None
    _plot_series(
        _vol_pct, "Volume (%)", "Volume Size Distribution (%) — comparison",
        "comparison_volume_pct.png",
    )

    # ---- 2b. Volume — raw (μm³) -------------------------------------------
    def _vol_raw(i, r):
        vol = binned[i][1]
        return vol if vol.sum() else None
    _plot_series(
        _vol_raw, "Gas volume in bin (μm³)",
        "Volume Size Distribution (raw) — comparison",
        "comparison_volume_raw.png",
    )

    # ---- 3. Cumulative (number + volume) ----------------------------------
    fig, (ax_n, ax_v) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for i, r in enumerate(results_list):
        d = r.diameters
        v = r.volumes
        if len(d) == 0:
            continue
        order = np.argsort(d)
        d_sorted = d[order]
        cum_n = np.arange(1, len(d_sorted) + 1) / len(d_sorted) * 100
        ax_n.plot(d_sorted, cum_n, color=_color(i), linewidth=2, label=r.sample_name)
        if v.sum() > 0:
            cum_v = np.cumsum(v[order]) / v.sum() * 100
            ax_v.plot(d_sorted, cum_v, color=_color(i), linewidth=2, label=r.sample_name)
    for a, ttl in [(ax_n, "Number"), (ax_v, "Volume")]:
        a.set_xlabel("Bubble Diameter (μm)")
        a.set_title(f"Cumulative {ttl} (%)")
        a.grid(True, alpha=0.3)
        a.set_xlim(p.min_diameter_um, p.max_diameter_um)
        a.set_ylim(0, 100)
    ax_n.set_ylabel("Cumulative (%)")
    ax_n.legend(fontsize=9)
    fig.suptitle("Cumulative distributions — comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_cumulative.png", dpi=150)
    plt.close(fig)

    # ---- 4. Combined per-bin CSV ------------------------------------------
    rows = []
    for r in results_list:
        d = r.diameters
        v = r.volumes
        if len(d) == 0:
            continue
        counts, _ = np.histogram(d, bins=edges)
        vol_per_bin = np.array([
            v[(d >= edges[j]) & (d < edges[j + 1])].sum()
            for j in range(len(centers))
        ])
        total_vol_uL = r.parameters.sample_volume_per_frame_uL * r.num_frames
        for j in range(len(centers)):
            rows.append({
                "sample": r.sample_name,
                "bin_center_um": centers[j],
                "count": int(counts[j]),
                "concentration_per_uL": counts[j] / total_vol_uL if total_vol_uL else 0,
                "number_pct": counts[j] / counts.sum() * 100 if counts.sum() else 0,
                "volume_pct": vol_per_bin[j] / v.sum() * 100 if v.sum() else 0,
            })
    pd.DataFrame(rows).to_csv(output_dir / "comparison_histogram.csv", index=False)

    # ---- 5. One-row-per-sample summary CSV --------------------------------
    summary_rows = []
    for r in results_list:
        d = r.diameters
        total_vol_uL = r.parameters.sample_volume_per_frame_uL * r.num_frames
        summary_rows.append({
            "sample": r.sample_name,
            "n_frames": r.num_frames,
            "n_bubbles": r.total_bubbles,
            "n_rejected": r.total_rejected,
            "mean_diameter_um": float(np.mean(d)) if len(d) else 0.0,
            "median_diameter_um": float(np.median(d)) if len(d) else 0.0,
            "std_diameter_um": float(np.std(d)) if len(d) else 0.0,
            "concentration_per_uL": r.total_bubbles / total_vol_uL if total_vol_uL else 0.0,
        })
    pd.DataFrame(summary_rows).to_csv(output_dir / "comparison_summary.csv", index=False)

    return {"n_samples": len(results_list)}
