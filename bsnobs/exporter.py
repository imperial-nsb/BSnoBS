import json
from datetime import datetime
from pathlib import Path
from typing import Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import io

from .data import AnalysisResults, FrameResult
from .params import AnalysisParameters


class ResultsExporter:
    def __init__(self, results: AnalysisResults, output_dir: Union[str, Path]):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(self):
        self.export_bubble_data()
        self.export_static_rejects()
        self.export_summary_statistics()
        self.export_histogram_data()
        self.create_distribution_plots()
        self.create_overlays()
        self.export_log()

    def export_static_rejects(self):
        rows = [
            {
                "frame": b.frame_name,
                "bubble_id": b.bubble_id,
                "diameter_um": b.diameter_um,
                "centroid_x_px": b.centroid_x,
                "centroid_y_px": b.centroid_y,
            }
            for b in self.results.all_bubbles
            if getattr(b, "is_static", False)
        ]
        if not rows:
            return
        df = pd.DataFrame(rows)
        df.to_csv(
            self.output_dir / f"{self.results.sample_name}_static_rejects.csv",
            index=False,
        )

    # ------------------------------------------------------------------
    # Tabular outputs
    # ------------------------------------------------------------------

    def export_bubble_data(self):
        rows = [
            {
                "frame": b.frame_name,
                "bubble_id": b.bubble_id,
                "diameter_um": b.diameter_um,
                "volume_um3": b.volume_um3,
                "centroid_x_px": b.centroid_x,
                "centroid_y_px": b.centroid_y,
                "circularity": b.circularity,
                "aspect_ratio": b.aspect_ratio,
            }
            for b in self.results.all_bubbles
            if b.is_valid
        ]
        df = pd.DataFrame(rows)
        base = self.output_dir / f"{self.results.sample_name}_bubble_data_full"
        df.to_excel(f"{base}.xlsx", index=False)
        df.to_csv(f"{base}.csv", index=False)

    def export_summary_statistics(self):
        params = self.results.parameters
        diameters = self.results.diameters
        volumes = self.results.volumes

        if len(diameters) == 0:
            return

        total_vol_uL = params.sample_volume_per_frame_uL * self.results.num_frames
        total_vol_um3 = float(np.sum(volumes))

        n_mean = float(np.mean(diameters))
        n_std = float(np.std(diameters))
        v_mean = float(np.sum(diameters * volumes) / total_vol_um3)
        v_std = float(
            np.sqrt(np.sum(volumes * (diameters - v_mean) ** 2) / total_vol_um3)
        )
        pdi = (n_std / n_mean) ** 2
        concentration = self.results.total_bubbles / total_vol_uL
        mean_vol = (4 / 3) * np.pi * (n_mean / 2) ** 3
        gas_dose_uL = mean_vol * concentration * params.injection_volume_uL * 1e-9

        stats = {
            "number_weighted_mean_um": n_mean,
            "number_weighted_std_um": n_std,
            "number_weighted_median_um": float(np.median(diameters)),
            "volume_weighted_mean_um": v_mean,
            "volume_weighted_std_um": v_std,
            "polydispersity_index": pdi,
            "number_concentration_per_uL": concentration,
            "gas_volume_dose_uL": gas_dose_uL,
            "total_sample_volume_uL": total_vol_uL,
            "number_of_frames": self.results.num_frames,
            "total_bubbles": self.results.total_bubbles,
            "total_rejected": self.results.total_rejected,
        }

        base = self.output_dir / f"{self.results.sample_name}_summary"
        pd.DataFrame([stats]).to_excel(f"{base}.xlsx", index=False)
        with open(f"{base}.json", "w") as f:
            json.dump(stats, f, indent=2)

    def export_histogram_data(self):
        params = self.results.parameters
        diameters = self.results.diameters
        volumes = self.results.volumes

        if len(diameters) == 0:
            return

        edges = np.arange(
            params.min_diameter_um,
            params.max_diameter_um + params.bin_size_um,
            params.bin_size_um,
        )
        counts, _ = np.histogram(diameters, bins=edges)
        vol_per_bin = np.array(
            [np.sum(volumes[(diameters >= edges[i]) & (diameters < edges[i + 1])])
             for i in range(len(counts))]
        )
        total_vol_uL = params.sample_volume_per_frame_uL * self.results.num_frames

        df = pd.DataFrame(
            {
                "bin_center_um": (edges[:-1] + edges[1:]) / 2,
                "bin_min_um": edges[:-1],
                "bin_max_um": edges[1:],
                "count": counts,
                "concentration_per_uL": counts / total_vol_uL,
                "number_pct": counts / len(diameters) * 100,
                "volume_pct": vol_per_bin / np.sum(volumes) * 100,
            }
        )
        df.to_excel(
            self.output_dir / f"{self.results.sample_name}_histogram.xlsx", index=False
        )

    # ------------------------------------------------------------------
    # Distribution plots
    # ------------------------------------------------------------------

    def create_distribution_plots(self):
        params = self.results.parameters
        diameters = self.results.diameters
        volumes = self.results.volumes

        if len(diameters) == 0:
            return

        edges = np.arange(
            params.min_diameter_um,
            params.max_diameter_um + params.bin_size_um,
            params.bin_size_um,
        )
        centers = (edges[:-1] + edges[1:]) / 2
        title_base = self.results.sample_name.replace("_", " ")

        BLUE = "#5778A4"
        ORANGE = "#E49444"

        # Number distribution
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(
            diameters, bins=edges, color=BLUE, edgecolor="white", alpha=0.85,
            weights=np.ones_like(diameters) / len(diameters) * 100,
        )
        ax.set_xlabel("Bubble Diameter (μm)", fontsize=12)
        ax.set_ylabel("Number (%)", fontsize=12)
        ax.set_title(f"{title_base}\nNumber Size Distribution", fontsize=14)
        ax.set_xlim(params.min_diameter_um, params.max_diameter_um)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"{self.results.sample_name}_histogram_number.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()

        # Volume distribution
        vol_per_bin = np.array(
            [np.sum(volumes[(diameters >= edges[i]) & (diameters < edges[i + 1])])
             for i in range(len(centers))]
        )
        vol_pct = vol_per_bin / np.sum(volumes) * 100

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(centers, vol_pct, width=params.bin_size_um * 0.9,
               color=ORANGE, edgecolor="white", alpha=0.85)
        ax.set_xlabel("Bubble Diameter (μm)", fontsize=12)
        ax.set_ylabel("Volume (%)", fontsize=12)
        ax.set_title(f"{title_base}\nVolume Size Distribution", fontsize=14)
        ax.set_xlim(params.min_diameter_um, params.max_diameter_um)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"{self.results.sample_name}_histogram_volume.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()

        # Cumulative
        sorted_d = np.sort(diameters)
        cum_n = np.arange(1, len(sorted_d) + 1) / len(sorted_d) * 100
        sort_idx = np.argsort(diameters)
        cum_v = np.cumsum(volumes[sort_idx]) / np.sum(volumes) * 100

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(sorted_d, cum_n, color=BLUE, linewidth=2, label="Number")
        ax.plot(sorted_d, cum_v, color=ORANGE, linewidth=2, label="Volume")
        ax.set_xlabel("Bubble Diameter (μm)", fontsize=12)
        ax.set_ylabel("Cumulative (%)", fontsize=12)
        ax.set_title(f"{title_base}\nCumulative Size Distribution", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(params.min_diameter_um, params.max_diameter_um)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"{self.results.sample_name}_cumulative.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close()

    # ------------------------------------------------------------------
    # Per-frame overlay images
    # ------------------------------------------------------------------

    def create_overlays(self):
        """Save a PNG for each frame with detected bubbles drawn as circles."""
        scale = self.results.parameters.scale_um_per_pixel
        for frame in self.results.frames:
            self._create_frame_overlay(frame, scale)

    def _create_frame_overlay(self, frame: FrameResult, scale_um_per_pixel: float):
        # Load original image
        if frame.image_path is None or not frame.image_path.exists():
            return

        image = io.imread(str(frame.image_path))
        if image.ndim == 3:
            # Convert to grayscale for display
            display = np.mean(image[:, :, :3], axis=-1)
        else:
            display = image

        valid = [b for b in frame.bubbles if b.is_valid]
        static = [b for b in frame.bubbles if getattr(b, "is_static", False)]
        rejected = [b for b in frame.bubbles if not b.is_valid and not getattr(b, "is_static", False)]

        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        ax.imshow(display, cmap="gray", interpolation="nearest")
        # Lock the axes to image bounds so output dims don't vary with content.
        ax.set_xlim(-0.5, display.shape[1] - 0.5)
        ax.set_ylim(display.shape[0] - 0.5, -0.5)

        for b in rejected:
            r_px = (b.diameter_um / 2) / scale_um_per_pixel
            ax.add_patch(
                mpatches.Circle(
                    (b.centroid_x, b.centroid_y), r_px,
                    fill=False, edgecolor="#FF1744", linewidth=1.6, alpha=0.95,
                )
            )

        for b in static:
            r_px = (b.diameter_um / 2) / scale_um_per_pixel
            ax.add_patch(
                mpatches.Circle(
                    (b.centroid_x, b.centroid_y), r_px,
                    fill=False, edgecolor="#FFEA00", linewidth=1.6, alpha=1.0,
                    linestyle="--",
                )
            )

        for b in valid:
            r_px = (b.diameter_um / 2) / scale_um_per_pixel
            ax.add_patch(
                mpatches.Circle(
                    (b.centroid_x, b.centroid_y), r_px,
                    fill=False, edgecolor="#00E676", linewidth=1.8, alpha=1.0,
                )
            )
            ax.text(
                b.centroid_x + r_px, b.centroid_y - r_px,
                f"{b.diameter_um:.1f}",
                color="#00E676", fontsize=5, alpha=1.0,
                ha="left", va="bottom",
            )

        # Scale bar: pick a round-number length that's ~10% of image width
        img_width_um = display.shape[1] * scale_um_per_pixel
        bar_um = _nice_scale_bar_length(img_width_um * 0.10)
        bar_px = bar_um / scale_um_per_pixel

        margin_x = display.shape[1] * 0.03
        margin_y = display.shape[0] * 0.05
        bar_y = display.shape[0] - margin_y
        bar_x0 = display.shape[1] - margin_x - bar_px
        bar_x1 = display.shape[1] - margin_x

        ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color="white", linewidth=2)
        ax.text(
            (bar_x0 + bar_x1) / 2, bar_y - display.shape[0] * 0.015,
            f"{bar_um:g} μm", color="white", fontsize=8,
            ha="center", va="bottom",
        )

        # Legend
        legend_handles = [
            mpatches.Patch(facecolor="none", edgecolor="#00FF7F", label=f"Valid ({len(valid)})"),
            mpatches.Patch(facecolor="none", edgecolor="#FF6B6B", label=f"Rejected ({len(rejected)})"),
        ]
        if static:
            legend_handles.append(
                mpatches.Patch(facecolor="none", edgecolor="#FFD166", label=f"Static dirt ({len(static)})")
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
                  framealpha=0.6, facecolor="#111111", labelcolor="white")

        ax.set_title(frame.frame_name, fontsize=10, color="white", pad=4)
        ax.axis("off")
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        out_path = self.output_dir / f"{frame.frame_name}_overlay.png"
        plt.savefig(out_path, dpi=150, facecolor="black")
        plt.close()

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def export_log(self):
        p = self.results.parameters
        d = self.results.diameters
        lines = [
            "=" * 60,
            "BUBBLE COUNTING ANALYSIS LOG",
            "=" * 60,
            f"Sample:    {self.results.sample_name}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "PARAMETERS",
            f"  Scale:              {p.scale_um_per_pixel} μm/pixel",
            f"  Volume/frame:       {p.sample_volume_per_frame_uL} μL",
            f"  Diameter range:     {p.min_diameter_um}–{p.max_diameter_um} μm",
            f"  Model:              {p.pretrained_model}",
            "",
            "RESULTS",
            f"  Frames:             {self.results.num_frames}",
            f"  Bubbles accepted:   {self.results.total_bubbles}",
            f"  Bubbles rejected:   {self.results.total_rejected}",
            f"  Static dirt rejected: {sum(1 for b in self.results.all_bubbles if getattr(b, 'is_static', False))}",
        ]
        if len(d) > 0:
            lines += [
                f"  Mean diameter:      {np.mean(d):.2f} ± {np.std(d):.2f} μm",
                f"  Median diameter:    {np.median(d):.2f} μm",
                f"  Range:              {np.min(d):.2f}–{np.max(d):.2f} μm",
            ]
        lines.append("=" * 60)

        log_path = self.output_dir / f"{self.results.sample_name}_log.txt"
        log_path.write_text("\n".join(lines), encoding="utf-8")


def _nice_scale_bar_length(target_um: float) -> float:
    """Return the nearest 'nice' number (1, 2, 5, 10, …) to target_um."""
    if target_um <= 0:
        return 1.0
    magnitude = 10 ** int(np.floor(np.log10(target_um)))
    for step in (1, 2, 5, 10):
        candidate = step * magnitude
        if candidate >= target_um:
            return candidate
    return 10 * magnitude
