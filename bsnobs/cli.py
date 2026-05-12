"""bsnobs — microbubble sizing CLI (YOLO student only)."""
from __future__ import annotations

from datetime import datetime
from importlib.resources import files as _pkg_files
from pathlib import Path
from typing import List, Optional

import json
import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from skimage import io

from .compare import render_comparison
from .data import AnalysisResults
from .exporter import ResultsExporter
from .infer import StudentAnalyzer, list_images
from .params import AnalysisParameters
from .static_filter import StaticFilterConfig, apply_static_filter

console = Console()
app = typer.Typer(
    name="bsnobs",
    help="BubSize NO BS — microbubble sizing from optical microscopy (YOLO student).",
    add_completion=False,
)

_SUBCOMMANDS = {"run", "compare", "gui"}


def default_student_weights() -> Optional[Path]:
    try:
        p = Path(str(_pkg_files("bsnobs").joinpath("models/student.pt")))
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    return p if p.exists() else None


def main():
    """Entry point: default to `run` when no subcommand is given."""
    import sys
    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-") and argv[0] not in _SUBCOMMANDS and argv[0] not in {"--help", "-h"}:
        sys.argv = [sys.argv[0], "run", *argv]
    app()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command("run")
def run_cmd(
    image_dir: Path = typer.Argument(..., help="Directory containing microscopy images"),
    weights: Optional[Path] = typer.Option(None, "--weights", "-w", help="YOLO student weights. Defaults to bundled bsnobs/models/student.pt."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory (default: <image_dir>_count_conf<NN>[_static]_<timestamp>)"),
    scale: float = typer.Option(0.0825, "--scale", help="Micrometres per pixel"),
    volume: float = typer.Option(0.00089, "--volume", help="Sample volume per frame (μL)"),
    min_diam: float = typer.Option(0.5, "--min-diam", help="Minimum bubble diameter (μm)"),
    max_diam: float = typer.Option(20.0, "--max-diam", help="Maximum bubble diameter (μm)"),
    conf: float = typer.Option(0.55, "--conf", help="YOLO confidence threshold"),
    imgsz: int = typer.Option(640, "--imgsz", help="YOLO inference image size"),
    max_det: int = typer.Option(1000, "--max-det", help="Max detections per image"),
    device: Optional[str] = typer.Option(None, "--device", help="mps | cpu | 0 (cuda). Autodetect if omitted."),
    reject_static: bool = typer.Option(False, "--reject-static", help="Reject detections that repeat at the same pixel location across many frames (dirt)."),
    static_frac: float = typer.Option(0.4, "--static-frac"),
    static_tol: float = typer.Option(6.0, "--static-tol"),
    static_diam_tol: float = typer.Option(0.5, "--static-diam-tol"),
):
    _print_header()
    image_dir = image_dir.expanduser().resolve()
    if not image_dir.exists():
        console.print(f"[red]Error:[/red] directory not found: {image_dir}")
        raise typer.Exit(1)

    weights = _resolve_weights(weights)

    if output is None:
        conf_tag = f"conf{int(round(conf * 100)):02d}"
        suffix = "_static" if reject_static else ""
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = (image_dir.parent / f"{image_dir.name}_count_{conf_tag}{suffix}_{ts}").resolve()
    else:
        output_dir = output.expanduser().resolve()

    run_pipeline(
        image_dir=image_dir, output_dir=output_dir, weights=weights,
        scale=scale, volume=volume, min_diam=min_diam, max_diam=max_diam,
        conf=conf, imgsz=imgsz, max_det=max_det, device=device,
        reject_static=reject_static, static_frac=static_frac,
        static_tol=static_tol, static_diam_tol=static_diam_tol,
        analyzer=None, print_params=True, print_summary=True,
    )


def _resolve_weights(weights: Optional[Path]) -> Path:
    if weights is None:
        bundled = default_student_weights()
        if bundled is None:
            console.print("[red]Error:[/red] no --weights given and bundled student.pt not found.")
            raise typer.Exit(1)
        weights = bundled
        console.print(f"[dim]Using bundled student weights:[/dim] {weights}")
    weights = weights.expanduser().resolve()
    if not weights.exists():
        console.print(f"[red]Error:[/red] weights not found: {weights}")
        raise typer.Exit(1)
    return weights


def run_pipeline(
    *, image_dir: Path, output_dir: Path, weights: Path,
    scale: float, volume: float, min_diam: float, max_diam: float,
    conf: float, imgsz: int, max_det: int, device: Optional[str],
    reject_static: bool, static_frac: float, static_tol: float, static_diam_tol: float,
    analyzer: Optional[StudentAnalyzer], print_params: bool, print_summary: bool,
) -> AnalysisResults:
    params = AnalysisParameters(
        scale_um_per_pixel=scale,
        sample_volume_per_frame_uL=volume,
        min_diameter_um=min_diam,
        max_diameter_um=max_diam,
        pretrained_model=str(weights),
    )

    if print_params:
        _print_params(image_dir, output_dir, params)

    if analyzer is None:
        with console.status("[bold cyan]Loading student (YOLO) model…[/bold cyan]"):
            try:
                analyzer = StudentAnalyzer(
                    weights=weights, params=params,
                    conf=conf, imgsz=imgsz, max_det=max_det, device=device,
                )
            except Exception as exc:
                console.print(f"[red]Failed to load student:[/red] {exc}")
                raise typer.Exit(1)
    else:
        analyzer.params = params

    image_files = list_images(image_dir)
    if not image_files:
        console.print(f"[red]No images found in {image_dir}[/red]")
        raise typer.Exit(1)

    results = AnalysisResults(sample_name=image_dir.name, parameters=params)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"[cyan]Analysing {image_dir.name}[/cyan]", total=len(image_files))
        for img_path in image_files:
            progress.update(task, description=f"[cyan]{img_path.name}[/cyan]")
            image = io.imread(str(img_path))
            frame = analyzer.analyze_image(image, img_path.stem, img_path)
            results.frames.append(frame)
            results.all_bubbles.extend(frame.bubbles)
            progress.advance(task)

    if reject_static:
        with console.status("[bold cyan]Filtering static (dirt) detections…[/bold cyan]"):
            summary = apply_static_filter(
                results,
                StaticFilterConfig(
                    min_frame_frac=static_frac,
                    tol_px=static_tol,
                    diameter_tol_frac=static_diam_tol,
                ),
            )
        console.print(
            f"[yellow]Static filter:[/yellow] rejected "
            f"[bold]{summary['n_static_rejected']}[/bold] detections "
            f"across [bold]{summary['n_clusters']}[/bold] persistent locations."
        )

    with console.status("[bold cyan]Exporting results…[/bold cyan]"):
        ResultsExporter(results, output_dir).export_all()

    metadata = {
        "command": "run",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "weights": str(weights),
        "weights_bundled": str(weights) == str(default_student_weights()),
        "yolo": {"conf": conf, "imgsz": imgsz, "max_det": max_det, "device": device},
        "physics": {
            "scale_um_per_pixel": scale,
            "sample_volume_per_frame_uL": volume,
            "min_diameter_um": min_diam,
            "max_diameter_um": max_diam,
        },
        "static_filter": {
            "enabled": reject_static,
            "min_frame_frac": static_frac,
            "tol_px": static_tol,
            "diameter_tol_frac": static_diam_tol,
        },
        "n_frames": results.num_frames,
        "n_bubbles_accepted": results.total_bubbles,
        "n_bubbles_rejected": results.total_rejected,
        "n_static_rejected": sum(1 for b in results.all_bubbles if getattr(b, "is_static", False)),
    }
    (output_dir / f"{results.sample_name}_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    if print_summary:
        _print_summary(results, output_dir)
    return results


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


@app.command("compare")
def compare_cmd(
    image_dirs: List[Path] = typer.Argument(None, help="One or more sample directories."),
    parent: Optional[Path] = typer.Option(None, "--parent", "-p", help="Parent directory whose subfolders are samples."),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
    weights: Optional[Path] = typer.Option(None, "--weights", "-w"),
    scale: float = typer.Option(0.0825, "--scale"),
    volume: float = typer.Option(0.00089, "--volume"),
    min_diam: float = typer.Option(0.5, "--min-diam"),
    max_diam: float = typer.Option(20.0, "--max-diam"),
    conf: float = typer.Option(0.55, "--conf"),
    imgsz: int = typer.Option(640, "--imgsz"),
    max_det: int = typer.Option(1000, "--max-det"),
    device: Optional[str] = typer.Option(None, "--device"),
    reject_static: bool = typer.Option(False, "--reject-static"),
    static_frac: float = typer.Option(0.4, "--static-frac"),
    static_tol: float = typer.Option(6.0, "--static-tol"),
    static_diam_tol: float = typer.Option(0.5, "--static-diam-tol"),
):
    """Run on multiple sample folders and emit comparative plots."""
    _print_header()

    if parent is not None:
        parent = parent.expanduser().resolve()
        if not parent.exists():
            console.print(f"[red]Parent not found:[/red] {parent}")
            raise typer.Exit(1)
        image_dirs = sorted(d for d in parent.iterdir() if d.is_dir() and list_images(d))
        if not image_dirs:
            console.print(f"[red]No sample subdirectories with images found in {parent}[/red]")
            raise typer.Exit(1)
        anchor = parent
    else:
        if not image_dirs:
            console.print("[red]Pass at least one sample dir, or use --parent.[/red]")
            raise typer.Exit(1)
        image_dirs = [d.expanduser().resolve() for d in image_dirs]
        for d in image_dirs:
            if not d.exists():
                console.print(f"[red]Directory not found:[/red] {d}")
                raise typer.Exit(1)
        anchor = image_dirs[0].parent

    weights = _resolve_weights(weights)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    conf_tag = f"conf{int(round(conf * 100)):02d}"
    suffix = "_static" if reject_static else ""
    if output is None:
        output_dir = (anchor / f"comparison_{conf_tag}{suffix}_{ts}").resolve()
    else:
        output_dir = output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[cyan]Samples ({len(image_dirs)}):[/cyan] " + ", ".join(d.name for d in image_dirs))
    console.print(f"[cyan]Output:[/cyan] {output_dir}\n")

    with console.status("[bold cyan]Loading student (YOLO) model…[/bold cyan]"):
        analyzer = StudentAnalyzer(
            weights=weights, params=AnalysisParameters(pretrained_model=str(weights)),
            conf=conf, imgsz=imgsz, max_det=max_det, device=device,
        )

    all_results: list[AnalysisResults] = []
    for d in image_dirs:
        per = output_dir / d.name
        per.mkdir(parents=True, exist_ok=True)
        res = run_pipeline(
            image_dir=d, output_dir=per, weights=weights,
            scale=scale, volume=volume, min_diam=min_diam, max_diam=max_diam,
            conf=conf, imgsz=imgsz, max_det=max_det, device=device,
            reject_static=reject_static, static_frac=static_frac,
            static_tol=static_tol, static_diam_tol=static_diam_tol,
            analyzer=analyzer, print_params=False, print_summary=False,
        )
        all_results.append(res)

    with console.status("[bold cyan]Rendering comparison plots…[/bold cyan]"):
        render_comparison(all_results, output_dir)

    console.print(Panel(
        f"Wrote comparison across [bold]{len(all_results)}[/bold] samples to\n[cyan]{output_dir}[/cyan]",
        title="[bold green]Comparison ready[/bold green]", border_style="green", expand=False,
    ))


# ---------------------------------------------------------------------------
# gui
# ---------------------------------------------------------------------------


@app.command("gui")
def gui_cmd():
    """Launch the PySide6 GUI."""
    from .gui.app import main as gui_main
    gui_main()


# ---------------------------------------------------------------------------
# Rich helpers
# ---------------------------------------------------------------------------


def _print_header():
    console.print()
    console.print(Panel(
        "[bold white]bsnobs[/bold white]  [dim]microbubble sizing · YOLO student[/dim]",
        expand=False, border_style="bright_cyan", padding=(0, 2),
    ))
    console.print()


def _print_params(image_dir: Path, output_dir: Path, p: AnalysisParameters):
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim"); t.add_column()
    t.add_row("Input",   str(image_dir))
    t.add_row("Output",  str(output_dir))
    t.add_row("Model",   str(p.pretrained_model))
    t.add_row("Scale",   f"{p.scale_um_per_pixel} μm/pixel")
    t.add_row("Volume",  f"{p.sample_volume_per_frame_uL} μL/frame")
    t.add_row("Range",   f"{p.min_diameter_um}–{p.max_diameter_um} μm")
    console.print(Panel(t, title="[bold]Parameters[/bold]", border_style="cyan", expand=False))
    console.print()


def _print_summary(results: AnalysisResults, output_dir: Path):
    d = results.diameters
    p = results.parameters
    total_vol = p.sample_volume_per_frame_uL * results.num_frames
    conc = results.total_bubbles / total_vol if total_vol > 0 else 0

    stats = Table(show_header=False, box=None, padding=(0, 2))
    stats.add_column(style="dim", no_wrap=True); stats.add_column(justify="right")
    stats.add_row("Frames analysed", str(results.num_frames))
    stats.add_row("Bubbles accepted", f"[bold green]{results.total_bubbles}[/bold green]")
    stats.add_row("Bubbles rejected", f"[yellow]{results.total_rejected}[/yellow]")
    if len(d) > 0:
        stats.add_row("", "")
        stats.add_row("Mean diameter",  f"{np.mean(d):.2f} ± {np.std(d):.2f} μm")
        stats.add_row("Median diameter", f"{np.median(d):.2f} μm")
        stats.add_row("Range",          f"{np.min(d):.2f}–{np.max(d):.2f} μm")
        stats.add_row("", "")
        stats.add_row("Concentration",  f"{conc:.3e} bubbles/μL")
    console.print()
    console.print(Panel(stats, title="[bold green]Results[/bold green]", border_style="green", expand=False))
    console.print()
