# BSNOBS — BubSize NO BS

Microbubble sizing from optical microscopy using a tiny distilled YOLO student. No Cellpose dependency — just the student model.

Ships a Typer CLI and a PySide6 GUI.

## Install

```bash
uv tool install /path/to/BSNOBS
```

The student weights (`bsnobs/models/student.pt`) are bundled in the package.

## CLI

Default command runs inference on a folder of microscopy images:

```bash
bsnobs <image_dir> --conf 0.4 --reject-static
```

Compare multiple samples on shared axes:

```bash
bsnobs compare dir1 dir2 dir3 --conf 0.4 --reject-static
bsnobs compare --parent /path/to/samples --conf 0.4
```

Key flags:
- `--conf`           YOLO confidence threshold (default 0.55)
- `--max-diam`       Max bubble diameter, μm (default 20)
- `--reject-static`  Filter out persistent dirt detections across the batch
- `--static-frac`    Min frame coverage for a detection to count as static (default 0.4)
- `--static-tol`     Centroid match tolerance in pixels (default 6)
- `--scale`          μm per pixel (default 0.0825)
- `--volume`         μL imaged per frame (default 0.00089)

## GUI

```bash
bsnobs-gui
```

Folder selector → settings panel → scrollable image viewer with overlay → **Export Results**.
