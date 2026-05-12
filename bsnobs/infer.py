"""YOLO student inference over a folder of images."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np
from skimage import io

from .data import AnalysisResults, BubbleData, FrameResult
from .params import AnalysisParameters

_IMAGE_PATTERNS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


def list_images(directory: Union[str, Path]) -> List[Path]:
    directory = Path(directory)
    for pattern in _IMAGE_PATTERNS:
        files = sorted(directory.glob(pattern))
        if files:
            return files
    return []


class StudentAnalyzer:
    """YOLO-detection student wrapped with size/area conversion + shape gating."""

    def __init__(
        self,
        weights: Union[str, Path],
        params: Optional[AnalysisParameters] = None,
        conf: float = 0.55,
        imgsz: int = 640,
        max_det: int = 1000,
        device: Optional[str] = None,
    ):
        from ultralytics import YOLO

        self.params = params or AnalysisParameters()
        self.conf = conf
        self.imgsz = imgsz
        self.max_det = max_det
        self.device = device
        self.model = YOLO(str(weights))

    def analyze_image(
        self,
        image: np.ndarray,
        frame_name: str = "frame",
        image_path: Optional[Path] = None,
    ) -> FrameResult:
        h, w = image.shape[:2]

        if image.ndim == 2:
            rgb = np.stack([image] * 3, axis=-1)
        else:
            rgb = image[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        result = self.model.predict(
            source=rgb,
            imgsz=self.imgsz,
            conf=self.conf,
            max_det=self.max_det,
            device=self.device,
            save=False,
            verbose=False,
        )[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
        else:
            boxes = np.zeros((0, 4))

        sx = w / result.orig_shape[1]
        sy = h / result.orig_shape[0]
        boxes = boxes * np.array([sx, sy, sx, sy])

        bubbles: list[BubbleData] = []
        num_rejected = 0
        scale = self.params.scale_um_per_pixel

        for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1
            r_px = (min(bw, bh) + float(np.sqrt(bw * bh))) / 4
            area_px = float(np.pi * r_px * r_px)
            diameter_um = (2 * r_px) * scale
            radius_um = diameter_um / 2
            volume_um3 = (4 / 3) * np.pi * radius_um ** 3

            is_valid = self.params.min_diameter_um <= diameter_um <= self.params.max_diameter_um
            if not is_valid:
                num_rejected += 1

            bubbles.append(
                BubbleData(
                    frame_name=frame_name,
                    bubble_id=i,
                    diameter_um=diameter_um,
                    volume_um3=volume_um3,
                    area_px=area_px,
                    centroid_x=float(cx),
                    centroid_y=float(cy),
                    is_valid=is_valid,
                )
            )

        return FrameResult(
            frame_name=frame_name,
            num_bubbles=sum(1 for b in bubbles if b.is_valid),
            num_rejected=num_rejected,
            bubbles=bubbles,
            image_path=image_path,
        )

    def analyze_directory(
        self,
        directory: Union[str, Path],
        sample_name: Optional[str] = None,
        progress_callback: Optional[Callable[[Path, FrameResult, int, int], None]] = None,
    ) -> AnalysisResults:
        directory = Path(directory)
        image_files = list_images(directory)
        if not image_files:
            raise FileNotFoundError(f"No image files found in {directory}")

        results = AnalysisResults(
            sample_name=sample_name or directory.name,
            parameters=self.params,
        )

        n = len(image_files)
        for idx, img_path in enumerate(image_files):
            image = io.imread(str(img_path))
            frame = self.analyze_image(image, img_path.stem, img_path)
            results.frames.append(frame)
            results.all_bubbles.extend(frame.bubbles)
            if progress_callback is not None:
                progress_callback(img_path, frame, idx + 1, n)

        return results
