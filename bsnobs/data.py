from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .params import AnalysisParameters


@dataclass
class BubbleData:
    frame_name: str
    bubble_id: int
    diameter_um: float
    volume_um3: float
    area_px: float
    centroid_x: float
    centroid_y: float
    circularity: float = 1.0
    aspect_ratio: float = 1.0
    is_valid: bool = True
    is_static: bool = False


@dataclass
class FrameResult:
    frame_name: str
    num_bubbles: int
    num_rejected: int
    bubbles: list = field(default_factory=list)
    image_path: Optional[Path] = None


@dataclass
class AnalysisResults:
    sample_name: str
    frames: list = field(default_factory=list)
    all_bubbles: list = field(default_factory=list)
    parameters: AnalysisParameters = field(default_factory=AnalysisParameters)

    @property
    def diameters(self) -> np.ndarray:
        return np.array([b.diameter_um for b in self.all_bubbles if b.is_valid])

    @property
    def volumes(self) -> np.ndarray:
        return np.array([b.volume_um3 for b in self.all_bubbles if b.is_valid])

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def total_bubbles(self) -> int:
        return sum(1 for b in self.all_bubbles if b.is_valid)

    @property
    def total_rejected(self) -> int:
        return sum(f.num_rejected for f in self.frames)
