"""Inference worker — runs `run_pipeline` on a QThread so the GUI stays responsive."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal
from skimage import io

from ..data import AnalysisResults
from ..infer import StudentAnalyzer, list_images
from ..params import AnalysisParameters
from ..static_filter import StaticFilterConfig, apply_static_filter


class InferenceWorker(QObject):
    progress = Signal(int, int, str)   # current, total, frame_name
    finished = Signal(object)          # AnalysisResults
    failed = Signal(str)
    status = Signal(str)

    def __init__(
        self,
        image_dir: Path,
        params: AnalysisParameters,
        conf: float,
        imgsz: int,
        max_det: int,
        reject_static: bool,
        static_cfg: StaticFilterConfig,
        analyzer: StudentAnalyzer,
    ):
        super().__init__()
        self.image_dir = image_dir
        self.params = params
        self.conf = conf
        self.imgsz = imgsz
        self.max_det = max_det
        self.reject_static = reject_static
        self.static_cfg = static_cfg
        self.analyzer = analyzer
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            analyzer = self.analyzer
            analyzer.params = self.params
            analyzer.conf = self.conf
            analyzer.imgsz = self.imgsz
            analyzer.max_det = self.max_det

            files = list_images(self.image_dir)
            if not files:
                self.failed.emit(f"No images in {self.image_dir}")
                return

            results = AnalysisResults(sample_name=self.image_dir.name, parameters=self.params)
            n = len(files)
            for i, p in enumerate(files):
                if self._cancel:
                    self.failed.emit("Cancelled.")
                    return
                self.progress.emit(i, n, p.name)
                image = io.imread(str(p))
                frame = analyzer.analyze_image(image, p.stem, p)
                results.frames.append(frame)
                results.all_bubbles.extend(frame.bubbles)
            self.progress.emit(n, n, "done")

            if self.reject_static:
                self.status.emit("Filtering static dirt…")
                apply_static_filter(results, self.static_cfg)

            self.finished.emit(results)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def make_thread(worker: InferenceWorker) -> QThread:
    """Wrap worker in a QThread, wiring start → worker.run and cleanup."""
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    return thread
