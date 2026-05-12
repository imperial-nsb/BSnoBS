"""BSnoBS PySide6 GUI."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSlider, QSpinBox, QSplitter,
    QStatusBar, QToolBar, QVBoxLayout, QWidget,
)

from ..cli import default_student_weights
from ..data import AnalysisResults
from ..exporter import ResultsExporter
from ..infer import StudentAnalyzer, list_images
from ..params import AnalysisParameters
from ..static_filter import StaticFilterConfig
from .viewer import FrameViewer
from .worker import InferenceWorker


def _available_devices() -> list[str]:
    """Probe torch for actually-available devices. Always includes auto + cpu."""
    devices = ["auto", "cpu"]
    try:
        import torch
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            devices.append("mps")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(str(i))
    except Exception:
        pass
    return devices


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BSnoBS — BubSize no BS")
        self.resize(1400, 900)

        self._image_dir: Optional[Path] = None
        self._weights: Optional[Path] = default_student_weights()
        self._results: Optional[AnalysisResults] = None
        self._worker: Optional[InferenceWorker] = None
        self._thread: Optional[QThread] = None
        self._analyzer: Optional[StudentAnalyzer] = None
        self._loaded_weights: Optional[Path] = None
        self._loaded_device: Optional[str] = None

        # ---- toolbar -------------------------------------------------------
        tb = QToolBar("Main")
        self.addToolBar(tb)
        open_act = QAction("Select folder…", self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(self._select_folder)
        tb.addAction(open_act)

        # ---- central split -------------------------------------------------
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # ---- status --------------------------------------------------------
        sb = QStatusBar()
        self.setStatusBar(sb)
        self._progress = QProgressBar()
        self._progress.setFixedWidth(220)
        sb.addPermanentWidget(self._progress)
        self._status_label = QLabel("Pick a folder of microscopy images to begin.")
        sb.addWidget(self._status_label, 1)

        self._update_weights_label()
        self._on_model_config_changed()

    # ---- panels ------------------------------------------------------------

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 8)

        # Folder summary
        self._folder_label = QLabel("<i>no folder selected</i>")
        self._folder_label.setWordWrap(True)
        layout.addWidget(self._folder_label)

        # ---- Model panel ---------------------------------------------------
        model_box = QGroupBox("Model")
        model_layout = QVBoxLayout(model_box)
        self._weights_label = QLabel()
        self._weights_label.setWordWrap(True)
        self._weights_label.setStyleSheet("color: #888; font-family: monospace;")
        model_layout.addWidget(self._weights_label)

        weights_row = QHBoxLayout()
        self.btn_pick_weights = QPushButton("Pick .pt file…")
        self.btn_pick_weights.clicked.connect(self._select_weights)
        self.btn_reset_weights = QPushButton("Use bundled")
        self.btn_reset_weights.clicked.connect(self._reset_weights)
        weights_row.addWidget(self.btn_pick_weights)
        weights_row.addWidget(self.btn_reset_weights)
        model_layout.addLayout(weights_row)

        device_row = QHBoxLayout()
        device_row.addWidget(QLabel("Device:"))
        self.s_device = QComboBox(); self.s_device.addItems(_available_devices())
        self.s_device.currentTextChanged.connect(lambda _: self._on_model_config_changed())
        device_row.addWidget(self.s_device, 1)
        model_layout.addLayout(device_row)

        self.btn_load_model = QPushButton("Load model")
        self.btn_load_model.clicked.connect(self._load_model)
        model_layout.addWidget(self.btn_load_model)

        self._model_status = QLabel("Not loaded.")
        self._model_status.setStyleSheet("color: #aaa; font-family: monospace;")
        self._model_status.setWordWrap(True)
        model_layout.addWidget(self._model_status)

        layout.addWidget(model_box)

        # Settings: detection (per-run, doesn't require reload)
        det_box = QGroupBox("Detection")
        det_form = QFormLayout(det_box)
        self.s_conf = QDoubleSpinBox(); self.s_conf.setRange(0.01, 1.0); self.s_conf.setSingleStep(0.05); self.s_conf.setValue(0.55); self.s_conf.setDecimals(2)
        self.s_imgsz = QSpinBox(); self.s_imgsz.setRange(128, 2048); self.s_imgsz.setSingleStep(32); self.s_imgsz.setValue(640)
        self.s_imgsz.setToolTip("Resolution images are resized to before YOLO runs (in px, on the long edge). Higher = slower but may find smaller bubbles.")
        self.s_max_det = QSpinBox(); self.s_max_det.setRange(10, 10000); self.s_max_det.setSingleStep(100); self.s_max_det.setValue(1000)
        det_form.addRow("Confidence", self.s_conf)
        det_form.addRow("Inference resolution (px)", self.s_imgsz)
        det_form.addRow("Max detections", self.s_max_det)
        layout.addWidget(det_box)

        # Settings: physics
        phys_box = QGroupBox("Physics")
        phys_form = QFormLayout(phys_box)
        self.s_scale = QDoubleSpinBox(); self.s_scale.setRange(0.001, 10.0); self.s_scale.setDecimals(4); self.s_scale.setValue(0.0825); self.s_scale.setSingleStep(0.005)
        self.s_volume = QDoubleSpinBox(); self.s_volume.setRange(1e-6, 1.0); self.s_volume.setDecimals(6); self.s_volume.setValue(0.00089); self.s_volume.setSingleStep(0.0001)
        self.s_min_d = QDoubleSpinBox(); self.s_min_d.setRange(0.0, 100.0); self.s_min_d.setValue(0.5); self.s_min_d.setSingleStep(0.1)
        self.s_max_d = QDoubleSpinBox(); self.s_max_d.setRange(0.1, 500.0); self.s_max_d.setValue(20.0); self.s_max_d.setSingleStep(1.0)
        phys_form.addRow("Scale (μm/px)", self.s_scale)
        phys_form.addRow("Volume (μL/frame)", self.s_volume)
        phys_form.addRow("Min diam (μm)", self.s_min_d)
        phys_form.addRow("Max diam (μm)", self.s_max_d)
        layout.addWidget(phys_box)

        # Settings: static dirt filter
        st_box = QGroupBox("Static dirt rejection")
        st_form = QFormLayout(st_box)
        self.s_reject_static = QCheckBox("Reject persistent locations")
        self.s_static_frac = QDoubleSpinBox(); self.s_static_frac.setRange(0.05, 1.0); self.s_static_frac.setValue(0.4); self.s_static_frac.setSingleStep(0.05); self.s_static_frac.setDecimals(2)
        self.s_static_tol = QDoubleSpinBox(); self.s_static_tol.setRange(0.5, 50.0); self.s_static_tol.setValue(6.0); self.s_static_tol.setSingleStep(0.5)
        self.s_static_diam_tol = QDoubleSpinBox(); self.s_static_diam_tol.setRange(0.0, 5.0); self.s_static_diam_tol.setValue(0.5); self.s_static_diam_tol.setSingleStep(0.05)
        st_form.addRow(self.s_reject_static)
        st_form.addRow("Min frame frac", self.s_static_frac)
        st_form.addRow("Centroid tol (px)", self.s_static_tol)
        st_form.addRow("Diam tol", self.s_static_diam_tol)
        layout.addWidget(st_box)

        # Run / Export
        self.btn_run = QPushButton("Run inference")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run_inference)
        layout.addWidget(self.btn_run)

        self.btn_export = QPushButton("Export results…")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_results)
        layout.addWidget(self.btn_export)

        # Summary readout
        self._summary_label = QLabel("")
        self._summary_label.setStyleSheet("color: #aaa; font-family: monospace;")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        layout.addStretch(1)
        return w

    def _build_center_panel(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)

        self.viewer = FrameViewer()
        self.viewer.frame_changed.connect(self._on_frame_changed)
        layout.addWidget(self.viewer, 1)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀"); self.btn_prev.clicked.connect(self.viewer.prev_frame)
        self.btn_next = QPushButton("▶"); self.btn_next.clicked.connect(self.viewer.next_frame)
        self.slider = QSlider(Qt.Horizontal); self.slider.setMinimum(0); self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._slider_changed)
        self.lbl_frame = QLabel("—/—")
        self.lbl_frame.setMinimumWidth(80)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.slider, 1)
        nav.addWidget(self.btn_next)
        nav.addWidget(self.lbl_frame)
        layout.addLayout(nav)

        self.lbl_frame_stats = QLabel("")
        self.lbl_frame_stats.setStyleSheet("color: #aaa; font-family: monospace;")
        layout.addWidget(self.lbl_frame_stats)
        return w

    # ---- folder / weights --------------------------------------------------

    def _select_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Pick folder of images", "")
        if not d:
            return
        path = Path(d)
        if not list_images(path):
            QMessageBox.warning(self, "No images", f"No .tif/.png/.jpg images found in {path}.")
            return
        self._image_dir = path
        n = len(list_images(path))
        self._folder_label.setText(f"<b>Folder:</b> {path}<br/><span style='color:#888'>{n} images</span>")
        self._refresh_run_enabled()

    def _select_weights(self):
        f, _ = QFileDialog.getOpenFileName(self, "Pick YOLO weights", "", "PyTorch weights (*.pt)")
        if not f:
            return
        self._weights = Path(f)
        self._update_weights_label()
        self._on_model_config_changed()

    def _reset_weights(self):
        bundled = default_student_weights()
        if bundled is None:
            QMessageBox.warning(self, "No bundled weights", "Bundled student.pt not found in the install.")
            return
        self._weights = bundled
        self._update_weights_label()
        self._on_model_config_changed()

    def _update_weights_label(self):
        if self._weights is None:
            self._weights_label.setText("<i>no weights found — pick a .pt file</i>")
            return
        bundled = default_student_weights()
        if bundled is not None and self._weights == bundled:
            tag = "<b>bundled (default)</b>"
        else:
            tag = "<b>custom</b>"
        self._weights_label.setText(f"{tag}<br/>{self._weights}")

    def _model_is_current(self) -> bool:
        return (
            self._analyzer is not None
            and self._loaded_weights == self._weights
            and self._loaded_device == self._device_value()
        )

    def _on_model_config_changed(self):
        if self._model_is_current():
            self.btn_load_model.setText("Reload model")
            self.btn_load_model.setEnabled(False)
        else:
            if self._analyzer is None:
                self.btn_load_model.setText("Load model")
            else:
                self.btn_load_model.setText("Reload model")
            self.btn_load_model.setEnabled(self._weights is not None)
        self._refresh_run_enabled()

    def _refresh_run_enabled(self):
        self.btn_run.setEnabled(self._image_dir is not None and self._analyzer is not None)

    def _load_model(self):
        if self._weights is None:
            return
        self.btn_load_model.setEnabled(False)
        self._model_status.setText("Loading…")
        QApplication.processEvents()
        try:
            self._analyzer = StudentAnalyzer(
                weights=self._weights,
                params=self._build_params(),
                conf=self.s_conf.value(),
                imgsz=self.s_imgsz.value(),
                max_det=self.s_max_det.value(),
                device=self._device_value(),
            )
            self._loaded_weights = self._weights
            self._loaded_device = self._device_value()
            dev_show = self._loaded_device or "auto"
            self._model_status.setText(f"Loaded.\nDevice: {dev_show}")
            self._status_label.setText(f"Model loaded ({self._weights.name}, device={dev_show}).")
        except Exception as exc:
            self._analyzer = None
            self._loaded_weights = None
            self._loaded_device = None
            self._model_status.setText(f"Load failed: {exc}")
            QMessageBox.critical(self, "Model load failed", f"{type(exc).__name__}: {exc}")
        self._on_model_config_changed()

    # ---- run ---------------------------------------------------------------

    def _build_params(self) -> AnalysisParameters:
        return AnalysisParameters(
            scale_um_per_pixel=self.s_scale.value(),
            sample_volume_per_frame_uL=self.s_volume.value(),
            min_diameter_um=self.s_min_d.value(),
            max_diameter_um=self.s_max_d.value(),
            pretrained_model=str(self._weights) if self._weights else "",
        )

    def _device_value(self) -> Optional[str]:
        v = self.s_device.currentText()
        return None if v == "auto" else v

    def _run_inference(self):
        if self._image_dir is None or self._analyzer is None:
            return
        self.btn_run.setEnabled(False)
        self.btn_export.setEnabled(False)
        self._results = None
        self._summary_label.setText("")
        self._progress.setRange(0, 0)
        self._status_label.setText("Running inference…")

        worker = InferenceWorker(
            image_dir=self._image_dir,
            params=self._build_params(),
            conf=self.s_conf.value(),
            imgsz=self.s_imgsz.value(),
            max_det=self.s_max_det.value(),
            reject_static=self.s_reject_static.isChecked(),
            static_cfg=StaticFilterConfig(
                min_frame_frac=self.s_static_frac.value(),
                tol_px=self.s_static_tol.value(),
                diameter_tol_frac=self.s_static_diam_tol.value(),
            ),
            analyzer=self._analyzer,
        )
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.status.connect(self._status_label.setText)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        self._worker = worker
        self._thread = thread
        thread.start()

    def _on_progress(self, i: int, n: int, name: str):
        self._progress.setRange(0, n)
        self._progress.setValue(i)
        self._status_label.setText(f"[{i}/{n}] {name}")

    def _on_failed(self, msg: str):
        self._refresh_run_enabled()
        self._progress.setRange(0, 1); self._progress.setValue(0)
        self._status_label.setText(msg)
        QMessageBox.critical(self, "Inference failed", msg)

    def _on_finished(self, results: AnalysisResults):
        self._results = results
        self._refresh_run_enabled()
        self.btn_export.setEnabled(True)
        self._progress.setRange(0, 1); self._progress.setValue(1)
        self._status_label.setText(f"Done — {results.total_bubbles} bubbles across {results.num_frames} frames.")
        self.viewer.set_data(results.frames, results.parameters.scale_um_per_pixel)
        self.slider.setMaximum(max(0, results.num_frames - 1))
        self.slider.setValue(0)
        self._refresh_summary()

    def _refresh_summary(self):
        if not self._results:
            self._summary_label.setText("")
            return
        d = self._results.diameters
        p = self._results.parameters
        total_vol = p.sample_volume_per_frame_uL * self._results.num_frames
        conc = self._results.total_bubbles / total_vol if total_vol else 0
        n_static = sum(1 for b in self._results.all_bubbles if getattr(b, "is_static", False))
        lines = [
            f"Frames        : {self._results.num_frames}",
            f"Accepted      : {self._results.total_bubbles}",
            f"Rejected      : {self._results.total_rejected}",
            f"  of which static: {n_static}",
        ]
        if len(d):
            lines += [
                f"Mean diameter : {np.mean(d):.2f} ± {np.std(d):.2f} μm",
                f"Median diam   : {np.median(d):.2f} μm",
                f"Concentration : {conc:.3e} /μL",
            ]
        self._summary_label.setText("\n".join(lines))

    # ---- frame nav ---------------------------------------------------------

    def _slider_changed(self, v: int):
        if self.viewer.current_index() != v:
            self.viewer.show_frame(v)

    def _on_frame_changed(self, idx: int):
        if self.slider.value() != idx:
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)
        if not self._results:
            return
        n = self._results.num_frames
        self.lbl_frame.setText(f"{idx + 1}/{n}")
        f = self._results.frames[idx]
        n_static = sum(1 for b in f.bubbles if getattr(b, "is_static", False))
        self.lbl_frame_stats.setText(
            f"frame: {f.frame_name}   valid: {f.num_bubbles}   "
            f"rejected: {f.num_rejected}   static: {n_static}"
        )

    # ---- export ------------------------------------------------------------

    def _export_results(self):
        if self._results is None or self._image_dir is None:
            return
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        conf_tag = f"conf{int(round(self.s_conf.value() * 100)):02d}"
        suffix = "_static" if self.s_reject_static.isChecked() else ""
        default_name = f"{self._image_dir.name}_count_{conf_tag}{suffix}_{ts}"
        default_path = str(self._image_dir.parent / default_name)
        out = QFileDialog.getExistingDirectory(self, "Choose output directory (will be created)", default_path)
        # On macOS, the dialog won't return a non-existent path. Use a workaround: ask for parent then create child.
        if not out:
            # Fallback: file dialog for save name
            name, _ = QFileDialog.getSaveFileName(self, "Output folder name", default_path)
            if not name:
                return
            out = name
        out_dir = Path(out)
        out_dir.mkdir(parents=True, exist_ok=True)

        self._status_label.setText(f"Exporting to {out_dir}…")
        QApplication.processEvents()
        try:
            ResultsExporter(self._results, out_dir).export_all()
            metadata = {
                "command": "gui",
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "image_dir": str(self._image_dir),
                "output_dir": str(out_dir),
                "weights": str(self._weights),
                "yolo": {
                    "conf": self.s_conf.value(),
                    "imgsz": self.s_imgsz.value(),
                    "max_det": self.s_max_det.value(),
                    "device": self._device_value(),
                },
                "physics": {
                    "scale_um_per_pixel": self.s_scale.value(),
                    "sample_volume_per_frame_uL": self.s_volume.value(),
                    "min_diameter_um": self.s_min_d.value(),
                    "max_diameter_um": self.s_max_d.value(),
                },
                "static_filter": {
                    "enabled": self.s_reject_static.isChecked(),
                    "min_frame_frac": self.s_static_frac.value(),
                    "tol_px": self.s_static_tol.value(),
                    "diameter_tol_frac": self.s_static_diam_tol.value(),
                },
                "n_frames": self._results.num_frames,
                "n_bubbles_accepted": self._results.total_bubbles,
                "n_bubbles_rejected": self._results.total_rejected,
                "n_static_rejected": sum(1 for b in self._results.all_bubbles if getattr(b, "is_static", False)),
            }
            (out_dir / f"{self._results.sample_name}_run_metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", f"{type(exc).__name__}: {exc}")
            self._status_label.setText("Export failed.")
            return
        self._status_label.setText(f"Wrote outputs to {out_dir}")
        QMessageBox.information(self, "Export complete", f"Wrote results to:\n{out_dir}")


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
