"""Image viewer with detection overlays. Shows one frame at a time."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QBrush, QColor, QImage, QPen, QPixmap, QPainter, QFont, QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsScene, QGraphicsView, QGraphicsSimpleTextItem,
)
from skimage import io

from ..data import FrameResult


# Colours match the matplotlib overlays.
_C_VALID    = QColor("#00E676")
_C_REJECTED = QColor("#FF1744")
_C_STATIC   = QColor("#FFEA00")


def _ndarray_to_qimage(arr: np.ndarray) -> QImage:
    """Convert a uint8 grayscale or RGB ndarray to QImage (deep-copied)."""
    if arr.ndim == 2:
        a = arr if arr.dtype == np.uint8 else np.clip(arr, 0, 255).astype(np.uint8)
        h, w = a.shape
        a = np.ascontiguousarray(a)
        return QImage(a.data, w, h, w, QImage.Format_Grayscale8).copy()
    a = arr[..., :3] if arr.ndim == 3 else arr
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    a = np.ascontiguousarray(a)
    h, w, _ = a.shape
    return QImage(a.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class FrameViewer(QGraphicsView):
    """Scene-based image viewer. Use `show_frame()` to display + overlay detections."""

    frame_changed = Signal(int)  # emitted when wheel/arrow changes frame

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        self.setBackgroundBrush(QBrush(QColor("#111111")))
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self._frames: List[FrameResult] = []
        self._scale_um_per_pixel: float = 0.0825
        self._index: int = -1
        self._fit_on_next = True

    def set_data(self, frames: List[FrameResult], scale_um_per_pixel: float):
        self._frames = frames
        self._scale_um_per_pixel = scale_um_per_pixel
        self._index = 0 if frames else -1
        self._fit_on_next = True
        if frames:
            self.show_frame(0)

    def num_frames(self) -> int:
        return len(self._frames)

    def current_index(self) -> int:
        return self._index

    def show_frame(self, idx: int):
        if not self._frames or not (0 <= idx < len(self._frames)):
            return
        self._index = idx
        frame = self._frames[idx]
        self._scene.clear()

        if frame.image_path and Path(frame.image_path).exists():
            arr = io.imread(str(frame.image_path))
            qimg = _ndarray_to_qimage(arr)
            pix = QPixmap.fromImage(qimg)
            self._scene.addPixmap(pix)
            self._scene.setSceneRect(QRectF(0, 0, pix.width(), pix.height()))
        else:
            self._scene.addText("(image not found)", QFont("monospace", 14))

        for b in frame.bubbles:
            if getattr(b, "is_static", False):
                color, dashed = _C_STATIC, True
            elif b.is_valid:
                color, dashed = _C_VALID, False
            else:
                color, dashed = _C_REJECTED, False

            r_px = (b.diameter_um / 2) / self._scale_um_per_pixel
            x = b.centroid_x - r_px
            y = b.centroid_y - r_px
            ell = QGraphicsEllipseItem(QRectF(x, y, 2 * r_px, 2 * r_px))
            pen = QPen(color)
            pen.setWidthF(1.6)
            pen.setCosmetic(True)
            if dashed:
                pen.setStyle(Qt.DashLine)
            ell.setPen(pen)
            ell.setBrush(Qt.NoBrush)
            self._scene.addItem(ell)

            if b.is_valid:
                label = QGraphicsSimpleTextItem(f"{b.diameter_um:.1f}")
                label.setBrush(QBrush(_C_VALID))
                f = QFont(); f.setPointSizeF(6.0)
                label.setFont(f)
                label.setPos(b.centroid_x + r_px, b.centroid_y - r_px - 6)
                self._scene.addItem(label)

        if self._fit_on_next:
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
            self._fit_on_next = False

        self.frame_changed.emit(idx)

    def fit_view(self):
        if self._scene.sceneRect().isValid():
            self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def next_frame(self):
        if self._frames and self._index < len(self._frames) - 1:
            self.show_frame(self._index + 1)

    def prev_frame(self):
        if self._frames and self._index > 0:
            self.show_frame(self._index - 1)

    def wheelEvent(self, ev: QWheelEvent):
        # Ctrl + wheel = zoom; plain wheel = next/prev frame.
        if ev.modifiers() & Qt.ControlModifier:
            factor = 1.15 if ev.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            if ev.angleDelta().y() > 0:
                self.prev_frame()
            else:
                self.next_frame()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Right, Qt.Key_Down, Qt.Key_PageDown, Qt.Key_Space):
            self.next_frame()
        elif ev.key() in (Qt.Key_Left, Qt.Key_Up, Qt.Key_PageUp):
            self.prev_frame()
        elif ev.key() == Qt.Key_F:
            self.fit_view()
        else:
            super().keyPressEvent(ev)
