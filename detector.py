"""
detector.py - Object detection logic using Ultralytics YOLOv8

Installation:
    pip install ultralytics opencv-python pillow

This module is intentionally decoupled from any GUI code so it can be
tested independently or swapped out for a different model family.
"""

import time

import cv2
import numpy as np
from ultralytics import YOLO

# ── Colour palette for bounding boxes (one colour per class, cycling) ─────────
_PALETTE = [
    (255,  56,  56), (255, 157,  51), ( 34, 197, 255), ( 99, 255,  80),
    (255,  99, 204), ( 56, 255, 165), (255, 255,  51), (  0, 161, 255),
    (173,   3, 255), (255,  99,  71),
]


class ObjectDetector:
    """
    Wraps a YOLOv8 model and exposes a single ``detect()`` method that
    accepts a raw BGR frame and returns an annotated copy together with
    structured detection results and a live FPS reading.
    """

    def __init__(self, model_name: str = "yolov8n", conf_threshold: float = 0.50):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.model: YOLO | None = None

        # FPS tracking
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = time.perf_counter()

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_model(self, model_name: str | None = None) -> None:
        """
        Download (first run) and load a YOLOv8 model.

        Args:
            model_name: e.g. ``"yolov8n"``, ``"yolov8s"``, ``"yolov8m"``, …
                        Falls back to ``self.model_name`` if *None*.
        """
        if model_name:
            self.model_name = model_name

        print(f"[Detector] Loading model '{self.model_name}' …")
        # YOLO() auto-downloads the .pt file on the first call
        self.model = YOLO(f"{self.model_name}.pt")
        print(f"[Detector] Model ready: {self.model_name}")

        # Reset FPS counters whenever a new model is loaded
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.perf_counter()

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, list[tuple], float]:
        """
        Run inference on a single BGR frame.

        Args:
            frame: Raw frame from ``cv2.VideoCapture``.

        Returns:
            annotated_frame : BGR copy of *frame* with boxes, labels, and an FPS
                              overlay drawn on it.
            detections      : List of ``(label, confidence, (x1, y1, x2, y2))``
                              for every detection above ``conf_threshold``.
            fps             : Smoothed frames-per-second estimate.
        """
        if self.model is None:
            return frame, [], 0.0

        self._tick()

        # ── Inference ──────────────────────────────────────────────────────────
        # verbose=False suppresses the per-frame console spam from ultralytics
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)

        # ── Render detections ──────────────────────────────────────────────────
        annotated = frame.copy()
        detections: list[tuple] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = result.names[cls_id]
                color = _PALETTE[cls_id % len(_PALETTE)]

                # Filled rectangle for the bounding box outline
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Semi-transparent filled background for the label chip
                tag = f"{label}  {conf:.0%}"
                (tw, th), bl = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                chip_top = max(y1 - th - bl - 6, 0)
                cv2.rectangle(annotated, (x1, chip_top), (x1 + tw + 6, y1), color, cv2.FILLED)
                cv2.putText(
                    annotated, tag,
                    (x1 + 3, y1 - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )

                detections.append((label, conf, (x1, y1, x2, y2)))

        # ── FPS overlay (bottom-left) ──────────────────────────────────────────
        fps_text = f"FPS: {self._fps:5.1f}"
        # Dark shadow for readability on any background
        cv2.putText(annotated, fps_text, (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(annotated, fps_text, (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 255, 100), 2, cv2.LINE_AA)

        return annotated, detections, self._fps

    def set_conf_threshold(self, value: float) -> None:
        """Clamp and update the confidence threshold (0.05 – 0.99)."""
        self.conf_threshold = max(0.05, min(0.99, float(value)))

    # ── Private helpers ────────────────────────────────────────────────────────

    def _tick(self) -> None:
        """Update the rolling FPS counter once per second."""
        self._frame_count += 1
        elapsed = time.perf_counter() - self._fps_timer
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = time.perf_counter()
