"""Local YOLOv8 person detector for robot_motion.

Intentionally kept inside this package so person follow and head tracking do
not depend on the robot_vision package or any of its stereo-pipeline defaults.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


@dataclass(frozen=True)
class Detection:
    """One detected person bounding box."""

    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    confidence: float
    class_id: int


@dataclass(frozen=True)
class InferenceResult:
    """Result of one YOLO inference pass."""

    detections: list[Detection]
    inference_time_ms: float
    image_width: int
    image_height: int


class YOLOWrapper:
    """Wrap Ultralytics YOLO and return COCO person detections only."""

    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        device: str = "cpu",
        input_size: int = 640,
    ) -> None:
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. "
                "Install with: pip install ultralytics"
            )

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.input_size = input_size
        self.model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.model = YOLO(self.model_path)
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            self.model.predict(
                dummy,
                device=self.device,
                verbose=False,
                conf=self.confidence_threshold,
            )
        except Exception as exc:
            self.model = None
            raise RuntimeError(
                f"Failed to load YOLO model from {self.model_path!r}: {exc}"
            ) from exc

    def detect(self, image: np.ndarray) -> InferenceResult:
        if self.model is None:
            raise RuntimeError("YOLO model is not loaded")

        image_height, image_width = image.shape[:2]
        start_time = time.perf_counter()

        results = self.model.predict(
            image,
            device=self.device,
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            imgsz=self.input_size,
            classes=[self.PERSON_CLASS_ID],
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        detections: list[Detection] = []

        if results and len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())

                if class_id != self.PERSON_CLASS_ID:
                    continue
                if confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(image_width, int(x2))
                y2 = min(image_height, int(y2))

                bbox_w = x2 - x1
                bbox_h = y2 - y1
                if bbox_w <= 0 or bbox_h <= 0:
                    continue

                detections.append(
                    Detection(
                        bbox_x=x1,
                        bbox_y=y1,
                        bbox_w=bbox_w,
                        bbox_h=bbox_h,
                        confidence=confidence,
                        class_id=class_id,
                    )
                )

        return InferenceResult(
            detections=detections,
            inference_time_ms=inference_time_ms,
            image_width=image_width,
            image_height=image_height,
        )
