"""
YOLO Model Wrapper for Person Detection.

Provides a clean interface around the Ultralytics YOLO model,
filtering for person class only and applying confidence thresholds.
"""

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import torch
except ImportError:
    torch = None


@dataclass
class Detection:
    """A single person detection result."""

    bbox_x: int       # Top-left x coordinate
    bbox_y: int       # Top-left y coordinate
    bbox_w: int       # Bounding box width
    bbox_h: int       # Bounding box height
    confidence: float  # Detection confidence [0, 1]
    class_id: int     # Class ID (0 = person in COCO)


@dataclass
class InferenceResult:
    """Result of a single inference pass."""

    detections: List[Detection]
    inference_time_ms: float
    image_width: int
    image_height: int


class YOLOWrapper:
    """Wraps Ultralytics YOLO model for person detection."""

    PERSON_CLASS_ID = 0  # COCO class ID for "person"

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        device: str = "cpu",
        input_size: int = 640,
    ):
        """
        Initialize the YOLO wrapper.

        Args:
            model_path: Path to YOLO model weights file.
            confidence_threshold: Minimum confidence to keep a detection.
            nms_threshold: IoU threshold for Non-Maximum Suppression.
            device: Inference device ('cuda', 'cpu', or 'mps').
            input_size: Model input image size (square).
        """
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. "
                "Install with: pip install ultralytics"
            )

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.requested_device = device
        self.device = self._resolve_device(device)
        self.input_size = input_size
        self.model: Optional[YOLO] = None

        self._load_model()

    def _resolve_device(self, device: str) -> str:
        """Return a usable inference device, falling back from CUDA to CPU."""
        requested = str(device).strip().lower()
        if requested in ("cuda", "cuda:0", "0"):
            if torch is None or not torch.cuda.is_available():
                print(
                    "CUDA requested for YOLO, but torch cannot use CUDA on this "
                    "system. Falling back to CPU.",
                    flush=True,
                )
                return "cpu"
        return device

    def _load_model(self) -> None:
        """Load the YOLO model from disk."""
        try:
            self.model = YOLO(self.model_path)
            # Warm up the model with a dummy inference
            dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
            self.model.predict(
                dummy,
                device=self.device,
                verbose=False,
                conf=self.confidence_threshold,
            )
        except Exception as e:
            self.model = None
            raise RuntimeError(f"Failed to load YOLO model from '{self.model_path}': {e}")

    def reload_model(self, model_path: Optional[str] = None) -> None:
        """
        Reload the model, optionally from a new path.

        Args:
            model_path: New model path, or None to reload current.
        """
        if model_path is not None:
            self.model_path = model_path
        self._load_model()

    def detect(self, image: np.ndarray) -> InferenceResult:
        """
        Run person detection on an image.

        Args:
            image: BGR or RGB numpy array (H, W, 3).

        Returns:
            InferenceResult with filtered person detections.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        h, w = image.shape[:2]

        start_time = time.perf_counter()

        results = self.model.predict(
            image,
            device=self.device,
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            imgsz=self.input_size,
            classes=[self.PERSON_CLASS_ID],  # Filter for person only
        )

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        detections: List[Detection] = []

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    # Double-check person class and confidence
                    if cls_id != self.PERSON_CLASS_ID:
                        continue
                    if conf < self.confidence_threshold:
                        continue

                    # Get bounding box in xyxy format and convert to xywh
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Clip to image boundaries
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(w, int(x2))
                    y2 = min(h, int(y2))

                    bbox_w = x2 - x1
                    bbox_h = y2 - y1

                    if bbox_w <= 0 or bbox_h <= 0:
                        continue

                    detections.append(Detection(
                        bbox_x=x1,
                        bbox_y=y1,
                        bbox_w=bbox_w,
                        bbox_h=bbox_h,
                        confidence=conf,
                        class_id=cls_id,
                    ))

        return InferenceResult(
            detections=detections,
            inference_time_ms=inference_time_ms,
            image_width=w,
            image_height=h,
        )
