"""Unit tests for the YOLO wrapper."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo_person_detector.yolo_wrapper import YOLOWrapper, Detection, InferenceResult


class TestDetectionDataclass:
    """Test the Detection dataclass."""

    def test_detection_creation(self):
        det = Detection(
            bbox_x=10, bbox_y=20, bbox_w=100, bbox_h=200,
            confidence=0.85, class_id=0,
        )
        assert det.bbox_x == 10
        assert det.bbox_y == 20
        assert det.bbox_w == 100
        assert det.bbox_h == 200
        assert det.confidence == 0.85
        assert det.class_id == 0

    def test_inference_result_creation(self):
        dets = [
            Detection(10, 20, 100, 200, 0.9, 0),
            Detection(300, 100, 80, 150, 0.7, 0),
        ]
        result = InferenceResult(
            detections=dets,
            inference_time_ms=5.2,
            image_width=640,
            image_height=480,
        )
        assert len(result.detections) == 2
        assert result.inference_time_ms == 5.2
        assert result.image_width == 640
        assert result.image_height == 480


class TestYOLOWrapper:
    """Test the YOLO wrapper (requires ultralytics installed)."""

    @pytest.fixture
    def yolo(self):
        """Create a YOLO wrapper instance."""
        try:
            return YOLOWrapper(
                model_path='yolov8n.pt',
                confidence_threshold=0.5,
                device='cpu',
                input_size=320,  # Smaller for faster tests
            )
        except (ImportError, RuntimeError) as e:
            pytest.skip(f'YOLO model not available: {e}')

    def test_model_loads(self, yolo):
        """Test that model loads successfully."""
        assert yolo.model is not None

    def test_detect_empty_image(self, yolo):
        """Test detection on a blank image returns no detections."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        result = yolo.detect(blank)
        assert isinstance(result, InferenceResult)
        assert result.image_width == 640
        assert result.image_height == 480
        assert result.inference_time_ms > 0

    def test_detect_returns_valid_boxes(self, yolo):
        """Test that any detections have valid bounding boxes."""
        # Create a random image (unlikely to have persons, but tests the pipeline)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = yolo.detect(img)

        for det in result.detections:
            assert det.bbox_x >= 0
            assert det.bbox_y >= 0
            assert det.bbox_w > 0
            assert det.bbox_h > 0
            assert det.bbox_x + det.bbox_w <= 640
            assert det.bbox_y + det.bbox_h <= 480
            assert 0.0 <= det.confidence <= 1.0
            assert det.class_id == 0  # Person class only

    def test_confidence_threshold(self, yolo):
        """Test that all detections meet confidence threshold."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = yolo.detect(img)

        for det in result.detections:
            assert det.confidence >= yolo.confidence_threshold

    def test_person_class_only(self, yolo):
        """Test that only person class detections are returned."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = yolo.detect(img)

        for det in result.detections:
            assert det.class_id == YOLOWrapper.PERSON_CLASS_ID


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
