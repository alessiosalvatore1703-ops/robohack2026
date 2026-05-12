"""Image conversion helpers for the X2 person tracking nodes.

The forced stereo camera publishes raw sensor_msgs/Image frames, so these
helpers avoid cv_bridge. That keeps the nodes working on ROS Humble systems
where cv_bridge was built against NumPy 1.x but user pip packages installed
NumPy 2.x.
"""

from __future__ import annotations

import numpy as np


def _image_rows(msg, channels: int) -> np.ndarray:
    height = int(msg.height)
    width = int(msg.width)
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size {width}x{height}")

    min_step = width * channels
    step = int(msg.step) if int(msg.step) > 0 else min_step
    if step < min_step:
        raise ValueError(
            f"Image step {step} is too small for {width}px x {channels} channels"
        )

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected_size = step * height
    if raw.size < expected_size:
        raise ValueError(
            f"Image data has {raw.size} bytes, expected at least {expected_size}"
        )

    rows = raw[:expected_size].reshape(height, step)
    return rows[:, :min_step].reshape(height, width, channels)


def image_msg_to_bgr8(msg) -> np.ndarray:
    """Convert a raw sensor_msgs/Image message into a contiguous BGR8 array."""

    encoding = str(msg.encoding).lower()

    if encoding in {"bgr8", "8uc3"}:
        return np.ascontiguousarray(_image_rows(msg, 3))

    if encoding == "rgb8":
        return np.ascontiguousarray(_image_rows(msg, 3)[:, :, ::-1])

    if encoding == "bgra8":
        return np.ascontiguousarray(_image_rows(msg, 4)[:, :, :3])

    if encoding == "rgba8":
        return np.ascontiguousarray(_image_rows(msg, 4)[:, :, [2, 1, 0]])

    if encoding in {"mono8", "8uc1"}:
        mono = _image_rows(msg, 1)[:, :, 0]
        return np.ascontiguousarray(np.repeat(mono[:, :, None], 3, axis=2))

    raise ValueError(
        f"Unsupported image encoding {msg.encoding!r}; expected raw stereo "
        "Image encoding bgr8, rgb8, bgra8, rgba8, mono8, 8UC1, or 8UC3"
    )


def compressed_image_msg_to_bgr8(msg) -> np.ndarray:
    """Convert a sensor_msgs/CompressedImage message into a BGR8 array."""

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError(
            "Compressed image conversion needs opencv-python. The forced stereo "
            "person tracking path should subscribe to raw sensor_msgs/Image."
        ) from exc

    data = np.frombuffer(msg.data, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode compressed image format {msg.format!r}")

    return np.ascontiguousarray(image)
