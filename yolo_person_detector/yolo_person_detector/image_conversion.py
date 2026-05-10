"""Image conversion helpers that do not depend on cv_bridge."""

import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image


def image_msg_to_bgr8(msg: Image):
    """Convert a sensor_msgs/Image into an OpenCV BGR uint8 image."""
    encoding = msg.encoding.lower()
    dtype = np.uint8
    channels = 1

    if encoding in ("rgb8", "bgr8"):
        channels = 3
    elif encoding in ("rgba8", "bgra8"):
        channels = 4
    elif encoding in ("mono16", "16uc1"):
        dtype = np.uint16
    elif encoding not in ("mono8", "8uc1"):
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    itemsize = np.dtype(dtype).itemsize
    row_items = msg.step // itemsize
    expected_items = msg.width * channels
    if row_items < expected_items:
        raise ValueError(
            f"Image step {msg.step} is too small for {msg.width}x{msg.encoding}"
        )

    arr = np.frombuffer(msg.data, dtype=dtype)
    image = arr.reshape((msg.height, row_items))[:, :expected_items]

    if channels > 1:
        image = image.reshape((msg.height, msg.width, channels))
    else:
        image = image.reshape((msg.height, msg.width))

    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    if encoding == "bgra8":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if encoding in ("mono16", "16uc1"):
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if encoding in ("mono8", "8uc1"):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(image)


def compressed_imgmsg_to_bgr8(msg: CompressedImage):
    """Decode a sensor_msgs/CompressedImage into an OpenCV BGR uint8 image."""
    data = np.frombuffer(msg.data, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode compressed image")
    return image


def bgr8_to_image_msg(image, header=None) -> Image:
    """Convert an OpenCV BGR uint8 image into sensor_msgs/Image."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image with shape HxWx3")
    if image.dtype != np.uint8:
        raise ValueError("Expected a uint8 BGR image")

    contiguous = np.ascontiguousarray(image)
    msg = Image()
    if header is not None:
        msg.header = header
    msg.height = int(contiguous.shape[0])
    msg.width = int(contiguous.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(contiguous.shape[1] * 3)
    msg.data = contiguous.tobytes()
    return msg


def bgr8_to_compressed_imgmsg(image, header=None, jpeg_quality: int = 85):
    """JPEG-encode an OpenCV BGR image into sensor_msgs/CompressedImage."""
    quality = min(max(int(jpeg_quality), 1), 100)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    ok, encoded = cv2.imencode(".jpg", image, encode_params)
    if not ok:
        return None

    msg = CompressedImage()
    if header is not None:
        msg.header = header
    msg.format = "jpeg"
    msg.data = encoded.tobytes()
    return msg
