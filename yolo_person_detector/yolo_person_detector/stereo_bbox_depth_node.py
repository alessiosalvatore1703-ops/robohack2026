"""Estimate person bbox depth from the Agibot X2 front stereo pair."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from vision_msgs.msg import Detection2DArray

from .image_conversion import (
    bgr8_to_compressed_imgmsg,
    bgr8_to_image_msg,
    image_msg_to_bgr8,
)


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
)

RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

CAMERA_INFO_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


@dataclass
class SyncedPair:
    left_msg: Image
    right_msg: Image


class StereoBboxDepthNode(Node):
    """Compute bbox depth using StereoSGBM disparity."""

    def __init__(self):
        super().__init__("stereo_bbox_depth")

        self.declare_parameter(
            "left_image_topic", "/aima/hal/sensor/stereo_head_front_left/rgb_image"
        )
        self.declare_parameter(
            "right_image_topic", "/aima/hal/sensor/stereo_head_front_right/rgb_image"
        )
        self.declare_parameter(
            "left_camera_info_topic",
            "/aima/hal/sensor/stereo_head_front_left/camera_info",
        )
        self.declare_parameter(
            "right_camera_info_topic",
            "/aima/hal/sensor/stereo_head_front_right/camera_info",
        )
        self.declare_parameter("detections_topic", "/stereo_person/detections")
        self.declare_parameter("depth_topic", "/stereo_person/person_depth")
        self.declare_parameter("debug_image_topic", "/stereo_person/depth_debug_image")
        self.declare_parameter(
            "debug_compressed_topic",
            "/stereo_person/depth_debug_image/compressed",
        )
        self.declare_parameter("jpeg_quality", 85)
        self.declare_parameter("baseline_m", 0.0)
        self.declare_parameter("sync_slop_sec", 0.05)
        self.declare_parameter("min_depth_m", 0.3)
        self.declare_parameter("max_depth_m", 8.0)
        self.declare_parameter("roi_shrink", 0.5)
        self.declare_parameter("min_valid_disparity_pixels", 80)
        self.declare_parameter("stereo_num_disparities", 96)
        self.declare_parameter("stereo_block_size", 7)

        self.left_image_topic = self.get_parameter("left_image_topic").value
        self.right_image_topic = self.get_parameter("right_image_topic").value
        self.sync_slop_sec = float(self.get_parameter("sync_slop_sec").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.roi_shrink = float(self.get_parameter("roi_shrink").value)
        self.min_valid_disparity_pixels = int(
            self.get_parameter("min_valid_disparity_pixels").value
        )
        self.baseline_m = float(self.get_parameter("baseline_m").value)
        self.jpeg_quality = self._jpeg_quality()

        num_disparities = int(self.get_parameter("stereo_num_disparities").value)
        num_disparities = max(16, int(np.ceil(num_disparities / 16.0)) * 16)
        block_size = int(self.get_parameter("stereo_block_size").value)
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(5, block_size)
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size * block_size,
            P2=32 * 3 * block_size * block_size,
            uniquenessRatio=8,
            speckleWindowSize=80,
            speckleRange=2,
            disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        self.left_info: Optional[CameraInfo] = None
        self.right_info: Optional[CameraInfo] = None
        self.left_msg: Optional[Image] = None
        self.right_msg: Optional[Image] = None
        self.latest_detections: Optional[Detection2DArray] = None
        self.maps = None

        self.create_subscription(Image, self.left_image_topic, self._left_callback, SENSOR_QOS)
        self.create_subscription(
            Image, self.right_image_topic, self._right_callback, SENSOR_QOS
        )
        self.create_subscription(
            CameraInfo,
            self.get_parameter("left_camera_info_topic").value,
            self._left_info_callback,
            CAMERA_INFO_QOS,
        )
        self.create_subscription(
            CameraInfo,
            self.get_parameter("right_camera_info_topic").value,
            self._right_info_callback,
            CAMERA_INFO_QOS,
        )
        self.create_subscription(
            Detection2DArray,
            self.get_parameter("detections_topic").value,
            self._detections_callback,
            RELIABLE_QOS,
        )

        self.depth_pub = self.create_publisher(
            PointStamped, self.get_parameter("depth_topic").value, RELIABLE_QOS
        )
        self.debug_pub = self.create_publisher(
            Image, self.get_parameter("debug_image_topic").value, SENSOR_QOS
        )
        self.debug_compressed_pub = self.create_publisher(
            CompressedImage,
            self.get_parameter("debug_compressed_topic").value,
            SENSOR_QOS,
        )

        self.get_logger().info(
            f"Stereo bbox depth listening on {self.left_image_topic} and {self.right_image_topic}"
        )

    def _left_callback(self, msg: Image) -> None:
        self.left_msg = msg
        self._try_compute()

    def _right_callback(self, msg: Image) -> None:
        self.right_msg = msg
        self._try_compute()

    def _left_info_callback(self, msg: CameraInfo) -> None:
        self.left_info = msg
        self.maps = None

    def _right_info_callback(self, msg: CameraInfo) -> None:
        self.right_info = msg
        self.maps = None

    def _detections_callback(self, msg: Detection2DArray) -> None:
        self.latest_detections = msg
        self._try_compute()

    def _try_compute(self) -> None:
        pair = self._synced_pair()
        if pair is None or self.latest_detections is None:
            return
        if not self.latest_detections.detections:
            return

        target = max(
            self.latest_detections.detections,
            key=lambda det: det.bbox.size_x * det.bbox.size_y,
        )

        try:
            left = image_msg_to_bgr8(pair.left_msg)
            right = image_msg_to_bgr8(pair.right_msg)
        except Exception as exc:
            self.get_logger().warn(f"Stereo image conversion failed: {exc}")
            return

        left_rect, right_rect = self._rectify_if_possible(left, right)
        depth = self._depth_from_bbox(left_rect, right_rect, target)
        if depth is None:
            return

        x_m, y_m, z_m, valid_count = depth
        point = PointStamped()
        point.header = pair.left_msg.header
        point.point.x = float(x_m)
        point.point.y = float(y_m)
        point.point.z = float(z_m)
        self.depth_pub.publish(point)
        self._publish_debug(left_rect, target, z_m, valid_count, pair.left_msg.header)

    def _synced_pair(self) -> Optional[SyncedPair]:
        if self.left_msg is None or self.right_msg is None:
            return None
        dt = abs(self._stamp_sec(self.left_msg) - self._stamp_sec(self.right_msg))
        if dt > self.sync_slop_sec:
            return None
        return SyncedPair(self.left_msg, self.right_msg)

    def _rectify_if_possible(self, left, right):
        if self.left_info is None or self.right_info is None:
            return left, right
        if self.maps is None:
            self.maps = self._build_rectification_maps(left.shape[1], left.shape[0])
        if self.maps is None:
            return left, right
        lmap1, lmap2, rmap1, rmap2 = self.maps
        return (
            cv2.remap(left, lmap1, lmap2, cv2.INTER_LINEAR),
            cv2.remap(right, rmap1, rmap2, cv2.INTER_LINEAR),
        )

    def _build_rectification_maps(self, width: int, height: int):
        try:
            left_k = np.array(self.left_info.k, dtype=np.float64).reshape(3, 3)
            right_k = np.array(self.right_info.k, dtype=np.float64).reshape(3, 3)
            left_d = np.array(self.left_info.d, dtype=np.float64)
            right_d = np.array(self.right_info.d, dtype=np.float64)
            left_r = np.array(self.left_info.r, dtype=np.float64).reshape(3, 3)
            right_r = np.array(self.right_info.r, dtype=np.float64).reshape(3, 3)
            left_p = np.array(self.left_info.p, dtype=np.float64).reshape(3, 4)
            right_p = np.array(self.right_info.p, dtype=np.float64).reshape(3, 4)
            size = (width, height)
            lmap1, lmap2 = cv2.initUndistortRectifyMap(
                left_k, left_d, left_r, left_p[:, :3], size, cv2.CV_32FC1
            )
            rmap1, rmap2 = cv2.initUndistortRectifyMap(
                right_k, right_d, right_r, right_p[:, :3], size, cv2.CV_32FC1
            )
            return lmap1, lmap2, rmap1, rmap2
        except Exception as exc:
            self.get_logger().warn(f"Could not build rectification maps: {exc}")
            return None

    def _depth_from_bbox(self, left, right, detection):
        gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

        x1, y1, x2, y2 = self._bbox_xyxy(detection, left.shape[1], left.shape[0])
        x1, y1, x2, y2 = self._shrink_roi(x1, y1, x2, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        roi = disparity[y1:y2, x1:x2]
        valid = roi[np.isfinite(roi) & (roi > 0.5)]
        if valid.size < self.min_valid_disparity_pixels:
            return None

        disp = float(np.median(valid))
        fx = self._fx()
        baseline = self._baseline()
        if fx <= 0.0 or baseline <= 0.0:
            self.get_logger().warn(
                "No usable stereo baseline. Set baseline_m or provide calibrated CameraInfo.",
                throttle_duration_sec=3.0,
            )
            return None

        z_m = fx * baseline / disp
        if z_m < self.min_depth_m or z_m > self.max_depth_m:
            return None

        cx = self._cx()
        cy = self._cy()
        bbox_cx = detection.bbox.center.position.x
        bbox_cy = detection.bbox.center.position.y
        x_m = (bbox_cx - cx) * z_m / fx if fx > 0 else 0.0
        y_m = (bbox_cy - cy) * z_m / fx if fx > 0 else 0.0
        return x_m, y_m, z_m, int(valid.size)

    def _bbox_xyxy(self, detection, width: int, height: int):
        bbox = detection.bbox
        cx = float(bbox.center.position.x)
        cy = float(bbox.center.position.y)
        half_w = float(bbox.size_x) / 2.0
        half_h = float(bbox.size_y) / 2.0
        x1 = max(0, int(cx - half_w))
        y1 = max(0, int(cy - half_h))
        x2 = min(width, int(cx + half_w))
        y2 = min(height, int(cy + half_h))
        return x1, y1, x2, y2

    def _shrink_roi(self, x1: int, y1: int, x2: int, y2: int):
        shrink = min(max(self.roi_shrink, 0.1), 1.0)
        if shrink >= 0.999:
            return x1, y1, x2, y2
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        half_w = (x2 - x1) * shrink / 2.0
        half_h = (y2 - y1) * shrink / 2.0
        return int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h)

    def _fx(self) -> float:
        if self.left_info is not None and self.left_info.p[0] > 0:
            return float(self.left_info.p[0])
        if self.left_info is not None and self.left_info.k[0] > 0:
            return float(self.left_info.k[0])
        return 0.0

    def _cx(self) -> float:
        if self.left_info is not None and self.left_info.p[2] > 0:
            return float(self.left_info.p[2])
        if self.left_info is not None and self.left_info.k[2] > 0:
            return float(self.left_info.k[2])
        return 0.0

    def _cy(self) -> float:
        if self.left_info is not None and self.left_info.p[6] > 0:
            return float(self.left_info.p[6])
        if self.left_info is not None and self.left_info.k[5] > 0:
            return float(self.left_info.k[5])
        return 0.0

    def _baseline(self) -> float:
        if self.baseline_m > 0.0:
            return self.baseline_m
        if self.right_info is not None and self.right_info.p[0] != 0.0:
            baseline = -float(self.right_info.p[3]) / float(self.right_info.p[0])
            if baseline > 0.0:
                return baseline
        return 0.0

    def _publish_debug(self, image, detection, depth_m: float, valid_count: int, header) -> None:
        debug = image.copy()
        x1, y1, x2, y2 = self._bbox_xyxy(detection, image.shape[1], image.shape[0])
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{depth_m:.2f} m ({valid_count} px)"
        cv2.putText(
            debug,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        out = bgr8_to_image_msg(debug, header)
        self.debug_pub.publish(out)

        compressed = self._cv2_to_compressed_imgmsg(debug, header)
        if compressed is not None:
            self.debug_compressed_pub.publish(compressed)

    def _cv2_to_compressed_imgmsg(self, image, header):
        msg = bgr8_to_compressed_imgmsg(
            image, header=header, jpeg_quality=self.jpeg_quality
        )
        if msg is None:
            self.get_logger().warn("Failed to JPEG-encode depth debug image")
        return msg

    def _jpeg_quality(self) -> int:
        quality = int(self.get_parameter("jpeg_quality").value)
        return min(max(quality, 1), 100)

    @staticmethod
    def _stamp_sec(msg) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = StereoBboxDepthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
