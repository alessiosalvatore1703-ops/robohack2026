"""Final compressed stereo person annotation pipeline.

This node subscribes directly to the front stereo head compressed image pair,
runs person detection on the left image, estimates per-person depth from the
stereo pair when possible, and publishes a single final compressed JPEG stream.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32

from .image_conversion import bgr8_to_compressed_imgmsg, compressed_imgmsg_to_bgr8
from .yolo_wrapper import Detection, YOLOWrapper


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
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
)

DEFAULT_BASELINE_M = 0.0578
DEFAULT_FOCAL_PX = 695.7059016837189
DEFAULT_CX = 1027.041296886738
DEFAULT_CY = 776.242452818167


@dataclass
class DepthEstimate:
    x_m: float
    y_m: float
    z_m: float
    valid_pixels: int


class StereoFinalAnnotatorNode(Node):
    """Publish final left-camera person annotations with best-effort depth."""

    def __init__(self):
        super().__init__("stereo_final_annotator")

        self.declare_parameter(
            "left_image_topic",
            "/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed",
        )
        self.declare_parameter(
            "right_image_topic",
            "/aima/hal/sensor/stereo_head_front_right/rgb_image/compressed",
        )
        self.declare_parameter(
            "left_camera_info_topic",
            "/aima/hal/sensor/stereo_head_front_left/camera_info",
        )
        self.declare_parameter(
            "right_camera_info_topic",
            "/aima/hal/sensor/stereo_head_front_right/camera_info",
        )
        self.declare_parameter(
            "output_topic", "/stereo_person/final_annotated_image/compressed"
        )
        self.declare_parameter("inference_time_topic", "/stereo_person/inference_time")
        self.declare_parameter("jpeg_quality", 75)
        self.declare_parameter("output_width", 960)
        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("nms_threshold", 0.45)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("input_size", 320)
        self.declare_parameter("processing_width", 512)
        self.declare_parameter("max_processing_fps", 6.0)
        self.declare_parameter("baseline_m", DEFAULT_BASELINE_M)
        self.declare_parameter("sync_slop_sec", 0.10)
        self.declare_parameter("right_buffer_size", 20)
        self.declare_parameter("min_depth_m", 0.3)
        self.declare_parameter("max_depth_m", 8.0)
        self.declare_parameter("roi_shrink", 0.5)
        self.declare_parameter("min_valid_disparity_pixels", 80)
        self.declare_parameter("stereo_num_disparities", 96)
        self.declare_parameter("stereo_block_size", 7)

        self.left_image_topic = self.get_parameter("left_image_topic").value
        self.right_image_topic = self.get_parameter("right_image_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.jpeg_quality = self._jpeg_quality()
        self.output_width = max(0, int(self.get_parameter("output_width").value))
        self.baseline_m = float(self.get_parameter("baseline_m").value)
        self.sync_slop_sec = float(self.get_parameter("sync_slop_sec").value)
        self.right_buffer_size = max(
            1, int(self.get_parameter("right_buffer_size").value)
        )
        self.processing_width = max(
            0, int(self.get_parameter("processing_width").value)
        )
        max_processing_fps = float(self.get_parameter("max_processing_fps").value)
        self.processing_period_sec = 1.0 / max(max_processing_fps, 0.1)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.roi_shrink = float(self.get_parameter("roi_shrink").value)
        self.min_valid_disparity_pixels = int(
            self.get_parameter("min_valid_disparity_pixels").value
        )

        self.yolo = YOLOWrapper(
            model_path=self.get_parameter("model_path").value,
            confidence_threshold=float(
                self.get_parameter("confidence_threshold").value
            ),
            nms_threshold=float(self.get_parameter("nms_threshold").value),
            device=self.get_parameter("device").value,
            input_size=int(self.get_parameter("input_size").value),
        )
        self.stereo = self._create_stereo_matcher()

        self.right_msgs = deque(maxlen=self.right_buffer_size)
        self.latest_left_msg: Optional[CompressedImage] = None
        self.processing = False
        self.left_info: Optional[CameraInfo] = None
        self.right_info: Optional[CameraInfo] = None
        self.maps = None
        self.maps_scale = None
        self.frame_count = 0

        self.create_subscription(
            CompressedImage, self.left_image_topic, self._left_callback, SENSOR_QOS
        )
        self.create_subscription(
            CompressedImage, self.right_image_topic, self._right_callback, SENSOR_QOS
        )
        self.create_timer(self.processing_period_sec, self._process_latest)
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

        self.output_pub = self.create_publisher(
            CompressedImage, self.output_topic, SENSOR_QOS
        )
        self.inference_pub = self.create_publisher(
            Float32, self.get_parameter("inference_time_topic").value, RELIABLE_QOS
        )

        self.get_logger().info(
            f"Final stereo annotator: {self.left_image_topic} + "
            f"{self.right_image_topic} -> {self.output_topic}"
        )

    def _left_callback(self, msg: CompressedImage) -> None:
        self.latest_left_msg = msg

    def _process_latest(self) -> None:
        if self.processing or self.latest_left_msg is None:
            return
        self.processing = True
        msg = self.latest_left_msg
        try:
            self._process_left(msg)
        finally:
            self.processing = False

    def _process_left(self, msg: CompressedImage) -> None:
        try:
            left = compressed_imgmsg_to_bgr8(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode left image: {exc}")
            return
        left_proc, scale = self._processing_image(left)

        try:
            result = self.yolo.detect(left_proc)
        except Exception as exc:
            self.get_logger().error(f"YOLO inference failed: {exc}")
            return
        detections = self._scale_detections(result.detections, 1.0 / scale)

        self._publish_inference_time(result.inference_time_ms)

        right = self._synced_right_image(msg)
        depths: Dict[int, DepthEstimate] = {}
        if right is not None and detections:
            right_proc = self._resize_like(right, left_proc)
            depths = self._estimate_depths(left_proc, right_proc, detections, scale)
        detections, depths = self._select_closest_detection(detections, depths)

        annotated = self._annotate(
            left, detections, depths, result.inference_time_ms
        )
        annotated = self._output_image(annotated)
        out = bgr8_to_compressed_imgmsg(
            annotated, header=msg.header, jpeg_quality=self.jpeg_quality
        )
        if out is None:
            self.get_logger().warn("Failed to JPEG-encode final annotated image")
            return

        self.output_pub.publish(out)
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"Frame {self.frame_count}: closest of {len(result.detections)} person(s), "
                f"{len(depths)} depth label(s), {result.inference_time_ms:.1f} ms"
            )

    def _right_callback(self, msg: CompressedImage) -> None:
        self.right_msgs.append(msg)

    def _left_info_callback(self, msg: CameraInfo) -> None:
        self.left_info = msg
        self.maps = None
        self.maps_scale = None
        self.get_logger().info(
            f"Left CameraInfo received: fx={self._fx():.2f}, cx={self._cx():.2f}",
            once=True,
        )

    def _right_info_callback(self, msg: CameraInfo) -> None:
        self.right_info = msg
        self.maps = None
        self.maps_scale = None
        self.get_logger().info(
            f"Right CameraInfo received: fx={msg.p[0]:.2f}, Tx={msg.p[3]:.4f}, "
            f"baseline={self._baseline():.4f}m",
            once=True,
        )

    def _synced_right_image(self, left_msg: CompressedImage):
        if not self.right_msgs:
            return None
        left_stamp = self._stamp_sec(left_msg)
        right_msg = min(
            self.right_msgs,
            key=lambda candidate: abs(left_stamp - self._stamp_sec(candidate)),
        )
        dt = abs(left_stamp - self._stamp_sec(right_msg))
        if dt > self.sync_slop_sec:
            self.get_logger().warn(
                f"No synced right image for left frame; dt={dt:.3f}s",
                throttle_duration_sec=3.0,
            )
            return None
        try:
            return compressed_imgmsg_to_bgr8(right_msg)
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to decode synced right image: {exc}",
                throttle_duration_sec=3.0,
            )
            return None

    def _processing_image(self, image):
        if self.processing_width <= 0 or image.shape[1] <= self.processing_width:
            return image, 1.0
        scale = self.processing_width / float(image.shape[1])
        height = max(1, int(round(image.shape[0] * scale)))
        resized = cv2.resize(image, (self.processing_width, height))
        return resized, scale

    def _resize_like(self, image, reference):
        if image.shape[:2] == reference.shape[:2]:
            return image
        return cv2.resize(image, (reference.shape[1], reference.shape[0]))

    def _output_image(self, image):
        if self.output_width <= 0 or image.shape[1] <= self.output_width:
            return image
        scale = self.output_width / float(image.shape[1])
        height = max(1, int(round(image.shape[0] * scale)))
        return cv2.resize(image, (self.output_width, height))

    def _scale_detections(self, detections, scale: float):
        if scale == 1.0:
            return detections
        scaled = []
        for detection in detections:
            scaled.append(
                Detection(
                    bbox_x=int(round(detection.bbox_x * scale)),
                    bbox_y=int(round(detection.bbox_y * scale)),
                    bbox_w=int(round(detection.bbox_w * scale)),
                    bbox_h=int(round(detection.bbox_h * scale)),
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                )
            )
        return scaled

    def _select_closest_detection(self, detections, depths):
        if not detections:
            return [], {}
        if depths:
            selected_idx = min(depths, key=lambda idx: depths[idx].z_m)
        else:
            selected_idx = max(
                range(len(detections)),
                key=lambda idx: detections[idx].bbox_w * detections[idx].bbox_h,
            )

        selected_depths = {}
        if selected_idx in depths:
            selected_depths[0] = depths[selected_idx]
        return [detections[selected_idx]], selected_depths

    def _estimate_depths(self, left, right, detections, processing_scale: float):
        left_rect, right_rect = self._rectify_if_possible(left, right, processing_scale)
        if left_rect.shape[:2] != right_rect.shape[:2]:
            self.get_logger().warn(
                "Left/right stereo images have different sizes; skipping depth",
                throttle_duration_sec=3.0,
            )
            return {}

        fx = self._fx() * processing_scale
        baseline = self._baseline()
        if fx <= 0.0:
            self.get_logger().warn(
                "No usable left CameraInfo yet; skipping stereo depth.",
                throttle_duration_sec=3.0,
            )
            return {}
        if baseline <= 0.0:
            return {}

        gray_left = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

        depths = {}
        for idx, detection in enumerate(detections):
            depth_detection = self._scale_detection(detection, processing_scale)
            depth = self._depth_from_detection(
                disparity, depth_detection, fx, baseline, processing_scale
            )
            if depth is not None:
                depths[idx] = depth
        return depths

    def _depth_from_detection(
        self,
        disparity,
        detection: Detection,
        fx: float,
        baseline: float,
        processing_scale: float,
    ) -> Optional[DepthEstimate]:
        height, width = disparity.shape[:2]
        x1, y1, x2, y2 = self._bbox_xyxy(detection, width, height)
        x1, y1, x2, y2 = self._shrink_roi(x1, y1, x2, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        roi = disparity[y1:y2, x1:x2]
        valid = roi[np.isfinite(roi) & (roi > 0.5)]
        if valid.size < self.min_valid_disparity_pixels:
            return None

        disp = float(np.median(valid))
        z_m = fx * baseline / disp
        if z_m < self.min_depth_m or z_m > self.max_depth_m:
            return None

        bbox_cx = detection.bbox_x + detection.bbox_w / 2.0
        bbox_cy = detection.bbox_y + detection.bbox_h / 2.0
        x_m = (bbox_cx - self._cx() * processing_scale) * z_m / fx
        y_m = (bbox_cy - self._cy() * processing_scale) * z_m / fx
        return DepthEstimate(x_m=x_m, y_m=y_m, z_m=z_m, valid_pixels=int(valid.size))

    def _scale_detection(self, detection: Detection, scale: float) -> Detection:
        if scale == 1.0:
            return detection
        return Detection(
            bbox_x=int(round(detection.bbox_x * scale)),
            bbox_y=int(round(detection.bbox_y * scale)),
            bbox_w=int(round(detection.bbox_w * scale)),
            bbox_h=int(round(detection.bbox_h * scale)),
            confidence=detection.confidence,
            class_id=detection.class_id,
        )

    def _rectify_if_possible(self, left, right, camera_scale: float):
        if self.left_info is None or self.right_info is None:
            return left, right
        if self.maps is None or self.maps_scale != camera_scale:
            self.maps = self._build_rectification_maps(
                left.shape[1], left.shape[0], camera_scale
            )
            self.maps_scale = camera_scale
        if self.maps is None:
            return left, right
        lmap1, lmap2, rmap1, rmap2 = self.maps
        return (
            cv2.remap(left, lmap1, lmap2, cv2.INTER_LINEAR),
            cv2.remap(right, rmap1, rmap2, cv2.INTER_LINEAR),
        )

    def _build_rectification_maps(self, width: int, height: int, camera_scale: float):
        try:
            left_k = self._scaled_k(self.left_info, camera_scale)
            right_k = self._scaled_k(self.right_info, camera_scale)
            left_d = np.array(self.left_info.d, dtype=np.float64)
            right_d = np.array(self.right_info.d, dtype=np.float64)
            left_r = np.array(self.left_info.r, dtype=np.float64).reshape(3, 3)
            right_r = np.array(self.right_info.r, dtype=np.float64).reshape(3, 3)
            left_p = self._scaled_p(self.left_info, camera_scale)
            right_p = self._scaled_p(self.right_info, camera_scale)
            size = (width, height)
            if self.left_info.distortion_model == "fisheye":
                lmap1, lmap2 = cv2.fisheye.initUndistortRectifyMap(
                    left_k, left_d, left_r, left_p[:, :3], size, cv2.CV_32FC1
                )
            else:
                lmap1, lmap2 = cv2.initUndistortRectifyMap(
                    left_k, left_d, left_r, left_p[:, :3], size, cv2.CV_32FC1
                )
            if self.right_info.distortion_model == "fisheye":
                rmap1, rmap2 = cv2.fisheye.initUndistortRectifyMap(
                    right_k, right_d, right_r, right_p[:, :3], size, cv2.CV_32FC1
                )
            else:
                rmap1, rmap2 = cv2.initUndistortRectifyMap(
                    right_k, right_d, right_r, right_p[:, :3], size, cv2.CV_32FC1
                )
            return lmap1, lmap2, rmap1, rmap2
        except Exception as exc:
            self.get_logger().warn(
                f"Could not build rectification maps: {exc}",
                throttle_duration_sec=3.0,
            )
            return None

    @staticmethod
    def _scaled_k(info: CameraInfo, scale: float):
        k = np.array(info.k, dtype=np.float64).reshape(3, 3)
        k[0, 0] *= scale
        k[0, 2] *= scale
        k[1, 1] *= scale
        k[1, 2] *= scale
        return k

    @staticmethod
    def _scaled_p(info: CameraInfo, scale: float):
        p = np.array(info.p, dtype=np.float64).reshape(3, 4)
        p[0, :] *= scale
        p[1, :] *= scale
        return p

    def _annotate(self, image, detections, depths, inference_time_ms: float):
        annotated = image.copy()
        for idx, detection in enumerate(detections):
            x1 = detection.bbox_x
            y1 = detection.bbox_y
            x2 = detection.bbox_x + detection.bbox_w
            y2 = detection.bbox_y + detection.bbox_h
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"Person {detection.confidence:.2f}"
            depth = depths.get(idx)
            if depth is not None:
                label = f"{label} {depth.z_m:.2f}m"
            self._draw_label(annotated, label, x1, y1, color)

        stats = (
            f"{inference_time_ms:.1f} ms | {len(detections)} person(s) | "
            f"{len(depths)} depth"
        )
        cv2.putText(
            annotated,
            stats,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        return annotated

    def _draw_label(self, image, label: str, x: int, y: int, color) -> None:
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(label_size[1] + 6, y)
        cv2.rectangle(
            image,
            (x, label_y - label_size[1] - 6),
            (x + label_size[0] + 6, label_y),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x + 3, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
        )

    def _bbox_xyxy(self, detection: Detection, width: int, height: int):
        x1 = max(0, int(detection.bbox_x))
        y1 = max(0, int(detection.bbox_y))
        x2 = min(width, int(detection.bbox_x + detection.bbox_w))
        y2 = min(height, int(detection.bbox_y + detection.bbox_h))
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
        return DEFAULT_FOCAL_PX

    def _cx(self) -> float:
        if self.left_info is not None and self.left_info.p[2] > 0:
            return float(self.left_info.p[2])
        if self.left_info is not None and self.left_info.k[2] > 0:
            return float(self.left_info.k[2])
        return DEFAULT_CX

    def _cy(self) -> float:
        if self.left_info is not None and self.left_info.p[6] > 0:
            return float(self.left_info.p[6])
        if self.left_info is not None and self.left_info.k[5] > 0:
            return float(self.left_info.k[5])
        return DEFAULT_CY

    def _baseline(self) -> float:
        if self.baseline_m > 0.0:
            return self.baseline_m
        if self.right_info is not None and self.right_info.p[0] != 0.0:
            baseline = abs(float(self.right_info.p[3]) / float(self.right_info.p[0]))
            if baseline > 0.0:
                return baseline
        return DEFAULT_BASELINE_M

    def _create_stereo_matcher(self):
        num_disparities = int(self.get_parameter("stereo_num_disparities").value)
        num_disparities = max(16, int(np.ceil(num_disparities / 16.0)) * 16)
        block_size = int(self.get_parameter("stereo_block_size").value)
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(5, block_size)
        return cv2.StereoSGBM_create(
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

    def _publish_inference_time(self, inference_time_ms: float) -> None:
        msg = Float32()
        msg.data = float(inference_time_ms)
        self.inference_pub.publish(msg)

    def _jpeg_quality(self) -> int:
        quality = int(self.get_parameter("jpeg_quality").value)
        return min(max(quality, 1), 100)

    @staticmethod
    def _stamp_sec(msg) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = StereoFinalAnnotatorNode()
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
