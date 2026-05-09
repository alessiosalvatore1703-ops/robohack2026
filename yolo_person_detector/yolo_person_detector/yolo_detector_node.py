"""
YOLO Detector ROS2 Node.

Subscribes to camera images, runs YOLO person detection,
and publishes detection results and annotated images.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    ObjectHypothesisWithPose,
    BoundingBox2D,
    Pose2D)

import numpy as np
import cv2

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None

from .yolo_wrapper import YOLOWrapper, InferenceResult


# QoS for subscribing to camera images (best effort, like camera drivers)
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

# QoS for publishing detections (reliable)
RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)


class YOLODetectorNode(Node):
    """ROS2 node that runs YOLO person detection on incoming images."""

    def __init__(self):
        super().__init__('yolo_detector')

        # Declare parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.45)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('input_size', 640)
        self.declare_parameter('input_topic', '/aima/hal/sensor/rgb_head_front_center/rgb_image')
        self.declare_parameter('publish_annotated', True)

        # Get parameters
        model_path = self.get_parameter('model_path').value
        confidence = self.get_parameter('confidence_threshold').value
        nms_thresh = self.get_parameter('nms_threshold').value
        device = self.get_parameter('device').value
        input_size = self.get_parameter('input_size').value
        input_topic = self.get_parameter('input_topic').value
        self.publish_annotated = self.get_parameter('publish_annotated').value

        # Initialize CV bridge
        if CvBridge is None:
            self.get_logger().error('cv_bridge not available. Install with: sudo apt install ros-humble-cv-bridge')
            raise RuntimeError('cv_bridge not available')
        self.bridge = CvBridge()

        # Initialize YOLO model
        self.get_logger().info(f'Loading YOLO model: {model_path} on {device}')
        try:
            self.yolo = YOLOWrapper(
                model_path=model_path,
                confidence_threshold=confidence,
                nms_threshold=nms_thresh,
                device=device,
                input_size=input_size,
            )
            self.get_logger().info('YOLO model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {e}')
            raise

        # Subscribers. Pick QoS that matches whichever publisher is live,
        # so we don't silently miss frames on a reliability/durability
        # mismatch (the X2 HAL uses a variety of profiles).
        pub_infos = self.get_publishers_info_by_topic(input_topic)
        if pub_infos:
            pub_qos = pub_infos[0].qos_profile
            image_qos = QoSProfile(
                reliability=pub_qos.reliability,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
                durability=pub_qos.durability,
            )
            self.get_logger().info(
                f"Adopting publisher QoS on '{input_topic}': "
                f"reliability={pub_qos.reliability.name}, "
                f"durability={pub_qos.durability.name}"
            )
        else:
            image_qos = SENSOR_QOS
            self.get_logger().warn(
                f"No publisher on '{input_topic}' yet — falling back to "
                f'BEST_EFFORT sensor QoS. If messages never arrive, check '
                f'`ros2 topic info -v {input_topic}`.'
            )

        self.image_sub = self.create_subscription(
            Image,
            input_topic,
            self._image_callback,
            image_qos,
        )
        self.get_logger().info(f'Subscribed to: {input_topic}')

        # Publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/yolo/detections',
            RELIABLE_QOS,
        )
        self.inference_time_pub = self.create_publisher(
            Float32,
            '/yolo/inference_time',
            RELIABLE_QOS,
        )
        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(
                Image,
                '/yolo/detection_image',
                SENSOR_QOS,
            )

        # Stats
        self._frame_count = 0
        self._error_count = 0

    def _image_callback(self, msg: Image) -> None:
        """Process incoming image through YOLO detector."""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'Failed to convert image: {e}')
            return

        # Run detection
        try:
            result = self.yolo.detect(cv_image)
            self._error_count = 0
        except Exception as e:
            self._error_count += 1
            self.get_logger().error(f'Inference failed ({self._error_count}): {e}')
            if self._error_count > 10:
                self.get_logger().fatal('Too many consecutive inference failures')
            return

        self._frame_count += 1

        # Publish detection results
        self._publish_detections(result, msg.header)

        # Publish inference time
        time_msg = Float32()
        time_msg.data = result.inference_time_ms
        self.inference_time_pub.publish(time_msg)

        # Publish annotated image
        if self.publish_annotated:
            self._publish_annotated_image(cv_image, result, msg.header)

        # Log stats periodically
        if self._frame_count % 30 == 0:
            self.get_logger().info(
                f'Frame {self._frame_count}: '
                f'{len(result.detections)} persons detected, '
                f'{result.inference_time_ms:.1f}ms inference'
            )

    def _publish_detections(self, result: InferenceResult, header) -> None:
        """Convert detections to Detection2DArray and publish."""
        det_array = Detection2DArray()
        det_array.header = header

        for det in result.detections:
            d2d = Detection2D()
            d2d.header = header

            # Bounding box (center_x, center_y, size_x, size_y)
            bbox = BoundingBox2D()
            bbox.center = Pose2D()
            bbox.center.position.x = float(det.bbox_x + det.bbox_w / 2.0)
            bbox.center.position.y = float(det.bbox_y + det.bbox_h / 2.0)
            bbox.center.theta = 0.0  # No rotation for axis-aligned bounding boxes
            bbox.size_x = float(det.bbox_w)
            bbox.size_y = float(det.bbox_h)
            d2d.bbox = bbox

            # Hypothesis
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(det.class_id)
            hyp.hypothesis.score = det.confidence
            d2d.results.append(hyp)

            det_array.detections.append(d2d)

        self.detection_pub.publish(det_array)

    def _publish_annotated_image(self, image: np.ndarray, result: InferenceResult, header) -> None:
        """Draw bounding boxes on image and publish."""
        annotated = image.copy()

        for det in result.detections:
            # Draw bounding box
            color = (0, 255, 0)  # Green
            cv2.rectangle(
                annotated,
                (det.bbox_x, det.bbox_y),
                (det.bbox_x + det.bbox_w, det.bbox_y + det.bbox_h),
                color,
                2,
            )

            # Draw label
            label = f'Person {det.confidence:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (det.bbox_x, det.bbox_y - label_size[1] - 4),
                (det.bbox_x + label_size[0], det.bbox_y),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (det.bbox_x, det.bbox_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        # Add inference time overlay
        cv2.putText(
            annotated,
            f'{result.inference_time_ms:.1f}ms | {len(result.detections)} persons',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # Publish
        img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        img_msg.header = header
        self.annotated_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
