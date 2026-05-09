"""
Visualization ROS2 Node.

Subscribes to detection results and input images, overlays bounding boxes,
and publishes annotated images for monitoring via WebRTC or rqt_image_view.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

import numpy as np
import cv2

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)


class VisualizationNode(Node):
    """Overlays detection results on camera images for visualization."""

    def __init__(self):
        super().__init__('yolo_visualization')

        # Parameters
        self.declare_parameter('bbox_color_r', 0)
        self.declare_parameter('bbox_color_g', 255)
        self.declare_parameter('bbox_color_b', 0)
        self.declare_parameter('bbox_thickness', 2)
        self.declare_parameter('show_confidence', True)
        self.declare_parameter('font_scale', 0.6)

        self.bbox_color = (
            self.get_parameter('bbox_color_b').value,
            self.get_parameter('bbox_color_g').value,
            self.get_parameter('bbox_color_r').value,
        )
        self.bbox_thickness = self.get_parameter('bbox_thickness').value
        self.show_confidence = self.get_parameter('show_confidence').value
        self.font_scale = self.get_parameter('font_scale').value

        if CvBridge is None:
            self.get_logger().error('cv_bridge not available')
            raise RuntimeError('cv_bridge not available')
        self.bridge = CvBridge()

        # State
        self._latest_image = None
        self._latest_detections = None

        # Subscribers
        self.create_subscription(
            Image, '/yolo/input_image', self._image_callback, SENSOR_QOS
        )
        self.create_subscription(
            Detection2DArray, '/yolo/detections', self._detection_callback, RELIABLE_QOS
        )

        # Publisher
        self.vis_pub = self.create_publisher(
            Image, '/yolo/visualization_image', SENSOR_QOS
        )

        # Timer to publish visualization at ~10 Hz
        self.create_timer(0.1, self._publish_visualization)

        self.get_logger().info('Visualization node started')

    def _image_callback(self, msg: Image) -> None:
        """Store latest image."""
        try:
            self._latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'Image conversion failed: {e}')

    def _detection_callback(self, msg: Detection2DArray) -> None:
        """Store latest detections."""
        self._latest_detections = msg

    def _publish_visualization(self) -> None:
        """Render and publish annotated image."""
        if self._latest_image is None:
            return

        vis = self._latest_image.copy()

        if self._latest_detections is not None:
            for det in self._latest_detections.detections:
                bbox = det.bbox
                cx = bbox.center.x
                cy = bbox.center.y
                w = bbox.size_x
                h = bbox.size_y

                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                x2 = int(cx + w / 2)
                y2 = int(cy + h / 2)

                # Draw box
                cv2.rectangle(vis, (x1, y1), (x2, y2), self.bbox_color, self.bbox_thickness)

                # Draw label
                if self.show_confidence and det.results:
                    conf = det.results[0].hypothesis.score
                    label = f'Person {conf:.2f}'
                    cv2.putText(
                        vis, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                        self.bbox_color, 1,
                    )

        # Publish
        img_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        self.vis_pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
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
