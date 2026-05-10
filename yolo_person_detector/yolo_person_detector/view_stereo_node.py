"""Realtime OpenCV viewer for the final stereo annotation stream."""

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from .image_conversion import compressed_imgmsg_to_bgr8


class FinalStereoViewer(Node):
    """Display the final compressed annotated image stream."""

    def __init__(self):
        super().__init__("final_stereo_viewer")

        self.declare_parameter(
            "image_topic", "/stereo_person/final_annotated_image/compressed"
        )
        self.declare_parameter("scale", 1.0)

        self.image_topic = self.get_parameter("image_topic").value
        self.scale = max(0.05, float(self.get_parameter("scale").value))

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            CompressedImage, self.image_topic, self._image_callback, qos
        )

        self.get_logger().info("Final stereo viewer started")
        self.get_logger().info(f"Image: {self.image_topic}")
        cv2.namedWindow("Final stereo annotation", cv2.WINDOW_NORMAL)

    def _image_callback(self, msg: CompressedImage):
        try:
            image = compressed_imgmsg_to_bgr8(msg)
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to decode final annotated image: {exc}",
                throttle_duration_sec=3.0,
            )
            return

        if self.scale != 1.0:
            image = cv2.resize(image, None, fx=self.scale, fy=self.scale)

        cv2.imshow("Final stereo annotation", image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = FinalStereoViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
