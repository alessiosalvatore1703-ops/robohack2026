"""Simple stereo camera image relay for the Agibot X2.

This node intentionally stays separate from the existing /yolo pipeline. It
subscribes to one selected stereo camera stream and republishes it on a small
/stereo_person namespace so the image path can be verified before detection.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

from .image_conversion import (
    bgr8_to_image_msg,
    compressed_imgmsg_to_bgr8,
    image_msg_to_bgr8,
)


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
)

CAMERA_TOPICS = {
    "stereo_head_front_left": {
        "raw": "/aima/hal/sensor/stereo_head_front_left/rgb_image",
        "compressed": "/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed",
    },
    "stereo_head_front_right": {
        "raw": "/aima/hal/sensor/stereo_head_front_right/rgb_image",
        "compressed": "/aima/hal/sensor/stereo_head_front_right/rgb_image/compressed",
    },
}


def qos_from_publishers(infos) -> QoSProfile:
    if not infos:
        return SENSOR_QOS
    pub_qos = infos[0].qos_profile
    return QoSProfile(
        reliability=pub_qos.reliability,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        durability=pub_qos.durability,
    )


class StereoImageViewNode(Node):
    """Relay one stereo image stream to /stereo_person/input_image."""

    def __init__(self):
        super().__init__("stereo_image_view")

        self.declare_parameter("camera", "stereo_head_front_left")
        self.declare_parameter("topic_type", "rgb_image")
        self.declare_parameter("image_topic", "")
        self.declare_parameter("output_topic", "/stereo_person/input_image")
        self.declare_parameter("view_topic", "/stereo_person/view_image")
        self.declare_parameter("timeout_sec", 2.0)

        self.camera = self.get_parameter("camera").value
        self.topic_type = self.get_parameter("topic_type").value
        self.image_topic_override = self.get_parameter("image_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.view_topic = self.get_parameter("view_topic").value
        self.timeout_sec = float(self.get_parameter("timeout_sec").value)

        if self.camera not in CAMERA_TOPICS:
            self.get_logger().warn(
                f"Unknown camera '{self.camera}', using stereo_head_front_left"
            )
            self.camera = "stereo_head_front_left"

        if self.topic_type not in ("rgb_image", "rgb_image_compressed"):
            self.get_logger().warn(
                f"Unknown topic_type '{self.topic_type}', using rgb_image"
            )
            self.topic_type = "rgb_image"

        self.image_topic = self._resolve_image_topic()
        self.frame_count = 0
        self.last_image_time = 0.0

        infos = self.get_publishers_info_by_topic(self.image_topic)
        image_qos = qos_from_publishers(infos)

        self.image_pub = self.create_publisher(Image, self.output_topic, SENSOR_QOS)
        self.view_pub = self.create_publisher(Image, self.view_topic, SENSOR_QOS)

        if self.topic_type == "rgb_image_compressed":
            self.image_sub = self.create_subscription(
                CompressedImage, self.image_topic, self._compressed_callback, image_qos
            )
        else:
            self.image_sub = self.create_subscription(
                Image, self.image_topic, self._image_callback, image_qos
            )

        self.create_timer(1.0, self._check_timeout)
        self.get_logger().info(
            f"Relaying {self.image_topic} -> {self.output_topic} and {self.view_topic}"
        )

    def _resolve_image_topic(self) -> str:
        if self.image_topic_override:
            return self.image_topic_override
        key = "compressed" if self.topic_type == "rgb_image_compressed" else "raw"
        return CAMERA_TOPICS[self.camera][key]

    def _image_callback(self, msg: Image) -> None:
        try:
            cv_image = image_msg_to_bgr8(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to convert image to bgr8: {exc}")
            return

        out = bgr8_to_image_msg(cv_image, msg.header)
        self._mark_frame(out.width, out.height, out.encoding, msg.encoding)
        self.image_pub.publish(out)
        self.view_pub.publish(out)

    def _compressed_callback(self, msg: CompressedImage) -> None:
        try:
            cv_image = compressed_imgmsg_to_bgr8(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode compressed image: {exc}")
            return

        out = bgr8_to_image_msg(cv_image, msg.header)
        self._mark_frame(out.width, out.height, out.encoding, msg.format)
        self.image_pub.publish(out)
        self.view_pub.publish(out)

    def _mark_frame(
        self, width: int, height: int, encoding: str, source_encoding: str
    ) -> None:
        self.frame_count += 1
        self.last_image_time = time.time()
        if self.frame_count == 1:
            self.get_logger().info(
                f"First image received ({width}x{height}, "
                f"source={source_encoding}, published={encoding})"
            )

    def _check_timeout(self) -> None:
        if self.last_image_time == 0.0:
            return
        elapsed = time.time() - self.last_image_time
        if elapsed > self.timeout_sec:
            self.get_logger().warn(
                f"No images from {self.image_topic} for {elapsed:.1f}s"
            )


def main(args=None):
    rclpy.init(args=args)
    node = StereoImageViewNode()
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
