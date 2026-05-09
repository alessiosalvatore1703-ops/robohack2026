"""
Camera Selector ROS2 Node.

Subscribes to one of the Agibot X2 cameras and republishes the images on
a unified topic (`/yolo/input_image`) for the rest of the pipeline.

Robust-to-reality behavior:
  - Discovers publishers on the configured topic and adopts their QoS
    profile (so we never miss messages because of a BEST_EFFORT vs
    RELIABLE mismatch on the X2 HAL).
  - If no publisher exists on the configured topic, lists image-like
    topics currently advertised in the graph so the operator can see
    what's actually available.
  - Warns clearly when the input stream goes silent.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from sensor_msgs.msg import Image, CameraInfo


# Default QoS used when no publisher is found yet. Sensor-style.
DEFAULT_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
)

# Available cameras on Agibot X2 (defaults; overridable via parameters).
CAMERA_TOPICS = {
    'rgbd_head_front': '/aima/hal/sensor/rgbd_head_front/rgb_image',
    'rgb_head_rear': '/aima/hal/sensor/rgb_head_rear/rgb_image',
    'stereo_head_front_left': '/aima/hal/sensor/stereo_head_front_left/rgb_image',
    'stereo_head_front_right': '/aima/hal/sensor/stereo_head_front_right/rgb_image',
}

CAMERA_INFO_TOPICS = {
    'rgbd_head_front': '/aima/hal/sensor/rgbd_head_front/rgb_camera_info',
    'rgb_head_rear': '/aima/hal/sensor/rgb_head_rear/camera_info',
    'stereo_head_front_left': '/aima/hal/sensor/stereo_head_front_left/camera_info',
    'stereo_head_front_right': '/aima/hal/sensor/stereo_head_front_right/camera_info',
}


def _qos_from_publisher_infos(infos) -> QoSProfile:
    """Pick a QoS compatible with the first publisher we find."""
    if not infos:
        return DEFAULT_SENSOR_QOS

    pub_qos = infos[0].qos_profile
    return QoSProfile(
        reliability=pub_qos.reliability,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        durability=pub_qos.durability,
    )


class CameraSelectorNode(Node):
    """Selects and republishes camera images from one of the robot's cameras."""

    def __init__(self):
        super().__init__('camera_selector')

        # Parameters
        self.declare_parameter('active_camera', 'rgbd_head_front')
        self.declare_parameter('timeout_sec', 2.0)
        # Escape hatch: let the user pin an exact topic name if their robot
        # uses a non-default camera path.
        self.declare_parameter('image_topic_override', '')
        self.declare_parameter('info_topic_override', '')
        self.declare_parameter('discovery_retry_sec', 2.0)

        self.active_camera = self.get_parameter('active_camera').value
        self.timeout_sec = self.get_parameter('timeout_sec').value
        self._image_topic_override = self.get_parameter('image_topic_override').value
        self._info_topic_override = self.get_parameter('info_topic_override').value
        self._discovery_retry_sec = self.get_parameter('discovery_retry_sec').value

        if self.active_camera not in CAMERA_TOPICS and not self._image_topic_override:
            self.get_logger().warn(
                f"Unknown camera '{self.active_camera}', "
                f"available: {list(CAMERA_TOPICS.keys())}. "
                f"Defaulting to 'rgbd_head_front'."
            )
            self.active_camera = 'rgbd_head_front'

        # Publishers
        self.image_pub = self.create_publisher(
            Image, '/yolo/input_image', DEFAULT_SENSOR_QOS,
        )
        self.info_pub = self.create_publisher(
            CameraInfo, '/yolo/camera_info', DEFAULT_SENSOR_QOS,
        )

        self._cam_subs = {}
        self._image_topic = ''
        self._info_topic = ''
        self._last_image_time = 0.0
        self._frame_count = 0

        # Resolve topics and subscribe (with retry until a publisher appears).
        self._resolve_and_subscribe()

        # Periodic checks: topic liveness + publisher discovery if we still
        # haven't latched onto anyone.
        self._liveness_timer = self.create_timer(1.0, self._check_timeout)
        self._discovery_timer = self.create_timer(
            self._discovery_retry_sec, self._retry_discovery,
        )

    # ------------------------------------------------------------------ #
    # Topic resolution / subscription                                     #
    # ------------------------------------------------------------------ #

    def _resolve_and_subscribe(self) -> None:
        """Determine image/info topic names and create subscriptions."""
        if self._image_topic_override:
            self._image_topic = self._image_topic_override
        else:
            self._image_topic = CAMERA_TOPICS[self.active_camera]

        if self._info_topic_override:
            self._info_topic = self._info_topic_override
        else:
            self._info_topic = CAMERA_INFO_TOPICS.get(self.active_camera, '')

        self._subscribe_to_topics()

    def _subscribe_to_topics(self) -> None:
        """(Re)subscribe using QoS matched to the current publisher(s)."""
        # Drop any existing subscriptions.
        for sub in self._cam_subs.values():
            self.destroy_subscription(sub)
        self._cam_subs.clear()

        img_infos = self.get_publishers_info_by_topic(self._image_topic)
        img_qos = _qos_from_publisher_infos(img_infos)

        self._cam_subs['image'] = self.create_subscription(
            Image, self._image_topic, self._image_callback, img_qos,
        )
        self.get_logger().info(
            f"Subscribed to image topic '{self._image_topic}' "
            f"(publishers={len(img_infos)}, "
            f"reliability={img_qos.reliability.name}, "
            f"durability={img_qos.durability.name})"
        )

        if not img_infos:
            self._log_available_image_topics()

        if self._info_topic:
            info_infos = self.get_publishers_info_by_topic(self._info_topic)
            info_qos = _qos_from_publisher_infos(info_infos)
            self._cam_subs['info'] = self.create_subscription(
                CameraInfo, self._info_topic, self._info_callback, info_qos,
            )
            self.get_logger().info(
                f"Subscribed to camera_info topic '{self._info_topic}' "
                f"(publishers={len(info_infos)})"
            )

    def _log_available_image_topics(self) -> None:
        """Help the operator by listing plausible image topics in the graph."""
        try:
            all_topics = self.get_topic_names_and_types()
        except Exception as exc:
            self.get_logger().warn(f'Could not enumerate topics: {exc}')
            return

        image_like = [
            name for name, types in all_topics
            if 'sensor_msgs/msg/Image' in types
        ]
        if not image_like:
            self.get_logger().warn(
                'No sensor_msgs/Image publishers found anywhere in the graph. '
                'Is the camera driver running? '
                'Check with `ros2 topic list` and confirm ROS_DOMAIN_ID / '
                'RMW_IMPLEMENTATION match the robot.'
            )
            return

        self.get_logger().warn(
            f"Topic '{self._image_topic}' has no publisher. "
            f'sensor_msgs/Image topics currently visible: {image_like}. '
            f'Set image_topic_override:=<topic> to use one of them.'
        )

    def _retry_discovery(self) -> None:
        """If we're still not receiving, re-pick QoS from current publishers."""
        if self._frame_count > 0:
            return

        infos = self.get_publishers_info_by_topic(self._image_topic)
        if not infos:
            # Nobody's publishing yet; re-log every few seconds.
            self._log_available_image_topics()
            return

        # Publisher appeared after we subscribed; re-subscribe so QoS matches.
        self.get_logger().info(
            f"Publisher now visible on '{self._image_topic}' — re-subscribing "
            f'with matching QoS.'
        )
        self._subscribe_to_topics()

    # ------------------------------------------------------------------ #
    # Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def _image_callback(self, msg: Image) -> None:
        self._last_image_time = time.time()
        self._frame_count += 1
        self.image_pub.publish(msg)
        if self._frame_count == 1:
            self.get_logger().info(
                f'First frame received ({msg.width}x{msg.height}, '
                f'encoding={msg.encoding}) — relaying to /yolo/input_image'
            )

    def _info_callback(self, msg: CameraInfo) -> None:
        self.info_pub.publish(msg)

    def _check_timeout(self) -> None:
        if self._last_image_time == 0.0:
            # Never received one. _retry_discovery handles reporting.
            return
        elapsed = time.time() - self._last_image_time
        if elapsed > self.timeout_sec:
            self.get_logger().warn(
                f'No images from {self.active_camera} '
                f"('{self._image_topic}') for {elapsed:.1f}s"
            )

    # ------------------------------------------------------------------ #
    # Runtime control                                                     #
    # ------------------------------------------------------------------ #

    def switch_camera(self, camera_name: str) -> bool:
        if camera_name not in CAMERA_TOPICS:
            self.get_logger().error(
                f"Cannot switch to unknown camera '{camera_name}'. "
                f'Available: {list(CAMERA_TOPICS.keys())}'
            )
            return False
        self.active_camera = camera_name
        self._image_topic_override = ''
        self._info_topic_override = ''
        self._frame_count = 0
        self._last_image_time = 0.0
        self._resolve_and_subscribe()
        return True


def main(args=None):
    rclpy.init(args=args)
    node = CameraSelectorNode()
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
