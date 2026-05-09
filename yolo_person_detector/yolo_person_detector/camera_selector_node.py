"""
Camera Selector ROS2 Node.

Subscribes to one of the Agibot X2 cameras and republishes the images on
a unified topic (`/yolo/input_image`) for the rest of the pipeline.

Reference: py_examples/echo_camera_rgbd.py provided by Agibot SDK.
Topics + QoS here mirror that example exactly, and we additionally
support the compressed RGB stream which is typically cheaper on the
network.

Per Agibot X2 HAL (rgb_head_front_center, plain RGB):
  - rgb_image                   sensor_msgs/Image            raw RGB
  - rgb_image/compressed        sensor_msgs/CompressedImage  JPEG
  - camera_info                 sensor_msgs/CameraInfo

All use SensorDataQoS (BEST_EFFORT + KEEP_LAST + VOLATILE, depth 5).

The active camera is fixed at launch time via the `active_camera` and
`topic_type` parameters. There is no runtime switching.
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
from sensor_msgs.msg import Image, CameraInfo, CompressedImage

try:
    from cv_bridge import CvBridge
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False


# SensorDataQoS equivalent, matching the Agibot echo_camera_rgbd example.
SENSOR_DATA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
    durability=DurabilityPolicy.VOLATILE,
)


# Per-camera topic layout. All X2 head cameras follow the same RGB layout:
# /rgb_image (raw), /rgb_image/compressed (JPEG), /camera_info.
CAMERA_LAYOUTS = {
    'rgb_head_front_center': {
        'rgb_raw':        '/aima/hal/sensor/rgb_head_front_center/rgb_image',
        'rgb_compressed': '/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed',
        'rgb_info':       '/aima/hal/sensor/rgb_head_front_center/camera_info',
    },
    'rgb_head_rear': {
        'rgb_raw':        '/aima/hal/sensor/rgb_head_rear/rgb_image',
        'rgb_compressed': '/aima/hal/sensor/rgb_head_rear/rgb_image/compressed',
        'rgb_info':       '/aima/hal/sensor/rgb_head_rear/camera_info',
    },
    'stereo_head_front_left': {
        'rgb_raw':        '/aima/hal/sensor/stereo_head_front_left/rgb_image',
        'rgb_compressed': '/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed',
        'rgb_info':       '/aima/hal/sensor/stereo_head_front_left/camera_info',
    },
    'stereo_head_front_right': {
        'rgb_raw':        '/aima/hal/sensor/stereo_head_front_right/rgb_image',
        'rgb_compressed': '/aima/hal/sensor/stereo_head_front_right/rgb_image/compressed',
        'rgb_info':       '/aima/hal/sensor/stereo_head_front_right/camera_info',
    },
}


def _qos_from_publisher_infos(infos) -> QoSProfile:
    """Return a QoS matching the first publisher, or SensorDataQoS default."""
    if not infos:
        return SENSOR_DATA_QOS
    pub_qos = infos[0].qos_profile
    return QoSProfile(
        reliability=pub_qos.reliability,
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        durability=pub_qos.durability,
    )


class CameraSelectorNode(Node):
    """Subscribes to one Agibot X2 camera; republishes on /yolo/input_image."""

    def __init__(self):
        super().__init__('camera_selector')

        # --- Parameters --------------------------------------------------
        self.declare_parameter('active_camera', 'stereo_head_front_left')
        # 'rgb_image' (raw) or 'rgb_image_compressed'. Mirrors the
        # topic_type names used by echo_camera_rgbd for familiarity.
        self.declare_parameter('topic_type', 'rgb_image')
        self.declare_parameter('timeout_sec', 2.0)
        self.declare_parameter('discovery_retry_sec', 2.0)
        # Escape hatches for non-standard deployments.
        self.declare_parameter('image_topic_override', '')
        self.declare_parameter('info_topic_override', '')

        self.active_camera = self.get_parameter('active_camera').value
        self.topic_type = self.get_parameter('topic_type').value
        self.timeout_sec = self.get_parameter('timeout_sec').value
        self._discovery_retry_sec = self.get_parameter('discovery_retry_sec').value
        self._image_topic_override = self.get_parameter('image_topic_override').value
        self._info_topic_override = self.get_parameter('info_topic_override').value

        if self.active_camera not in CAMERA_LAYOUTS and not self._image_topic_override:
            self.get_logger().warn(
                f"Unknown camera '{self.active_camera}', "
                f"available: {list(CAMERA_LAYOUTS.keys())}. "
                f"Defaulting to 'stereo_head_front_left'."
            )
            self.active_camera = 'stereo_head_front_left'

        if self.topic_type not in ('rgb_image', 'rgb_image_compressed'):
            self.get_logger().warn(
                f"Unknown topic_type '{self.topic_type}', falling back to 'rgb_image'. "
                f"Supported: rgb_image, rgb_image_compressed."
            )
            self.topic_type = 'rgb_image'

        if self.topic_type == 'rgb_image_compressed' and not CV_AVAILABLE:
            self.get_logger().warn(
                'topic_type=rgb_image_compressed requested but cv_bridge/cv2 '
                'is not available. Falling back to rgb_image (raw).'
            )
            self.topic_type = 'rgb_image'

        self._bridge = CvBridge() if CV_AVAILABLE else None

        # --- Publishers (always raw Image on /yolo/input_image) ---------
        self.image_pub = self.create_publisher(
            Image, '/yolo/input_image', SENSOR_DATA_QOS,
        )
        self.info_pub = self.create_publisher(
            CameraInfo, '/yolo/camera_info', SENSOR_DATA_QOS,
        )

        # --- State -------------------------------------------------------
        self._cam_subs = {}
        self._image_topic = ''
        self._info_topic = ''
        self._last_image_time = 0.0
        self._frame_count = 0

        self._resolve_and_subscribe()

        self._liveness_timer = self.create_timer(1.0, self._check_timeout)
        self._discovery_timer = self.create_timer(
            self._discovery_retry_sec, self._retry_discovery,
        )

    # ------------------------------------------------------------------ #
    # Topic resolution                                                    #
    # ------------------------------------------------------------------ #

    def _resolve_topics(self) -> None:
        layout = CAMERA_LAYOUTS.get(self.active_camera, {})

        if self._image_topic_override:
            self._image_topic = self._image_topic_override
        elif self.topic_type == 'rgb_image_compressed':
            self._image_topic = layout.get('rgb_compressed', '')
        else:
            self._image_topic = layout.get('rgb_raw', '')

        if self._info_topic_override:
            self._info_topic = self._info_topic_override
        else:
            self._info_topic = layout.get('rgb_info', '')

    def _resolve_and_subscribe(self) -> None:
        self._resolve_topics()
        self._subscribe_to_topics()

    def _subscribe_to_topics(self) -> None:
        for sub in self._cam_subs.values():
            self.destroy_subscription(sub)
        self._cam_subs.clear()

        if not self._image_topic:
            self.get_logger().error(
                f"No image topic resolved for camera={self.active_camera} "
                f"topic_type={self.topic_type}."
            )
            return

        img_infos = self.get_publishers_info_by_topic(self._image_topic)
        img_qos = _qos_from_publisher_infos(img_infos)

        if self.topic_type == 'rgb_image_compressed':
            self._cam_subs['image'] = self.create_subscription(
                CompressedImage,
                self._image_topic,
                self._compressed_callback,
                img_qos,
            )
        else:
            self._cam_subs['image'] = self.create_subscription(
                Image,
                self._image_topic,
                self._image_callback,
                img_qos,
            )

        self.get_logger().info(
            f"Subscribed to {self.topic_type}: '{self._image_topic}' "
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
                CameraInfo,
                self._info_topic,
                self._info_callback,
                info_qos,
            )
            self.get_logger().info(
                f"Subscribed to camera_info: '{self._info_topic}' "
                f"(publishers={len(info_infos)})"
            )

    def _log_available_image_topics(self) -> None:
        try:
            all_topics = self.get_topic_names_and_types()
        except Exception as exc:
            self.get_logger().warn(f'Could not enumerate topics: {exc}')
            return

        image_like = [
            name for name, types in all_topics
            if 'sensor_msgs/msg/Image' in types
            or 'sensor_msgs/msg/CompressedImage' in types
        ]
        if not image_like:
            self.get_logger().warn(
                'No Image/CompressedImage publishers visible in the graph. '
                'Is the HAL camera driver running on the robot? '
                'Check `ros2 topic list` and confirm ROS_DOMAIN_ID / '
                'RMW_IMPLEMENTATION match the robot.'
            )
            return

        self.get_logger().warn(
            f"Topic '{self._image_topic}' has no publisher yet. "
            f'Image-like topics currently visible: {image_like}. '
            f'Set image_topic_override:=<topic> if the name differs.'
        )

    def _retry_discovery(self) -> None:
        if self._frame_count > 0:
            return
        infos = self.get_publishers_info_by_topic(self._image_topic)
        if not infos:
            self._log_available_image_topics()
            return
        self.get_logger().info(
            f"Publisher now visible on '{self._image_topic}' — "
            f'resubscribing with matching QoS.'
        )
        self._subscribe_to_topics()

    # ------------------------------------------------------------------ #
    # Callbacks                                                           #
    # ------------------------------------------------------------------ #

    def _image_callback(self, msg: Image) -> None:
        self._mark_received(msg.width, msg.height, msg.encoding, compressed=False)
        self.image_pub.publish(msg)

    def _compressed_callback(self, msg: CompressedImage) -> None:
        """Decode JPEG -> raw Image and republish on /yolo/input_image."""
        if self._bridge is None:
            self.get_logger().error(
                'cv_bridge is required to decode CompressedImage but is unavailable.'
            )
            return
        try:
            cv_img = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warn(f'Failed to decode CompressedImage: {exc}')
            return

        out = self._bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
        out.header = msg.header
        self._mark_received(out.width, out.height, out.encoding, compressed=True)
        self.image_pub.publish(out)

    def _info_callback(self, msg: CameraInfo) -> None:
        self.info_pub.publish(msg)

    def _mark_received(self, w: int, h: int, encoding: str, compressed: bool) -> None:
        self._last_image_time = time.time()
        self._frame_count += 1
        if self._frame_count == 1:
            tag = 'CompressedImage' if compressed else 'Image'
            self.get_logger().info(
                f'First {tag} received ({w}x{h}, encoding={encoding}) — '
                f'relaying to /yolo/input_image'
            )

    def _check_timeout(self) -> None:
        if self._last_image_time == 0.0:
            return
        elapsed = time.time() - self._last_image_time
        if elapsed > self.timeout_sec:
            self.get_logger().warn(
                f'No images from {self.active_camera} '
                f"('{self._image_topic}') for {elapsed:.1f}s"
            )


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
