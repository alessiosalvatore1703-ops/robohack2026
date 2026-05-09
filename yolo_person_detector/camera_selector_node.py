"""
Camera Selector ROS2 Node.

Subscribes to multiple camera topics on the Agibot X2 and republishes
the selected camera's images to a unified topic for the detection pipeline.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import SetBool

import time

# QoS for camera subscriptions
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)

# Available cameras on Agibot X2 (from topics_and_services)
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


class CameraSelectorNode(Node):
    """Selects and republishes camera images from one of the robot's cameras."""

    def __init__(self):
        super().__init__('camera_selector')

        # Parameters
        self.declare_parameter('active_camera', 'rgbd_head_front')
        self.declare_parameter('timeout_sec', 2.0)

        self.active_camera = self.get_parameter('active_camera').value
        self.timeout_sec = self.get_parameter('timeout_sec').value

        if self.active_camera not in CAMERA_TOPICS:
            self.get_logger().warn(
                f"Unknown camera '{self.active_camera}', "
                f"available: {list(CAMERA_TOPICS.keys())}. "
                f"Defaulting to 'rgbd_head_front'."
            )
            self.active_camera = 'rgbd_head_front'

        # Publishers
        self.image_pub = self.create_publisher(Image, '/yolo/input_image', SENSOR_QOS)
        self.info_pub = self.create_publisher(CameraInfo, '/yolo/camera_info', SENSOR_QOS)

        # Subscribe to active camera (use _cam_subs to avoid conflict with Node._subscriptions)
        self._cam_subs = {}
        self._subscribe_to_camera(self.active_camera)

        # Timeout tracking
        self._last_image_time = time.time()
        self._timeout_timer = self.create_timer(1.0, self._check_timeout)

        self.get_logger().info(
            f'Camera selector started. Active: {self.active_camera} '
            f'({CAMERA_TOPICS[self.active_camera]})'
        )

    def _subscribe_to_camera(self, camera_name: str) -> None:
        """Subscribe to the specified camera's image and info topics."""
        # Unsubscribe from previous
        for sub in self._cam_subs.values():
            self.destroy_subscription(sub)
        self._cam_subs.clear()

        topic = CAMERA_TOPICS[camera_name]
        self._cam_subs['image'] = self.create_subscription(
            Image, topic, self._image_callback, SENSOR_QOS
        )

        if camera_name in CAMERA_INFO_TOPICS:
            info_topic = CAMERA_INFO_TOPICS[camera_name]
            self._cam_subs['info'] = self.create_subscription(
                CameraInfo, info_topic, self._info_callback, SENSOR_QOS
            )

        self.get_logger().info(f'Subscribed to camera: {camera_name} ({topic})')

    def _image_callback(self, msg: Image) -> None:
        """Forward image from active camera to unified topic."""
        self._last_image_time = time.time()
        self.image_pub.publish(msg)

    def _info_callback(self, msg: CameraInfo) -> None:
        """Forward camera info to unified topic."""
        self.info_pub.publish(msg)

    def _check_timeout(self) -> None:
        """Check if camera stream has timed out."""
        elapsed = time.time() - self._last_image_time
        if elapsed > self.timeout_sec:
            self.get_logger().warn(
                f'No images from {self.active_camera} for {elapsed:.1f}s'
            )

    def switch_camera(self, camera_name: str) -> bool:
        """
        Switch to a different camera.

        Args:
            camera_name: Name of camera to switch to.

        Returns:
            True if switch was successful.
        """
        if camera_name not in CAMERA_TOPICS:
            self.get_logger().error(
                f"Cannot switch to unknown camera '{camera_name}'. "
                f"Available: {list(CAMERA_TOPICS.keys())}"
            )
            return False

        self.active_camera = camera_name
        self._subscribe_to_camera(camera_name)
        self._last_image_time = time.time()
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
