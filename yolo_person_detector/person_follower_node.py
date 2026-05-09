"""
Person Follower Node for Agibot X2.

Subscribes to /yolo/detections and steers the robot to follow the
largest (closest) detected person using the robot's locomotion API.

Uses visual servoing (no depth required):
  - Horizontal error (person vs image center) -> angular velocity
  - Bounding box height (larger = closer) -> forward velocity

Safety:
  - Stops when no person detected
  - Stops when watchdog timer expires
  - Respects robot's minimum/maximum velocity thresholds
"""

import time
import signal
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

try:
    from aimdk_msgs.msg import McLocomotionVelocity, MessageHeader
    from aimdk_msgs.srv import SetMcInputSource
    AIMDK_AVAILABLE = True
except ImportError:
    AIMDK_AVAILABLE = False


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


class PersonFollowerNode(Node):
    """Steers the robot toward the detected person using visual servoing."""

    def __init__(self):
        super().__init__('person_follower')

        if not AIMDK_AVAILABLE:
            self.get_logger().fatal(
                'aimdk_msgs not available. This node must run on the Agibot X2.'
            )
            raise RuntimeError('aimdk_msgs not available')

        # Parameters
        self.declare_parameter('enabled', False)  # Safety: start disabled
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('target_bbox_height_ratio', 0.6)  # Target person height as fraction of image
        self.declare_parameter('forward_gain', 0.8)
        self.declare_parameter('angular_gain', 1.5)
        self.declare_parameter('max_forward_speed', 0.6)
        self.declare_parameter('min_forward_speed', 0.2)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('min_angular_speed', 0.1)
        self.declare_parameter('center_deadzone_px', 50)  # Ignore small horizontal errors
        self.declare_parameter('watchdog_timeout_sec', 0.5)  # Stop if no detection for this long
        self.declare_parameter('control_rate_hz', 20.0)

        self.enabled = self.get_parameter('enabled').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.target_bbox_height_ratio = self.get_parameter('target_bbox_height_ratio').value
        self.forward_gain = self.get_parameter('forward_gain').value
        self.angular_gain = self.get_parameter('angular_gain').value
        self.max_forward_speed = self.get_parameter('max_forward_speed').value
        self.min_forward_speed = self.get_parameter('min_forward_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        self.min_angular_speed = self.get_parameter('min_angular_speed').value
        self.center_deadzone_px = self.get_parameter('center_deadzone_px').value
        self.watchdog_timeout_sec = self.get_parameter('watchdog_timeout_sec').value
        control_rate = self.get_parameter('control_rate_hz').value

        # State
        self._last_detection_time = 0.0
        self._target_center_x = None
        self._target_bbox_height = None
        self._forward_vel = 0.0
        self._angular_vel = 0.0

        # ROS interfaces
        self.vel_pub = self.create_publisher(
            McLocomotionVelocity,
            '/aima/mc/locomotion/velocity',
            RELIABLE_QOS,
        )
        self.input_source_client = self.create_client(
            SetMcInputSource,
            '/aimdk_5Fmsgs/srv/SetMcInputSource',
        )
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/yolo/detections',
            self._detection_callback,
            RELIABLE_QOS,
        )
        # Subscribe to input_image to auto-update image dimensions
        self.image_sub = self.create_subscription(
            Image,
            '/yolo/input_image',
            self._image_callback,
            SENSOR_QOS,
        )

        # Control loop timer
        self.control_timer = self.create_timer(1.0 / control_rate, self._control_loop)

        self.get_logger().info(
            f'Person follower started. enabled={self.enabled}. '
            f'Publish Bool(true) on /yolo/follower/enable to activate.'
        )

        if self.enabled:
            self._register_input_source()

    def _register_input_source(self) -> bool:
        """Register as a locomotion input source (required before publishing velocity)."""
        timeout_sec = 8.0
        start = self.get_clock().now().nanoseconds / 1e9

        while not self.input_source_client.wait_for_service(timeout_sec=2.0):
            now = self.get_clock().now().nanoseconds / 1e9
            if now - start > timeout_sec:
                self.get_logger().error('Locomotion service unavailable')
                return False
            self.get_logger().info('Waiting for locomotion input service...')

        req = SetMcInputSource.Request()
        req.action.value = 1001  # ADD
        req.input_source.name = 'person_follower'
        req.input_source.priority = 40
        req.input_source.timeout = 1000

        for attempt in range(8):
            req.request.header.stamp = self.get_clock().now().to_msg()
            future = self.input_source_client.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.25)
            if future.done():
                break

        if future.done() and future.result() is not None:
            self.get_logger().info('Input source registered as "person_follower"')
            return True

        self.get_logger().error('Failed to register input source')
        return False

    def _image_callback(self, msg: Image) -> None:
        """Update image dimensions from input stream."""
        if self.image_width != msg.width or self.image_height != msg.height:
            self.image_width = msg.width
            self.image_height = msg.height
            self.get_logger().info(
                f'Image size updated: {self.image_width}x{self.image_height}'
            )

    def _detection_callback(self, msg: Detection2DArray) -> None:
        """Pick the largest (closest) person and store target."""
        if not msg.detections:
            self._target_center_x = None
            self._target_bbox_height = None
            return

        # Pick the detection with largest bbox area (closest person)
        largest = max(
            msg.detections,
            key=lambda d: d.bbox.size_x * d.bbox.size_y,
        )

        self._target_center_x = largest.bbox.center.x
        self._target_bbox_height = largest.bbox.size_y
        self._last_detection_time = time.time()

    def _control_loop(self) -> None:
        """Compute and publish velocity commands at fixed rate."""
        now = time.time()
        detection_age = now - self._last_detection_time

        # Watchdog: stop if detection is stale or follower disabled
        if not self.enabled or detection_age > self.watchdog_timeout_sec:
            self._forward_vel = 0.0
            self._angular_vel = 0.0
            self._publish_velocity()
            return

        if self._target_center_x is None or self._target_bbox_height is None:
            self._forward_vel = 0.0
            self._angular_vel = 0.0
            self._publish_velocity()
            return

        # --- Angular control: keep person centered horizontally ---
        center_x = self.image_width / 2.0
        err_x = self._target_center_x - center_x

        if abs(err_x) < self.center_deadzone_px:
            angular_vel = 0.0
        else:
            # Normalize error to [-1, 1] then scale by gain
            normalized_err = err_x / (self.image_width / 2.0)
            # Negative because positive err_x (person on right) means robot should turn right
            # In ROS convention, positive angular_vel = counter-clockwise (left turn)
            angular_vel = -self.angular_gain * normalized_err
            angular_vel = self._clamp_velocity(
                angular_vel,
                self.min_angular_speed,
                self.max_angular_speed,
            )

        # --- Forward control: keep person at target size (closer = larger bbox) ---
        target_height_px = self.target_bbox_height_ratio * self.image_height
        height_err = target_height_px - self._target_bbox_height
        # Positive err = person too far (bbox too small) -> move forward
        normalized_height_err = height_err / target_height_px
        forward_vel = self.forward_gain * normalized_height_err
        forward_vel = self._clamp_velocity(
            forward_vel,
            self.min_forward_speed,
            self.max_forward_speed,
        )

        self._forward_vel = forward_vel
        self._angular_vel = angular_vel
        self._publish_velocity()

    def _clamp_velocity(self, v: float, v_min: float, v_max: float) -> float:
        """Clamp velocity respecting robot's dead-band and max limits."""
        if abs(v) < v_min:
            return 0.0
        if v > v_max:
            return v_max
        if v < -v_max:
            return -v_max
        return v

    def _publish_velocity(self) -> None:
        """Publish velocity command to robot."""
        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = 'person_follower'
        msg.forward_velocity = float(self._forward_vel)
        msg.lateral_velocity = 0.0
        msg.angular_velocity = float(self._angular_vel)
        self.vel_pub.publish(msg)

    def stop(self) -> None:
        """Emergency stop."""
        self._forward_vel = 0.0
        self._angular_vel = 0.0
        self._publish_velocity()


# Global for signal handler
_global_node = None


def _signal_handler(sig, frame):
    if _global_node is not None:
        _global_node.get_logger().info('Signal received, stopping robot')
        _global_node.stop()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(0)


def main(args=None):
    global _global_node
    rclpy.init(args=args)
    node = PersonFollowerNode()
    _global_node = node

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
