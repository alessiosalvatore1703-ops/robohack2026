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

Agibot X2 motion-mode handling:
  The X2 motion controller enforces a state-transition diagram:

      PASSIVE_DEFAULT -> DAMPING_DEFAULT -> JOINT_DEFAULT
      JOINT_DEFAULT   -> STAND_DEFAULT == LOCOMOTION_DEFAULT  (unified)

  Any non-adjacent transition is rejected. When the follower is enabled
  (either via the `enabled` parameter or a Bool(true) on
  `/yolo/follower/enable`) and `auto_enable_locomotion` is true, the node
  will walk the state machine itself via the SetMcAction service.
"""

import time
import signal
import sys
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2DArray

try:
    from aimdk_msgs.msg import (
        McLocomotionVelocity,
        MessageHeader,
        RequestHeader,
        CommonState,
        McActionCommand,
    )
    from aimdk_msgs.srv import SetMcInputSource, SetMcAction
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

# Ordered path up the X2 motion-mode state machine toward Stable Stand.
# Each step is a legal one-step transition; rejections on states we are
# already past are expected and non-fatal.
STABLE_STAND_STATE_SEQUENCE = (
    'DAMPING_DEFAULT',
    'JOINT_DEFAULT',
    'STAND_DEFAULT',
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
        self.declare_parameter('auto_enable_locomotion', True)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('target_bbox_height_ratio', 0.6)
        self.declare_parameter('forward_gain', 0.8)
        self.declare_parameter('angular_gain', 1.5)
        self.declare_parameter('max_forward_speed', 0.6)
        self.declare_parameter('min_forward_speed', 0.2)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('min_angular_speed', 0.1)
        self.declare_parameter('center_deadzone_px', 50)
        self.declare_parameter('watchdog_timeout_sec', 0.5)
        self.declare_parameter('control_rate_hz', 20.0)

        self.enabled = bool(self.get_parameter('enabled').value)
        self._auto_enable_locomotion = bool(
            self.get_parameter('auto_enable_locomotion').value
        )
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
        self._registered_input_source = False
        self._activation_lock = threading.Lock()
        self._activation_in_progress = False

        # Reentrant group so service calls inside subscription callbacks don't
        # deadlock when spun by a MultiThreadedExecutor.
        self._cb_group = ReentrantCallbackGroup()

        # ROS interfaces
        self.vel_pub = self.create_publisher(
            McLocomotionVelocity,
            '/aima/mc/locomotion/velocity',
            RELIABLE_QOS,
        )
        self.input_source_client = self.create_client(
            SetMcInputSource,
            '/aimdk_5Fmsgs/srv/SetMcInputSource',
            callback_group=self._cb_group,
        )
        self.mc_action_client = self.create_client(
            SetMcAction,
            '/aimdk_5Fmsgs/srv/SetMcAction',
            callback_group=self._cb_group,
        )
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/yolo/detections',
            self._detection_callback,
            RELIABLE_QOS,
            callback_group=self._cb_group,
        )
        self.image_sub = self.create_subscription(
            Image,
            '/yolo/input_image',
            self._image_callback,
            SENSOR_QOS,
            callback_group=self._cb_group,
        )
        self.enable_sub = self.create_subscription(
            Bool,
            '/yolo/follower/enable',
            self._enable_callback,
            RELIABLE_QOS,
            callback_group=self._cb_group,
        )

        # Control loop timer
        self.control_timer = self.create_timer(
            1.0 / control_rate,
            self._control_loop,
            callback_group=self._cb_group,
        )

        self.get_logger().info(
            f'Person follower started. enabled={self.enabled}, '
            f'auto_enable_locomotion={self._auto_enable_locomotion}. '
            f'Publish Bool(true) on /yolo/follower/enable to activate.'
        )

        # If enabled at startup, kick off activation in a worker thread so we
        # don't block node construction.
        if self.enabled:
            # Temporarily flip off while activation runs; _activate_follower
            # will set it back to True on success.
            self.enabled = False
            self._spawn_activation()

    # ------------------------------------------------------------------ #
    # Enable / activation                                                 #
    # ------------------------------------------------------------------ #

    def _enable_callback(self, msg: Bool) -> None:
        """Runtime enable/disable via /yolo/follower/enable."""
        if msg.data:
            if self.enabled or self._activation_in_progress:
                return
            self.get_logger().info('Received enable=True, activating follower')
            self._spawn_activation()
        else:
            if not self.enabled:
                return
            self.get_logger().info('Received enable=False, stopping follower')
            self.enabled = False
            self.stop()

    def _spawn_activation(self) -> None:
        """Run the activation sequence on a worker thread."""
        with self._activation_lock:
            if self._activation_in_progress:
                return
            self._activation_in_progress = True
        threading.Thread(target=self._activate_follower, daemon=True).start()

    def _activate_follower(self) -> None:
        """Put the robot in Stable Stand and register as a velocity source."""
        try:
            if self._auto_enable_locomotion:
                if not self._ensure_stable_stand_mode():
                    self.get_logger().error(
                        'Could not transition robot to STAND_DEFAULT. '
                        'Put it there manually with '
                        '`ros2 run py_examples set_mc_action SD`.'
                    )
                    return
            else:
                self.get_logger().warn(
                    'auto_enable_locomotion=false. Make sure the robot is in '
                    'STAND_DEFAULT before expecting motion.'
                )

            if not self._registered_input_source:
                if not self._register_input_source():
                    return
                self._registered_input_source = True

            self.enabled = True
            self.get_logger().info('Follower ENABLED — robot will move on detections.')
        finally:
            with self._activation_lock:
                self._activation_in_progress = False

    # ------------------------------------------------------------------ #
    # Motion-mode state machine                                           #
    # ------------------------------------------------------------------ #

    def _ensure_stable_stand_mode(self) -> bool:
        """Walk the X2 state diagram until we reach STAND_DEFAULT.

        Strategy:
          1. Try STAND_DEFAULT directly. If we were already in STAND/LOCOMOTION
             or JOINT, this single call should win.
          2. Otherwise walk DD -> JD -> SD; rejections on states we're already
             past are logged at WARN and ignored.
          3. Confirm with a final STAND_DEFAULT request.
        """
        self.get_logger().info('Requesting STAND_DEFAULT directly...')
        if self._send_mc_action('STAND_DEFAULT'):
            return True

        self.get_logger().info(
            'Direct transition rejected, walking state machine: '
            'DAMPING_DEFAULT -> JOINT_DEFAULT -> STAND_DEFAULT'
        )
        for action in STABLE_STAND_STATE_SEQUENCE:
            ok = self._send_mc_action(action)
            if ok:
                self.get_logger().info(f'Transitioned to {action}')
            else:
                self.get_logger().warn(
                    f'Transition to {action} rejected (probably already past it)'
                )
            # Give the robot time to settle between stages.
            time.sleep(0.8)

        # Final confirmation. If the robot is now in SD this is a no-op and
        # the service should return success.
        return self._send_mc_action('STAND_DEFAULT')

    def _send_mc_action(self, action_name: str) -> bool:
        """Invoke SetMcAction; True iff the controller reports SUCCESS."""
        if not self.mc_action_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('SetMcAction service unavailable')
            return False

        req = SetMcAction.Request()
        req.header = RequestHeader()
        cmd = McActionCommand()
        cmd.action_desc = action_name
        req.command = cmd

        future = None
        for attempt in range(8):
            req.header.stamp = self.get_clock().now().to_msg()
            future = self.mc_action_client.call_async(req)
            deadline = time.time() + 0.5
            while rclpy.ok() and not future.done() and time.time() < deadline:
                time.sleep(0.02)
            if future.done():
                break
            self.get_logger().debug(f'SetMcAction({action_name}) retry [{attempt}]')

        if future is None or not future.done():
            self.get_logger().error(f'SetMcAction({action_name}) timed out')
            return False

        resp = future.result()
        if resp is None:
            self.get_logger().error(f'SetMcAction({action_name}) returned None')
            return False

        if resp.response.status.value == CommonState.SUCCESS:
            self.get_logger().info(f'SetMcAction({action_name}) OK')
            return True

        self.get_logger().warn(
            f'SetMcAction({action_name}) rejected: {resp.response.message}'
        )
        return False

    # ------------------------------------------------------------------ #
    # Input-source registration                                           #
    # ------------------------------------------------------------------ #

    def _register_input_source(self) -> bool:
        """Register as a locomotion input source before publishing velocity."""
        if not self.input_source_client.wait_for_service(timeout_sec=4.0):
            self.get_logger().error('SetMcInputSource service unavailable')
            return False

        req = SetMcInputSource.Request()
        req.action.value = 1001  # ADD
        req.input_source.name = 'person_follower'
        req.input_source.priority = 40
        req.input_source.timeout = 1000

        future = None
        for attempt in range(8):
            req.request.header.stamp = self.get_clock().now().to_msg()
            future = self.input_source_client.call_async(req)
            deadline = time.time() + 0.5
            while rclpy.ok() and not future.done() and time.time() < deadline:
                time.sleep(0.02)
            if future.done():
                break
            self.get_logger().debug(f'SetMcInputSource retry [{attempt}]')

        if future is None or not future.done() or future.result() is None:
            self.get_logger().error('Failed to register input source')
            return False

        self.get_logger().info('Input source registered as "person_follower"')
        return True

    # ------------------------------------------------------------------ #
    # Detection / control                                                 #
    # ------------------------------------------------------------------ #

    def _image_callback(self, msg: Image) -> None:
        if self.image_width != msg.width or self.image_height != msg.height:
            self.image_width = msg.width
            self.image_height = msg.height
            self.get_logger().info(
                f'Image size updated: {self.image_width}x{self.image_height}'
            )

    def _detection_callback(self, msg: Detection2DArray) -> None:
        if not msg.detections:
            self._target_center_x = None
            self._target_bbox_height = None
            return

        largest = max(
            msg.detections,
            key=lambda d: d.bbox.size_x * d.bbox.size_y,
        )
        self._target_center_x = largest.bbox.center.position.x
        self._target_bbox_height = largest.bbox.size_y
        self._last_detection_time = time.time()

    def _control_loop(self) -> None:
        now = time.time()
        detection_age = now - self._last_detection_time

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

        # Angular control: keep person horizontally centered.
        center_x = self.image_width / 2.0
        err_x = self._target_center_x - center_x
        if abs(err_x) < self.center_deadzone_px:
            angular_vel = 0.0
        else:
            normalized_err = err_x / (self.image_width / 2.0)
            angular_vel = -self.angular_gain * normalized_err
            angular_vel = self._clamp_velocity(
                angular_vel,
                self.min_angular_speed,
                self.max_angular_speed,
            )

        # Forward control: keep person at target bbox size.
        target_height_px = self.target_bbox_height_ratio * self.image_height
        height_err = target_height_px - self._target_bbox_height
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
        if abs(v) < v_min:
            return 0.0
        if v > v_max:
            return v_max
        if v < -v_max:
            return -v_max
        return v

    def _publish_velocity(self) -> None:
        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = 'person_follower'
        msg.forward_velocity = float(self._forward_vel)
        msg.lateral_velocity = 0.0
        msg.angular_velocity = float(self._angular_vel)
        self.vel_pub.publish(msg)

    def stop(self) -> None:
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

    # MultiThreadedExecutor lets the enable-subscription callback make
    # synchronous service calls without deadlocking the control-loop timer.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.stop()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
