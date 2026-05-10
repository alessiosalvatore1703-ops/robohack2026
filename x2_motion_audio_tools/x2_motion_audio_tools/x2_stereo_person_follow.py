#!/usr/bin/env python3
"""Follow a stereo person target with the X2 locomotion controller.

This node consumes the stereo target point published by the vision pipeline and
publishes high-level locomotion velocity commands. It does not command leg
joints directly.
"""

from __future__ import annotations

import math
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

try:
    from aimdk_msgs.msg import (
        CommonState,
        JointStateArray,
        McActionCommand,
        McLocomotionVelocity,
        MessageHeader,
        RequestHeader,
    )
    from aimdk_msgs.srv import SetMcAction, SetMcInputSource

    AIMDK_AVAILABLE = True
except ImportError:
    AIMDK_AVAILABLE = False


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

SOURCE_NAME = "stereo_person_follow"
WAIST_YAW_JOINT = "waist_yaw_joint"
STABLE_STAND_STATE_SEQUENCE = (
    "DAMPING_DEFAULT",
    "JOINT_DEFAULT",
    "STAND_DEFAULT",
)


@dataclass
class TargetState:
    x_m: float
    z_m: float
    stamp_sec: float


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def bool_param(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class X2StereoPersonFollow(Node):
    """Convert stereo person target points into conservative base motion."""

    def __init__(self) -> None:
        super().__init__("x2_stereo_person_follow")

        self.declare_parameter("target_topic", "/stereo_person/target_point")
        self.declare_parameter("waist_state_topic", "/aima/hal/joint/waist/state")
        self.declare_parameter("enabled", False)
        self.declare_parameter("dry_run", True)
        self.declare_parameter("auto_enable_stable_stand", False)
        self.declare_parameter("auto_enable_locomotion", False)
        self.declare_parameter("source_name", SOURCE_NAME)
        self.declare_parameter("target_distance_m", 0.75)
        self.declare_parameter("stop_min_m", 0.5)
        self.declare_parameter("stop_max_m", 1.0)
        self.declare_parameter("min_valid_depth_m", 0.3)
        self.declare_parameter("max_valid_depth_m", 4.0)
        self.declare_parameter("forward_gain", 0.25)
        self.declare_parameter("angular_gain", 0.8)
        self.declare_parameter("max_forward_speed", 0.12)
        self.declare_parameter("max_reverse_speed", 0.08)
        self.declare_parameter("max_angular_speed", 0.25)
        self.declare_parameter("min_forward_speed", 0.0)
        self.declare_parameter("min_angular_speed", 0.0)
        self.declare_parameter("center_deadzone_deg", 4.0)
        self.declare_parameter("max_forward_bearing_deg", 10.0)
        self.declare_parameter("target_timeout_sec", 0.5)
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("reverse_enabled", False)
        self.declare_parameter("invert_angular", False)
        self.declare_parameter("require_waist_neutral_for_forward", True)
        self.declare_parameter("waist_neutral_limit_deg", 5.0)
        self.declare_parameter("waist_state_timeout_sec", 0.5)

        self.target_topic = str(self.get_parameter("target_topic").value)
        self.waist_state_topic = str(self.get_parameter("waist_state_topic").value)
        self.enabled = bool_param(self.get_parameter("enabled").value)
        self.dry_run = bool_param(self.get_parameter("dry_run").value)
        self.auto_enable_stable_stand = bool_param(
            self.get_parameter("auto_enable_stable_stand").value
        ) or bool_param(
            self.get_parameter("auto_enable_locomotion").value
        )
        self.source_name = str(self.get_parameter("source_name").value)
        self.target_distance_m = float(self.get_parameter("target_distance_m").value)
        self.stop_min_m = float(self.get_parameter("stop_min_m").value)
        self.stop_max_m = float(self.get_parameter("stop_max_m").value)
        self.min_valid_depth_m = float(self.get_parameter("min_valid_depth_m").value)
        self.max_valid_depth_m = float(self.get_parameter("max_valid_depth_m").value)
        self.forward_gain = float(self.get_parameter("forward_gain").value)
        self.angular_gain = float(self.get_parameter("angular_gain").value)
        self.max_forward_speed = float(self.get_parameter("max_forward_speed").value)
        self.max_reverse_speed = float(self.get_parameter("max_reverse_speed").value)
        self.max_angular_speed = float(self.get_parameter("max_angular_speed").value)
        self.min_forward_speed = float(self.get_parameter("min_forward_speed").value)
        self.min_angular_speed = float(self.get_parameter("min_angular_speed").value)
        self.center_deadzone_rad = math.radians(
            float(self.get_parameter("center_deadzone_deg").value)
        )
        self.max_forward_bearing_rad = math.radians(
            float(self.get_parameter("max_forward_bearing_deg").value)
        )
        self.target_timeout_sec = float(self.get_parameter("target_timeout_sec").value)
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.control_period_sec = 1.0 / max(control_rate_hz, 0.1)
        self.reverse_enabled = bool_param(self.get_parameter("reverse_enabled").value)
        self.invert_angular = bool_param(self.get_parameter("invert_angular").value)
        self.require_waist_neutral_for_forward = bool_param(
            self.get_parameter("require_waist_neutral_for_forward").value
        )
        self.waist_neutral_limit_rad = math.radians(
            float(self.get_parameter("waist_neutral_limit_deg").value)
        )
        self.waist_state_timeout_sec = float(
            self.get_parameter("waist_state_timeout_sec").value
        )

        self.target: Optional[TargetState] = None
        self.waist_yaw: Optional[float] = None
        self.last_waist_state_time = -float("inf")
        self.registered_input_source = False
        self.activation_lock = threading.Lock()
        self.activation_in_progress = False
        self.last_log_time = 0.0
        self.last_stop_publish_time = 0.0

        if not AIMDK_AVAILABLE and not self.dry_run:
            self.get_logger().fatal(
                "aimdk_msgs not available. Use dry_run=true off robot."
            )
            raise RuntimeError("aimdk_msgs not available")

        self.cb_group = ReentrantCallbackGroup()
        self.create_subscription(
            PointStamped,
            self.target_topic,
            self.target_callback,
            RELIABLE_QOS,
            callback_group=self.cb_group,
        )
        if AIMDK_AVAILABLE:
            self.create_subscription(
                JointStateArray,
                self.waist_state_topic,
                self.waist_state_callback,
                SENSOR_QOS,
                callback_group=self.cb_group,
            )
            self.velocity_pub = self.create_publisher(
                McLocomotionVelocity,
                "/aima/mc/locomotion/velocity",
                RELIABLE_QOS,
            )
            self.input_source_client = self.create_client(
                SetMcInputSource,
                "/aimdk_5Fmsgs/srv/SetMcInputSource",
                callback_group=self.cb_group,
            )
            self.mc_action_client = self.create_client(
                SetMcAction,
                "/aimdk_5Fmsgs/srv/SetMcAction",
                callback_group=self.cb_group,
            )
        else:
            self.velocity_pub = None
            self.input_source_client = None
            self.mc_action_client = None

        self.create_subscription(
            Bool,
            "/stereo_person/follow/enable",
            self.enable_callback,
            RELIABLE_QOS,
            callback_group=self.cb_group,
        )
        self.create_timer(
            self.control_period_sec,
            self.control_loop,
            callback_group=self.cb_group,
        )

        self.get_logger().info(
            "Stereo person follow started: "
            f"enabled={self.enabled}, dry_run={self.dry_run}, "
            f"auto_enable_stable_stand={self.auto_enable_stable_stand}, "
            f"target_topic={self.target_topic}, stop_band=[{self.stop_min_m:.2f}, "
            f"{self.stop_max_m:.2f}]m, max_v={self.max_forward_speed:.2f}m/s, "
            f"max_w={self.max_angular_speed:.2f}rad/s"
        )

        if self.enabled and not self.dry_run:
            self.enabled = False
            self.spawn_activation()

    def enable_callback(self, msg: Bool) -> None:
        if msg.data:
            if self.enabled or self.activation_in_progress:
                return
            if self.dry_run:
                self.enabled = True
                self.get_logger().info("Follow dry-run enabled.")
            else:
                self.get_logger().info("Activating stereo person follow.")
                self.spawn_activation()
        else:
            if not self.enabled and not self.activation_in_progress:
                return
            self.enabled = False
            self.get_logger().info("Follow disabled; publishing stop.")
            self.publish_stop()

    def spawn_activation(self) -> None:
        with self.activation_lock:
            if self.activation_in_progress:
                return
            self.activation_in_progress = True
        threading.Thread(target=self.activate_follower, daemon=True).start()

    def activate_follower(self) -> None:
        try:
            if self.dry_run:
                self.enabled = True
                return

            if self.auto_enable_stable_stand:
                if not self.ensure_stable_stand_mode():
                    self.get_logger().error(
                        "Could not enter STAND_DEFAULT. Try "
                        "`ros2 run py_examples set_mc_action SD` manually."
                    )
                    return
            else:
                self.get_logger().warn(
                    "auto_enable_stable_stand=false. Make sure the robot is in "
                    "STAND_DEFAULT before enabling walking commands."
                )

            if not self.registered_input_source:
                if not self.register_input_source():
                    return
                self.registered_input_source = True

            self.enabled = True
            self.get_logger().info("Stereo person follow ENABLED; robot may move.")
        finally:
            with self.activation_lock:
                self.activation_in_progress = False

    def ensure_stable_stand_mode(self) -> bool:
        self.get_logger().info("Requesting STAND_DEFAULT directly...")
        if self.send_mc_action("STAND_DEFAULT"):
            return True

        self.get_logger().info(
            "Direct transition rejected; walking DD -> JD -> SD."
        )
        for action in STABLE_STAND_STATE_SEQUENCE:
            ok = self.send_mc_action(action)
            if ok:
                self.get_logger().info(f"Transitioned to {action}")
            else:
                self.get_logger().warn(
                    f"Transition to {action} rejected; continuing."
            )
            time.sleep(0.8)

        return self.send_mc_action("STAND_DEFAULT")

    def send_mc_action(self, action_name: str) -> bool:
        if self.mc_action_client is None:
            return False
        if not self.mc_action_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("SetMcAction service unavailable")
            return False

        req = SetMcAction.Request()
        req.header = RequestHeader()
        req.command = McActionCommand()
        req.command.action_desc = action_name

        future = None
        for attempt in range(8):
            req.header.stamp = self.get_clock().now().to_msg()
            future = self.mc_action_client.call_async(req)
            deadline = time.monotonic() + 0.5
            while rclpy.ok() and not future.done() and time.monotonic() < deadline:
                time.sleep(0.02)
            if future.done():
                break
            self.get_logger().debug(f"SetMcAction({action_name}) retry {attempt + 1}")

        if future is None or not future.done() or future.result() is None:
            self.get_logger().error(f"SetMcAction({action_name}) failed")
            return False

        resp = future.result()
        if resp.response.status.value == CommonState.SUCCESS:
            self.get_logger().info(f"SetMcAction({action_name}) OK")
            return True

        self.get_logger().warn(
            f"SetMcAction({action_name}) rejected: {resp.response.message}"
        )
        return False

    def register_input_source(self) -> bool:
        if self.input_source_client is None:
            return False
        if not self.input_source_client.wait_for_service(timeout_sec=4.0):
            self.get_logger().error("SetMcInputSource service unavailable")
            return False

        req = SetMcInputSource.Request()
        req.action.value = 1001
        req.input_source.name = self.source_name
        req.input_source.priority = 40
        req.input_source.timeout = 1000

        future = None
        for attempt in range(8):
            req.request.header.stamp = self.get_clock().now().to_msg()
            future = self.input_source_client.call_async(req)
            deadline = time.monotonic() + 0.5
            while rclpy.ok() and not future.done() and time.monotonic() < deadline:
                time.sleep(0.02)
            if future.done():
                break
            self.get_logger().debug(
                f"SetMcInputSource retry [{attempt + 1}/8]"
            )

        if future is None or not future.done() or future.result() is None:
            self.get_logger().error("Input source registration failed")
            return False

        self.get_logger().info(f'Input source registered as "{self.source_name}"')
        return True

    def target_callback(self, msg: PointStamped) -> None:
        x_m = float(msg.point.x)
        z_m = float(msg.point.z)
        if not math.isfinite(x_m) or not math.isfinite(z_m):
            return
        self.target = TargetState(x_m=x_m, z_m=z_m, stamp_sec=time.monotonic())

    def waist_state_callback(self, msg) -> None:
        for joint in msg.joints:
            if getattr(joint, "name", "") == WAIST_YAW_JOINT:
                self.waist_yaw = float(joint.position)
                self.last_waist_state_time = time.monotonic()
                return
        if msg.joints:
            self.waist_yaw = float(msg.joints[0].position)
            self.last_waist_state_time = time.monotonic()

    def control_loop(self) -> None:
        forward, angular, reason = self.compute_velocity()
        if not self.enabled:
            return

        if self.dry_run:
            self.log_throttled(reason, forward, angular)
            return

        self.publish_velocity(forward, angular)

    def compute_velocity(self) -> tuple[float, float, str]:
        now = time.monotonic()
        if self.target is None:
            return 0.0, 0.0, "no_target"

        target_age = now - self.target.stamp_sec
        if target_age > self.target_timeout_sec:
            return 0.0, 0.0, "target_timeout"

        z_m = self.target.z_m
        if z_m < self.min_valid_depth_m or z_m > self.max_valid_depth_m:
            return 0.0, 0.0, "invalid_depth"

        bearing_rad = math.atan2(self.target.x_m, z_m)
        angular = self.angular_velocity_for_bearing(bearing_rad)

        waist_neutral = self.waist_is_neutral(now)
        if not waist_neutral:
            return 0.0, angular, "waist_not_neutral"

        if abs(bearing_rad) > self.max_forward_bearing_rad:
            return 0.0, angular, "turning_to_center"

        if self.stop_min_m <= z_m <= self.stop_max_m:
            return 0.0, angular, "inside_stop_band"

        if z_m > self.stop_max_m:
            distance_error = z_m - self.target_distance_m
            forward = clamp(
                self.forward_gain * distance_error,
                0.0,
                self.max_forward_speed,
            )
            forward = self.apply_min_velocity(forward, self.min_forward_speed)
            return forward, angular, "approaching"

        if z_m < self.stop_min_m and self.reverse_enabled:
            distance_error = self.target_distance_m - z_m
            reverse = -clamp(
                self.forward_gain * distance_error,
                0.0,
                self.max_reverse_speed,
            )
            reverse = -self.apply_min_velocity(abs(reverse), self.min_forward_speed)
            return reverse, angular, "reversing"

        return 0.0, angular, "too_close_stop"

    def angular_velocity_for_bearing(self, bearing_rad: float) -> float:
        if abs(bearing_rad) < self.center_deadzone_rad:
            return 0.0

        angular = -self.angular_gain * bearing_rad
        if self.invert_angular:
            angular *= -1.0
        angular = clamp(angular, -self.max_angular_speed, self.max_angular_speed)
        if abs(angular) < self.min_angular_speed:
            return math.copysign(self.min_angular_speed, angular)
        return angular

    def waist_is_neutral(self, now: float) -> bool:
        if not self.require_waist_neutral_for_forward:
            return True
        if self.waist_yaw is None:
            return False
        if now - self.last_waist_state_time > self.waist_state_timeout_sec:
            return False
        return abs(self.waist_yaw) <= self.waist_neutral_limit_rad

    def apply_min_velocity(self, value: float, min_abs: float) -> float:
        if value <= 0.0:
            return 0.0
        if min_abs <= 0.0:
            return value
        return max(value, min_abs)

    def publish_velocity(self, forward_velocity: float, angular_velocity: float) -> None:
        if self.velocity_pub is None:
            return

        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = self.source_name
        msg.forward_velocity = float(forward_velocity)
        msg.lateral_velocity = 0.0
        msg.angular_velocity = float(angular_velocity)
        self.velocity_pub.publish(msg)

    def publish_stop(self) -> None:
        now = time.monotonic()
        if now - self.last_stop_publish_time < 0.05:
            return
        self.last_stop_publish_time = now
        if not self.dry_run:
            self.publish_velocity(0.0, 0.0)

    def log_throttled(self, reason: str, forward: float, angular: float) -> None:
        now = time.monotonic()
        if now - self.last_log_time < 1.0:
            return
        self.last_log_time = now
        self.get_logger().info(
            f"follow={reason}: forward={forward:.3f}m/s, "
            f"angular={angular:.3f}rad/s"
        )

    def stop(self) -> None:
        self.publish_velocity(0.0, 0.0)


_GLOBAL_NODE: Optional[X2StereoPersonFollow] = None


def signal_handler(sig, _frame) -> None:
    if _GLOBAL_NODE is not None:
        _GLOBAL_NODE.get_logger().info(f"Received signal {sig}; stopping robot")
        _GLOBAL_NODE.stop()
    if rclpy.ok():
        rclpy.shutdown()
    sys.exit(0)


def main(args=None) -> None:
    global _GLOBAL_NODE
    rclpy.init(args=args)
    node = X2StereoPersonFollow()
    _GLOBAL_NODE = node

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
