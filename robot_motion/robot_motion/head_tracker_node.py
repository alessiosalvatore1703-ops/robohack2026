#!/usr/bin/env python3
"""Track the stereo target point with the X2 head yaw joint only."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

try:
    import ruckig
except ImportError:
    ruckig = None

try:
    from aimdk_msgs.msg import (
        JointCommand,
        JointCommandArray,
        JointStateArray,
        MessageHeader,
    )

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


@dataclass(frozen=True)
class JointInfo:
    name: str
    lower_limit: float
    upper_limit: float
    kp: float
    kd: float


HEAD_JOINTS = [
    JointInfo("head_yaw_joint", -0.366, 0.366, 20.0, 2.0),
    JointInfo("head_pitch_joint", -0.3838, 0.3838, 20.0, 2.0),
]
HEAD_YAW_INDEX = 0
HEAD_PITCH_INDEX = 1


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def move_toward(value: float, target: float, max_delta: float) -> float:
    if target > value + max_delta:
        return value + max_delta
    if target < value - max_delta:
        return value - max_delta
    return target


def bool_param(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class X2HeadYawTracker(Node):
    """Slew head_yaw_joint toward a PointStamped target in the camera frame."""

    def __init__(self) -> None:
        super().__init__("head_tracker")

        if not AIMDK_AVAILABLE:
            self.get_logger().fatal(
                "aimdk_msgs not available. This node must run on the Agibot X2."
            )
            raise RuntimeError("aimdk_msgs not available")

        self.declare_parameter("target_topic", "/stereo_person/target_point")
        self.declare_parameter("head_state_topic", "/aima/hal/joint/head/state")
        self.declare_parameter("head_command_topic", "/aima/hal/joint/head/command")
        self.declare_parameter("enabled", True)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("yaw_gain", 0.6)
        self.declare_parameter("center_deadzone_deg", 2.0)
        self.declare_parameter("use_ruckig", True)
        self.declare_parameter("max_yaw_velocity", 1.0)
        self.declare_parameter("max_yaw_acceleration", 1.0)
        self.declare_parameter("max_yaw_jerk", 25.0)
        self.declare_parameter("target_timeout_sec", 0.5)
        self.declare_parameter("control_rate_hz", 500.0)
        self.declare_parameter("hold_on_lost", True)
        self.declare_parameter("invert_yaw", False)
        self.declare_parameter("soft_limit_deg", 20.0)
        self.declare_parameter("min_depth_m", 0.2)

        self.target_topic = self.get_parameter("target_topic").value
        self.head_state_topic = self.get_parameter("head_state_topic").value
        self.head_command_topic = self.get_parameter("head_command_topic").value
        self.enabled = bool_param(self.get_parameter("enabled").value)
        self.dry_run = bool_param(self.get_parameter("dry_run").value)
        self.yaw_gain = float(self.get_parameter("yaw_gain").value)
        self.center_deadzone_rad = math.radians(
            float(self.get_parameter("center_deadzone_deg").value)
        )
        self.use_ruckig = bool_param(self.get_parameter("use_ruckig").value)
        self.max_yaw_velocity = float(self.get_parameter("max_yaw_velocity").value)
        self.max_yaw_acceleration = float(
            self.get_parameter("max_yaw_acceleration").value
        )
        self.max_yaw_jerk = float(self.get_parameter("max_yaw_jerk").value)
        self.target_timeout_sec = float(self.get_parameter("target_timeout_sec").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.control_period_sec = 1.0 / max(self.control_rate_hz, 0.1)
        self.hold_on_lost = bool_param(self.get_parameter("hold_on_lost").value)
        self.invert_yaw = bool_param(self.get_parameter("invert_yaw").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)

        soft_limit_rad = math.radians(float(self.get_parameter("soft_limit_deg").value))
        yaw_joint = HEAD_JOINTS[HEAD_YAW_INDEX]
        if soft_limit_rad > 0.0:
            self.yaw_lower_limit = max(yaw_joint.lower_limit, -soft_limit_rad)
            self.yaw_upper_limit = min(yaw_joint.upper_limit, soft_limit_rad)
        else:
            self.yaw_lower_limit = yaw_joint.lower_limit
            self.yaw_upper_limit = yaw_joint.upper_limit

        self.head_positions: Optional[list[float]] = None
        self.head_velocities: Optional[list[float]] = None
        self.command_positions: Optional[list[float]] = None
        self.last_target_bearing_rad: Optional[float] = None
        self.target_yaw_position: Optional[float] = None
        self.ruckig_controller = None
        self.ruckig_input = None
        self.ruckig_output = None
        self.ruckig_initialized = False
        self.last_target_time = -float("inf")
        self.last_state_warning = 0.0
        self.last_dry_run_log = 0.0
        self.last_ruckig_warning = 0.0

        if self.use_ruckig:
            if ruckig is None:
                self.get_logger().warn(
                    "ruckig is not installed; using velocity-limited fallback."
                )
                self.use_ruckig = False
            else:
                self.ruckig_controller = ruckig.Ruckig(
                    len(HEAD_JOINTS), self.control_period_sec
                )
                self.ruckig_input = ruckig.InputParameter(len(HEAD_JOINTS))
                self.ruckig_output = ruckig.OutputParameter(len(HEAD_JOINTS))
                self.ruckig_input.max_velocity = self.ruckig_max_velocity()
                self.ruckig_input.max_acceleration = self.ruckig_max_acceleration()
                self.ruckig_input.max_jerk = self.ruckig_max_jerk()

        self.create_subscription(
            PointStamped,
            self.target_topic,
            self.target_callback,
            RELIABLE_QOS,
        )
        self.create_subscription(
            JointStateArray,
            self.head_state_topic,
            self.head_state_callback,
            SENSOR_QOS,
        )
        self.command_pub = self.create_publisher(
            JointCommandArray,
            self.head_command_topic,
            RELIABLE_QOS,
        )
        self.create_timer(self.control_period_sec, self.control_loop)

        self.get_logger().info(
            "Head yaw tracker started: "
            f"target={self.target_topic}, state={self.head_state_topic}, "
            f"command={self.head_command_topic}, enabled={self.enabled}, "
            f"dry_run={self.dry_run}, use_ruckig={self.use_ruckig}, "
            f"period={self.control_period_sec:.4f}s, "
            f"yaw_limits=[{self.yaw_lower_limit:.3f}, {self.yaw_upper_limit:.3f}]"
        )

    def target_callback(self, msg: PointStamped) -> None:
        if msg.point.z < self.min_depth_m:
            return
        bearing_rad = math.atan2(float(msg.point.x), float(msg.point.z))
        self.last_target_bearing_rad = bearing_rad
        self.last_target_time = time.monotonic()
        start_positions = self.command_positions or self.head_positions
        if start_positions is not None:
            self.target_yaw_position = self.desired_yaw_for_bearing(
                bearing_rad, start_positions[HEAD_YAW_INDEX]
            )

    def head_state_callback(self, msg: JointStateArray) -> None:
        by_name = {
            joint.name: joint
            for joint in msg.joints
            if getattr(joint, "name", "")
        }

        if all(joint.name in by_name for joint in HEAD_JOINTS):
            self.head_positions = [
                float(by_name[joint.name].position) for joint in HEAD_JOINTS
            ]
            self.head_velocities = [
                float(getattr(by_name[joint.name], "velocity", 0.0))
                for joint in HEAD_JOINTS
            ]
        elif len(msg.joints) >= len(HEAD_JOINTS):
            self.head_positions = [
                float(msg.joints[i].position) for i in range(len(HEAD_JOINTS))
            ]
            self.head_velocities = [
                float(getattr(msg.joints[i], "velocity", 0.0))
                for i in range(len(HEAD_JOINTS))
            ]
        else:
            return

        self.head_positions = self.clamp_head_positions(self.head_positions)
        if self.command_positions is None:
            self.command_positions = list(self.head_positions)
        if self.use_ruckig and not self.ruckig_initialized:
            self.initialize_ruckig()

    def initialize_ruckig(self) -> None:
        if (
            self.ruckig_input is None
            or self.head_positions is None
            or self.head_velocities is None
        ):
            return

        self.ruckig_input.current_position = list(self.head_positions)
        self.ruckig_input.current_velocity = list(self.head_velocities)
        self.ruckig_input.current_acceleration = [0.0] * len(HEAD_JOINTS)
        self.ruckig_input.target_position = list(self.head_positions)
        self.ruckig_input.target_velocity = [0.0] * len(HEAD_JOINTS)
        self.ruckig_input.target_acceleration = [0.0] * len(HEAD_JOINTS)
        self.ruckig_initialized = True
        self.get_logger().info("Ruckig head tracker initialized from joint state.")

    def control_loop(self) -> None:
        if not self.enabled:
            return

        if self.head_positions is None:
            now = time.monotonic()
            if now - self.last_state_warning > 3.0:
                self.last_state_warning = now
                self.get_logger().warn(
                    f"No head joint state messages yet on {self.head_state_topic}"
                )
            return

        target_is_stale = time.monotonic() - self.last_target_time > self.target_timeout_sec
        if target_is_stale or self.last_target_bearing_rad is None:
            self.target_yaw_position = None
            if self.hold_on_lost:
                self.publish_head_command(
                    self.command_positions or self.head_positions,
                    [0.0] * len(HEAD_JOINTS),
                )
            return

        start_positions = self.command_positions or self.head_positions
        if self.target_yaw_position is None:
            self.target_yaw_position = self.desired_yaw_for_bearing(
                self.last_target_bearing_rad,
                start_positions[HEAD_YAW_INDEX],
            )

        target_positions = list(start_positions)
        target_positions[HEAD_YAW_INDEX] = self.target_yaw_position

        if self.use_ruckig:
            next_positions, velocities = self.next_ruckig_step(target_positions)
        else:
            next_positions, velocities = self.next_head_step(target_positions)
        self.publish_head_command(next_positions, velocities)
        self.command_positions = list(next_positions)

    def desired_yaw_for_bearing(self, bearing_rad: float, current_yaw: float) -> float:
        if self.invert_yaw:
            bearing_rad = -bearing_rad
        if abs(bearing_rad) < self.center_deadzone_rad:
            return clamp(current_yaw, self.yaw_lower_limit, self.yaw_upper_limit)

        # Positive camera x is image-right; X2 joint-positive yaw is treated
        # as left by default, so subtract bearing unless invert_yaw is set.
        return clamp(
            current_yaw - self.yaw_gain * bearing_rad,
            self.yaw_lower_limit,
            self.yaw_upper_limit,
        )

    def next_ruckig_step(
        self, target_positions: list[float]
    ) -> tuple[list[float], list[float]]:
        if (
            self.ruckig_controller is None
            or self.ruckig_input is None
            or self.ruckig_output is None
            or not self.ruckig_initialized
        ):
            return self.next_head_step(target_positions)

        target_positions = self.clamp_head_positions(target_positions)
        self.ruckig_input.target_position = target_positions
        self.ruckig_input.target_velocity = [0.0] * len(HEAD_JOINTS)
        self.ruckig_input.target_acceleration = [0.0] * len(HEAD_JOINTS)
        self.ruckig_input.max_velocity = self.ruckig_max_velocity()
        self.ruckig_input.max_acceleration = self.ruckig_max_acceleration()
        self.ruckig_input.max_jerk = self.ruckig_max_jerk()

        result = self.ruckig_controller.update(
            self.ruckig_input, self.ruckig_output
        )
        if result not in [ruckig.Result.Working, ruckig.Result.Finished]:
            now = time.monotonic()
            if now - self.last_ruckig_warning > 3.0:
                self.last_ruckig_warning = now
                self.get_logger().warn(
                    "Ruckig head tracker step failed; using velocity-limited fallback."
                )
            return self.next_head_step(target_positions)

        positions = self.clamp_head_positions(list(self.ruckig_output.new_position))
        velocities = list(self.ruckig_output.new_velocity)
        self.ruckig_input.current_position = positions
        self.ruckig_input.current_velocity = velocities
        self.ruckig_input.current_acceleration = list(
            self.ruckig_output.new_acceleration
        )
        return positions, velocities

    def next_head_step(
        self, target_positions: list[float]
    ) -> tuple[list[float], list[float]]:
        start_positions = self.command_positions or self.head_positions
        if start_positions is None:
            return self.clamp_head_positions(target_positions), [0.0] * len(
                HEAD_JOINTS
            )

        max_delta = self.max_yaw_velocity * self.control_period_sec
        positions = list(start_positions)
        positions[HEAD_YAW_INDEX] = move_toward(
            start_positions[HEAD_YAW_INDEX],
            target_positions[HEAD_YAW_INDEX],
            max_delta,
        )
        positions[HEAD_PITCH_INDEX] = start_positions[HEAD_PITCH_INDEX]
        positions = self.clamp_head_positions(positions)
        velocities = [
            (positions[i] - start_positions[i]) / self.control_period_sec
            for i in range(len(HEAD_JOINTS))
        ]
        return positions, velocities

    def ruckig_max_velocity(self) -> list[float]:
        return [self.max_yaw_velocity, self.max_yaw_velocity]

    def ruckig_max_acceleration(self) -> list[float]:
        return [self.max_yaw_acceleration, self.max_yaw_acceleration]

    def ruckig_max_jerk(self) -> list[float]:
        return [self.max_yaw_jerk, self.max_yaw_jerk]

    def publish_head_command(
        self, positions: list[float], velocities: Optional[list[float]] = None
    ) -> None:
        positions = self.clamp_head_positions(positions)
        if velocities is None:
            velocities = [0.0] * len(HEAD_JOINTS)
        if self.dry_run:
            now = time.monotonic()
            if now - self.last_dry_run_log > 1.0:
                self.last_dry_run_log = now
                bearing = self.last_target_bearing_rad
                bearing_deg = (
                    math.degrees(bearing) if bearing is not None else float("nan")
                )
                self.get_logger().info(
                    "dry_run head command: "
                    f"yaw={positions[HEAD_YAW_INDEX]:+.3f}rad, "
                    f"pitch={positions[HEAD_PITCH_INDEX]:+.3f}rad, "
                    f"bearing={bearing_deg:+.1f}deg"
                )
            return

        self.command_pub.publish(self.make_head_command(positions, velocities))

    def make_head_command(
        self, positions: list[float], velocities: list[float]
    ) -> JointCommandArray:
        cmd = JointCommandArray()
        try:
            cmd.header = MessageHeader()
            cmd.header.stamp = self.get_clock().now().to_msg()
        except Exception:
            pass

        positions = self.clamp_head_positions(positions)
        for i, joint_info in enumerate(HEAD_JOINTS):
            joint = JointCommand()
            joint.name = joint_info.name
            joint.position = positions[i]
            joint.velocity = float(velocities[i])
            joint.effort = 0.0
            joint.stiffness = joint_info.kp
            joint.damping = joint_info.kd
            cmd.joints.append(joint)

        return cmd

    def clamp_head_positions(self, positions: list[float]) -> list[float]:
        clamped = []
        for i, joint in enumerate(HEAD_JOINTS):
            lower_limit = joint.lower_limit
            upper_limit = joint.upper_limit
            if i == HEAD_YAW_INDEX:
                lower_limit = self.yaw_lower_limit
                upper_limit = self.yaw_upper_limit
            clamped.append(clamp(positions[i], lower_limit, upper_limit))
        return clamped


def main(args=None):
    rclpy.init(args=args)
    node = X2HeadYawTracker()
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
