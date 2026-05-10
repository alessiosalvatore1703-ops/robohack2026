#!/usr/bin/env python3
"""Standalone HAL arm pose test for the chair-assist demo.

This node intentionally does not publish locomotion velocity, does not request
motion-controller modes, and does not command waist or torso joints.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

try:
    from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray
    from aimdk_msgs.msg import MessageHeader

    AIMDK_AVAILABLE = True
except ImportError:
    AIMDK_AVAILABLE = False
    JointCommand = JointCommandArray = JointStateArray = MessageHeader = None

try:
    import ruckig
except ImportError:
    ruckig = None


SUBSCRIBER_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    durability=DurabilityPolicy.VOLATILE,
)

PUBLISHER_QOS = QoSProfile(
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


ARM_JOINTS: List[JointInfo] = [
    JointInfo("left_shoulder_pitch_joint", -3.08, 2.04, 20.0, 2.0),
    JointInfo("left_shoulder_roll_joint", -0.061, 2.993, 20.0, 2.0),
    JointInfo("left_shoulder_yaw_joint", -2.556, 2.556, 20.0, 2.0),
    JointInfo("left_elbow_joint", -2.3556, 0.0, 20.0, 2.0),
    JointInfo("left_wrist_yaw_joint", -2.556, 2.556, 20.0, 2.0),
    JointInfo("left_wrist_pitch_joint", -0.558, 0.558, 20.0, 2.0),
    JointInfo("left_wrist_roll_joint", -1.571, 0.724, 20.0, 2.0),
    JointInfo("right_shoulder_pitch_joint", -3.08, 2.04, 20.0, 2.0),
    JointInfo("right_shoulder_roll_joint", -2.993, 0.061, 20.0, 2.0),
    JointInfo("right_shoulder_yaw_joint", -2.556, 2.556, 20.0, 2.0),
    JointInfo("right_elbow_joint", -2.3556, 0.0, 20.0, 2.0),
    JointInfo("right_wrist_yaw_joint", -2.556, 2.556, 20.0, 2.0),
    JointInfo("right_wrist_pitch_joint", -0.558, 0.558, 20.0, 2.0),
    JointInfo("right_wrist_roll_joint", -0.724, 1.571, 20.0, 2.0),
]


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def smoothstep(alpha: float) -> float:
    alpha = clamp(alpha, 0.0, 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * alpha)


class ArmAssistPose(Node):
    """Move both arms to an assist-ready pose using shaped HAL commands."""

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("x2_arm_assist_pose")
        if not AIMDK_AVAILABLE:
            self.get_logger().fatal("aimdk_msgs is not available in this environment.")
            raise RuntimeError("aimdk_msgs is not available")

        self.args = args
        self.period_sec = 1.0 / max(args.control_hz, 1.0)
        self.arm_positions: Optional[List[float]] = None
        self.arm_velocities: Optional[List[float]] = None
        self.warned_unnamed_state = False

        self.arm_pub = self.create_publisher(
            JointCommandArray, args.arm_command_topic, PUBLISHER_QOS
        )
        self.create_subscription(
            JointStateArray,
            args.arm_state_topic,
            self.arm_state_callback,
            SUBSCRIBER_QOS,
        )

    def arm_state_callback(self, msg: JointStateArray) -> None:
        by_name = {
            joint.name: joint
            for joint in msg.joints
            if getattr(joint, "name", "")
        }
        if all(joint.name in by_name for joint in ARM_JOINTS):
            self.arm_positions = [
                float(by_name[joint.name].position) for joint in ARM_JOINTS
            ]
            self.arm_velocities = [
                float(getattr(by_name[joint.name], "velocity", 0.0))
                for joint in ARM_JOINTS
            ]
            return

        if len(msg.joints) < len(ARM_JOINTS):
            return

        if not self.warned_unnamed_state:
            self.warned_unnamed_state = True
            self.get_logger().warn(
                "Arm state does not contain all expected joint names; using "
                "the SDK model order fallback."
            )
        self.arm_positions = [
            float(msg.joints[i].position) for i in range(len(ARM_JOINTS))
        ]
        self.arm_velocities = [
            float(getattr(msg.joints[i], "velocity", 0.0))
            for i in range(len(ARM_JOINTS))
        ]

    def wait_for_arm_state(self, timeout_sec: float) -> bool:
        self.get_logger().info(f"Waiting for arm joint state on {self.args.arm_state_topic}")
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.arm_positions is not None:
                self.get_logger().info("Arm joint state received.")
                return True
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().error("Timed out waiting for arm joint state.")
        return False

    def assist_ready_target(self) -> List[float]:
        if self.arm_positions is None:
            raise RuntimeError("Arm state is unavailable.")

        angle_rad = math.radians(self.args.arm_angle_deg)
        roll = self.args.shoulder_roll_rad
        elbow = self.args.elbow_angle_rad
        target_by_name: Dict[str, float] = {
            "left_shoulder_pitch_joint": -angle_rad,
            "left_shoulder_roll_joint": roll,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": elbow,
            "left_wrist_yaw_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": -angle_rad,
            "right_shoulder_roll_joint": -roll,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": elbow,
            "right_wrist_yaw_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
        }

        target = list(self.arm_positions)
        for index, joint in enumerate(ARM_JOINTS):
            target[index] = clamp(
                target_by_name[joint.name],
                joint.lower_limit,
                joint.upper_limit,
            )
        return target

    def make_arm_command(
        self, positions: List[float], velocities: Optional[List[float]] = None
    ) -> JointCommandArray:
        if velocities is None:
            velocities = [0.0] * len(ARM_JOINTS)

        cmd = JointCommandArray()
        cmd.header = MessageHeader()
        cmd.header.stamp = self.get_clock().now().to_msg()
        for index, joint_info in enumerate(ARM_JOINTS):
            joint = JointCommand()
            joint.name = joint_info.name
            joint.position = clamp(
                positions[index],
                joint_info.lower_limit,
                joint_info.upper_limit,
            )
            joint.velocity = float(velocities[index])
            joint.effort = 0.0
            joint.stiffness = joint_info.kp
            joint.damping = joint_info.kd
            cmd.joints.append(joint)
        return cmd

    def publish_arm_pose(
        self, positions: List[float], velocities: Optional[List[float]] = None
    ) -> None:
        self.arm_pub.publish(self.make_arm_command(positions, velocities))

    def print_plan(self, target: List[float]) -> None:
        current = self.arm_positions or [0.0] * len(ARM_JOINTS)
        print("Assist-ready arm pose plan:")
        print(f"  command_topic: {self.args.arm_command_topic}")
        print(f"  move_seconds: {self.args.move_seconds:.2f}")
        print(f"  hold_seconds: {self.args.hold_seconds:.2f}")
        print(f"  control_hz: {self.args.control_hz:.1f}")
        print("  joints:")
        for joint, current_pos, target_pos in zip(ARM_JOINTS, current, target):
            print(
                f"    {joint.name}: current={current_pos:+.3f} rad, "
                f"target={target_pos:+.3f} rad"
            )

    def move_with_ruckig(self, target: List[float]) -> None:
        if ruckig is None:
            self.get_logger().warn(
                "ruckig Python package not found; using cosine interpolation fallback."
            )
            self.move_with_fallback(target)
            return
        if self.arm_positions is None:
            raise RuntimeError("Arm state is unavailable.")

        dofs = len(ARM_JOINTS)
        otg = ruckig.Ruckig(dofs, self.period_sec)
        inp = ruckig.InputParameter(dofs)
        out = ruckig.OutputParameter(dofs)

        inp.current_position = list(self.arm_positions)
        inp.current_velocity = list(self.arm_velocities or [0.0] * dofs)
        inp.current_acceleration = [0.0] * dofs
        inp.target_position = target
        inp.target_velocity = [0.0] * dofs
        inp.target_acceleration = [0.0] * dofs
        inp.max_velocity = [self.args.max_velocity] * dofs
        inp.max_acceleration = [self.args.max_acceleration] * dofs
        inp.max_jerk = [self.args.max_jerk] * dofs

        self.get_logger().info("Moving arms with Ruckig trajectory.")
        while rclpy.ok():
            result = otg.update(inp, out)
            if result not in (ruckig.Result.Working, ruckig.Result.Finished):
                raise RuntimeError(f"Ruckig trajectory generation failed: {result}")

            self.publish_arm_pose(out.new_position, out.new_velocity)
            inp.current_position = list(out.new_position)
            inp.current_velocity = list(out.new_velocity)
            inp.current_acceleration = list(out.new_acceleration)
            rclpy.spin_once(self, timeout_sec=0.0)

            if result == ruckig.Result.Finished:
                break
            time.sleep(self.period_sec)

    def move_with_fallback(self, target: List[float]) -> None:
        if self.arm_positions is None:
            raise RuntimeError("Arm state is unavailable.")

        start = list(self.arm_positions)
        start_time = time.monotonic()
        while rclpy.ok():
            elapsed = time.monotonic() - start_time
            alpha = smoothstep(elapsed / max(self.args.move_seconds, 0.1))
            positions = [
                s + (t - s) * alpha
                for s, t in zip(start, target)
            ]
            self.publish_arm_pose(positions)
            rclpy.spin_once(self, timeout_sec=0.0)

            if elapsed >= self.args.move_seconds:
                break
            time.sleep(self.period_sec)

    def hold_pose(self, target: List[float]) -> None:
        if self.args.hold_seconds <= 0.0:
            self.get_logger().info("Arm target reached; not holding with repeated commands.")
            return

        self.get_logger().info(f"Holding arm pose for {self.args.hold_seconds:.2f}s.")
        deadline = time.monotonic() + self.args.hold_seconds
        while rclpy.ok() and time.monotonic() < deadline:
            self.publish_arm_pose(target)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(self.period_sec)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move both X2 arms to a cautious assist-ready HAL pose."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the active-mode safety prompt.",
    )
    parser.add_argument(
        "--arm-state-topic",
        default="/aima/hal/joint/arm/state",
    )
    parser.add_argument(
        "--arm-command-topic",
        default="/aima/hal/joint/arm/command",
    )
    parser.add_argument("--arm-angle-deg", type=float, default=70.0)
    parser.add_argument("--shoulder-roll-rad", type=float, default=0.25)
    parser.add_argument("--elbow-angle-rad", type=float, default=-0.35)
    parser.add_argument("--move-seconds", type=float, default=5.0)
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    parser.add_argument("--control-hz", type=float, default=200.0)
    parser.add_argument("--max-velocity", type=float, default=0.5)
    parser.add_argument("--max-acceleration", type=float, default=0.8)
    parser.add_argument("--max-jerk", type=float, default=15.0)
    parser.add_argument("--state-timeout-sec", type=float, default=5.0)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    positive = [
        ("move-seconds", args.move_seconds),
        ("control-hz", args.control_hz),
        ("max-velocity", args.max_velocity),
        ("max-acceleration", args.max_acceleration),
        ("max-jerk", args.max_jerk),
        ("state-timeout-sec", args.state_timeout_sec),
    ]
    for name, value in positive:
        if value <= 0.0:
            raise ValueError(f"--{name} must be > 0")
    if args.hold_seconds < 0.0:
        raise ValueError("--hold-seconds must be >= 0")
    if args.arm_angle_deg < 0.0:
        raise ValueError("--arm-angle-deg must be >= 0")


def main(args=None) -> int:
    parsed = parse_args(sys.argv[1:] if args is None else args)
    validate_args(parsed)

    rclpy.init()
    node = ArmAssistPose(parsed)

    def signal_handler(sig, _frame):
        node.get_logger().info(f"Received signal {sig}; shutting down arm pose node.")
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if not node.wait_for_arm_state(parsed.state_timeout_sec):
            return 2

        target = node.assist_ready_target()
        node.print_plan(target)

        if parsed.dry_run:
            node.get_logger().info("Dry-run complete; no arm command published.")
            return 0

        if not parsed.yes:
            node.get_logger().warning(
                "This will publish low-level HAL arm commands only. Confirm the "
                "robot is stable, arms are clear, and no other node is commanding "
                "the arms. Press Enter to continue."
            )
            input()

        node.move_with_ruckig(target)
        node.hold_pose(target)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        rclpy.logging.get_logger("x2_arm_assist_pose").fatal(
            f"Program exited with exception: {exc}"
        )
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
