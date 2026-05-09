#!/usr/bin/env python3
"""Move to a point beside a detected person, then raise both arms to 90 degrees.

Input convention:
  - distance is meters from the robot to the person/target
  - angle is degrees from straight ahead, positive to the robot's left

This is open-loop: it converts the requested coordinate into turn/step/turn
velocity commands. Later, face recognition can pass the same distance and angle
through --distance-m and --angle-deg instead of using the interactive prompts.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray
from aimdk_msgs.msg import McLocomotionVelocity, MessageHeader
from aimdk_msgs.srv import SetMcInputSource

try:
    import ruckig
except ImportError:
    ruckig = None


SOURCE_NAME = "coordinate_offset_arms"

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


@dataclass(frozen=True)
class TravelPlan:
    target_x_m: float
    target_y_m: float
    goal_x_m: float
    goal_y_m: float
    goal_distance_m: float
    turn_to_goal_rad: float
    final_turn_rad: float
    side: str
    side_offset_m: float


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


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def plan_offset_goal(
    distance_m: float,
    angle_deg: float,
    side: str,
    side_offset_m: float,
    face_target: bool,
) -> TravelPlan:
    """Convert person polar coordinates into an offset waypoint plan."""
    angle_rad = math.radians(angle_deg)
    target_x = distance_m * math.cos(angle_rad)
    target_y = distance_m * math.sin(angle_rad)

    side_sign = 1.0 if side == "left" else -1.0
    left_normal_x = -math.sin(angle_rad)
    left_normal_y = math.cos(angle_rad)
    goal_x = target_x + side_sign * side_offset_m * left_normal_x
    goal_y = target_y + side_sign * side_offset_m * left_normal_y

    goal_distance = math.hypot(goal_x, goal_y)
    turn_to_goal = math.atan2(goal_y, goal_x) if goal_distance > 0.001 else 0.0

    if face_target:
        final_heading = math.atan2(target_y - goal_y, target_x - goal_x)
        final_turn = normalize_angle(final_heading - turn_to_goal)
    else:
        final_turn = 0.0

    return TravelPlan(
        target_x_m=target_x,
        target_y_m=target_y,
        goal_x_m=goal_x,
        goal_y_m=goal_y,
        goal_distance_m=goal_distance,
        turn_to_goal_rad=turn_to_goal,
        final_turn_rad=final_turn,
        side=side,
        side_offset_m=side_offset_m,
    )


class CoordinateOffsetRaiseArms(Node):
    def __init__(self, control_hz: float):
        super().__init__("x2_coordinate_offset_raise_arms")
        self.control_hz = control_hz
        self.period = 1.0 / control_hz

        self.velocity_pub = self.create_publisher(
            McLocomotionVelocity, "/aima/mc/locomotion/velocity", 10
        )
        self.arm_pub = self.create_publisher(
            JointCommandArray, "/aima/hal/joint/arm/command", PUBLISHER_QOS
        )
        self.input_source_client = self.create_client(
            SetMcInputSource, "/aimdk_5Fmsgs/srv/SetMcInputSource"
        )
        self.create_subscription(
            JointStateArray,
            "/aima/hal/joint/arm/state",
            self.arm_state_callback,
            SUBSCRIBER_QOS,
        )

        self.arm_positions: Optional[List[float]] = None
        self.arm_velocities: Optional[List[float]] = None

    def arm_state_callback(self, msg: JointStateArray) -> None:
        by_name = {joint.name: joint for joint in msg.joints}
        if all(joint.name in by_name for joint in ARM_JOINTS):
            self.arm_positions = [by_name[joint.name].position for joint in ARM_JOINTS]
            self.arm_velocities = [
                getattr(by_name[joint.name], "velocity", 0.0)
                for joint in ARM_JOINTS
            ]
            return

        if len(msg.joints) >= len(ARM_JOINTS):
            self.arm_positions = [
                msg.joints[i].position for i in range(len(ARM_JOINTS))
            ]
            self.arm_velocities = [
                getattr(msg.joints[i], "velocity", 0.0)
                for i in range(len(ARM_JOINTS))
            ]

    def register_input_source(self) -> bool:
        self.get_logger().info("Registering locomotion input source...")
        timeout_sec = 8.0
        start = self.get_clock().now().nanoseconds / 1e9

        while not self.input_source_client.wait_for_service(timeout_sec=2.0):
            now = self.get_clock().now().nanoseconds / 1e9
            if now - start > timeout_sec:
                self.get_logger().error("Waiting for input source service timed out.")
                return False
            self.get_logger().info("Waiting for input source service...")

        request = SetMcInputSource.Request()
        request.action.value = 1001
        request.input_source.name = SOURCE_NAME
        request.input_source.priority = 40
        request.input_source.timeout = 1000

        future = None
        for attempt in range(8):
            request.request.header.stamp = self.get_clock().now().to_msg()
            future = self.input_source_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.25)
            if future.done():
                break
            self.get_logger().info(
                f"Trying to register input source... [{attempt + 1}/8]"
            )

        if future is None or not future.done():
            self.get_logger().error("Input source registration failed or timed out.")
            return False

        try:
            response = future.result()
            state = response.response.state.value
            self.get_logger().info(
                f"Input source registered: state={state}, "
                f"task_id={response.response.task_id}"
            )
            return True
        except Exception as exc:
            self.get_logger().error(f"Input source registration exception: {exc}")
            return False

    def publish_velocity(
        self,
        forward_velocity: float,
        lateral_velocity: float = 0.0,
        angular_velocity: float = 0.0,
    ) -> None:
        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = SOURCE_NAME
        msg.forward_velocity = float(forward_velocity)
        msg.lateral_velocity = float(lateral_velocity)
        msg.angular_velocity = float(angular_velocity)
        self.velocity_pub.publish(msg)

    def hold_velocity(
        self,
        seconds: float,
        forward_velocity: float = 0.0,
        lateral_velocity: float = 0.0,
        angular_velocity: float = 0.0,
    ) -> None:
        end_time = time.monotonic() + max(0.0, seconds)
        while rclpy.ok() and time.monotonic() < end_time:
            self.publish_velocity(
                forward_velocity=forward_velocity,
                lateral_velocity=lateral_velocity,
                angular_velocity=angular_velocity,
            )
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(self.period)

    def stop_velocity(self, seconds: float = 0.5) -> None:
        self.hold_velocity(seconds)

    def rotate_by(
        self,
        angle_rad: float,
        angular_speed: float,
        invert_turn_direction: bool,
    ) -> None:
        angle_rad = normalize_angle(angle_rad)
        if abs(angle_rad) < math.radians(1.0):
            self.get_logger().info("Turn skipped; angle is already close.")
            return

        direction = 1.0 if angle_rad > 0.0 else -1.0
        if invert_turn_direction:
            direction *= -1.0

        duration = abs(angle_rad) / angular_speed
        self.get_logger().info(
            f"Turning {math.degrees(angle_rad):+.1f} deg "
            f"for {duration:.2f} s at {angular_speed:.2f} rad/s."
        )
        self.hold_velocity(duration, angular_velocity=direction * angular_speed)
        self.stop_velocity(0.4)

    def walk_forward_in_step_pulses(
        self,
        distance_m: float,
        forward_speed: float,
        step_length_m: float,
        step_pause_sec: float,
    ) -> None:
        remaining = max(0.0, distance_m)
        step_index = 1
        estimated_steps = max(1, math.ceil(remaining / step_length_m))
        self.get_logger().info(
            f"Walking {distance_m:.2f} m as about {estimated_steps} step pulses."
        )

        while rclpy.ok() and remaining > 0.01:
            segment = min(step_length_m, remaining)
            duration = segment / forward_speed
            self.get_logger().info(
                f"Step pulse {step_index}/{estimated_steps}: "
                f"{segment:.2f} m for {duration:.2f} s."
            )
            self.hold_velocity(duration, forward_velocity=forward_speed)
            self.stop_velocity(step_pause_sec)
            remaining -= segment
            step_index += 1

        self.stop_velocity(0.75)

    def wait_for_arm_state(self, timeout_sec: float) -> bool:
        self.get_logger().info("Waiting for arm joint state...")
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.arm_positions is not None:
                self.get_logger().info("Arm joint state received.")
                return True
            rclpy.spin_once(self, timeout_sec=0.05)

        self.get_logger().error("Timed out waiting for arm joint state.")
        return False

    def arm_target_from_current(self, arm_angle_deg: float) -> List[float]:
        if self.arm_positions is None:
            raise RuntimeError("Arm state is unavailable.")

        target_by_name = {
            "left_shoulder_pitch_joint": -math.radians(arm_angle_deg),
            "left_shoulder_roll_joint": 0.25,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.20,
            "left_wrist_yaw_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": -math.radians(arm_angle_deg),
            "right_shoulder_roll_joint": -0.25,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.20,
            "right_wrist_yaw_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
        }

        target = list(self.arm_positions)
        for index, joint in enumerate(ARM_JOINTS):
            if joint.name in target_by_name:
                target[index] = clamp(
                    target_by_name[joint.name],
                    joint.lower_limit,
                    joint.upper_limit,
                )
        return target

    def make_arm_command(
        self,
        positions: List[float],
        velocities: Optional[List[float]] = None,
    ) -> JointCommandArray:
        cmd = JointCommandArray()
        cmd.header = MessageHeader()
        cmd.header.stamp = self.get_clock().now().to_msg()
        if velocities is None:
            velocities = [0.0] * len(ARM_JOINTS)

        for index, joint_info in enumerate(ARM_JOINTS):
            joint = JointCommand()
            joint.name = joint_info.name
            joint.position = clamp(
                positions[index],
                joint_info.lower_limit,
                joint_info.upper_limit,
            )
            joint.velocity = velocities[index]
            joint.effort = 0.0
            joint.stiffness = joint_info.kp
            joint.damping = joint_info.kd
            cmd.joints.append(joint)

        return cmd

    def publish_arm_pose(
        self,
        positions: List[float],
        velocities: Optional[List[float]] = None,
    ) -> None:
        self.arm_pub.publish(self.make_arm_command(positions, velocities))

    def raise_arms_to_angle(
        self,
        arm_angle_deg: float,
        move_seconds: float,
        arm_control_hz: float,
        hold_seconds: Optional[float],
    ) -> None:
        if self.arm_positions is None:
            raise RuntimeError("Arm state is unavailable.")

        target = self.arm_target_from_current(arm_angle_deg)
        self.get_logger().info(
            f"Raising both arms to {arm_angle_deg:.1f} deg shoulder pitch."
        )

        if ruckig is None:
            self.get_logger().warning(
                "ruckig Python package not found; using timed cosine fallback. "
                "Install ruckig on the robot for smoother arm motion."
            )
            self.raise_arms_with_fallback(
                target=target,
                move_seconds=move_seconds,
                arm_control_hz=arm_control_hz,
            )
        else:
            self.raise_arms_with_ruckig(
                target=target,
                arm_control_hz=arm_control_hz,
                minimum_duration=move_seconds,
            )

        self.hold_arm_pose(
            target=target,
            arm_control_hz=arm_control_hz,
            hold_seconds=hold_seconds,
        )

    def raise_arms_with_ruckig(
        self,
        target: List[float],
        arm_control_hz: float,
        minimum_duration: float,
    ) -> None:
        if self.arm_positions is None:
            raise RuntimeError("Arm state is unavailable.")
        if ruckig is None:
            raise RuntimeError("ruckig is unavailable.")

        dofs = len(ARM_JOINTS)
        period = 1.0 / arm_control_hz
        max_velocity = [1.0] * dofs
        current_velocity = list(self.arm_velocities or [0.0] * dofs)
        if len(current_velocity) != dofs:
            current_velocity = [0.0] * dofs
        current_velocity = [
            clamp(velocity, -limit, limit)
            for velocity, limit in zip(current_velocity, max_velocity)
        ]

        otg = ruckig.Ruckig(dofs, period)
        inp = ruckig.InputParameter(dofs)
        out = ruckig.OutputParameter(dofs)

        inp.current_position = [
            clamp(position, joint.lower_limit, joint.upper_limit)
            for position, joint in zip(self.arm_positions, ARM_JOINTS)
        ]
        inp.current_velocity = current_velocity
        inp.current_acceleration = [0.0] * dofs
        inp.target_position = target
        inp.target_velocity = [0.0] * dofs
        inp.target_acceleration = [0.0] * dofs
        inp.max_velocity = max_velocity
        inp.max_acceleration = [1.0] * dofs
        inp.max_jerk = [25.0] * dofs
        if minimum_duration > 0.0 and hasattr(inp, "minimum_duration"):
            inp.minimum_duration = minimum_duration

        self.get_logger().info(
            f"Publishing Ruckig arm trajectory at {arm_control_hz:.1f} Hz."
        )

        while rclpy.ok():
            result = otg.update(inp, out)
            if result not in [ruckig.Result.Working, ruckig.Result.Finished]:
                raise RuntimeError(f"Ruckig arm trajectory failed: {result}")

            positions = list(out.new_position)
            velocities = list(out.new_velocity)
            self.publish_arm_pose(positions, velocities)
            inp.current_position = positions
            inp.current_velocity = velocities
            inp.current_acceleration = list(out.new_acceleration)
            rclpy.spin_once(self, timeout_sec=0.0)

            if result == ruckig.Result.Finished:
                break
            time.sleep(period)

        self.publish_arm_pose(target)

    def raise_arms_with_fallback(
        self,
        target: List[float],
        move_seconds: float,
        arm_control_hz: float,
    ) -> None:
        if self.arm_positions is None:
            raise RuntimeError("Arm state is unavailable.")

        start = list(self.arm_positions)
        period = 1.0 / arm_control_hz
        start_time = time.monotonic()
        previous_positions = list(start)

        while rclpy.ok():
            elapsed = time.monotonic() - start_time
            alpha = smoothstep(elapsed / move_seconds)
            positions = [s + (t - s) * alpha for s, t in zip(start, target)]
            velocities = [
                (position - previous) / period
                for position, previous in zip(positions, previous_positions)
            ]
            self.publish_arm_pose(positions, velocities)
            previous_positions = list(positions)
            rclpy.spin_once(self, timeout_sec=0.0)

            if elapsed >= move_seconds:
                break
            time.sleep(period)

        self.publish_arm_pose(target)

    def hold_arm_pose(
        self,
        target: List[float],
        arm_control_hz: float,
        hold_seconds: Optional[float],
    ) -> None:
        period = 1.0 / arm_control_hz

        if hold_seconds is None:
            self.get_logger().info(
                "Holding arm pose until this node is interrupted."
            )
            while rclpy.ok():
                self.publish_arm_pose(target)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(period)
            return

        if hold_seconds <= 0.0:
            self.get_logger().info("Arm target reached.")
            return

        self.get_logger().info(f"Holding arm pose for {hold_seconds:.1f} s.")
        hold_until = time.monotonic() + hold_seconds
        while rclpy.ok() and time.monotonic() < hold_until:
            self.publish_arm_pose(target)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

    def run_travel_plan(self, plan: TravelPlan, args: argparse.Namespace) -> None:
        self.get_logger().info(format_plan(plan, args.step_length_m))

        self.rotate_by(
            plan.turn_to_goal_rad,
            args.turn_speed,
            args.invert_turn_direction,
        )
        self.walk_forward_in_step_pulses(
            plan.goal_distance_m,
            args.forward_speed,
            args.step_length_m,
            args.step_pause_sec,
        )
        if args.face_target:
            self.rotate_by(
                plan.final_turn_rad,
                args.turn_speed,
                args.invert_turn_direction,
            )
        self.stop_velocity(args.settle_seconds)


def format_plan(plan: TravelPlan, step_length_m: float) -> str:
    estimated_steps = max(1, math.ceil(plan.goal_distance_m / step_length_m))
    return (
        "Plan: "
        f"person=({plan.target_x_m:.2f}m forward, {plan.target_y_m:.2f}m left), "
        f"goal={plan.side_offset_m:.2f}m {plan.side} of person "
        f"at ({plan.goal_x_m:.2f}m forward, {plan.goal_y_m:.2f}m left), "
        f"turn={math.degrees(plan.turn_to_goal_rad):+.1f}deg, "
        f"walk={plan.goal_distance_m:.2f}m in ~{estimated_steps} steps, "
        f"final_turn={math.degrees(plan.final_turn_rad):+.1f}deg"
    )


def prompt_float(label: str, minimum: Optional[float] = None) -> float:
    while True:
        raw_value = input(f"{label}: ").strip()
        try:
            value = float(raw_value)
        except ValueError:
            print("Please enter a number.")
            continue

        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move to a coordinate 50cm beside a target, then raise both arms."
        )
    )
    parser.add_argument(
        "positional_distance_m",
        nargs="?",
        type=float,
        help="Target distance in meters. If omitted, the script prompts.",
    )
    parser.add_argument(
        "positional_angle_deg",
        nargs="?",
        type=float,
        help="Target angle in degrees, positive left. If omitted, prompts.",
    )
    parser.add_argument(
        "--distance-m",
        type=float,
        default=None,
        help="Target distance in meters; face recognition can fill this.",
    )
    parser.add_argument(
        "--angle-deg",
        type=float,
        default=None,
        help="Target angle in degrees, positive left; face recognition can fill this.",
    )
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--side-offset-m", type=float, default=0.50)
    parser.add_argument("--forward-speed", type=float, default=0.20)
    parser.add_argument("--turn-speed", type=float, default=0.35)
    parser.add_argument("--step-length-m", type=float, default=0.25)
    parser.add_argument("--step-pause-sec", type=float, default=0.25)
    parser.add_argument("--settle-seconds", type=float, default=0.75)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--arm-angle-deg", type=float, default=90.0)
    parser.add_argument(
        "--arm-control-hz",
        type=float,
        default=500.0,
        help="HAL arm command frequency; 500 Hz matches the 2 ms Ruckig example.",
    )
    parser.add_argument(
        "--arm-move-seconds",
        type=float,
        default=4.0,
        help="Minimum Ruckig duration when supported; fallback move duration.",
    )
    parser.add_argument(
        "--arm-hold-seconds",
        type=float,
        default=None,
        help=(
            "Seconds to keep publishing the raised-arm pose. "
            "Omit to hold until interrupted."
        ),
    )
    parser.add_argument(
        "--no-face-target",
        dest="face_target",
        action="store_false",
        help="Skip the final turn toward the original target coordinate.",
    )
    parser.set_defaults(face_target=True)
    parser.add_argument(
        "--skip-arms",
        action="store_true",
        help="Only move to the offset coordinate; do not raise arms.",
    )
    parser.add_argument(
        "--no-arm-prompt",
        action="store_true",
        help="Do not pause before publishing low-level HAL arm commands.",
    )
    parser.add_argument(
        "--invert-turn-direction",
        action="store_true",
        help="Use this if positive angular velocity turns the robot the wrong way.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the computed plan without connecting to ROS or moving.",
    )
    return parser.parse_args(argv)


def resolve_distance_and_angle(args: argparse.Namespace) -> tuple[float, float]:
    distance = args.distance_m
    if distance is None:
        distance = args.positional_distance_m

    angle = args.angle_deg
    if angle is None:
        angle = args.positional_angle_deg

    if distance is None:
        distance = prompt_float("Distance to target in meters", minimum=0.0)
    if angle is None:
        angle = prompt_float("Angle to target in degrees, positive left")

    if distance < 0.0:
        raise ValueError("Distance must be >= 0.")

    return distance, angle


def validate_args(args: argparse.Namespace) -> None:
    checks = [
        ("side-offset-m", args.side_offset_m),
        ("forward-speed", args.forward_speed),
        ("turn-speed", args.turn_speed),
        ("step-length-m", args.step_length_m),
        ("control-hz", args.control_hz),
        ("arm-control-hz", args.arm_control_hz),
        ("arm-move-seconds", args.arm_move_seconds),
    ]
    for name, value in checks:
        if value <= 0.0:
            raise ValueError(f"--{name} must be > 0.")

    if args.step_pause_sec < 0.0:
        raise ValueError("--step-pause-sec must be >= 0.")
    if args.settle_seconds < 0.0:
        raise ValueError("--settle-seconds must be >= 0.")
    if args.arm_hold_seconds is not None and args.arm_hold_seconds < 0.0:
        raise ValueError("--arm-hold-seconds must be >= 0.")


def main(args=None) -> int:
    parsed = parse_args(sys.argv[1:] if args is None else args)
    validate_args(parsed)
    distance_m, angle_deg = resolve_distance_and_angle(parsed)
    plan = plan_offset_goal(
        distance_m=distance_m,
        angle_deg=angle_deg,
        side=parsed.side,
        side_offset_m=parsed.side_offset_m,
        face_target=parsed.face_target,
    )

    if parsed.dry_run:
        print(format_plan(plan, parsed.step_length_m))
        return 0

    rclpy.init()
    node = CoordinateOffsetRaiseArms(control_hz=parsed.control_hz)

    def signal_handler(sig, _frame):
        node.get_logger().info(f"Received signal {sig}; stopping velocity.")
        node.stop_velocity(seconds=0.5)
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        if not node.register_input_source():
            return 2

        node.run_travel_plan(plan, parsed)

        if parsed.skip_arms:
            node.get_logger().info("Movement complete; arm raise skipped.")
            return 0

        if not parsed.no_arm_prompt:
            node.get_logger().warning(
                "Next step publishes low-level HAL arm joint commands. "
                "Only continue when the robot is stable and your HAL safety "
                "procedure is satisfied. Press Enter to continue."
            )
            input()

        if not node.wait_for_arm_state(timeout_sec=5.0):
            return 3

        node.raise_arms_to_angle(
            arm_angle_deg=parsed.arm_angle_deg,
            move_seconds=parsed.arm_move_seconds,
            arm_control_hz=parsed.arm_control_hz,
            hold_seconds=parsed.arm_hold_seconds,
        )
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        rclpy.logging.get_logger("main").fatal(
            f"Program exited with exception: {exc}"
        )
        return 1
    finally:
        try:
            node.stop_velocity(seconds=0.5)
            node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
