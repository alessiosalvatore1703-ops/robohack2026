#!/usr/bin/env python3
"""Rotate the X2 waist/torso toward a person and say "On my way".

This version follows the AimDK joint-control example pattern:
  - creates a dedicated waist joint controller
  - uses Ruckig to generate the waist trajectory
  - publishes JointCommandArray to /aima/hal/joint/waist/command

It does not publish locomotion velocity, so it should not step the legs.
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import rclpy
import rclpy.logging
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from aimdk_msgs.msg import JointCommand, JointCommandArray, JointStateArray
from aimdk_msgs.srv import PlayTts

try:
    import ruckig
except ImportError:
    ruckig = None


SOURCE_NAME = "turn_to_person_tts"

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


WAIST_JOINTS = [
    JointInfo("waist_yaw_joint", -3.43, 2.382, 20.0, 4.0),
    JointInfo("waist_pitch_joint", -0.314, 0.314, 20.0, 4.0),
    JointInfo("waist_roll_joint", -0.488, 0.488, 20.0, 4.0),
]
WAIST_YAW_INDEX = 0


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def prompt_float(label: str, default: Optional[float] = None) -> float:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw_value = input(f"{label}{suffix}: ").strip()
        if not raw_value and default is not None:
            return default

        try:
            return float(raw_value)
        except ValueError:
            print("Please enter a number.")


class PlayTtsClient(Node):
    def __init__(self) -> None:
        super().__init__("x2_turn_to_person_tts_speaker")
        self.client = self.create_client(PlayTts, "/aimdk_5Fmsgs/srv/PlayTts")

    def say(self, text: str) -> bool:
        self.get_logger().info("Waiting for TTS service...")
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("TTS service unavailable, waiting...")

        request = PlayTts.Request()
        request.tts_req.text = text
        request.tts_req.domain = SOURCE_NAME
        request.tts_req.trace_id = "turn_to_person"
        request.tts_req.is_interrupted = True
        request.tts_req.priority_weight = 0
        request.tts_req.priority_level.value = 6

        self.get_logger().info(f"Sending TTS: {text}")
        future = None
        for attempt in range(8):
            try:
                request.header.header.stamp = self.get_clock().now().to_msg()
            except Exception:
                pass

            future = self.client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=0.25)
            if future.done():
                break
            self.get_logger().info(f"Retrying TTS request... [{attempt + 1}/8]")

        if future is None or not future.done():
            self.get_logger().error("TTS service call timed out.")
            return False

        response = future.result()
        if response is None:
            self.get_logger().error("TTS service returned no response.")
            return False

        if response.tts_resp.is_success:
            self.get_logger().info("TTS sent successfully.")
            return True

        self.get_logger().error("TTS request failed.")
        return False


class WaistJointController(Node):
    def __init__(
        self,
        waist_state_topic: str,
        waist_command_topic: str,
        control_period_sec: float,
        waist_soft_limit_deg: float,
    ) -> None:
        super().__init__("x2_turn_to_person_tts_waist")
        if ruckig is None:
            raise ImportError(
                "ruckig is required for waist joint control. Install it on the "
                "robot with: python3 -m pip install ruckig"
            )

        self.joint_info = WAIST_JOINTS
        self.dofs = len(WAIST_JOINTS)
        self.control_period_sec = control_period_sec
        self.ruckig = ruckig.Ruckig(self.dofs, control_period_sec)
        self.input = ruckig.InputParameter(self.dofs)
        self.output = ruckig.OutputParameter(self.dofs)

        self.input.current_position = [0.0] * self.dofs
        self.input.current_velocity = [0.0] * self.dofs
        self.input.current_acceleration = [0.0] * self.dofs
        self.input.max_velocity = [1.0] * self.dofs
        self.input.max_acceleration = [1.0] * self.dofs
        self.input.max_jerk = [25.0] * self.dofs

        yaw_joint = WAIST_JOINTS[WAIST_YAW_INDEX]
        soft_limit_rad = math.radians(waist_soft_limit_deg)
        if soft_limit_rad > 0.0:
            self.waist_yaw_lower_limit = max(yaw_joint.lower_limit, -soft_limit_rad)
            self.waist_yaw_upper_limit = min(yaw_joint.upper_limit, soft_limit_rad)
        else:
            self.waist_yaw_lower_limit = yaw_joint.lower_limit
            self.waist_yaw_upper_limit = yaw_joint.upper_limit

        self.state_received = False
        self.sub = self.create_subscription(
            JointStateArray,
            waist_state_topic,
            self.joint_state_callback,
            SUBSCRIBER_QOS,
        )
        self.pub = self.create_publisher(
            JointCommandArray,
            waist_command_topic,
            PUBLISHER_QOS,
        )
        self.get_logger().info(
            f"Using waist state={waist_state_topic}, command={waist_command_topic}"
        )

    def joint_state_callback(self, msg: JointStateArray) -> None:
        positions = self.positions_from_state(msg)
        if positions is None:
            return

        if not self.state_received:
            self.input.current_position = positions
            self.input.current_velocity = self.velocities_from_state(msg)
            self.input.current_acceleration = [0.0] * self.dofs
            self.state_received = True

    def positions_from_state(self, msg: JointStateArray) -> Optional[list[float]]:
        by_name = {joint.name: joint for joint in msg.joints}
        if all(joint.name in by_name for joint in self.joint_info):
            return [by_name[joint.name].position for joint in self.joint_info]

        if len(msg.joints) >= self.dofs:
            return [msg.joints[i].position for i in range(self.dofs)]

        return None

    def velocities_from_state(self, msg: JointStateArray) -> list[float]:
        by_name = {joint.name: joint for joint in msg.joints}
        if all(joint.name in by_name for joint in self.joint_info):
            return [by_name[joint.name].velocity for joint in self.joint_info]

        if len(msg.joints) >= self.dofs:
            return [msg.joints[i].velocity for i in range(self.dofs)]

        return [0.0] * self.dofs

    def capture_state_briefly(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while rclpy.ok() and time.monotonic() < deadline:
            if self.state_received:
                return
            rclpy.spin_once(self, timeout_sec=0.02)

        self.get_logger().warn(
            "No waist state received before command; using zero waist pose as "
            "the Ruckig start state, matching the SDK joint-control example."
        )

    def set_waist_yaw_target(
        self,
        yaw_target_rad: float,
        relative_from_current: bool,
    ) -> None:
        target = list(self.input.current_position)
        if not self.state_received:
            target = [0.0] * self.dofs

        if relative_from_current:
            yaw_target_rad = target[WAIST_YAW_INDEX] + yaw_target_rad

        yaw_target_rad = clamp(
            yaw_target_rad,
            self.waist_yaw_lower_limit,
            self.waist_yaw_upper_limit,
        )
        target[WAIST_YAW_INDEX] = yaw_target_rad
        target[1] = clamp(target[1], self.joint_info[1].lower_limit, self.joint_info[1].upper_limit)
        target[2] = clamp(target[2], self.joint_info[2].lower_limit, self.joint_info[2].upper_limit)

        self.input.target_position = target
        self.input.target_velocity = [0.0] * self.dofs
        self.input.target_acceleration = [0.0] * self.dofs

        self.get_logger().info(
            f"Commanding waist_yaw_joint to {math.degrees(yaw_target_rad):+.1f} deg"
        )
        self.control_to_target(WAIST_YAW_INDEX)

    def control_to_target(self, joint_idx: int) -> None:
        tolerance = 1e-4
        last_publish = time.monotonic()

        while rclpy.ok():
            result = self.ruckig.update(self.input, self.output)
            if result not in [ruckig.Result.Working, ruckig.Result.Finished]:
                raise RuntimeError(f"Ruckig trajectory generation failed: {result}")

            self.input.current_position = list(self.output.new_position)
            self.input.current_velocity = list(self.output.new_velocity)
            self.input.current_acceleration = list(self.output.new_acceleration)

            self.publish_command(
                list(self.output.new_position),
                list(self.output.new_velocity),
            )
            rclpy.spin_once(self, timeout_sec=0.0)

            current = self.output.new_position[joint_idx]
            target = self.input.target_position[joint_idx]
            if result == ruckig.Result.Finished or abs(current - target) < tolerance:
                break

            elapsed = time.monotonic() - last_publish
            if elapsed < self.control_period_sec:
                time.sleep(self.control_period_sec - elapsed)
            last_publish = time.monotonic()

        self.publish_command(
            list(self.input.target_position),
            [0.0] * self.dofs,
        )

    def publish_command(self, positions: list[float], velocities: list[float]) -> None:
        cmd = JointCommandArray()
        for i, joint_info in enumerate(self.joint_info):
            joint = JointCommand()
            joint.name = joint_info.name
            joint.position = clamp(
                positions[i],
                self.limit_for_joint(i)[0],
                self.limit_for_joint(i)[1],
            )
            joint.velocity = velocities[i]
            joint.effort = 0.0
            joint.stiffness = joint_info.kp
            joint.damping = joint_info.kd
            cmd.joints.append(joint)

        self.pub.publish(cmd)

    def limit_for_joint(self, index: int) -> tuple[float, float]:
        joint = self.joint_info[index]
        if index == WAIST_YAW_INDEX:
            return self.waist_yaw_lower_limit, self.waist_yaw_upper_limit
        return joint.lower_limit, joint.upper_limit


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Rotate the torso toward a person and say "On my way".'
    )
    parser.add_argument(
        "positional_distance_m",
        nargs="?",
        type=float,
        help="Distance to person in meters. Stored/logged for future integration.",
    )
    parser.add_argument(
        "positional_angle_deg",
        nargs="?",
        type=float,
        help="Bearing to person in degrees, positive left.",
    )
    parser.add_argument("--distance-m", type=float, default=None)
    parser.add_argument("--angle-deg", type=float, default=None)
    parser.add_argument("--text", default="On my way")
    parser.add_argument("--waist-soft-limit-deg", type=float, default=90.0)
    parser.add_argument(
        "--waist-state-topic",
        default="/aima/hal/joint/waist/state",
    )
    parser.add_argument(
        "--waist-command-topic",
        default="/aima/hal/joint/waist/command",
    )
    parser.add_argument(
        "--relative-from-current",
        action="store_true",
        help="Add the input angle to the current waist yaw instead of using it as an absolute yaw target.",
    )
    parser.add_argument(
        "--invert-waist-direction",
        "--invert-turn-direction",
        dest="invert_waist_direction",
        action="store_true",
        help="Use this if positive angle turns the torso the wrong way.",
    )
    parser.add_argument("--control-period-sec", type=float, default=0.002)
    parser.add_argument("--state-wait-sec", type=float, default=0.5)
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def resolve_distance_and_angle(args: argparse.Namespace) -> tuple[float, float]:
    distance = args.distance_m
    if distance is None:
        distance = args.positional_distance_m

    angle = args.angle_deg
    if angle is None:
        angle = args.positional_angle_deg

    if angle is None:
        if distance is None:
            distance = prompt_float("Distance to person in meters", default=0.0)
        angle = prompt_float("Angle to person in degrees, positive left")
    elif distance is None:
        distance = 0.0

    if distance < 0.0:
        raise ValueError("Distance must be >= 0.")
    return distance, angle


def validate_args(args: argparse.Namespace) -> None:
    if args.control_period_sec <= 0.0:
        raise ValueError("--control-period-sec must be > 0.")
    if args.state_wait_sec < 0.0:
        raise ValueError("--state-wait-sec must be >= 0.")


def main(args=None) -> int:
    parsed = parse_args(sys.argv[1:] if args is None else args)
    validate_args(parsed)

    distance_m, angle_deg = resolve_distance_and_angle(parsed)
    yaw_target_rad = normalize_angle(math.radians(angle_deg))
    if parsed.invert_waist_direction:
        yaw_target_rad = -yaw_target_rad

    if parsed.dry_run:
        mode = "relative" if parsed.relative_from_current else "absolute"
        print(
            f"Plan: distance={distance_m:.2f}m, "
            f"waist_yaw_target={math.degrees(yaw_target_rad):+.1f}deg, "
            f"mode={mode}, text={parsed.text!r}"
        )
        return 0

    rclpy.init()
    speaker = PlayTtsClient()
    waist = WaistJointController(
        waist_state_topic=parsed.waist_state_topic,
        waist_command_topic=parsed.waist_command_topic,
        control_period_sec=parsed.control_period_sec,
        waist_soft_limit_deg=parsed.waist_soft_limit_deg,
    )

    def signal_handler(sig, _frame):
        waist.get_logger().info(f"Received signal {sig}; stopping.")
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        speaker.get_logger().info(
            f"Person input: distance={distance_m:.2f}m, angle={angle_deg:+.1f}deg"
        )
        if not parsed.skip_tts:
            speaker.say(parsed.text)

        waist.capture_state_briefly(parsed.state_wait_sec)
        waist.set_waist_yaw_target(
            yaw_target_rad=yaw_target_rad,
            relative_from_current=parsed.relative_from_current,
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
        speaker.destroy_node()
        waist.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
