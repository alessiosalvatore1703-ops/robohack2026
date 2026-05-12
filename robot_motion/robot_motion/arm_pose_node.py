#!/usr/bin/env python3
"""Hold both arms in a forearms-forward assist pose.

This is based on the arm pose routine from the `origin/marcos` branch, adjusted
for the chair-assist demo: the pose can be held indefinitely or for a finite
assist window, and the node can be triggered once from the stereo follow stop
state.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

try:
    from aimdk_msgs.msg import (
        CommonRequest,
        JointCommand,
        JointCommandArray,
        JointStateArray,
        MessageHeader,
    )
    from aimdk_msgs.srv import GetMcAction

    AIMDK_AVAILABLE = True
except ImportError:
    AIMDK_AVAILABLE = False
    CommonRequest = JointCommand = JointCommandArray = JointStateArray = MessageHeader = None
    GetMcAction = None


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

ARM_COMMAND_TOPIC = "/aima/hal/joint/arm/command"
ARM_STATE_TOPIC = "/aima/hal/joint/arm/state"
MC_ACTION_SERVICE = "/aimdk_5Fmsgs/srv/GetMcAction"
MC_ACTION_RUNNING = 100
BALANCED_MOTION_MODES = {"LOCOMOTION_DEFAULT", "STAND_DEFAULT"}


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


def bool_param(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


class RaiseArmsPose(Node):
    def __init__(self) -> None:
        super().__init__("arm_pose")
        if not AIMDK_AVAILABLE:
            self.get_logger().fatal("aimdk_msgs is not available in this environment.")
            raise RuntimeError("aimdk_msgs is not available")

        self.declare_parameter("auto_start", True)
        self.declare_parameter("trigger_topic", "/x2/assist/raise_arms_trigger")
        self.declare_parameter("run_once", True)
        self.declare_parameter("shoulder_pitch_deg", 10.0)
        self.declare_parameter("shoulder_roll_deg", 0.0)
        self.declare_parameter("shoulder_yaw_deg", 0.0)
        self.declare_parameter("elbow_bend_deg", 90.0)
        self.declare_parameter("move_seconds", 3.0)
        self.declare_parameter("hold_indefinitely", True)
        self.declare_parameter("hold_seconds", 0.0)
        self.declare_parameter("release_seconds", 0.0)
        self.declare_parameter("move_stiffness", 8.0)
        self.declare_parameter("move_damping", 0.8)
        self.declare_parameter("hold_stiffness", 8.0)
        self.declare_parameter("hold_damping", 0.8)
        self.declare_parameter("require_no_other_arm_publishers", True)
        self.declare_parameter("require_balanced_mode", True)
        self.declare_parameter("require_balanced_status", False)
        self.declare_parameter("control_hz", 200.0)
        self.declare_parameter("arm_state_timeout_s", 5.0)

        self.auto_start = bool_param(self.get_parameter("auto_start").value)
        self.trigger_topic = str(self.get_parameter("trigger_topic").value)
        self.run_once = bool_param(self.get_parameter("run_once").value)
        self.shoulder_pitch_deg = float(self.get_parameter("shoulder_pitch_deg").value)
        self.shoulder_roll_deg = float(self.get_parameter("shoulder_roll_deg").value)
        self.shoulder_yaw_deg = float(self.get_parameter("shoulder_yaw_deg").value)
        self.elbow_bend_deg = float(self.get_parameter("elbow_bend_deg").value)
        self.move_seconds = float(self.get_parameter("move_seconds").value)
        self.hold_indefinitely = bool_param(
            self.get_parameter("hold_indefinitely").value
        )
        self.hold_seconds = float(self.get_parameter("hold_seconds").value)
        self.release_seconds = float(self.get_parameter("release_seconds").value)
        self.move_stiffness = float(self.get_parameter("move_stiffness").value)
        self.move_damping = float(self.get_parameter("move_damping").value)
        self.hold_stiffness = float(self.get_parameter("hold_stiffness").value)
        self.hold_damping = float(self.get_parameter("hold_damping").value)
        self.require_no_other_arm_publishers = bool_param(
            self.get_parameter("require_no_other_arm_publishers").value
        )
        self.require_balanced_mode = bool_param(
            self.get_parameter("require_balanced_mode").value
        )
        self.require_balanced_status = bool_param(
            self.get_parameter("require_balanced_status").value
        )
        self.control_hz = float(self.get_parameter("control_hz").value)
        self.arm_state_timeout_s = float(self.get_parameter("arm_state_timeout_s").value)

        if self.control_hz <= 0.0:
            raise ValueError("control_hz must be > 0")
        if self.move_seconds <= 0.0:
            raise ValueError("move_seconds must be > 0")
        if self.hold_seconds < 0.0:
            raise ValueError("hold_seconds must be >= 0")
        if self.release_seconds < 0.0:
            raise ValueError("release_seconds must be >= 0")

        self.period = 1.0 / self.control_hz
        self.arm_positions: Optional[List[float]] = None
        self.active_thread: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        self.has_started_once = False
        self.warned_unnamed_state = False

        self.mc_action_client = self.create_client(GetMcAction, MC_ACTION_SERVICE)
        self.arm_pub = None
        self.create_subscription(
            JointStateArray,
            ARM_STATE_TOPIC,
            self._arm_state_callback,
            SUBSCRIBER_QOS,
        )
        self.create_subscription(
            Bool,
            self.trigger_topic,
            self._trigger_callback,
            PUBLISHER_QOS,
        )

        self.get_logger().info(
            "Raise-arms pose node ready: "
            f"auto_start={self.auto_start}, trigger_topic={self.trigger_topic}, "
            f"run_once={self.run_once}, hold_indefinitely={self.hold_indefinitely}, "
            f"hold_seconds={self.hold_seconds:.2f}, "
            f"release_seconds={self.release_seconds:.2f}"
        )
        if self.auto_start:
            self.start_pose_thread()

    def _arm_state_callback(self, msg: JointStateArray) -> None:
        by_name = {
            joint.name: joint
            for joint in msg.joints
            if getattr(joint, "name", "")
        }
        if all(joint.name in by_name for joint in ARM_JOINTS):
            self.arm_positions = [
                float(by_name[joint.name].position) for joint in ARM_JOINTS
            ]
            return

        if len(msg.joints) >= len(ARM_JOINTS):
            if not self.warned_unnamed_state:
                self.warned_unnamed_state = True
                self.get_logger().warn(
                    "Arm state lacks expected joint names; using SDK model order."
                )
            self.arm_positions = [
                float(msg.joints[i].position) for i in range(len(ARM_JOINTS))
            ]

    def _trigger_callback(self, msg: Bool) -> None:
        if msg.data:
            self.get_logger().info("Received arm pose activate trigger.")
            self.start_pose_thread()
        else:
            self.get_logger().info("Received arm pose deactivate trigger.")
            self.cancel_event.set()

    def start_pose_thread(self) -> None:
        if self.run_once and self.has_started_once:
            self.get_logger().info("Arm pose trigger ignored; run_once already consumed.")
            return
        if self.active_thread is not None and self.active_thread.is_alive():
            self.get_logger().info("Arm pose trigger ignored; pose routine already active.")
            return

        self.has_started_once = True
        self.cancel_event.clear()
        self.get_logger().info("Starting arm pose routine.")
        self.active_thread = threading.Thread(target=self.run_pose, daemon=True)
        self.active_thread.start()

    def get_motion_mode(self) -> Optional[Tuple[str, int]]:
        if not self.mc_action_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(f"{MC_ACTION_SERVICE} service unavailable.")
            return None

        request = GetMcAction.Request()
        request.request = CommonRequest()
        future = None
        for attempt in range(8):
            request.request.header.stamp = self.get_clock().now().to_msg()
            future = self.mc_action_client.call_async(request)
            deadline = time.monotonic() + 0.25
            while rclpy.ok() and not future.done() and time.monotonic() < deadline:
                time.sleep(0.02)
            if future.done():
                break
            self.get_logger().debug(f"GetMcAction retry [{attempt + 1}/8]")

        if future is None or not future.done() or future.result() is None:
            self.get_logger().error("GetMcAction timed out.")
            return None

        response = future.result()
        mode = response.info.action_desc
        status = response.info.status.value
        self.get_logger().info(f"Current motion mode: {mode}, status={status}.")
        return mode, status

    def check_balanced_mode(self) -> bool:
        if not self.require_balanced_mode:
            return True

        mode_status = self.get_motion_mode()
        if mode_status is None:
            return False

        mode, status = mode_status
        if status != MC_ACTION_RUNNING and self.require_balanced_status:
            self.get_logger().error(
                f"Motion mode {mode} is not running yet (status={status})."
            )
            return False
        if status != MC_ACTION_RUNNING:
            self.get_logger().warn(
                f"Motion mode {mode} reported status={status}; continuing because "
                "require_balanced_status=false."
            )
        if mode in BALANCED_MOTION_MODES:
            return True

        allowed = ", ".join(sorted(BALANCED_MOTION_MODES))
        self.get_logger().error(
            f"Current motion mode is {mode}; expected one of: {allowed}."
        )
        return False

    def check_command_conflicts(self) -> bool:
        if not self.require_no_other_arm_publishers:
            return True

        publishers = self.get_publishers_info_by_topic(ARM_COMMAND_TOPIC)
        if not publishers:
            return True

        publisher_names = ", ".join(
            sorted(
                f"{info.node_namespace.rstrip('/')}/{info.node_name}".replace("//", "/")
                for info in publishers
            )
        )
        self.get_logger().error(
            f"Found publisher(s) already on {ARM_COMMAND_TOPIC}: {publisher_names}. "
            "Aborting to avoid conflicting arm commands."
        )
        return False

    def start_command_publisher(self) -> None:
        if self.arm_pub is None:
            self.arm_pub = self.create_publisher(
                JointCommandArray, ARM_COMMAND_TOPIC, PUBLISHER_QOS
            )

    def wait_for_arm_state(self) -> bool:
        self.get_logger().info("Waiting for arm joint state...")
        deadline = time.monotonic() + self.arm_state_timeout_s
        while rclpy.ok() and time.monotonic() < deadline:
            if self.arm_positions is not None:
                self.get_logger().info("Arm joint state received.")
                return True
            time.sleep(0.05)
        self.get_logger().error("Timed out waiting for arm joint state.")
        return False

    def build_target(self) -> List[float]:
        if self.arm_positions is None:
            raise RuntimeError("Arm state is unavailable.")

        shoulder_pitch = -math.radians(self.shoulder_pitch_deg)
        shoulder_roll = math.radians(self.shoulder_roll_deg)
        shoulder_yaw = math.radians(self.shoulder_yaw_deg)
        elbow = -math.radians(self.elbow_bend_deg)
        overrides = {
            "left_shoulder_pitch_joint": shoulder_pitch,
            "left_shoulder_roll_joint": shoulder_roll,
            "left_shoulder_yaw_joint": shoulder_yaw,
            "left_elbow_joint": elbow,
            "left_wrist_yaw_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "right_shoulder_pitch_joint": shoulder_pitch,
            "right_shoulder_roll_joint": -shoulder_roll,
            "right_shoulder_yaw_joint": shoulder_yaw,
            "right_elbow_joint": elbow,
            "right_wrist_yaw_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
        }

        target = list(self.arm_positions)
        for index, joint in enumerate(ARM_JOINTS):
            target[index] = clamp(
                overrides[joint.name], joint.lower_limit, joint.upper_limit
            )
        return target

    def make_arm_command(
        self, positions: List[float], stiffness: float, damping: float
    ) -> JointCommandArray:
        cmd = JointCommandArray()
        cmd.header = MessageHeader()
        cmd.header.stamp = self.get_clock().now().to_msg()

        for index, joint_info in enumerate(ARM_JOINTS):
            joint = JointCommand()
            joint.name = joint_info.name
            joint.position = clamp(
                positions[index], joint_info.lower_limit, joint_info.upper_limit
            )
            joint.velocity = 0.0
            joint.effort = 0.0
            joint.stiffness = stiffness
            joint.damping = damping
            cmd.joints.append(joint)
        return cmd

    def publish_arm_pose(
        self, positions: List[float], stiffness: float, damping: float
    ) -> None:
        if self.arm_pub is None:
            raise RuntimeError("Arm command publisher is not started.")
        self.arm_pub.publish(self.make_arm_command(positions, stiffness, damping))

    def run_pose(self) -> None:
        try:
            if not self.check_balanced_mode():
                return
            if not self.check_command_conflicts():
                return

            self.start_command_publisher()
            if not self.wait_for_arm_state():
                return

            start = list(self.arm_positions or [])
            target = self.build_target()
            self.get_logger().info(
                f"Raising arms: shoulder_pitch={self.shoulder_pitch_deg:.1f}deg, "
                f"elbow_bend={self.elbow_bend_deg:.1f}deg, "
                f"move={self.move_seconds:.2f}s."
            )

            start_time = time.monotonic()
            while rclpy.ok() and not self.cancel_event.is_set():
                elapsed = time.monotonic() - start_time
                alpha = smoothstep(elapsed / self.move_seconds)
                positions = [s + (t - s) * alpha for s, t in zip(start, target)]
                self.publish_arm_pose(
                    positions,
                    stiffness=self.move_stiffness,
                    damping=self.move_damping,
                )
                if elapsed >= self.move_seconds:
                    break
                time.sleep(self.period)

            if self.cancel_event.is_set():
                return

            self.get_logger().info("Arm pose reached; holding pose.")
            hold_until = time.monotonic() + self.hold_seconds
            while rclpy.ok() and not self.cancel_event.is_set():
                if not self.hold_indefinitely and time.monotonic() >= hold_until:
                    break
                self.publish_arm_pose(
                    target,
                    stiffness=self.hold_stiffness,
                    damping=self.hold_damping,
                )
                time.sleep(self.period)

            if self.cancel_event.is_set():
                return

            if not self.hold_indefinitely and self.release_seconds > 0.0:
                self.get_logger().info(
                    f"Releasing arm pose over {self.release_seconds:.2f}s."
                )
                release_start = list(target)
                release_target = list(start)
                release_start_time = time.monotonic()
                while rclpy.ok() and not self.cancel_event.is_set():
                    elapsed = time.monotonic() - release_start_time
                    alpha = smoothstep(elapsed / self.release_seconds)
                    positions = [
                        s + (t - s) * alpha
                        for s, t in zip(release_start, release_target)
                    ]
                    self.publish_arm_pose(
                        positions,
                        stiffness=self.move_stiffness,
                        damping=self.move_damping,
                    )
                    if elapsed >= self.release_seconds:
                        break
                    time.sleep(self.period)

            self.get_logger().info("Arm pose routine inactive.")
        except Exception as exc:
            self.get_logger().error(f"Arm pose routine failed: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RaiseArmsPose()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.cancel_event.set()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
