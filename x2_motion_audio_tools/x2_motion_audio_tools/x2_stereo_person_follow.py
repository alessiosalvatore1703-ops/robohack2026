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
        McControlArea,
        McLocomotionVelocity,
        McPresetMotion,
        MessageHeader,
        RequestHeader,
    )
    from aimdk_msgs.srv import SetMcAction, SetMcInputSource, SetMcPresetMotion

    AIMDK_AVAILABLE = True
except ImportError:
    AIMDK_AVAILABLE = False

try:
    from aimdk_msgs.srv import PlayTts

    PLAYTTS_AVAILABLE = True
except ImportError:
    PLAYTTS_AVAILABLE = False
    PlayTts = None

try:
    from aimdk_msgs.srv import PlayEmoji

    PLAYEMOJI_AVAILABLE = True
except ImportError:
    PLAYEMOJI_AVAILABLE = False
    PlayEmoji = None


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
        self.declare_parameter("stop_min_m", 0.45)
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
        self.declare_parameter("hold_base_in_stop_band", True)
        self.declare_parameter("arm_pose_trigger_enabled", False)
        self.declare_parameter(
            "arm_pose_trigger_topic", "/x2/assist/raise_arms_trigger"
        )
        self.declare_parameter("arm_pose_trigger_duration_sec", 2.0)
        self.declare_parameter("arm_pose_use_preset_motion", True)
        self.declare_parameter("arm_pose_preset_area_id", 3)
        self.declare_parameter("arm_pose_preset_motion_id", 1010)
        self.declare_parameter("assist_wait_seconds", 7.0)
        self.declare_parameter("announce_enabled", False)
        self.declare_parameter("announce_startup_text", "I'm on my way")
        self.declare_parameter("announce_stop_text", "Hold on")
        self.declare_parameter("announce_startup_emoji_id", 90)
        self.declare_parameter("announce_stop_emoji_id", 200)
        self.declare_parameter("announce_emoji_mode", 1)
        self.declare_parameter("announce_tts_priority", 6)
        self.declare_parameter("announce_emoji_priority", 10)
        self.declare_parameter("announce_domain", "x2_stereo_person_follow")
        self.declare_parameter("assist_head_pat_enabled", False)
        self.declare_parameter("assist_head_pat_topic", "/x2/assist/head_pat")
        self.declare_parameter("require_waist_neutral_for_forward", False)
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
        self.hold_base_in_stop_band = bool_param(
            self.get_parameter("hold_base_in_stop_band").value
        )
        self.arm_pose_trigger_enabled = bool_param(
            self.get_parameter("arm_pose_trigger_enabled").value
        )
        self.arm_pose_trigger_topic = str(
            self.get_parameter("arm_pose_trigger_topic").value
        )
        self.arm_pose_trigger_duration_sec = float(
            self.get_parameter("arm_pose_trigger_duration_sec").value
        )
        self.arm_pose_use_preset_motion = bool_param(
            self.get_parameter("arm_pose_use_preset_motion").value
        )
        self.arm_pose_preset_area_id = int(
            self.get_parameter("arm_pose_preset_area_id").value
        )
        self.arm_pose_preset_motion_id = int(
            self.get_parameter("arm_pose_preset_motion_id").value
        )
        self.assist_wait_seconds = float(
            self.get_parameter("assist_wait_seconds").value
        )
        self.announce_enabled = bool_param(
            self.get_parameter("announce_enabled").value
        )
        self.announce_startup_text = str(
            self.get_parameter("announce_startup_text").value
        )
        self.announce_stop_text = str(
            self.get_parameter("announce_stop_text").value
        )
        self.announce_startup_emoji_id = int(
            self.get_parameter("announce_startup_emoji_id").value
        )
        self.announce_stop_emoji_id = int(
            self.get_parameter("announce_stop_emoji_id").value
        )
        self.announce_emoji_mode = int(
            self.get_parameter("announce_emoji_mode").value
        )
        self.announce_tts_priority = int(
            self.get_parameter("announce_tts_priority").value
        )
        self.announce_emoji_priority = int(
            self.get_parameter("announce_emoji_priority").value
        )
        self.announce_domain = str(self.get_parameter("announce_domain").value)
        self.assist_head_pat_enabled = bool_param(
            self.get_parameter("assist_head_pat_enabled").value
        )
        self.assist_head_pat_topic = str(
            self.get_parameter("assist_head_pat_topic").value
        )
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
        self.arm_pose_triggered = False
        self.arms_used_once = False
        self.assist_wait_until = 0.0
        self.arm_pose_trigger_active_until = 0.0

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
            if self.require_waist_neutral_for_forward:
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
            self.preset_motion_client = self.create_client(
                SetMcPresetMotion,
                "/aimdk_5Fmsgs/srv/SetMcPresetMotion",
                callback_group=self.cb_group,
            )
        else:
            self.velocity_pub = None
            self.input_source_client = None
            self.mc_action_client = None
            self.preset_motion_client = None

        self.tts_client = None
        self.emoji_client = None
        if self.announce_enabled and AIMDK_AVAILABLE:
            if PLAYTTS_AVAILABLE:
                self.tts_client = self.create_client(
                    PlayTts,
                    "/aimdk_5Fmsgs/srv/PlayTts",
                    callback_group=self.cb_group,
                )
            else:
                self.get_logger().warn(
                    "PlayTts not available in aimdk_msgs; TTS announcements disabled."
                )
            if PLAYEMOJI_AVAILABLE:
                self.emoji_client = self.create_client(
                    PlayEmoji,
                    "/face_ui_proxy/play_emoji",
                    callback_group=self.cb_group,
                )
            else:
                self.get_logger().warn(
                    "PlayEmoji not available in aimdk_msgs; emoji announcements disabled."
                )

        self.create_subscription(
            Bool,
            "/stereo_person/follow/enable",
            self.enable_callback,
            RELIABLE_QOS,
            callback_group=self.cb_group,
        )
        self.create_subscription(
            Bool,
            self.assist_head_pat_topic,
            self.head_pat_callback,
            RELIABLE_QOS,
            callback_group=self.cb_group,
        )
        self.arm_pose_trigger_pub = self.create_publisher(
            Bool,
            self.arm_pose_trigger_topic,
            RELIABLE_QOS,
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
            f"require_waist_neutral={self.require_waist_neutral_for_forward}, "
            f"assist_wait={self.assist_wait_seconds:.2f}s, "
            f"target_topic={self.target_topic}, stop_band=[{self.stop_min_m:.2f}, "
            f"{self.stop_max_m:.2f}]m, max_v={self.max_forward_speed:.2f}m/s, "
            f"max_w={self.max_angular_speed:.2f}rad/s"
        )

        if self.enabled and not self.dry_run:
            self.enabled = False
            self.spawn_activation()
        elif self.enabled and self.dry_run:
            self.announce(
                "startup",
                self.announce_startup_text,
                self.announce_startup_emoji_id,
            )

    def enable_callback(self, msg: Bool) -> None:
        if msg.data:
            if self.enabled or self.activation_in_progress:
                return
            if self.dry_run:
                self.enabled = True
                self.get_logger().info("Follow dry-run enabled.")
                self.announce(
                    "startup",
                    self.announce_startup_text,
                    self.announce_startup_emoji_id,
                )
            else:
                self.get_logger().info("Activating stereo person follow.")
                self.spawn_activation()
        else:
            if not self.enabled and not self.activation_in_progress:
                return
            self.enabled = False
            self.get_logger().info("Follow disabled; publishing stop.")
            self.publish_stop()

    def head_pat_callback(self, msg: Bool) -> None:
        if not msg.data or not self.assist_head_pat_enabled:
            return

        self.get_logger().info(
            "Head-pat event ignored; integrated assist wait is time-based."
        )

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
                self.announce(
                    "startup",
                    self.announce_startup_text,
                    self.announce_startup_emoji_id,
                )
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
            self.announce(
                "startup",
                self.announce_startup_text,
                self.announce_startup_emoji_id,
            )
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

        self.update_assist_state(reason)
        if self.assist_wait_active():
            forward, angular, reason = 0.0, 0.0, "ASSIST_WAIT"
            self.log_throttled(reason, forward, angular)
            # Intentionally do NOT publish velocity here. Keeping the publish
            # stream alive holds our MC input source, which causes the MC to
            # reject preset-motion requests (e.g. the arm emote). Letting the
            # input source time out during the assist wait is required for the
            # preset path to be accepted.
            return

        self.log_throttled(reason, forward, angular)

        if self.dry_run:
            return

        if not self.registered_input_source:
            # Our input source timed out during assist wait. It will be
            # re-registered by the background thread spawned in
            # expire_assist_wait(); skip publishing until that completes.
            return

        self.publish_velocity(forward, angular)

    def compute_velocity(self) -> tuple[float, float, str]:
        now = time.monotonic()
        if self.target is None:
            return 0.0, 0.0, "NO_TARGET"

        target_age = now - self.target.stamp_sec
        if target_age > self.target_timeout_sec:
            return 0.0, 0.0, "NO_TARGET"

        z_m = self.target.z_m
        if z_m < self.min_valid_depth_m or z_m > self.max_valid_depth_m:
            return 0.0, 0.0, "INVALID_DEPTH"

        if self.stop_min_m <= z_m <= self.stop_max_m:
            return 0.0, 0.0, "STOP_BAND"

        if z_m < self.stop_min_m and self.reverse_enabled:
            distance_error = self.target_distance_m - z_m
            reverse = -clamp(
                self.forward_gain * distance_error,
                0.0,
                self.max_reverse_speed,
            )
            reverse = -self.apply_min_velocity(abs(reverse), self.min_forward_speed)
            return reverse, 0.0, "REVERSE"

        if z_m < self.stop_min_m:
            return 0.0, 0.0, "TOO_CLOSE"

        bearing_rad = math.atan2(self.target.x_m, z_m)
        angular = self.angular_velocity_for_bearing(bearing_rad)

        waist_neutral = self.waist_is_neutral(now)
        if not waist_neutral:
            return 0.0, angular, "WAIST_NOT_NEUTRAL"

        if abs(bearing_rad) > self.max_forward_bearing_rad:
            return 0.0, angular, "ALIGN"

        distance_error = z_m - self.target_distance_m
        forward = clamp(
            self.forward_gain * distance_error,
            0.0,
            self.max_forward_speed,
        )
        forward = self.apply_min_velocity(forward, self.min_forward_speed)
        return forward, angular, "APPROACH"

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
        if self.velocity_pub is None or not rclpy.ok():
            return

        msg = McLocomotionVelocity()
        msg.header = MessageHeader()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.source = self.source_name
        msg.forward_velocity = float(forward_velocity)
        msg.lateral_velocity = 0.0
        msg.angular_velocity = float(angular_velocity)
        try:
            self.velocity_pub.publish(msg)
        except Exception as exc:
            if rclpy.ok():
                self.get_logger().warn(f"Failed to publish velocity: {exc}")

    def update_assist_state(self, reason: str) -> None:
        self.expire_assist_wait()
        self.publish_pending_arm_trigger()

        if self.assist_wait_active():
            return

        if reason not in {"STOP_BAND", "TOO_CLOSE"}:
            return

        if not self.arm_pose_trigger_enabled:
            return

        if self.arms_used_once:
            return

        self.arms_used_once = True
        self.get_logger().info(
            f"Arrived at person; starting timed assist wait for "
            f"{max(0.0, self.assist_wait_seconds):.2f}s."
        )
        self.announce(
            "stop",
            self.announce_stop_text,
            self.announce_stop_emoji_id,
        )
        self.start_arm_pose_trigger()
        if self.assist_wait_seconds > 0.0:
            self.assist_wait_until = time.monotonic() + self.assist_wait_seconds

    def assist_wait_active(self) -> bool:
        return self.assist_wait_until > time.monotonic()

    def expire_assist_wait(self) -> None:
        if self.assist_wait_until <= 0.0:
            return
        if time.monotonic() < self.assist_wait_until:
            return
        self.assist_wait_until = 0.0
        # The MC will have timed out our input source during the assist wait
        # (we deliberately stopped publishing velocity). Re-register before the
        # control loop resumes so the next velocity command is honored.
        if not self.dry_run:
            self.registered_input_source = False
            threading.Thread(
                target=self._reregister_input_source, daemon=True
            ).start()
        self.get_logger().info(
            "Timed assist wait complete; re-registering input source and "
            "resuming normal follow."
        )

    def _reregister_input_source(self) -> None:
        if self.register_input_source():
            self.registered_input_source = True
        else:
            self.get_logger().error(
                "Failed to re-register input source after assist wait; "
                "follow will stay stopped until re-registration succeeds."
            )

    def announce(self, label: str, text: str, emoji_id: int) -> None:
        """Fire-and-forget TTS + emoji. Never raises; failures only log.

        Safe to run in dry_run because neither service causes robot motion.
        """
        if not self.announce_enabled:
            return
        threading.Thread(
            target=self._announce_worker,
            args=(label, text, emoji_id),
            daemon=True,
        ).start()

    def _announce_worker(self, label: str, text: str, emoji_id: int) -> None:
        try:
            self._play_tts(text)
        except Exception as exc:
            self.get_logger().warn(
                f"Announce TTS '{label}' failed: {exc}"
            )
        try:
            self._play_emoji(emoji_id)
        except Exception as exc:
            self.get_logger().warn(
                f"Announce emoji '{label}' failed: {exc}"
            )

    def _play_tts(self, text: str) -> None:
        if self.tts_client is None or PlayTts is None:
            return
        if not self.tts_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                "PlayTts service unavailable; skipping TTS."
            )
            return

        req = PlayTts.Request()
        req.tts_req.text = text
        req.tts_req.domain = self.announce_domain
        req.tts_req.trace_id = self.announce_domain
        req.tts_req.is_interrupted = True
        req.tts_req.priority_weight = 0
        req.tts_req.priority_level.value = self.announce_tts_priority

        future = None
        for _ in range(8):
            req.header.header.stamp = self.get_clock().now().to_msg()
            future = self.tts_client.call_async(req)
            deadline = time.monotonic() + 0.25
            while rclpy.ok() and not future.done() and time.monotonic() < deadline:
                time.sleep(0.02)
            if future.done():
                break

        if future is None or not future.done() or future.result() is None:
            self.get_logger().warn("PlayTts call timed out.")
            return

        resp = future.result()
        if resp.tts_resp.is_success:
            self.get_logger().info(f"TTS played: '{text}'")
        else:
            self.get_logger().warn(f"TTS failed for '{text}'")

    def _play_emoji(self, emoji_id: int) -> None:
        if self.emoji_client is None or PlayEmoji is None:
            return
        if not self.emoji_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn(
                "PlayEmoji service unavailable; skipping emoji."
            )
            return

        req = PlayEmoji.Request()
        req.emotion_id = int(emoji_id)
        req.mode = int(self.announce_emoji_mode)
        req.priority = int(self.announce_emoji_priority)

        future = None
        for _ in range(8):
            req.header.header.stamp = self.get_clock().now().to_msg()
            future = self.emoji_client.call_async(req)
            deadline = time.monotonic() + 0.25
            while rclpy.ok() and not future.done() and time.monotonic() < deadline:
                time.sleep(0.02)
            if future.done():
                break

        if future is None or not future.done() or future.result() is None:
            self.get_logger().warn("PlayEmoji call timed out.")
            return

        resp = future.result()
        if resp.success:
            self.get_logger().info(
                f"Emoji {emoji_id} played: {resp.message}"
            )
        else:
            self.get_logger().warn(
                f"Emoji {emoji_id} failed: {resp.message}"
            )

    def start_arm_pose_trigger(self) -> None:
        if not self.arm_pose_trigger_enabled:
            return

        now = time.monotonic()
        self.arm_pose_triggered = True
        self.arm_pose_trigger_active_until = (
            now + max(0.1, self.arm_pose_trigger_duration_sec)
        )
        if self.dry_run:
            mode = (
                "preset-motion"
                if self.arm_pose_use_preset_motion
                else f"bool-trigger on {self.arm_pose_trigger_topic}"
            )
            self.get_logger().info(
                f"Dry-run: would start one-shot arm pose ({mode})"
            )
            return

        if self.arm_pose_use_preset_motion:
            self.get_logger().info(
                "Starting one-shot arm pose via MC preset "
                f"area={self.arm_pose_preset_area_id}, "
                f"motion={self.arm_pose_preset_motion_id}"
            )
            threading.Thread(
                target=self._send_arm_preset_motion, daemon=True
            ).start()
            return

        self.get_logger().info(
            f"Starting one-shot arm pose trigger on {self.arm_pose_trigger_topic}"
        )
        self.publish_arm_pose_trigger(True)

    def _send_arm_preset_motion(self) -> None:
        if self.preset_motion_client is None:
            self.get_logger().error(
                "Preset motion client unavailable; cannot send arm preset."
            )
            return
        if not self.preset_motion_client.wait_for_service(timeout_sec=4.0):
            self.get_logger().error(
                "SetMcPresetMotion service unavailable; arm preset skipped."
            )
            return

        # The MC rejects preset-motion requests when an application input
        # source is actively holding locomotion. control_loop() stops
        # publishing velocity during the assist wait; pause here so the MC's
        # input-source timeout (1 s) can actually fire before we call the
        # service.
        pre_call_delay_sec = 1.5
        self.get_logger().info(
            f"Waiting {pre_call_delay_sec:.1f}s for input-source timeout "
            "before sending preset motion..."
        )
        time.sleep(pre_call_delay_sec)

        req = SetMcPresetMotion.Request()
        req.header = RequestHeader()
        req.area = McControlArea()
        req.motion = McPresetMotion()
        req.area.value = self.arm_pose_preset_area_id
        req.motion.value = self.arm_pose_preset_motion_id
        req.interrupt = True

        future = None
        for attempt in range(8):
            req.header.stamp = self.get_clock().now().to_msg()
            future = self.preset_motion_client.call_async(req)
            deadline = time.monotonic() + 0.5
            while rclpy.ok() and not future.done() and time.monotonic() < deadline:
                time.sleep(0.02)
            if future.done():
                break
            self.get_logger().debug(
                f"SetMcPresetMotion retry [{attempt + 1}/8]"
            )

        if future is None or not future.done() or future.result() is None:
            self.get_logger().error("SetMcPresetMotion call timed out.")
            return

        response = future.result()
        if response.response.header.code == 0:
            self.get_logger().info(
                f"Arm preset accepted: task_id={response.response.task_id}"
            )
            return
        if response.response.state.value == CommonState.RUNNING:
            self.get_logger().info(
                f"Arm preset running: task_id={response.response.task_id}"
            )
            return

        self.get_logger().error(
            "Arm preset rejected. Ensure robot is in Stable Stand."
        )

    def publish_pending_arm_trigger(self) -> None:
        if self.dry_run:
            return
        if self.arm_pose_use_preset_motion:
            return
        now = time.monotonic()
        if self.arm_pose_trigger_active_until <= now:
            return
        self.publish_arm_pose_trigger(True)

    def publish_arm_pose_trigger(self, active: bool) -> None:
        msg = Bool()
        msg.data = active
        self.arm_pose_trigger_pub.publish(msg)

    def maybe_trigger_arm_pose(self, reason: str) -> None:
        """Compatibility shim for older tests; use update_assist_state."""
        if not self.arm_pose_trigger_enabled or self.dry_run:
            return

        now = time.monotonic()
        if self.arm_pose_trigger_active_until > now:
            self.publish_arm_pose_trigger(True)
            return

        if self.arm_pose_triggered or reason not in {"STOP_BAND", "TOO_CLOSE"}:
            return

        self.arm_pose_triggered = True
        self.arm_pose_trigger_active_until = (
            now + max(0.1, self.arm_pose_trigger_duration_sec)
        )
        self.get_logger().info(
            f"Starting one-shot arm pose trigger on {self.arm_pose_trigger_topic}"
        )
        self.publish_arm_pose_trigger(True)

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
        target_text = "target=none"
        if self.target is not None:
            age_sec = now - self.target.stamp_sec
            if self.target.z_m != 0.0:
                bearing_deg = math.degrees(math.atan2(self.target.x_m, self.target.z_m))
            else:
                bearing_deg = 0.0
            target_text = (
                f"x={self.target.x_m:.2f}m, z={self.target.z_m:.2f}m, "
                f"bearing={bearing_deg:.1f}deg, age={age_sec:.2f}s"
            )
        self.get_logger().info(
            f"follow={reason}: forward={forward:.3f}m/s, "
            f"angular={angular:.3f}rad/s, {target_text}"
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
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
