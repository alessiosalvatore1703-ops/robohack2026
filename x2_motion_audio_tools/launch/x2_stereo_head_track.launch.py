"""Launch stereo person target detection with head tracking and base following."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    left_image_topic = LaunchConfiguration("left_image_topic")
    right_image_topic = LaunchConfiguration("right_image_topic")
    left_camera_info_topic = LaunchConfiguration("left_camera_info_topic")
    right_camera_info_topic = LaunchConfiguration("right_camera_info_topic")
    output_topic = LaunchConfiguration("output_topic")
    target_point_topic = LaunchConfiguration("target_point_topic")
    device = LaunchConfiguration("device")
    model = LaunchConfiguration("model")
    confidence = LaunchConfiguration("confidence")
    input_size = LaunchConfiguration("input_size")
    processing_width = LaunchConfiguration("processing_width")
    max_processing_fps = LaunchConfiguration("max_processing_fps")
    output_width = LaunchConfiguration("output_width")
    baseline_m = LaunchConfiguration("baseline_m")
    depth_disparity_percentile = LaunchConfiguration("depth_disparity_percentile")
    sync_slop_sec = LaunchConfiguration("sync_slop_sec")
    right_buffer_size = LaunchConfiguration("right_buffer_size")
    jpeg_quality = LaunchConfiguration("jpeg_quality")

    head_state_topic = LaunchConfiguration("head_state_topic")
    head_command_topic = LaunchConfiguration("head_command_topic")
    head_enabled = LaunchConfiguration("head_enabled")
    dry_run = LaunchConfiguration("dry_run")
    yaw_gain = LaunchConfiguration("yaw_gain")
    center_deadzone_deg = LaunchConfiguration("center_deadzone_deg")
    use_ruckig = LaunchConfiguration("use_ruckig")
    max_yaw_velocity = LaunchConfiguration("max_yaw_velocity")
    max_yaw_acceleration = LaunchConfiguration("max_yaw_acceleration")
    max_yaw_jerk = LaunchConfiguration("max_yaw_jerk")
    target_timeout_sec = LaunchConfiguration("target_timeout_sec")
    control_rate_hz = LaunchConfiguration("control_rate_hz")
    hold_on_lost = LaunchConfiguration("hold_on_lost")
    invert_yaw = LaunchConfiguration("invert_yaw")
    soft_limit_deg = LaunchConfiguration("soft_limit_deg")
    follow_enabled = LaunchConfiguration("follow_enabled")
    follow_dry_run = LaunchConfiguration("follow_dry_run")
    follow_auto_enable_stable_stand = LaunchConfiguration(
        "follow_auto_enable_stable_stand"
    )
    follow_auto_enable_locomotion = LaunchConfiguration("follow_auto_enable_locomotion")
    follow_target_distance_m = LaunchConfiguration("follow_target_distance_m")
    follow_stop_min_m = LaunchConfiguration("follow_stop_min_m")
    follow_stop_max_m = LaunchConfiguration("follow_stop_max_m")
    follow_max_forward_speed = LaunchConfiguration("follow_max_forward_speed")
    follow_min_forward_speed = LaunchConfiguration("follow_min_forward_speed")
    follow_max_angular_speed = LaunchConfiguration("follow_max_angular_speed")
    follow_forward_gain = LaunchConfiguration("follow_forward_gain")
    follow_angular_gain = LaunchConfiguration("follow_angular_gain")
    follow_center_deadzone_deg = LaunchConfiguration("follow_center_deadzone_deg")
    follow_max_forward_bearing_deg = LaunchConfiguration(
        "follow_max_forward_bearing_deg"
    )
    follow_control_rate_hz = LaunchConfiguration("follow_control_rate_hz")
    follow_reverse_enabled = LaunchConfiguration("follow_reverse_enabled")
    follow_invert_angular = LaunchConfiguration("follow_invert_angular")
    follow_hold_base_in_stop_band = LaunchConfiguration(
        "follow_hold_base_in_stop_band"
    )
    assist_arm_pose_enabled = LaunchConfiguration("assist_arm_pose_enabled")
    assist_arm_use_preset_motion = LaunchConfiguration(
        "assist_arm_use_preset_motion"
    )
    assist_arm_preset_area_id = LaunchConfiguration("assist_arm_preset_area_id")
    assist_arm_preset_motion_id = LaunchConfiguration(
        "assist_arm_preset_motion_id"
    )
    announce_enabled = LaunchConfiguration("announce_enabled")
    announce_startup_text = LaunchConfiguration("announce_startup_text")
    announce_stop_text = LaunchConfiguration("announce_stop_text")
    announce_startup_emoji_id = LaunchConfiguration(
        "announce_startup_emoji_id"
    )
    announce_stop_emoji_id = LaunchConfiguration("announce_stop_emoji_id")
    assist_arm_pose_trigger_topic = LaunchConfiguration(
        "assist_arm_pose_trigger_topic"
    )
    assist_arm_pose_trigger_duration_sec = LaunchConfiguration(
        "assist_arm_pose_trigger_duration_sec"
    )
    assist_wait_seconds = LaunchConfiguration("assist_wait_seconds")
    assist_arm_shoulder_pitch_deg = LaunchConfiguration(
        "assist_arm_shoulder_pitch_deg"
    )
    assist_arm_elbow_bend_deg = LaunchConfiguration("assist_arm_elbow_bend_deg")
    assist_arm_move_seconds = LaunchConfiguration("assist_arm_move_seconds")
    assist_arm_hold_seconds = LaunchConfiguration("assist_arm_hold_seconds")
    assist_arm_release_seconds = LaunchConfiguration("assist_arm_release_seconds")
    assist_arm_hold_indefinitely = LaunchConfiguration(
        "assist_arm_hold_indefinitely"
    )
    assist_head_pat_enabled = LaunchConfiguration("assist_head_pat_enabled")
    assist_head_pat_topic = LaunchConfiguration("assist_head_pat_topic")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "left_image_topic",
                default_value="/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed",
                description="Left stereo compressed image topic.",
            ),
            DeclareLaunchArgument(
                "right_image_topic",
                default_value="/aima/hal/sensor/stereo_head_front_right/rgb_image/compressed",
                description="Right stereo compressed image topic.",
            ),
            DeclareLaunchArgument(
                "left_camera_info_topic",
                default_value="/aima/hal/sensor/stereo_head_front_left/camera_info",
                description="Left stereo CameraInfo topic.",
            ),
            DeclareLaunchArgument(
                "right_camera_info_topic",
                default_value="/aima/hal/sensor/stereo_head_front_right/camera_info",
                description="Right stereo CameraInfo topic.",
            ),
            DeclareLaunchArgument(
                "output_topic",
                default_value="/stereo_person/final_annotated_image/compressed",
                description="Final annotated compressed JPEG output topic.",
            ),
            DeclareLaunchArgument(
                "target_point_topic",
                default_value="/stereo_person/target_point",
                description="Closest valid person center point in the left camera frame.",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cpu",
                description="YOLO inference device: cpu, cuda, or mps.",
            ),
            DeclareLaunchArgument(
                "model",
                default_value="yolov8n.pt",
                description="YOLO model file path.",
            ),
            DeclareLaunchArgument(
                "confidence",
                default_value="0.5",
                description="YOLO confidence threshold.",
            ),
            DeclareLaunchArgument(
                "input_size",
                default_value="320",
                description="YOLO model input image size.",
            ),
            DeclareLaunchArgument(
                "processing_width",
                default_value="512",
                description="Working image width for YOLO/depth; 0 keeps source size.",
            ),
            DeclareLaunchArgument(
                "max_processing_fps",
                default_value="6.0",
                description="Maximum perception processing rate.",
            ),
            DeclareLaunchArgument(
                "output_width",
                default_value="960",
                description="Final JPEG width; 0 publishes full source size.",
            ),
            DeclareLaunchArgument(
                "baseline_m",
                default_value="0.0578",
                description="Stereo baseline in meters.",
            ),
            DeclareLaunchArgument(
                "depth_disparity_percentile",
                default_value="70.0",
                description=(
                    "Disparity percentile used for person depth. Higher values "
                    "bias distance closer for safer following."
                ),
            ),
            DeclareLaunchArgument(
                "sync_slop_sec",
                default_value="0.10",
                description="Maximum timestamp difference between left and right frames.",
            ),
            DeclareLaunchArgument(
                "right_buffer_size",
                default_value="20",
                description="Number of recent right frames available for sync matching.",
            ),
            DeclareLaunchArgument(
                "jpeg_quality",
                default_value="75",
                description="JPEG quality for annotated compressed outputs.",
            ),
            DeclareLaunchArgument(
                "head_state_topic",
                default_value="/aima/hal/joint/head/state",
                description="HAL head joint state topic.",
            ),
            DeclareLaunchArgument(
                "head_command_topic",
                default_value="/aima/hal/joint/head/command",
                description="HAL head joint command topic.",
            ),
            DeclareLaunchArgument(
                "head_enabled",
                default_value="true",
                description="When true, publish head yaw commands.",
            ),
            DeclareLaunchArgument(
                "dry_run",
                default_value="false",
                description="When true, log head commands but do not publish them.",
            ),
            DeclareLaunchArgument(
                "yaw_gain",
                default_value="0.6",
                description="Head yaw gain from target bearing to joint target.",
            ),
            DeclareLaunchArgument(
                "center_deadzone_deg",
                default_value="2.0",
                description="Ignore target bearings smaller than this.",
            ),
            DeclareLaunchArgument(
                "use_ruckig",
                default_value="true",
                description="Use Ruckig trajectory planning for head yaw commands.",
            ),
            DeclareLaunchArgument(
                "max_yaw_velocity",
                default_value="1.0",
                description="Maximum commanded head yaw speed in rad/s.",
            ),
            DeclareLaunchArgument(
                "max_yaw_acceleration",
                default_value="1.0",
                description="Maximum commanded head yaw acceleration in rad/s^2.",
            ),
            DeclareLaunchArgument(
                "max_yaw_jerk",
                default_value="25.0",
                description="Maximum commanded head yaw jerk in rad/s^3.",
            ),
            DeclareLaunchArgument(
                "target_timeout_sec",
                default_value="0.5",
                description="Treat target point as lost after this age.",
            ),
            DeclareLaunchArgument(
                "control_rate_hz",
                default_value="500.0",
                description="Head controller update rate.",
            ),
            DeclareLaunchArgument(
                "hold_on_lost",
                default_value="true",
                description="Hold last head command when the target is lost.",
            ),
            DeclareLaunchArgument(
                "invert_yaw",
                default_value="false",
                description="Flip head yaw correction if the head turns away.",
            ),
            DeclareLaunchArgument(
                "soft_limit_deg",
                default_value="20.0",
                description="Symmetric software yaw limit, bounded by robot joint limits.",
            ),
            DeclareLaunchArgument(
                "follow_enabled",
                default_value="false",
                description="When true, enable high-level locomotion following.",
            ),
            DeclareLaunchArgument(
                "follow_dry_run",
                default_value="true",
                description="When true, log computed walking commands without publishing.",
            ),
            DeclareLaunchArgument(
                "follow_auto_enable_stable_stand",
                default_value="false",
                description="When true, request STAND_DEFAULT before publishing.",
            ),
            DeclareLaunchArgument(
                "follow_auto_enable_locomotion",
                default_value="false",
                description=(
                    "Deprecated alias for follow_auto_enable_stable_stand."
                ),
            ),
            DeclareLaunchArgument(
                "follow_target_distance_m",
                default_value="0.75",
                description="Nominal human standoff distance.",
            ),
            DeclareLaunchArgument(
                "follow_stop_min_m",
                default_value="0.45",
                description="Lower edge of the stop band.",
            ),
            DeclareLaunchArgument(
                "follow_stop_max_m",
                default_value="1.0",
                description="Upper edge of the stop band.",
            ),
            DeclareLaunchArgument(
                "follow_max_forward_speed",
                default_value="0.10",
                description="Conservative max forward velocity in m/s.",
            ),
            DeclareLaunchArgument(
                "follow_min_forward_speed",
                default_value="0.10",
                description="Minimum nonzero forward velocity in m/s.",
            ),
            DeclareLaunchArgument(
                "follow_max_angular_speed",
                default_value="0.20",
                description="Conservative max yaw velocity in rad/s.",
            ),
            DeclareLaunchArgument(
                "follow_forward_gain",
                default_value="0.25",
                description="Forward velocity gain from distance error.",
            ),
            DeclareLaunchArgument(
                "follow_angular_gain",
                default_value="0.8",
                description="Yaw velocity gain from target bearing.",
            ),
            DeclareLaunchArgument(
                "follow_center_deadzone_deg",
                default_value="4.0",
                description="Ignore target bearings smaller than this for base yaw.",
            ),
            DeclareLaunchArgument(
                "follow_max_forward_bearing_deg",
                default_value="10.0",
                description="Only walk forward when target bearing is within this.",
            ),
            DeclareLaunchArgument(
                "follow_control_rate_hz",
                default_value="20.0",
                description="Walking supervisor control-loop rate.",
            ),
            DeclareLaunchArgument(
                "follow_reverse_enabled",
                default_value="false",
                description="When true, allow backing up if the person is too close.",
            ),
            DeclareLaunchArgument(
                "follow_invert_angular",
                default_value="false",
                description="Flip base yaw direction if the robot turns away.",
            ),
            DeclareLaunchArgument(
                "follow_hold_base_in_stop_band",
                default_value="true",
                description=(
                    "Compatibility flag. Close-range stop now suppresses base "
                    "yaw before alignment is evaluated."
                ),
            ),
            DeclareLaunchArgument(
                "assist_arm_pose_enabled",
                default_value="false",
                description=(
                    "When true, launch the arm pose node and trigger it once "
                    "when the follow supervisor first enters STOP_BAND."
                ),
            ),
            DeclareLaunchArgument(
                "assist_arm_use_preset_motion",
                default_value="true",
                description=(
                    "When true, the follow supervisor fires the arm emote via "
                    "SetMcPresetMotion instead of a Bool trigger topic. In "
                    "preset mode the legacy x2_raise_arms_pose node is not "
                    "launched."
                ),
            ),
            DeclareLaunchArgument(
                "assist_arm_preset_area_id",
                default_value="3",
                description="McControlArea.value for the arm preset motion.",
            ),
            DeclareLaunchArgument(
                "assist_arm_preset_motion_id",
                default_value="1010",
                description=(
                    "McPresetMotion.value. 1010 = raise both hands in the "
                    "AimDK preset table."
                ),
            ),
            DeclareLaunchArgument(
                "announce_enabled",
                default_value="false",
                description=(
                    "When true, speak a short TTS line and play an emoji at "
                    "follow activation and at first stop-band arrival. Any "
                    "failure is logged and ignored; the follower keeps "
                    "running."
                ),
            ),
            DeclareLaunchArgument(
                "announce_startup_text",
                default_value="I'm on my way",
                description="TTS spoken when follow activation completes.",
            ),
            DeclareLaunchArgument(
                "announce_stop_text",
                default_value="Hold on",
                description="TTS spoken on first stop-band arrival.",
            ),
            DeclareLaunchArgument(
                "announce_startup_emoji_id",
                default_value="90",
                description=(
                    "PlayEmoji.emotion_id for the startup announcement "
                    "(90 = happy)."
                ),
            ),
            DeclareLaunchArgument(
                "announce_stop_emoji_id",
                default_value="200",
                description=(
                    "PlayEmoji.emotion_id for the stop announcement "
                    "(200 = adore)."
                ),
            ),
            DeclareLaunchArgument(
                "assist_arm_pose_trigger_topic",
                default_value="/x2/assist/raise_arms_trigger",
                description="Bool topic used to trigger the assist arm pose.",
            ),
            DeclareLaunchArgument(
                "assist_arm_pose_trigger_duration_sec",
                default_value="2.0",
                description="Seconds to repeatedly publish the one-shot arm trigger.",
            ),
            DeclareLaunchArgument(
                "assist_wait_seconds",
                default_value="7.0",
                description=(
                    "Seconds to keep the base stationary during the first timed "
                    "assist stop."
                ),
            ),
            DeclareLaunchArgument(
                "assist_arm_shoulder_pitch_deg",
                default_value="10.0",
                description="Assist pose shoulder pitch in degrees.",
            ),
            DeclareLaunchArgument(
                "assist_arm_elbow_bend_deg",
                default_value="90.0",
                description="Assist pose elbow bend in degrees.",
            ),
            DeclareLaunchArgument(
                "assist_arm_move_seconds",
                default_value="3.0",
                description="Seconds used to move into the assist arm pose.",
            ),
            DeclareLaunchArgument(
                "assist_arm_hold_seconds",
                default_value="3.0",
                description="Seconds to hold the finite assist arm pose.",
            ),
            DeclareLaunchArgument(
                "assist_arm_release_seconds",
                default_value="0.5",
                description="Seconds used to move out of the finite assist arm pose.",
            ),
            DeclareLaunchArgument(
                "assist_arm_hold_indefinitely",
                default_value="false",
                description="Keep publishing the assist arm pose after reaching it.",
            ),
            DeclareLaunchArgument(
                "assist_head_pat_enabled",
                default_value="false",
                description=(
                    "Deprecated compatibility flag. The integrated assist wait "
                    "is time-based and does not use head-pat input."
                ),
            ),
            DeclareLaunchArgument(
                "assist_head_pat_topic",
                default_value="/x2/assist/head_pat",
                description="Deprecated head-pat topic; ignored by timed assist.",
            ),
            Node(
                package="yolo_person_detector",
                executable="stereo_final_annotator_node",
                name="stereo_final_annotator",
                output="screen",
                parameters=[
                    {
                        "left_image_topic": left_image_topic,
                        "right_image_topic": right_image_topic,
                        "left_camera_info_topic": left_camera_info_topic,
                        "right_camera_info_topic": right_camera_info_topic,
                        "output_topic": output_topic,
                        "target_point_topic": target_point_topic,
                        "device": device,
                        "model_path": model,
                        "confidence_threshold": confidence,
                        "input_size": input_size,
                        "processing_width": processing_width,
                        "max_processing_fps": max_processing_fps,
                        "output_width": output_width,
                        "baseline_m": baseline_m,
                        "depth_disparity_percentile": ParameterValue(
                            depth_disparity_percentile, value_type=float
                        ),
                        "sync_slop_sec": sync_slop_sec,
                        "right_buffer_size": right_buffer_size,
                        "jpeg_quality": jpeg_quality,
                    }
                ],
            ),
            Node(
                package="x2_motion_audio_tools",
                executable="x2_head_yaw_tracker",
                name="x2_head_yaw_tracker",
                output="screen",
                parameters=[
                    {
                        "target_topic": target_point_topic,
                        "head_state_topic": head_state_topic,
                        "head_command_topic": head_command_topic,
                        "enabled": ParameterValue(head_enabled, value_type=bool),
                        "dry_run": ParameterValue(dry_run, value_type=bool),
                        "yaw_gain": ParameterValue(yaw_gain, value_type=float),
                        "center_deadzone_deg": ParameterValue(
                            center_deadzone_deg, value_type=float
                        ),
                        "use_ruckig": ParameterValue(use_ruckig, value_type=bool),
                        "max_yaw_velocity": ParameterValue(
                            max_yaw_velocity, value_type=float
                        ),
                        "max_yaw_acceleration": ParameterValue(
                            max_yaw_acceleration, value_type=float
                        ),
                        "max_yaw_jerk": ParameterValue(
                            max_yaw_jerk, value_type=float
                        ),
                        "target_timeout_sec": ParameterValue(
                            target_timeout_sec, value_type=float
                        ),
                        "control_rate_hz": ParameterValue(
                            control_rate_hz, value_type=float
                        ),
                        "hold_on_lost": ParameterValue(hold_on_lost, value_type=bool),
                        "invert_yaw": ParameterValue(invert_yaw, value_type=bool),
                        "soft_limit_deg": ParameterValue(soft_limit_deg, value_type=float),
                    }
                ],
            ),
            Node(
                package="x2_motion_audio_tools",
                executable="x2_stereo_person_follow",
                name="x2_stereo_person_follow",
                output="screen",
                parameters=[
                    {
                        "target_topic": target_point_topic,
                        "enabled": ParameterValue(follow_enabled, value_type=bool),
                        "dry_run": ParameterValue(follow_dry_run, value_type=bool),
                        "auto_enable_stable_stand": ParameterValue(
                            follow_auto_enable_stable_stand, value_type=bool
                        ),
                        "auto_enable_locomotion": ParameterValue(
                            follow_auto_enable_locomotion, value_type=bool
                        ),
                        "target_distance_m": ParameterValue(
                            follow_target_distance_m, value_type=float
                        ),
                        "stop_min_m": ParameterValue(
                            follow_stop_min_m, value_type=float
                        ),
                        "stop_max_m": ParameterValue(
                            follow_stop_max_m, value_type=float
                        ),
                        "max_forward_speed": ParameterValue(
                            follow_max_forward_speed, value_type=float
                        ),
                        "min_forward_speed": ParameterValue(
                            follow_min_forward_speed, value_type=float
                        ),
                        "max_angular_speed": ParameterValue(
                            follow_max_angular_speed, value_type=float
                        ),
                        "forward_gain": ParameterValue(
                            follow_forward_gain, value_type=float
                        ),
                        "angular_gain": ParameterValue(
                            follow_angular_gain, value_type=float
                        ),
                        "center_deadzone_deg": ParameterValue(
                            follow_center_deadzone_deg, value_type=float
                        ),
                        "max_forward_bearing_deg": ParameterValue(
                            follow_max_forward_bearing_deg, value_type=float
                        ),
                        "reverse_enabled": ParameterValue(
                            follow_reverse_enabled, value_type=bool
                        ),
                        "invert_angular": ParameterValue(
                            follow_invert_angular, value_type=bool
                        ),
                        "hold_base_in_stop_band": ParameterValue(
                            follow_hold_base_in_stop_band, value_type=bool
                        ),
                        "arm_pose_trigger_enabled": ParameterValue(
                            assist_arm_pose_enabled, value_type=bool
                        ),
                        "arm_pose_trigger_topic": assist_arm_pose_trigger_topic,
                        "arm_pose_trigger_duration_sec": ParameterValue(
                            assist_arm_pose_trigger_duration_sec, value_type=float
                        ),
                        "arm_pose_use_preset_motion": ParameterValue(
                            assist_arm_use_preset_motion, value_type=bool
                        ),
                        "arm_pose_preset_area_id": ParameterValue(
                            assist_arm_preset_area_id, value_type=int
                        ),
                        "arm_pose_preset_motion_id": ParameterValue(
                            assist_arm_preset_motion_id, value_type=int
                        ),
                        "announce_enabled": ParameterValue(
                            announce_enabled, value_type=bool
                        ),
                        "announce_startup_text": announce_startup_text,
                        "announce_stop_text": announce_stop_text,
                        "announce_startup_emoji_id": ParameterValue(
                            announce_startup_emoji_id, value_type=int
                        ),
                        "announce_stop_emoji_id": ParameterValue(
                            announce_stop_emoji_id, value_type=int
                        ),
                        "assist_wait_seconds": ParameterValue(
                            assist_wait_seconds, value_type=float
                        ),
                        "assist_head_pat_enabled": ParameterValue(
                            assist_head_pat_enabled, value_type=bool
                        ),
                        "assist_head_pat_topic": assist_head_pat_topic,
                        "target_timeout_sec": ParameterValue(
                            target_timeout_sec, value_type=float
                        ),
                        "control_rate_hz": ParameterValue(
                            follow_control_rate_hz, value_type=float
                        ),
                    }
                ],
            ),
            Node(
                package="x2_motion_audio_tools",
                executable="x2_raise_arms_pose",
                name="x2_raise_arms_pose",
                output="screen",
                condition=IfCondition(
                    PythonExpression(
                        [
                            "'",
                            assist_arm_pose_enabled,
                            "' == 'true' and '",
                            assist_arm_use_preset_motion,
                            "' == 'false'",
                        ]
                    )
                ),
                parameters=[
                    {
                        "auto_start": False,
                        "trigger_topic": assist_arm_pose_trigger_topic,
                        "run_once": True,
                        "shoulder_pitch_deg": ParameterValue(
                            assist_arm_shoulder_pitch_deg, value_type=float
                        ),
                        "elbow_bend_deg": ParameterValue(
                            assist_arm_elbow_bend_deg, value_type=float
                        ),
                        "move_seconds": ParameterValue(
                            assist_arm_move_seconds, value_type=float
                        ),
                        "hold_seconds": ParameterValue(
                            assist_arm_hold_seconds, value_type=float
                        ),
                        "release_seconds": ParameterValue(
                            assist_arm_release_seconds, value_type=float
                        ),
                        "hold_indefinitely": ParameterValue(
                            assist_arm_hold_indefinitely, value_type=bool
                        ),
                    }
                ],
            ),
        ]
    )
