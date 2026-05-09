"""Launch stereo-camera person detection, TTS greeting, and body following."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    device = LaunchConfiguration("device")
    tts_enabled = LaunchConfiguration("tts_enabled")
    tts_text = LaunchConfiguration("tts_text")
    tts_cooldown_sec = LaunchConfiguration("tts_cooldown_sec")
    tts_reset_after_lost_sec = LaunchConfiguration("tts_reset_after_lost_sec")
    follow_enabled = LaunchConfiguration("follow_enabled")
    stop_distance_m = LaunchConfiguration("stop_distance_m")
    stop_deadband_m = LaunchConfiguration("stop_deadband_m")
    forward_gain = LaunchConfiguration("forward_gain")
    angular_gain = LaunchConfiguration("angular_gain")
    max_forward_speed = LaunchConfiguration("max_forward_speed")
    max_angular_speed = LaunchConfiguration("max_angular_speed")
    max_forward_bearing_deg = LaunchConfiguration("max_forward_bearing_deg")
    lidar_window_deg = LaunchConfiguration("lidar_window_deg")
    lidar_angle_offset_deg = LaunchConfiguration("lidar_angle_offset_deg")
    lidar_min_range_m = LaunchConfiguration("lidar_min_range_m")
    lidar_max_range_m = LaunchConfiguration("lidar_max_range_m")
    waist_tracking_enabled = LaunchConfiguration("waist_tracking_enabled")
    waist_state_topic = LaunchConfiguration("waist_state_topic")
    waist_command_topic = LaunchConfiguration("waist_command_topic")
    waist_yaw_gain = LaunchConfiguration("waist_yaw_gain")
    waist_soft_limit_deg = LaunchConfiguration("waist_soft_limit_deg")
    waist_max_velocity = LaunchConfiguration("waist_max_velocity")
    waist_max_acceleration = LaunchConfiguration("waist_max_acceleration")
    waist_max_jerk = LaunchConfiguration("waist_max_jerk")
    waist_hold_on_lost = LaunchConfiguration("waist_hold_on_lost")
    waist_invert_direction = LaunchConfiguration("waist_invert_direction")
    input_source_retry_sec = LaunchConfiguration("input_source_retry_sec")
    visual_fallback_enabled = LaunchConfiguration("visual_fallback_enabled")
    visual_target_bbox_height_ratio = LaunchConfiguration(
        "visual_target_bbox_height_ratio"
    )
    visual_fallback_max_forward_speed = LaunchConfiguration(
        "visual_fallback_max_forward_speed"
    )
    visual_stop_deadband_ratio = LaunchConfiguration("visual_stop_deadband_ratio")
    publish_debug_image = LaunchConfiguration("publish_debug_image")
    publish_debug_markers = LaunchConfiguration("publish_debug_markers")
    publish_status = LaunchConfiguration("publish_status")
    start_image_view = LaunchConfiguration("start_image_view")
    start_rviz = LaunchConfiguration("start_rviz")
    debug_rviz_config = PathJoinSubstitution(
        [
            FindPackageShare("x2_motion_audio_tools"),
            "launch",
            "x2_person_follow_debug.rviz",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "device",
                default_value="cpu",
                description="YOLO inference device: cpu, cuda, or mps.",
            ),
            DeclareLaunchArgument(
                "tts_enabled",
                default_value="true",
                description="When true, say a greeting when a person is detected.",
            ),
            DeclareLaunchArgument(
                "tts_text",
                default_value="Hello",
                description="Greeting text sent to the X2 TTS service.",
            ),
            DeclareLaunchArgument(
                "tts_cooldown_sec",
                default_value="60.0",
                description="Minimum time before greeting a later person encounter.",
            ),
            DeclareLaunchArgument(
                "tts_reset_after_lost_sec",
                default_value="2.0",
                description="How long the person must be lost before greeting resets.",
            ),
            DeclareLaunchArgument(
                "follow_enabled",
                default_value="true",
                description="When true, publish locomotion commands to follow the person.",
            ),
            DeclareLaunchArgument(
                "stop_distance_m",
                default_value="1.0",
                description="Distance where the robot stops approaching.",
            ),
            DeclareLaunchArgument(
                "stop_deadband_m",
                default_value="0.12",
                description="Extra distance buffer around stop_distance_m to avoid creep.",
            ),
            DeclareLaunchArgument(
                "forward_gain",
                default_value="0.28",
                description="Forward speed gain from distance error to m/s.",
            ),
            DeclareLaunchArgument(
                "angular_gain",
                default_value="1.0",
                description="Angular speed gain from target bearing to rad/s.",
            ),
            DeclareLaunchArgument(
                "max_forward_speed",
                default_value="0.25",
                description="Maximum walking speed in m/s.",
            ),
            DeclareLaunchArgument(
                "max_angular_speed",
                default_value="0.45",
                description="Maximum turning speed in rad/s.",
            ),
            DeclareLaunchArgument(
                "max_forward_bearing_deg",
                default_value="25.0",
                description="Only walk forward when the target is within this bearing.",
            ),
            DeclareLaunchArgument(
                "lidar_window_deg",
                default_value="8.0",
                description="Angular sector around the camera bearing used for LiDAR distance.",
            ),
            DeclareLaunchArgument(
                "lidar_angle_offset_deg",
                default_value="0.0",
                description="Offset between camera bearing and lidar frame.",
            ),
            DeclareLaunchArgument(
                "lidar_min_range_m",
                default_value="0.05",
                description="Ignore LiDAR points closer than this distance.",
            ),
            DeclareLaunchArgument(
                "lidar_max_range_m",
                default_value="8.0",
                description="Ignore LiDAR points farther than this distance. Use <=0 for no max.",
            ),
            DeclareLaunchArgument(
                "waist_tracking_enabled",
                default_value="false",
                description="Legacy option. Keep false for body-follow mode.",
            ),
            DeclareLaunchArgument(
                "waist_state_topic",
                default_value="/aima/hal/joint/waist/state",
                description="HAL waist joint state topic.",
            ),
            DeclareLaunchArgument(
                "waist_command_topic",
                default_value="/aima/hal/joint/waist/command",
                description="HAL waist joint command topic.",
            ),
            DeclareLaunchArgument(
                "waist_yaw_gain",
                default_value="1.0",
                description="Visual-servo gain from camera bearing to waist yaw.",
            ),
            DeclareLaunchArgument(
                "waist_soft_limit_deg",
                default_value="90.0",
                description="Symmetric software yaw limit for torso tracking.",
            ),
            DeclareLaunchArgument(
                "waist_max_velocity",
                default_value="0.35",
                description="Maximum commanded waist joint speed in rad/s.",
            ),
            DeclareLaunchArgument(
                "waist_max_acceleration",
                default_value="0.25",
                description="Maximum commanded waist joint acceleration in rad/s^2.",
            ),
            DeclareLaunchArgument(
                "waist_max_jerk",
                default_value="3.0",
                description="Maximum commanded waist joint jerk in rad/s^3.",
            ),
            DeclareLaunchArgument(
                "waist_hold_on_lost",
                default_value="true",
                description="Keep publishing the last waist target when detection is lost.",
            ),
            DeclareLaunchArgument(
                "waist_invert_direction",
                default_value="false",
                description="Flip waist yaw direction if the torso turns away.",
            ),
            DeclareLaunchArgument(
                "input_source_retry_sec",
                default_value="2.0",
                description="Seconds between locomotion input-source registration retries.",
            ),
            DeclareLaunchArgument(
                "visual_fallback_enabled",
                default_value="true",
                description="Use bbox size for slow forward motion when LiDAR distance is unavailable.",
            ),
            DeclareLaunchArgument(
                "visual_target_bbox_height_ratio",
                default_value="0.55",
                description="Target person bbox height ratio for visual fallback.",
            ),
            DeclareLaunchArgument(
                "visual_fallback_max_forward_speed",
                default_value="0.12",
                description="Maximum visual-fallback forward speed in m/s.",
            ),
            DeclareLaunchArgument(
                "visual_stop_deadband_ratio",
                default_value="0.04",
                description="Visual fallback bbox-height deadband as image-height ratio.",
            ),
            DeclareLaunchArgument(
                "publish_debug_image",
                default_value="true",
                description="Publish annotated camera image on /x2/person_follow/debug_image.",
            ),
            DeclareLaunchArgument(
                "publish_debug_markers",
                default_value="true",
                description="Publish RViz markers on /x2/person_follow/debug_markers.",
            ),
            DeclareLaunchArgument(
                "publish_status",
                default_value="true",
                description="Publish JSON status on /x2/person_follow/status.",
            ),
            DeclareLaunchArgument(
                "start_image_view",
                default_value="false",
                description="Start rqt_image_view for the debug image.",
            ),
            DeclareLaunchArgument(
                "start_rviz",
                default_value="false",
                description="Start RViz with LiDAR and target marker displays.",
            ),
            Node(
                package="x2_motion_audio_tools",
                executable="x2_person_follow",
                name="x2_person_follow",
                output="screen",
                parameters=[
                    {
                        "device": device,
                        "tts_enabled": ParameterValue(
                            tts_enabled, value_type=bool
                        ),
                        "tts_text": tts_text,
                        "tts_cooldown_sec": ParameterValue(
                            tts_cooldown_sec, value_type=float
                        ),
                        "tts_reset_after_lost_sec": ParameterValue(
                            tts_reset_after_lost_sec, value_type=float
                        ),
                        "follow_enabled": ParameterValue(
                            follow_enabled, value_type=bool
                        ),
                        "stop_distance_m": ParameterValue(
                            stop_distance_m, value_type=float
                        ),
                        "stop_deadband_m": ParameterValue(
                            stop_deadband_m, value_type=float
                        ),
                        "forward_gain": ParameterValue(
                            forward_gain, value_type=float
                        ),
                        "angular_gain": ParameterValue(
                            angular_gain, value_type=float
                        ),
                        "max_forward_speed": ParameterValue(
                            max_forward_speed, value_type=float
                        ),
                        "max_angular_speed": ParameterValue(
                            max_angular_speed, value_type=float
                        ),
                        "max_forward_bearing_deg": ParameterValue(
                            max_forward_bearing_deg, value_type=float
                        ),
                        "lidar_window_deg": ParameterValue(
                            lidar_window_deg, value_type=float
                        ),
                        "lidar_angle_offset_deg": ParameterValue(
                            lidar_angle_offset_deg, value_type=float
                        ),
                        "lidar_min_range_m": ParameterValue(
                            lidar_min_range_m, value_type=float
                        ),
                        "lidar_max_range_m": ParameterValue(
                            lidar_max_range_m, value_type=float
                        ),
                        "waist_tracking_enabled": ParameterValue(
                            waist_tracking_enabled, value_type=bool
                        ),
                        "waist_state_topic": waist_state_topic,
                        "waist_command_topic": waist_command_topic,
                        "waist_yaw_gain": ParameterValue(
                            waist_yaw_gain, value_type=float
                        ),
                        "waist_soft_limit_deg": ParameterValue(
                            waist_soft_limit_deg, value_type=float
                        ),
                        "waist_max_velocity": ParameterValue(
                            waist_max_velocity, value_type=float
                        ),
                        "waist_max_acceleration": ParameterValue(
                            waist_max_acceleration, value_type=float
                        ),
                        "waist_max_jerk": ParameterValue(
                            waist_max_jerk, value_type=float
                        ),
                        "waist_hold_on_lost": ParameterValue(
                            waist_hold_on_lost, value_type=bool
                        ),
                        "waist_invert_direction": ParameterValue(
                            waist_invert_direction, value_type=bool
                        ),
                        "input_source_retry_sec": ParameterValue(
                            input_source_retry_sec, value_type=float
                        ),
                        "visual_fallback_enabled": ParameterValue(
                            visual_fallback_enabled, value_type=bool
                        ),
                        "visual_target_bbox_height_ratio": ParameterValue(
                            visual_target_bbox_height_ratio, value_type=float
                        ),
                        "visual_fallback_max_forward_speed": ParameterValue(
                            visual_fallback_max_forward_speed, value_type=float
                        ),
                        "visual_stop_deadband_ratio": ParameterValue(
                            visual_stop_deadband_ratio, value_type=float
                        ),
                        "publish_debug_image": ParameterValue(
                            publish_debug_image, value_type=bool
                        ),
                        "publish_debug_markers": ParameterValue(
                            publish_debug_markers, value_type=bool
                        ),
                        "publish_status": ParameterValue(
                            publish_status, value_type=bool
                        ),
                    }
                ],
            ),
            Node(
                package="rqt_image_view",
                executable="rqt_image_view",
                name="x2_person_follow_image_view",
                arguments=["/x2/person_follow/debug_image"],
                condition=IfCondition(start_image_view),
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="x2_person_follow_rviz",
                arguments=["-d", debug_rviz_config],
                condition=IfCondition(start_rviz),
                output="screen",
            ),
        ]
    )
