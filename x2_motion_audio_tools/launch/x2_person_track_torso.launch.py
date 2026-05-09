"""Launch stereo-camera person detection, TTS greeting, and torso tracking."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    device = LaunchConfiguration("device")
    tts_enabled = LaunchConfiguration("tts_enabled")
    tts_text = LaunchConfiguration("tts_text")
    tts_cooldown_sec = LaunchConfiguration("tts_cooldown_sec")
    tts_reset_after_lost_sec = LaunchConfiguration("tts_reset_after_lost_sec")
    follow_enabled = LaunchConfiguration("follow_enabled")
    stop_distance_m = LaunchConfiguration("stop_distance_m")
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
                default_value="false",
                description="When true, publish locomotion commands.",
            ),
            DeclareLaunchArgument(
                "stop_distance_m",
                default_value="1.2",
                description="Distance where the robot stops approaching.",
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
                default_value="true",
                description="When true, turn the torso toward visible people.",
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
            Node(
                package="x2_motion_audio_tools",
                executable="x2_person_track_torso",
                name="x2_person_track_torso",
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
                    }
                ],
            ),
        ]
    )
