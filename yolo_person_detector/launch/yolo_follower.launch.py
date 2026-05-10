"""
Launch file for YOLO Detection + Person Follower on Agibot X2.

Runs the full pipeline: camera selector -> YOLO detector -> person follower.
The follower is DISABLED by default for safety. Enable it with follower_enabled:=true
only when the robot is in Stable Stand.

Usage (detection only, robot does not move):
    ros2 launch yolo_person_detector yolo_follower.launch.py

Usage (detection + active following — robot will move):
    # The follower will transition the robot to STAND_DEFAULT itself
    # (DAMPING -> JOINT -> STAND) via SetMcAction. If you prefer to do
    # that manually, set auto_enable_locomotion:=false and run:
    #     ros2 run py_examples set_mc_action SD
    ros2 launch yolo_person_detector yolo_follower.launch.py follower_enabled:=true
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory('yolo_person_detector')
    config_file = os.path.join(pkg_dir, 'config', 'yolo_params.yaml')

    camera_arg = DeclareLaunchArgument(
        'camera',
        default_value='stereo_head_front_left',
        description=(
            'Camera: rgb_head_front_center, rgb_head_rear, '
            'stereo_head_front_left, stereo_head_front_right'
        ),
    )
    topic_type_arg = DeclareLaunchArgument(
        'topic_type',
        default_value='rgb_image',
        description='rgb_image (raw) or rgb_image_compressed (JPEG)',
    )
    device_arg = DeclareLaunchArgument(
        'device', default_value='cpu',
        description='Inference device: cpu, cuda, mps',
    )
    model_arg = DeclareLaunchArgument(
        'model', default_value='yolov8n.pt',
        description='YOLO model file',
    )
    follower_arg = DeclareLaunchArgument(
        'follower_enabled', default_value='false',
        description='Enable active robot following (robot will move!)',
    )

    camera_selector = Node(
        package='yolo_person_detector',
        executable='camera_selector_node',
        name='camera_selector',
        parameters=[
            config_file,
            {
                'active_camera': LaunchConfiguration('camera'),
                'topic_type': LaunchConfiguration('topic_type'),
            },
        ],
        output='screen',
    )

    yolo_detector = Node(
        package='yolo_person_detector',
        executable='yolo_detector_node',
        name='yolo_detector',
        parameters=[
            config_file,
            {
                'model_path': LaunchConfiguration('model'),
                'device': LaunchConfiguration('device'),
                'input_topic': '/yolo/input_image',
            },
        ],
        output='screen',
    )

    person_follower = Node(
        package='yolo_person_detector',
        executable='person_follower_node',
        name='person_follower',
        parameters=[
            config_file,
            {'enabled': LaunchConfiguration('follower_enabled')},
        ],
        output='screen',
    )

    return LaunchDescription([
        camera_arg,
        topic_type_arg,
        device_arg,
        model_arg,
        follower_arg,
        camera_selector,
        yolo_detector,
        person_follower,
    ])
