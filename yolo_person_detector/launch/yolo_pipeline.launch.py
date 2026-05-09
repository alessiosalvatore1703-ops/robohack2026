"""
Launch file for the YOLO Person Detection Pipeline on Agibot X2.

Usage:
    ros2 launch yolo_person_detector yolo_pipeline.launch.py
    ros2 launch yolo_person_detector yolo_pipeline.launch.py device:=cuda
    ros2 launch yolo_person_detector yolo_pipeline.launch.py camera:=rgb_head_rear
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package share directory for config files
    pkg_dir = get_package_share_directory('yolo_person_detector')
    config_file = os.path.join(pkg_dir, 'config', 'yolo_params.yaml')

    # Launch arguments
    camera_arg = DeclareLaunchArgument(
        'camera',
        default_value='stereo_head_front_left',
        description=(
            'Active camera (rgb_head_front_center, rgb_head_rear, '
            'stereo_head_front_left, stereo_head_front_right)'
        ),
    )
    topic_type_arg = DeclareLaunchArgument(
        'topic_type',
        default_value='rgb_image',
        description='rgb_image (raw) or rgb_image_compressed (JPEG)',
    )
    model_arg = DeclareLaunchArgument(
        'model',
        default_value='yolov8n.pt',
        description='YOLO model file path',
    )
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cpu',
        description='Inference device (cpu, cuda, mps)',
    )
    confidence_arg = DeclareLaunchArgument(
        'confidence',
        default_value='0.5',
        description='Detection confidence threshold',
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

    # YOLO Detector Node
    yolo_detector = Node(
        package='yolo_person_detector',
        executable='yolo_detector_node',
        name='yolo_detector',
        parameters=[
            config_file,
            {
                'model_path': LaunchConfiguration('model'),
                'device': LaunchConfiguration('device'),
                'confidence_threshold': LaunchConfiguration('confidence'),
                'input_topic': '/yolo/input_image',
            },
        ],
        output='screen',
    )

    # Visualization Node
    visualization = Node(
        package='yolo_person_detector',
        executable='visualization_node',
        name='yolo_visualization',
        parameters=[config_file],
        output='screen',
    )

    return LaunchDescription([
        camera_arg,
        topic_type_arg,
        model_arg,
        device_arg,
        confidence_arg,
        camera_selector,
        yolo_detector,
        visualization,
    ])
