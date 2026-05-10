"""Launch the final compressed stereo person annotation pipeline."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    left_image_topic = LaunchConfiguration("left_image_topic")
    right_image_topic = LaunchConfiguration("right_image_topic")
    left_camera_info_topic = LaunchConfiguration("left_camera_info_topic")
    right_camera_info_topic = LaunchConfiguration("right_camera_info_topic")
    output_topic = LaunchConfiguration("output_topic")
    device = LaunchConfiguration("device")
    model = LaunchConfiguration("model")
    confidence = LaunchConfiguration("confidence")
    input_size = LaunchConfiguration("input_size")
    processing_width = LaunchConfiguration("processing_width")
    max_processing_fps = LaunchConfiguration("max_processing_fps")
    output_width = LaunchConfiguration("output_width")
    baseline_m = LaunchConfiguration("baseline_m")
    sync_slop_sec = LaunchConfiguration("sync_slop_sec")
    right_buffer_size = LaunchConfiguration("right_buffer_size")
    jpeg_quality = LaunchConfiguration("jpeg_quality")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "left_image_topic",
                default_value="/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed",
                description="Left stereo compressed image topic",
            ),
            DeclareLaunchArgument(
                "right_image_topic",
                default_value="/aima/hal/sensor/stereo_head_front_right/rgb_image/compressed",
                description="Right stereo compressed image topic",
            ),
            DeclareLaunchArgument(
                "left_camera_info_topic",
                default_value="/aima/hal/sensor/stereo_head_front_left/camera_info",
                description="Left stereo camera info topic",
            ),
            DeclareLaunchArgument(
                "right_camera_info_topic",
                default_value="/aima/hal/sensor/stereo_head_front_right/camera_info",
                description="Right stereo camera info topic",
            ),
            DeclareLaunchArgument(
                "output_topic",
                default_value="/stereo_person/final_annotated_image/compressed",
                description="Final annotated compressed JPEG output topic",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="cpu",
                description="YOLO inference device: cpu, cuda, or mps",
            ),
            DeclareLaunchArgument(
                "model",
                default_value="yolov8n.pt",
                description="YOLO model file path",
            ),
            DeclareLaunchArgument(
                "confidence",
                default_value="0.5",
                description="YOLO confidence threshold",
            ),
            DeclareLaunchArgument(
                "input_size",
                default_value="416",
                description="YOLO model input image size",
            ),
            DeclareLaunchArgument(
                "processing_width",
                default_value="640",
                description="Working image width for YOLO/depth; 0 keeps source size",
            ),
            DeclareLaunchArgument(
                "max_processing_fps",
                default_value="5.0",
                description="Maximum processing rate; latest frames replace queued frames",
            ),
            DeclareLaunchArgument(
                "output_width",
                default_value="960",
                description="Final JPEG width; 0 publishes full source size",
            ),
            DeclareLaunchArgument(
                "baseline_m",
                default_value="0.0578",
                description="Stereo baseline in meters; default measured from CameraInfo",
            ),
            DeclareLaunchArgument(
                "sync_slop_sec",
                default_value="0.10",
                description="Maximum timestamp difference between left and right frames",
            ),
            DeclareLaunchArgument(
                "right_buffer_size",
                default_value="20",
                description="Number of recent right frames available for sync matching",
            ),
            DeclareLaunchArgument(
                "jpeg_quality",
                default_value="75",
                description="JPEG quality for annotated compressed outputs",
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
                        "device": device,
                        "model_path": model,
                        "confidence_threshold": confidence,
                        "input_size": input_size,
                        "processing_width": processing_width,
                        "max_processing_fps": max_processing_fps,
                        "output_width": output_width,
                        "baseline_m": baseline_m,
                        "sync_slop_sec": sync_slop_sec,
                        "right_buffer_size": right_buffer_size,
                        "jpeg_quality": jpeg_quality,
                    }
                ],
            ),
        ]
    )
