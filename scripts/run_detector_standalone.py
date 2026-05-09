#!/usr/bin/env python3
"""
Standalone YOLO Person Detector for Agibot X2.

This script runs the YOLO person detector directly without needing
a full colcon build. It subscribes to the robot's camera topic and
publishes detections.

Usage (on the robot or with ROS2 sourced):
    python3 run_detector_standalone.py
    python3 run_detector_standalone.py --device cuda
    python3 run_detector_standalone.py --camera rgb_head_rear
    python3 run_detector_standalone.py --model yolov8s.pt --confidence 0.6

Usage (with webcam for local testing):
    python3 run_detector_standalone.py --webcam
"""

import argparse
import sys
import os

# Add parent directory to path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time

from yolo_person_detector.yolo_wrapper import YOLOWrapper


# Agibot X2 camera topics
CAMERA_TOPICS = {
    'rgbd_head_front': '/aima/hal/sensor/rgbd_head_front/rgb_image',
    'rgb_head_rear': '/aima/hal/sensor/rgb_head_rear/rgb_image',
    'stereo_head_front_left': '/aima/hal/sensor/stereo_head_front_left/rgb_image',
    'stereo_head_front_right': '/aima/hal/sensor/stereo_head_front_right/rgb_image',
}


def run_webcam_mode(yolo: YOLOWrapper, cam_index: int = 0, save_video: str = None):
    """Run detection on local webcam for testing.

    Args:
        yolo: YOLOWrapper instance.
        cam_index: Camera device index (0 = first camera, 1 = second, etc.).
        save_video: If set, path to save the annotated output video (e.g. 'output.mp4').
    """
    print(f"Opening camera index {cam_index}...")
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera at index {cam_index}.")
        print()
        print("On macOS, you need to grant camera permission:")
        print("  System Settings → Privacy & Security → Camera")
        print("  Enable access for Terminal (or your IDE).")
        print()
        print("Tips:")
        print("  - Use --cam-index to try a different camera (e.g. --cam-index 1)")
        print("  - If running from VS Code/Kiro, try Terminal.app instead")
        print("  - Disconnect iPhone Continuity Camera if you want the built-in cam")
        return

    # Get camera properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Camera opened: {w}x{h} @ {fps:.0f}fps")

    # Set up video writer if saving
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_video, fourcc, fps, (w, h))
        if writer.isOpened():
            print(f"Recording to: {save_video}")
        else:
            print(f"WARNING: Could not open video writer for {save_video}")
            writer = None

    print("Running YOLO person detection. Press 'q' to quit.")
    print()

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed, stopping.")
            break

        result = yolo.detect(frame)
        frame_count += 1

        # Draw detections
        for det in result.detections:
            # Bounding box
            cv2.rectangle(
                frame,
                (det.bbox_x, det.bbox_y),
                (det.bbox_x + det.bbox_w, det.bbox_y + det.bbox_h),
                (0, 255, 0), 2,
            )
            # Label background
            label = f'Person {det.confidence:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(
                frame,
                (det.bbox_x, det.bbox_y - label_size[1] - 6),
                (det.bbox_x + label_size[0] + 4, det.bbox_y),
                (0, 255, 0), -1,
            )
            cv2.putText(
                frame, label, (det.bbox_x + 2, det.bbox_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1,
            )

        # Stats overlay
        elapsed = time.time() - start_time
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        stats = f'{result.inference_time_ms:.0f}ms | {len(result.detections)} persons | {avg_fps:.1f} fps'
        cv2.putText(
            frame, stats, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )

        # Save frame to video
        if writer is not None:
            writer.write(frame)

        # Show window
        cv2.imshow('YOLO Person Detector - Agibot X2', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
        print(f"\nVideo saved to: {os.path.abspath(save_video)}")
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\nProcessed {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps)")


def run_ros2_mode(yolo: YOLOWrapper, camera: str):
    """Run detection as a ROS2 subscriber/publisher."""
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float32
        from cv_bridge import CvBridge
    except ImportError as e:
        print(f"ERROR: ROS2 not available: {e}")
        print("Use --webcam flag for local testing without ROS2")
        sys.exit(1)

    SENSOR_QOS = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        durability=DurabilityPolicy.VOLATILE,
    )

    class DetectorNode(Node):
        def __init__(self):
            super().__init__('yolo_detector_standalone')
            self.bridge = CvBridge()
            self.yolo = yolo

            topic = CAMERA_TOPICS[camera]
            self.sub = self.create_subscription(Image, topic, self.callback, SENSOR_QOS)
            self.pub = self.create_publisher(Image, '/yolo/detection_image', SENSOR_QOS)
            self.get_logger().info(f'Listening on: {topic}')

        def callback(self, msg):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except Exception as e:
                self.get_logger().warn(f'Image conversion failed: {e}')
                return

            result = self.yolo.detect(cv_image)

            # Draw detections
            for det in result.detections:
                cv2.rectangle(
                    cv_image,
                    (det.bbox_x, det.bbox_y),
                    (det.bbox_x + det.bbox_w, det.bbox_y + det.bbox_h),
                    (0, 255, 0), 2,
                )
                label = f'Person {det.confidence:.2f}'
                cv2.putText(
                    cv_image, label, (det.bbox_x, det.bbox_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )

            cv2.putText(
                cv_image,
                f'{result.inference_time_ms:.1f}ms | {len(result.detections)} persons',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
            )

            img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            img_msg.header = msg.header
            self.pub.publish(img_msg)

            self.get_logger().info(
                f'{len(result.detections)} persons | {result.inference_time_ms:.1f}ms',
                throttle_duration_sec=1.0,
            )

    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description='YOLO Person Detector for Agibot X2')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', default='cpu', help='Device: cpu, cuda, mps (cpu is fastest for yolov8n on Mac)')
    parser.add_argument('--camera', default='rgbd_head_front',
                        choices=list(CAMERA_TOPICS.keys()),
                        help='ROS2 camera to use (robot mode)')
    parser.add_argument('--webcam', action='store_true', help='Use local webcam for testing')
    parser.add_argument('--cam-index', type=int, default=0,
                        help='Camera device index (0=first camera, 1=second, etc.)')
    parser.add_argument('--save-video', type=str, default=None,
                        help='Save annotated output to video file (e.g. output.mp4)')
    parser.add_argument('--input-size', type=int, default=640, help='Model input size')
    parser.add_argument('--list-cameras', action='store_true',
                        help='List available cameras and exit')

    args = parser.parse_args()

    # List cameras mode
    if args.list_cameras:
        print("Scanning cameras...")
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"  Index {i}: {w}x{h}")
                cap.release()
        return

    print(f"Loading YOLO model: {args.model} on {args.device}")
    yolo = YOLOWrapper(
        model_path=args.model,
        confidence_threshold=args.confidence,
        device=args.device,
        input_size=args.input_size,
    )
    print("Model loaded successfully!")

    if args.webcam:
        run_webcam_mode(yolo, cam_index=args.cam_index, save_video=args.save_video)
    else:
        run_ros2_mode(yolo, args.camera)


if __name__ == '__main__':
    main()
