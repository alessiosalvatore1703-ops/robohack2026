# YOLO Person Detector for Agibot X2

Real-time person detection pipeline using YOLOv8 for the Agibot X2 humanoid robot.

## Quick Start

### 1. Install dependencies

```bash
pip install ultralytics opencv-python numpy
```

### 2. Test locally with webcam

```bash
cd yolo_person_detector
python3 scripts/run_detector_standalone.py --webcam
```

This opens your webcam and runs person detection locally — no ROS2 needed.

### 3. Run on the robot (via SSH)

```bash
ssh agibot@<ROBOT_IP>

# Source ROS2
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# Run standalone detector on RGBD front camera
python3 run_detector_standalone.py --camera rgbd_head_front --device cpu

# Or with GPU
python3 run_detector_standalone.py --camera rgbd_head_front --device cuda
```

### 4. Full ROS2 pipeline (after colcon build)

```bash
# Build
cd ~/ros2_ws
colcon build --packages-select yolo_person_detector

# Launch
ros2 launch yolo_person_detector yolo_pipeline.launch.py device:=cuda
```

## Architecture

```
Camera Topic ──→ Camera Selector ──→ YOLO Detector ──→ Detections
                                          │
                                          ▼
                                   Annotated Image ──→ WebRTC/Visualization
```

## Camera Topics (Agibot X2)

| Camera | Topic |
|--------|-------|
| RGBD Head Front | `/aima/hal/sensor/rgbd_head_front/rgb_image` |
| RGB Head Rear | `/aima/hal/sensor/rgb_head_rear/rgb_image` |
| Stereo Left | `/aima/hal/sensor/stereo_head_front_left/rgb_image` |
| Stereo Right | `/aima/hal/sensor/stereo_head_front_right/rgb_image` |

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/yolo/detections` | `vision_msgs/Detection2DArray` | Person detections |
| `/yolo/detection_image` | `sensor_msgs/Image` | Annotated camera image |
| `/yolo/inference_time` | `std_msgs/Float32` | Inference time (ms) |
| `/yolo/visualization_image` | `sensor_msgs/Image` | Visualization output |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | `yolov8n.pt` | YOLO model weights |
| `confidence_threshold` | `0.5` | Min detection confidence |
| `device` | `cpu` | Inference device (cpu/cuda/mps) |
| `active_camera` | `rgbd_head_front` | Camera to use |

## Model Options

| Model | Speed (GPU) | Speed (CPU) | Accuracy |
|-------|-------------|-------------|----------|
| `yolov8n.pt` | ~3ms | ~50ms | Good |
| `yolov8s.pt` | ~5ms | ~100ms | Better |
| `yolov8m.pt` | ~10ms | ~200ms | Best |

For real-time on CPU, use `yolov8n.pt`. For GPU, any model works at 30+ fps.
