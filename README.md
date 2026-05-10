# YOLO Person Detector & Follower for Agibot X2

Real-time person detection and follow-me behavior for the Agibot X2 humanoid robot, built on YOLOv8 and ROS2 Humble.

**This repo contains the `yolo_person_detector` ROS2 package.** On the robot, clone it into `~/ros2_ws/src/robohack2026/` and build with `colcon`.

## What's in the box

| Node | Purpose |
|------|---------|
| `camera_selector_node` | Subscribes to one of the robot's cameras, republishes on a unified topic |
| `yolo_detector_node` | Runs YOLOv8 person detection, publishes bboxes + annotated image |
| `visualization_node` | Overlays detections on camera frames for monitoring (rqt / web video) |
| `person_follower_node` | Visual-servoing follower: robot centers on & approaches the detected person |

Two launch files:

- `yolo_pipeline.launch.py` — **detection only** (robot does not move)
- `yolo_follower.launch.py` — **detection + active following** (robot moves!)

---

## 1. Local testing on MacBook (no robot)

### Setup

```bash
conda create -n yolo_detector python=3.10 -y
conda activate yolo_detector
pip install -r requirements.txt
pip install pytest
```

### Run against your webcam

```bash
conda activate yolo_detector

# List available cameras
python3 yolo_person_detector/scripts/run_detector_standalone.py --list-cameras

# Run detection and save a debug video (index 1 = Continuity Camera on most Macs)
python3 yolo_person_detector/scripts/run_detector_standalone.py --webcam --cam-index 1 --save-video output.mp4
```

Press `q` to quit.

### Run unit tests

```bash
pytest yolo_person_detector/test/ -v
```

---

## 2. Deploy to the Agibot X2

The robot needs:
- ROS2 Humble (`/opt/ros/humble`)
- `aimdk_msgs` available — source whatever aimdk workspace you have on the robot
- `ros-humble-vision-msgs` (install with apt)
- Python packages: `ultralytics`, `opencv-python`, `numpy`

### Install missing dependencies on the robot

```bash
sudo apt update
sudo apt install -y ros-humble-vision-msgs ros-humble-cv-bridge
pip3 install --user ultralytics opencv-python numpy
```

### Clone + build the workspace

```bash
# On the robot (e.g. ssh agi@10.0.1.40)
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone git@github.com:alessiosalvatore1703-ops/robohack2026.git
cd ~/ros2_ws

# Source your environments (adjust aimdk path as needed)
source /opt/ros/humble/setup.bash
source ~/aimdk/install/setup.bash   # wherever aimdk_msgs lives on your robot

# Build (colcon discovers yolo_person_detector inside src/robohack2026/)
colcon build --packages-select yolo_person_detector --symlink-install
source install/setup.bash
```

> If `~/aimdk/install/setup.bash` doesn't exist, locate it with:
> `find ~ -name "setup.bash" -path "*/install/*" 2>/dev/null`

### Pull updates later

```bash
cd ~/ros2_ws/src/robohack2026
git pull
cd ~/ros2_ws
colcon build --packages-select yolo_person_detector --symlink-install
source install/setup.bash
```

If the build reports a missing launch file under
`src/robohack2026/install/yolo_person_detector/...`, a previous build was run
from inside the repository instead of the workspace root. Clean the stale nested
build products, then rebuild from `~/ros2_ws`:

```bash
cd ~/ros2_ws/src/robohack2026
rm -rf build install log
git pull

cd ~/ros2_ws
rm -rf build/yolo_person_detector install/yolo_person_detector
colcon build --packages-select yolo_person_detector --symlink-install
source install/setup.bash
```

---

## 3. Running on the robot

SSH in and source all environments:

```bash
source /opt/ros/humble/setup.bash
source ~/aimdk/install/setup.bash
source ~/ros2_ws/install/setup.bash
```

### A. Detection only (safe, robot will not move)

```bash
ros2 launch yolo_person_detector yolo_pipeline.launch.py
```

Optional arguments:

```bash
ros2 launch yolo_person_detector yolo_pipeline.launch.py \
    camera:=rgb_head_front_center \
    device:=cuda \
    confidence:=0.6
```

Monitor the output:

```bash
ros2 topic echo /yolo/detections                       # detection list
ros2 run rqt_image_view rqt_image_view /yolo/detection_image
```

### B. Person follower (robot WILL move)

**Safety checklist before enabling the follower:**

1. Robot is in an open, unobstructed area
2. E-stop ready (Ctrl+C stops publishing velocity; watchdog stops robot within 0.5s of lost detection)
3. Start with default conservative gains

**Start sequence:**

```bash
# Launch with following enabled — the node will put the robot into
# LOCOMOTION_DEFAULT itself (DD -> JD -> LD via SetMcAction).
ros2 launch yolo_person_detector yolo_follower.launch.py \
    follower_enabled:=true \
    device:=cuda
```

You can also leave `follower_enabled:=false` and flip the switch at runtime:

```bash
ros2 topic pub -1 /yolo/follower/enable std_msgs/Bool "data: true"
# ...and to stop following without killing the node:
ros2 topic pub -1 /yolo/follower/enable std_msgs/Bool "data: false"
```

If you prefer to drive the motion-mode state machine yourself (e.g. with
`ros2 run py_examples set_mc_action LD`), set `auto_enable_locomotion: false`
in `config/yolo_params.yaml` so the node won't issue `SetMcAction` calls.

Stop with `Ctrl+C` — watchdog zeros velocity within 50ms.

### How the follower behaves

Visual servoing (no depth required):

- **Horizontal**: rotates to keep the person's bbox centered
- **Forward**: walks toward the person until bbox height reaches ~60% of image height
- **Safety**: stops if no detection for > 0.5s
- **Target**: largest bounding box (closest person)

All parameters tunable in `config/yolo_params.yaml` → `person_follower`:

| Param | Default | Purpose |
|-------|---------|---------|
| `target_bbox_height_ratio` | 0.6 | Target person size (fraction of image height) |
| `forward_gain` | 0.8 | P gain on forward error |
| `angular_gain` | 1.5 | P gain on lateral error |
| `max_forward_speed` | 0.6 m/s | Cap on forward velocity |
| `max_angular_speed` | 0.8 rad/s | Cap on rotation |
| `center_deadzone_px` | 50 | Ignore horizontal errors smaller than this |
| `watchdog_timeout_sec` | 0.5 | Stop after this long without detection |

---

## 4. Topics & Messages

### Inputs (subscribed)

| Topic | Type |
|-------|------|
| `/aima/hal/sensor/rgb_head_front_center/rgb_image` | `sensor_msgs/Image` |
| `/aima/hal/sensor/rgb_head_rear/rgb_image` | `sensor_msgs/Image` |
| `/aima/hal/sensor/stereo_head_front_left/rgb_image` | `sensor_msgs/Image` |

### Outputs (published)

| Topic | Type | Description |
|-------|------|-------------|
| `/yolo/input_image` | `sensor_msgs/Image` | Selected camera stream |
| `/yolo/detections` | `vision_msgs/Detection2DArray` | Person detections (bbox + score) |
| `/yolo/detection_image` | `sensor_msgs/Image` | Image annotated with bboxes |
| `/yolo/inference_time` | `std_msgs/Float32` | Inference time per frame (ms) |
| `/yolo/visualization_image` | `sensor_msgs/Image` | Visualization node output |
| `/aima/mc/locomotion/velocity` | `aimdk_msgs/McLocomotionVelocity` | Velocity commands (follower mode only) |

---

## 5. Troubleshooting

**`aimdk_msgs not found` during build**
You forgot to source the aimdk workspace. Find it with:
```bash
find ~ -name "setup.bash" -path "*/install/*" 2>/dev/null
```
Source that file before running `colcon build`.

**`vision_msgs` error during build**
```bash
sudo apt install -y ros-humble-vision-msgs
```

**`cv_bridge` crashes with NumPy 2.x**
ROS Humble's `cv_bridge` is usually built against NumPy 1.x. If you see
`AttributeError: _ARRAY_API not found`, either use the stereo launch files in
this repo, which avoid `cv_bridge`, or pin the robot Python environment back to
NumPy 1.x:
```bash
pip3 install --user "numpy<2"
```

**Robot doesn't move with `follower_enabled:=true`**
Set locomotion mode first: `ros2 run py_examples set_mc_action LD`. Also check the log for `Input source registered as "person_follower"`.

**Detection slow (>100ms per frame)**
Use `device:=cuda` if the robot has a GPU. Check frame rate: `ros2 topic hz /yolo/detections`.

**YOLO model download fails (no internet)**
Copy `yolov8n.pt` from your Mac:
```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
scp ~/.cache/ultralytics/yolov8n.pt agi@10.0.1.40:~/ros2_ws/src/robohack2026/yolo_person_detector/
# Then launch with model:=~/ros2_ws/src/robohack2026/yolo_person_detector/yolov8n.pt
```

**Camera topic silent**
Check the sensor is running: `ros2 topic hz /aima/hal/sensor/rgb_head_front_center/rgb_image`. If empty, the HAL sensor service isn't started on the robot.
