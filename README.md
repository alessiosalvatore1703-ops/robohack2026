# Person Detection & Following for Agibot X2

Real-time stereo person detection and follow-me behaviour for the Agibot X2 humanoid robot, built on YOLOv8 and ROS2 Humble.

The repo contains two ROS2 packages:

| Package | Responsibility |
|---------|---------------|
| `robot_vision` | Stereo YOLO detection + depth estimation. Publishes the shared perception contract `/stereo_person/target_point`. |
| `robot_motion` | Head tracking, body following, arm pose. Consumes `/stereo_person/target_point` and drives the robot. |

On the robot, clone into `~/ros2_ws/src/robohack2026/` and build with `colcon`.

---

## What's in the box

### `robot_vision` nodes

| Node | Entry point | Purpose |
|------|-------------|---------|
| `stereo_detector` | `stereo_detector_node.py` | Subscribes to both stereo compressed streams, runs YOLO on the left image, estimates per-person stereo depth, publishes annotated JPEG and `/stereo_person/target_point` |
| `view_stereo` | `view_stereo_node.py` | OpenCV viewer for the annotated stream — run on a laptop to monitor the pipeline output |

### `robot_motion` nodes

| Node | Entry point | Purpose |
|------|-------------|---------|
| `head_tracker` | `head_tracker_node.py` | Slews head yaw joint toward the target point using Ruckig trajectory |
| `body_follower` | `body_follower_node.py` | High-level locomotion supervisor: walks toward the person, stops at the configured distance, fires the arm trigger on arrival |
| `arm_pose` | `arm_pose_node.py` | One-shot arm raise into the assist pose; holds until a head-pat Bool releases it |
| `person_follow` | `person_follow_node.py` | Alternative single-launch follower that bundles its own YOLO + LiDAR distance (no stereo depth required) |

### Launch files

| File | Package | What it starts |
|------|---------|---------------|
| `robot_motion/launch/person_follow.launch.py` | `robot_motion` | **Full pipeline** — stereo_detector + head_tracker + body_follower + arm_pose |
| `robot_motion/launch/person_follow_lidar.launch.py` | `robot_motion` | Alternative: person_follow node only (YOLO + LiDAR, no stereo depth) |
| `robot_vision/launch/stereo_detector.launch.py` | `robot_vision` | Vision only — no motion |

---

## 1. Deploy to the Agibot X2

### Dependencies

```bash
sudo apt update
sudo apt install -y ros-humble-vision-msgs ros-humble-cv-bridge
pip3 install --user ultralytics opencv-python numpy
```

### Clone and build

```bash
# On the robot (e.g. ssh agi@10.0.1.40)
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone git@github.com:alessiosalvatore1703-ops/robohack2026.git

cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source ~/aimdk/install/setup.bash   # adjust path as needed

colcon build --packages-select robot_vision robot_motion --symlink-install
source install/setup.bash
```

> If `~/aimdk/install/setup.bash` doesn't exist, find it with:
> `find ~ -name "setup.bash" -path "*/install/*" 2>/dev/null`

### Pull updates later

```bash
cd ~/ros2_ws/src/robohack2026
git pull
cd ~/ros2_ws
colcon build --packages-select robot_vision robot_motion --symlink-install
source install/setup.bash
```

---

## 2. Running on the robot

Source all environments in every new terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/aimdk/install/setup.bash
source ~/ros2_ws/install/setup.bash
```

Confirm the stereo streams are alive before anything else:

```bash
ros2 topic hz /aima/hal/sensor/stereo_head_front_left/rgb_image/compressed \
  --qos-reliability best_effort
ros2 topic hz /aima/hal/sensor/stereo_head_front_right/rgb_image/compressed \
  --qos-reliability best_effort
```

### A. Vision only (robot will not move)

```bash
ros2 launch robot_vision stereo_detector.launch.py device:=cpu confidence:=0.5
```

Monitor:

```bash
ros2 topic hz  /stereo_person/target_point
ros2 topic echo /stereo_person/target_point --once
ros2 topic hz  /stereo_person/final_annotated_image/compressed --qos-reliability best_effort

# Live annotated video (run on your laptop with ROS networking pointed at the robot)
ros2 run robot_vision view_stereo
# Optional: scale the window down
ros2 run robot_vision view_stereo --ros-args -p scale:=0.5
```

Pass `baseline_m` if `CameraInfo` does not carry a usable stereo baseline:

```bash
ros2 launch robot_vision stereo_detector.launch.py device:=cuda baseline_m:=0.06
```

### B. Full pipeline — perception + head + base + arms (robot WILL move)

**Safety checklist:**
1. Robot in open, unobstructed area
2. E-stop in reach — `Ctrl+C` stops all velocity publishing; watchdog zeros locomotion within 0.5 s
3. Start with `follow_dry_run:=true` (stage 1 below) before enabling motion

```bash
ros2 launch robot_motion person_follow.launch.py \
  device:=cpu \
  follow_enabled:=true \
  follow_dry_run:=false
```

Runtime enable/disable without killing the launch:

```bash
ros2 topic pub -1 /stereo_person/follow/enable std_msgs/Bool "data: false"
ros2 topic pub -1 /stereo_person/follow/enable std_msgs/Bool "data: true"
```

### C. Alternative: person_follow with LiDAR (no stereo depth)

```bash
ros2 launch robot_motion person_follow_lidar.launch.py device:=cpu follow_enabled:=true
```

---

## 3. Topics & Messages

### Perception contract (robot_vision → robot_motion)

| Topic | Type | Description |
|-------|------|-------------|
| `/stereo_person/target_point` | `geometry_msgs/PointStamped` | Closest detected person's 3D centre in the left camera frame (x, y, z in metres) |
| `/stereo_person/final_annotated_image/compressed` | `sensor_msgs/CompressedImage` | Annotated JPEG with bbox + depth label |
| `/stereo_person/inference_time` | `std_msgs/Float32` | YOLO inference time per frame (ms) |

### Motion inputs / outputs (robot_motion)

| Topic | Type | Description |
|-------|------|-------------|
| `/aima/hal/joint/head/command` | `aimdk_msgs/JointCommandArray` | Head yaw/pitch commands (head_tracker) |
| `/aima/mc/locomotion/velocity` | `aimdk_msgs/McLocomotionVelocity` | Forward + angular velocity (body_follower) |
| `/aima/hal/joint/arm/command` | `aimdk_msgs/JointCommandArray` | Arm pose commands (arm_pose) |
| `/x2/assist/raise_arms_trigger` | `std_msgs/Bool` | `true` starts the arm raise |
| `/x2/assist/head_pat` | `std_msgs/Bool` | `true` releases the arm hold and resumes following |
| `/stereo_person/follow/enable` | `std_msgs/Bool` | Runtime enable/disable of body_follower |

---

## 4. Troubleshooting

**`aimdk_msgs not found` during build**
Source the aimdk workspace before `colcon build`:
```bash
find ~ -name "setup.bash" -path "*/install/*" 2>/dev/null
```

**`cv_bridge` crashes with NumPy 2.x**
The stereo pipeline avoids `cv_bridge`. If another node triggers this, pin NumPy:
```bash
pip3 install --user "numpy<2"
```

**Robot doesn't move with `follow_enabled:=true`**
Put the robot in Stable Stand first:
```bash
ros2 run py_examples set_mc_action SD
ros2 run py_examples get_current_input_source
```
Also check the log for `Input source registered as "body_follower"` or `"person_follower"`.

**Detection slow (>100ms per frame)**
Use `device:=cuda` if a GPU is available. Check:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
ros2 topic hz /stereo_person/inference_time
```

**YOLO model download fails (no internet)**
Copy the weights from your laptop:
```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
scp ~/.cache/ultralytics/yolov8n.pt agi@10.0.1.40:~/ros2_ws/src/robohack2026/
```
Then launch with `model:=~/ros2_ws/src/robohack2026/yolov8n.pt`.

**Camera topic silent**
```bash
ros2 topic hz /aima/hal/sensor/stereo_head_front_left/rgb_image/compressed \
  --qos-reliability best_effort
```
If empty, the HAL sensor service isn't started on the robot.

---

## 5. End-to-end demo: perception → head → base → arm

The full chair-assist pipeline: `stereo_detector` publishes `/stereo_person/target_point`, `head_tracker` points the head, `body_follower` drives the base, and on first arrival into the stop band it fires `/x2/assist/raise_arms_trigger`. `arm_pose` holds the assist pose until a Bool `true` on `/x2/assist/head_pat` releases the arms and resumes following.

Run each stage in order. Do not skip stages.

### 5.0 One-time setup per SSH session

Open one terminal for the launch and a second for inspection. In every terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/aimdk/install/setup.bash
source ~/ros2_ws/install/setup.bash
```

### 5.1 Stage 1: dry run (no motion)

Confirm all nodes come up and the perception contract flows, without any motion commands reaching the hardware.

```bash
ros2 launch robot_motion person_follow.launch.py \
  device:=cpu \
  dry_run:=true \
  follow_enabled:=true \
  follow_dry_run:=true \
  assist_arm_pose_enabled:=true \
  assist_head_pat_enabled:=true
```

Inspect in the second terminal:

```bash
# Perception contract
ros2 topic hz   /stereo_person/target_point
ros2 topic echo /stereo_person/target_point --once

# Head tracker connected but not publishing
ros2 topic info /aima/hal/joint/head/state
ros2 topic info /aima/hal/joint/head/command

# Arm node subscribed but not publishing
ros2 node info /arm_pose
ros2 topic info /x2/assist/raise_arms_trigger
ros2 topic info /aima/hal/joint/arm/command

# Body follower alive, not publishing velocity
ros2 node info /body_follower
ros2 topic info /aima/mc/locomotion/velocity
```

Expected in the launch log:

- `stereo_detector` publishing annotated JPEGs and target points.
- `head_tracker` logging computed yaw targets; no command published.
- `body_follower` printing `APPROACH|STOP_BAND|ALIGN` state roughly once per second.
- `arm_pose` logging "Raise-arms pose node ready", `auto_start=false`, no "Starting arm pose routine."

Stop with `Ctrl+C` before stage 2.

### 5.2 Stage 2: gantry run (robot secured, walks in place)

Preflight:
1. Robot attached to gantry, straps verified.
2. Clear area in front of the stereo cameras; an operator acts as the target.
3. E-stop in reach.
4. Put the robot in Stable Stand:

```bash
ros2 run py_examples set_mc_action SD
ros2 run py_examples get_current_input_source
```

Launch with conservative speed limits:

```bash
ros2 launch robot_motion person_follow.launch.py \
  device:=cpu \
  follow_enabled:=true \
  follow_dry_run:=false \
  follow_max_forward_speed:=0.15 \
  follow_min_forward_speed:=0.10 \
  follow_max_angular_speed:=0.25 \
  follow_max_forward_bearing_deg:=20.0 \
  follow_stop_min_m:=0.45 \
  follow_stop_max_m:=1.0 \
  follow_target_distance_m:=0.85 \
  depth_disparity_percentile:=75.0 \
  assist_arm_pose_enabled:=true \
  assist_head_pat_enabled:=true
```

Verify motion in the second terminal:

```bash
ros2 topic hz   /aima/mc/locomotion/velocity
ros2 topic echo /aima/mc/locomotion/velocity --once
ros2 topic hz   /aima/hal/joint/head/command
# Arm traffic only appears after the first STOP_BAND arrival:
ros2 topic hz   /aima/hal/joint/arm/command
```

Expected physical behaviour:
- Operator steps in front of the cameras → head yaws toward them; base yaws and steps in the target direction.
- Operator moves into the stop band → base stops, head continues tracking, arms lift into assist pose.

Release arm hold and resume following:

```bash
ros2 topic pub -1 /x2/assist/head_pat std_msgs/Bool "data: true"
```

Runtime enable/disable:

```bash
ros2 topic pub -1 /stereo_person/follow/enable std_msgs/Bool "data: false"
ros2 topic pub -1 /stereo_person/follow/enable std_msgs/Bool "data: true"
```

Stop with `Ctrl+C`. The supervisor publishes a zero-velocity stop on shutdown.

### 5.3 Stage 3: off-gantry deployment

Use only after stage 2 has been reproduced reliably.

Preflight:
1. Clear area, no obstacles within 2 m.
2. One operator holds the physical E-stop; a second is the target.
3. Speed limits at or below the gantry numbers.

```bash
ros2 run py_examples set_mc_action SD

ros2 launch robot_motion person_follow.launch.py \
  device:=cpu \
  follow_enabled:=true \
  follow_dry_run:=false \
  follow_max_forward_speed:=0.10 \
  follow_min_forward_speed:=0.08 \
  follow_max_angular_speed:=0.20 \
  follow_max_forward_bearing_deg:=15.0 \
  follow_stop_min_m:=0.60 \
  follow_stop_max_m:=1.10 \
  follow_target_distance_m:=0.90 \
  depth_disparity_percentile:=75.0 \
  assist_arm_pose_enabled:=true \
  assist_head_pat_enabled:=true
```

Emergency stop order: physical E-stop first, then disable follow, then `Ctrl+C`:

```bash
ros2 topic pub -1 /stereo_person/follow/enable std_msgs/Bool "data: false"
```

### 5.4 Stand-alone arm pose sanity check

Run before stage 2 to verify the arm path in isolation.

```bash
ros2 run py_examples set_mc_action SD

ros2 launch robot_motion person_follow.launch.py \
  follow_enabled:=false \
  assist_arm_pose_enabled:=true \
  auto_start:=false \
  shoulder_pitch_deg:=10.0 \
  elbow_bend_deg:=90.0 \
  move_seconds:=3.0

# In a second terminal, trigger the pose:
ros2 topic pub -1 /x2/assist/raise_arms_trigger std_msgs/Bool "data: true"

# Release the hold:
ros2 topic pub -1 /x2/assist/raise_arms_trigger std_msgs/Bool "data: false"
```

### 5.5 Key launch arguments

| Argument | Effect |
|----------|--------|
| `dry_run:=true` | Head tracker logs only, no head-command publish |
| `follow_dry_run:=true` | Body follower logs only, no locomotion publish |
| `follow_enabled:=true/false` | Enable/disable base following at launch time |
| `assist_arm_pose_enabled:=true` | Include `arm_pose` node in the launch |
| `assist_head_pat_enabled:=true` | Body follower waits for `/x2/assist/head_pat` before resuming after first stop |
| `follow_invert_angular:=true` | Flip base yaw sign if the body turns the wrong way |
| `follow_auto_enable_stable_stand:=true` | Supervisor requests STAND_DEFAULT itself via SetMcAction |
| `device:=cuda` | Use GPU for YOLO inference |
| `depth_disparity_percentile:=75.0` | Percentile used for stereo depth estimate (higher = more conservative distance) |

---

## 6. Upgrading from the old package names

If you previously built this repo on the robot under the old package names (`yolo_person_detector` and `x2_motion_audio_tools`), the packages were renamed in this refactor and **the old build artifacts will not be automatically removed**. ROS2 will continue to find and source the old packages from `~/ros2_ws/install/`, meaning `ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py` may still appear to work — but it is running stale code that is no longer maintained or updated.

Before building the new packages, manually clean the old ones:

```bash
cd ~/ros2_ws
rm -rf build/yolo_person_detector  install/yolo_person_detector
rm -rf build/x2_motion_audio_tools install/x2_motion_audio_tools
```

Then build fresh:

```bash
colcon build --packages-select robot_vision robot_motion --symlink-install
source install/setup.bash
```

Any shell scripts, aliases, or saved launch commands on the robot that reference the old names will also need to be updated:

| Old command | New command |
|-------------|-------------|
| `ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py` | `ros2 launch robot_motion person_follow.launch.py` |
| `ros2 launch x2_motion_audio_tools x2_person_follow.launch.py` | `ros2 launch robot_motion person_follow_lidar.launch.py` |
| `ros2 launch yolo_person_detector stereo_person_pipeline.launch.py` | `ros2 launch robot_vision stereo_detector.launch.py` |
