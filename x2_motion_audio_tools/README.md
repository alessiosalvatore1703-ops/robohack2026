# X2 Motion and Audio Tools

ROS 2 helper nodes for AgiBot X2 audio, speech, open-loop coordinate motion,
forward/backward movement, torso person tracking, and cautious arm raising.

This package is intentionally separate from `yolo_person_detector` so the
existing detection and follower code remains unchanged. The torso tracker reuses
that package's YOLO wrapper, but commands only the X2 HAL waist joint by default.

## Build

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
source ~/aimdk/install/setup.bash
colcon build --packages-up-to x2_motion_audio_tools --symlink-install
source install/setup.bash
```

For the AWS voice assistant and transcription logger:

```bash
python3 -m pip install -r src/x2_motion_audio_tools/requirements-voice.txt
```

For person detection and torso tracking, install the YOLO package dependencies:

```bash
python3 -m pip install -r src/yolo_person_detector/requirements.txt
sudo apt install ros-humble-cv-bridge ros-humble-sensor-msgs-py
```

## Audio and Voice

Log raw sound level without AWS:

```bash
ros2 run x2_motion_audio_tools x2_sound_logger --stream-id 2
```

Log VAD activity and optionally transcribe utterances:

```bash
ros2 run x2_motion_audio_tools x2_mic_logger --stream-id 2 --verbose-levels
ros2 run x2_motion_audio_tools x2_mic_logger --stream-id 2 --transcribe
```

Run the Bedrock voice assistant:

```bash
export AWS_REGION=us-east-1
export BEDROCK_MODEL_ID=us.amazon.nova-2-lite-v1:0
export TRANSCRIBE_LANGUAGE_CODE=en-US
ros2 run x2_motion_audio_tools x2_bedrock_voice_assistant --stream-id 2
```

`--stream-id 1` uses the onboard mic, `--stream-id 2` uses the external mic,
and `--stream-id 0` accepts either stream.

## Coordinate Offset Motion

`x2_go_to_offset_raise_arms` accepts a target distance and bearing, computes a
point 50 cm to the left or right of that coordinate, walks there using
step-sized velocity pulses, optionally turns to face the target, then raises
both arms.

Angle convention:

- `0` degrees means straight ahead
- positive degrees are to the robot's left
- negative degrees are to the robot's right

Dry-run the math without moving:

```bash
ros2 run x2_motion_audio_tools x2_go_to_offset_raise_arms --dry-run
```

Run movement only first:

```bash
ros2 run py_examples set_mc_action LD
ros2 run x2_motion_audio_tools x2_go_to_offset_raise_arms --skip-arms
```

Run with known values:

```bash
ros2 run x2_motion_audio_tools x2_go_to_offset_raise_arms \
  --distance-m 2.0 \
  --angle-deg 15 \
  --side left
```

## Turn Toward Person With TTS

`x2_turn_to_person_tts` says "On my way" through the X2 TTS service, then
rotates the torso/waist toward an input bearing. It uses the HAL waist joint
topic, not locomotion velocity, so it should not step the legs. It accepts
distance too so face/person recognition can pass the same estimate shape later,
but the torso turn only uses the angle.

Dry-run the turn without moving:

```bash
ros2 run x2_motion_audio_tools x2_turn_to_person_tts --dry-run --angle-deg 25
```

Run it on the robot:

```bash
ros2 run x2_motion_audio_tools x2_turn_to_person_tts --angle-deg 25
```

If the torso rotates the wrong direction, add `--invert-waist-direction`.

Future face-recognition handoff:

```bash
ros2 run x2_motion_audio_tools x2_turn_to_person_tts \
  --distance-m FACE_DISTANCE \
  --angle-deg FACE_ANGLE
```

## Person Torso Tracking

`x2_person_follow` runs YOLO person detection on either top front stereo camera,
says "Hello" once per visible encounter, reads the chest `PointCloud2` LiDAR,
logs camera/YOLO/LiDAR details, and turns `waist_yaw_joint` so the torso keeps
facing the selected person. With `follow_enabled:=false`, it does not publish
locomotion velocity, so the legs should not move.

Run torso tracking without walking:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py follow_enabled:=false
```

By default this uses `/aima/hal/sensor/stereo_head_front_left/rgb_image` and
`/aima/hal/sensor/lidar_chest_front/lidar_pointcloud`. To use the right stereo
camera or compressed stream:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  camera_topic_type:=right_rgb_image

ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  camera_topic_type:=left_rgb_image_compressed
```

If you want detection logs only, disable waist tracking too:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  follow_enabled:=false \
  waist_tracking_enabled:=false
```

Expected detection logs include person count, selected bbox/confidence, camera
bearing, base bearing, camera FPS, LiDAR FPS, valid point count, sector point
count, and the estimated distance.

If the torso turns away from the person, flip the waist sign:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  follow_enabled:=false \
  waist_invert_direction:=true
```

If the LiDAR distance samples the wrong direction, tune the sector:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  follow_enabled:=false \
  lidar_angle_offset_deg:=180 \
  lidar_window_deg:=12
```

The waist tracker holds the last commanded torso pose when detections go stale.
Default motion limits are intentionally gentle:
`waist_max_velocity:=0.35`, `waist_max_acceleration:=0.25`, and
`waist_max_jerk:=3.0`. To make it slower:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  follow_enabled:=false \
  waist_max_velocity:=0.20 \
  waist_max_acceleration:=0.15 \
  waist_max_jerk:=1.5
```

Walking follow remains optional and separate:

```bash
ros2 run py_examples set_mc_action LD
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py follow_enabled:=true
```

## Forward, Backward, and Arms

Forward/backward movement only:

```bash
ros2 run py_examples set_mc_action LD
ros2 run x2_motion_audio_tools x2_forward_backward_steps
```

Forward/backward movement followed by a cautious arm raise:

```bash
ros2 run py_examples set_mc_action LD
ros2 run x2_motion_audio_tools x2_forward_back_raise_arms
```

The motion nodes publish high-level locomotion velocity commands. Arm raising
uses low-level HAL joint commands, so only run arm sections when the robot is
stable and your AimDK/HAL safety procedure is satisfied.
