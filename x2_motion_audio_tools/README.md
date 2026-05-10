# X2 Motion and Audio Tools

ROS 2 helper nodes for AgiBot X2 audio, speech, open-loop coordinate motion,
forward/backward movement, head and torso person tracking, stereo target
following, and cautious arm raising.

This package is intentionally separate from `yolo_person_detector`. Stereo
vision publishes `/stereo_person/target_point`; this package consumes that
target for head tracking, optional torso tracking, and conservative high-level
locomotion following.

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
ros2 run py_examples set_mc_action SD
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

## Stereo Head, Waist, and Walking Follow

The current stereo interaction stack is launched from
`x2_stereo_head_track.launch.py`. It starts the stereo vision pipeline, head yaw
tracker, optional waist yaw tracker, and the stereo walking supervisor.

Default behavior is safe:

- Head tracking is enabled.
- Waist tracking is disabled unless `torso_enabled:=true`.
- Walking follow is disabled unless `follow_enabled:=true`.
- Walking follow is dry-run unless `follow_dry_run:=false`.

Head tracking only:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py device:=cuda
```

Head plus bounded waist tracking:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py \
  device:=cuda \
  torso_enabled:=true \
  waist_soft_limit_deg:=10.0 \
  waist_start_threshold_deg:=12.0
```

Gantry dry-run walking follow:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py \
  device:=cuda \
  follow_enabled:=true \
  follow_dry_run:=true
```

Active gantry follow:

```bash
ros2 run py_examples set_mc_action SD
aima em stop-app rc

ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py \
  device:=cuda \
  follow_enabled:=true \
  follow_dry_run:=false \
  follow_max_forward_speed:=0.10 \
  follow_max_angular_speed:=0.20
```

Once the manual `SD` sequence is trusted, the launch can request Stable Stand
for you:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py \
  device:=cuda \
  follow_enabled:=true \
  follow_dry_run:=false \
  follow_auto_enable_stable_stand:=true \
  follow_max_forward_speed:=0.10 \
  follow_max_angular_speed:=0.20
```

The stereo walking supervisor publishes high-level
`aimdk_msgs/McLocomotionVelocity` commands on `/aima/mc/locomotion/velocity`.
It never commands leg joints directly. It consumes
`/stereo_person/target_point`, rotates the base toward the target, walks forward
only when the person is centered and farther than the stop band, and stops in
the `0.5-1.0 m` range by default.

Forward walking is blocked while the waist is away from neutral by default:
`follow_require_waist_neutral:=true` and
`follow_waist_neutral_limit_deg:=5.0`. This keeps torso offsets from becoming a
stability problem while the gait controller is stepping.

Runtime enable and disable:

```bash
ros2 topic pub -1 /stereo_person/follow/enable std_msgs/Bool "data: true"
ros2 topic pub -1 /stereo_person/follow/enable std_msgs/Bool "data: false"
```

If the base turns away from the person, relaunch with
`follow_invert_angular:=true`.

## Person Body Following

`x2_person_follow` runs YOLO person detection on either top front stereo camera,
says "Hello" once per visible encounter, reads the chest `PointCloud2` LiDAR,
logs camera/YOLO/LiDAR details, registers a locomotion input source, turns with
the legs, walks toward the selected person, and stops about one meter away. It
does not command the waist/torso joints by default.

Before running, put the robot in Stable Stand and release the remote-controller
channel:

```bash
ros2 run py_examples set_mc_action SD
aima em stop-app rc
```

Run body following:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py
```

By default this uses `/aima/hal/sensor/stereo_head_front_left/rgb_image` and
`/aima/hal/sensor/lidar_chest_front/lidar_pointcloud`, walks while the selected
person remains visible, stops when the target is lost, and stops approaching at
`stop_distance_m:=1.0` with a small `stop_deadband_m:=0.12`.

To use the right stereo camera or compressed stream:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  camera_topic_type:=right_rgb_image

ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  camera_topic_type:=left_rgb_image_compressed
```

If you want detection logs only:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  follow_enabled:=false
```

Expected detection logs include person count, selected bbox/confidence, camera
bearing, base bearing, camera FPS, LiDAR FPS, valid point count, sector point
count, and the estimated distance.

If the body turns away from the person, flip the angular sign by using a
negative angular gain:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  angular_gain:=-1.0
```

If the LiDAR distance samples the wrong direction, tune the sector:

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  lidar_angle_offset_deg:=180 \
  lidar_window_deg:=12
```

The follower only walks forward when the target is mostly centered. It turns in
place when the target bearing is outside `max_forward_bearing_deg:=25.0`.
Default speed limits are conservative:
`max_forward_speed:=0.25`, `max_angular_speed:=0.45`, and
`forward_gain:=0.28`.

```bash
ros2 launch x2_motion_audio_tools x2_person_follow.launch.py \
  stop_distance_m:=1.0 \
  max_forward_speed:=0.18 \
  max_angular_speed:=0.30
```

The previous torso-only tracker is preserved as `x2_person_track_torso`:

```bash
ros2 launch x2_motion_audio_tools x2_person_track_torso.launch.py follow_enabled:=false
```

## Control Standards

Use high-level locomotion velocity for walking. Do not command walking through
raw leg joint targets.

Use Ruckig-shaped HAL joint commands for head, waist, arms, wrists, and hands.
Initialize from live joint state, publish the full command array for the joint
group, and explicitly hold joints that are not being moved.

The detailed architecture, findings, and gantry/off-gantry workflow are in:

```bash
docs/HUMAN_FOLLOWING_AND_CONTROL.md
```

## Forward, Backward, and Arms

Forward/backward movement only:

```bash
ros2 run py_examples set_mc_action SD
ros2 run x2_motion_audio_tools x2_forward_backward_steps
```

Forward/backward movement followed by a cautious arm raise:

```bash
ros2 run py_examples set_mc_action SD
ros2 run x2_motion_audio_tools x2_forward_back_raise_arms
```

The motion nodes publish high-level locomotion velocity commands. Arm raising
uses low-level HAL joint commands, so only run arm sections when the robot is
stable and your AimDK/HAL safety procedure is satisfied.
