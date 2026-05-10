# Human Following And Control Context

This document is the working context for future agents and teammates building on
the stereo vision, head tracking, torso tracking, and locomotion-following work.

## Architecture

The current human-following stack is split into perception, upper-body tracking,
and base locomotion.

`yolo_person_detector` owns stereo perception. The main contract for motion code
is `/stereo_person/target_point` as `geometry_msgs/PointStamped`. The point is
the selected person's bounding-box center projected with stereo depth in the
left camera frame:

- `point.x`: lateral offset, positive image-right.
- `point.y`: vertical offset.
- `point.z`: forward depth in meters.

`x2_motion_audio_tools` owns robot motion helpers. Upper-body motion uses HAL
joint command topics. Walking uses the motion-controller locomotion API:

- Head yaw tracking: `x2_head_yaw_tracker`.
- Head keyboard teleop: `x2_head_keyboard_teleop`.
- Waist yaw tracking: `x2_waist_yaw_tracker`.
- Waist keyboard teleop: `x2_waist_keyboard_teleop`.
- Stereo target walking supervisor: `x2_stereo_person_follow`.

The walking supervisor consumes the stereo target point and publishes
`aimdk_msgs/McLocomotionVelocity` on `/aima/mc/locomotion/velocity`. It does not
command leg joints directly.

## Control Standards

Use the high-level locomotion controller for anything that steps the legs. The
known smooth walking examples and vendor demos go through the motion-controller
stack. Do not implement walking by sending raw leg joint commands.

Use low-level HAL joint commands for upper-body tracking and teleop only. For
head, waist, arms, wrists, and hands, command smooth trajectories initialized
from the live joint state. The head and waist nodes use Ruckig when available
and fall back to velocity-limited interpolation if Ruckig is missing.

When commanding a joint group, publish the whole command array for that group
and explicitly hold the joints that are not being moved. This prevents
uncommanded degrees of freedom from drifting.

Avoid abrupt target changes. The motor jitter seen during early head control was
transition jitter from direct setpoint changes, not steady-state holding error.
Trajectory shaping fixed the head and waist behavior and should be reused for
future hands, wrists, and arms.

## Walking Integration

Head, torso, and legs can run at the same time, but the responsibilities should
stay separate:

- Head tracks fast visual error.
- Waist tracking is optional and bounded.
- Base yaw handles large heading corrections during walking.
- Forward walking is blocked unless the waist is close to neutral.

The torso "null point" is waist yaw near zero. During follow mode, the waist
should either be disabled or limited to small offsets. The walking supervisor
has `require_waist_neutral_for_forward:=true` by default and blocks forward
velocity unless `abs(waist_yaw_joint)` is within
`waist_neutral_limit_deg:=5.0`.

The first stereo walking behavior is intentionally conservative:

- Stop if the target point is stale.
- Stop if stereo depth is invalid.
- Stop inside the `0.5-1.0 m` band.
- Rotate in place when the target is off center.
- Walk forward only when the target is centered and farther than `1.0 m`.
- Do not reverse by default.

If the base turns away from the human, flip `follow_invert_angular:=true`.

## Launch Recipes

Stereo vision only:

```bash
ros2 launch yolo_person_detector stereo_person_pipeline.launch.py device:=cuda
```

Head tracking with vision:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py device:=cuda
```

Head plus bounded torso tracking:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py \
  device:=cuda \
  torso_enabled:=true \
  waist_soft_limit_deg:=10.0 \
  waist_start_threshold_deg:=12.0
```

Gantry dry-run follow. This logs computed walking velocity but does not publish
locomotion commands:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py \
  device:=cuda \
  follow_enabled:=true \
  follow_dry_run:=true
```

Gantry active follow. Put the robot in Stable Stand first and release the
remote-controller channel if required by the platform workflow:

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

For a more automated demo, the follow supervisor can request Stable Stand during
activation:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py \
  device:=cuda \
  follow_enabled:=true \
  follow_dry_run:=false \
  follow_auto_enable_stable_stand:=true \
  follow_max_forward_speed:=0.10 \
  follow_max_angular_speed:=0.20
```

Runtime enable and disable:

```bash
ros2 topic pub -1 /stereo_person/follow/enable std_msgs/Bool "data: true"
ros2 topic pub -1 /stereo_person/follow/enable std_msgs/Bool "data: false"
```

## Test Workflow

Start with perception only. Confirm that `/stereo_person/target_point` is stable
and that `point.z` changes correctly as a person approaches.

Then run the follow supervisor in dry-run mode. Check logs:

- Person centered and farther than `1.0 m`: `approaching`.
- Person inside `0.5-1.0 m`: `inside_stop_band`.
- Person off center: `turning_to_center`.
- Waist not neutral: `waist_not_neutral`.
- Lost target: `target_timeout`.

Only after dry-run output is correct should locomotion be enabled on the gantry.
Use low speed limits first. The robot should walk in place on the gantry while
the target distance changes as the human approaches, then stop inside the
`0.5-1.0 m` band.

Off-gantry testing should use lower speed limits than gantry testing, a clear
area, and an active emergency stop. Keep reverse disabled until forward
approach and stopping behavior are reliable.

## Current Findings

The stereo pipeline is now the preferred perception input for follow behaviors
because it publishes a real 3D target point instead of requiring bbox-size
heuristics.

The older `x2_person_follow` node is still useful as the LiDAR-based reference
and as an example of `SetMcInputSource` plus `McLocomotionVelocity`, but the new
stereo supervisor should be the path for the camera-depth follow experiment.

The existing smooth walking examples are already using the correct interface:
high-level locomotion velocity through the motion controller.

The torso should not be treated as the primary way to aim the robot during
walking. Use it for small alignment and interaction motion. Use base yaw for
large heading changes.
