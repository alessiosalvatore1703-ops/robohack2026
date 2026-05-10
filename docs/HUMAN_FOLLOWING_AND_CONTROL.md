# Human Following And Control Context

This document is the working context for future agents and teammates building on
the stereo vision, head tracking, and locomotion-following work.

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
- Waist yaw tracking: `x2_waist_yaw_tracker`, proof-of-concept only.
- Waist keyboard teleop: `x2_waist_keyboard_teleop`, proof-of-concept only.
- Stereo target walking supervisor: `x2_stereo_person_follow`.

The walking supervisor consumes the stereo target point and publishes
`aimdk_msgs/McLocomotionVelocity` on `/aima/mc/locomotion/velocity`. It does not
command leg joints directly.

## Control Standards

Use the high-level locomotion controller for anything that steps the legs. The
known smooth walking examples and vendor demos go through the motion-controller
stack. Do not implement walking by sending raw leg joint commands.

Use low-level HAL joint commands for upper-body tracking and teleop only when
the current motion mode does not already own those joints. In `SD`
(`STAND_DEFAULT`), the controller owns the torso/body joints for balance, so the
walking demo must not command waist/torso HAL joints. Head HAL control is still
acceptable because `SD` does not command the head.

When commanding a joint group, publish the whole command array for that group
and explicitly hold the joints that are not being moved. This prevents
uncommanded degrees of freedom from drifting.

Avoid abrupt target changes. The motor jitter seen during early head control was
transition jitter from direct setpoint changes, not steady-state holding error.
Trajectory shaping fixed the head and waist behavior and should be reused for
future hands, wrists, and arms.

## Walking Integration

The `SD` walking demo is head-plus-base only:

- Head tracks fast visual error to keep the person in view.
- Base yaw handles body-facing direction.
- Forward velocity handles approach distance.
- Waist/torso tracking must stay off during the walking demo.

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
ros2 launch yolo_person_detector stereo_person_pipeline.launch.py device:=cpu
```

Head tracking with vision:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py device:=cpu
```

Gantry dry-run follow. This logs computed walking velocity but does not publish
locomotion commands:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py \
  device:=cpu \
  follow_enabled:=true \
  follow_dry_run:=true
```

Gantry active follow. Put the robot in Stable Stand first. If the robot image
does not expose an `rc` app, do not run `aima em stop-app rc`; use
`get_current_input_source` to inspect arbitration instead.

```bash
ros2 run py_examples set_mc_action SD

ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py \
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
  depth_disparity_percentile:=75.0
```

For a more automated demo, the follow supervisor can request Stable Stand during
activation:

```bash
ros2 launch x2_motion_audio_tools x2_stereo_head_track.launch.py \
  device:=cpu \
  follow_enabled:=true \
  follow_dry_run:=false \
  follow_auto_enable_stable_stand:=true \
  follow_max_forward_speed:=0.15 \
  follow_min_forward_speed:=0.10 \
  follow_max_angular_speed:=0.25 \
  follow_max_forward_bearing_deg:=20.0 \
  follow_stop_min_m:=0.45 \
  follow_stop_max_m:=1.0 \
  follow_target_distance_m:=0.85 \
  depth_disparity_percentile:=75.0
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

- Person centered and farther than `1.0 m`: `APPROACH`.
- Person inside the stop band: `STOP_BAND` with `forward=0.000` and
  `angular=0.000`.
- Person off center: `ALIGN`.
- Target too close: `TOO_CLOSE`.
- Lost target: `NO_TARGET`.
- Bad depth: `INVALID_DEPTH`.

Only after dry-run output is correct should locomotion be enabled on the gantry.
Use low speed limits first. The robot should walk in place on the gantry while
the target distance changes as the human approaches, then stop all base motion
inside the stop band while the head continues tracking.

Arm-only assist-ready pose test:

```bash
ros2 run x2_motion_audio_tools x2_arm_assist_pose --dry-run

ros2 run x2_motion_audio_tools x2_arm_assist_pose \
  --arm-angle-deg 45 \
  --move-seconds 5 \
  --hold-seconds 0
```

The arm-only script publishes full-group HAL arm commands only. It does not
publish locomotion velocity, does not switch robot modes, and does not command
waist or torso joints. The future head-tap trigger and chair-assist state
machine should integrate this behavior later, after the arm pose is smooth.

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

The torso must not be controlled by our HAL waist nodes during `SD` walking.
That causes a conflict with the balance controller. Use base yaw for all
body-facing corrections and keep torso tools separate from the walking demo.
