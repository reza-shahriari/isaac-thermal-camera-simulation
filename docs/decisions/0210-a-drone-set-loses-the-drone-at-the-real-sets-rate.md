# ADR 0210 — A drone set loses the drone at the real set's rate

**Status:** Accepted (EV.26)
**Date:** 2026-10-08

## Context

Every frame of a rendered drone set held the drone. A detector trained only on such frames learns
that a drone is always there, which is how a model trained on synthetic data starts calling
anything in the sky a drone. The real Anti-UAV RGBT set has frames without one: its per-frame
`exist` flag is 0 when the operator lost the drone and no box was drawn.

`scripts/absent_frames.py` reads those flags at the full 20 fps, over all 318 infrared clips:

- 3,692 of 296,901 frames (1.24 %) have no drone;
- they fall in 52 of the 318 clips (16 %);
- they come as 97 runs: median 14 frames, 90th percentile 70, shortest 1.

The sky-only split the detector is scored on (`EV.17`) has 1.2 % (train), 3.3 % (val) and 2.1 %
(test). The runs are written to `data/validation/anti_uav_absent_runs.csv`: split, clip, first
frame and length, and nothing from the frames themselves.

## Options considered

1. **Add empty-sky frames.** Render frames with no drone and mix them in. This is cheap, but an
   empty frame unlike its neighbours teaches nothing about the case that matters: a clip whose
   drone leaves and comes back.
2. **Lose the drone inside a clip.** The mount swings off the drone and back, as an operator's
   does. The empty frames then sit in the same sky, hour and camera as the frames around them.
3. **Hide the drone (switch it off for some frames).** This gives a clip with no aircraft and an
   unchanged camera, which no real camera produces.

## Decision

Option 2.

- **The planner** (`generate_aerial_dataset.draw_lost_runs`, its own RNG stream `[seed, 27]`):
  - The set's lost frames are sized over the whole set to `--absent-share`. The default, `real`,
    is the measured 1.24 %. A number raises or lowers it; 0 turns it off.
  - Each run's length is drawn from the 97 real lengths, in frames, capped at a quarter of the
    clip. A length that would carry the set past its target is redrawn.
  - Each run goes to a clip in proportion to its frames. Its start is uniform, with room for the
    swing either side and away from the clip's other runs.
  - Each run gets a direction uniform over 0–180°: right, up or left.
  - Scatter sets get none: their poses are unrelated, so there is no mount to lose the drone with.
- **The renderer** (`render_phantom4.py --lost-lock first:length:direction,...`; `irsim_isaac.
  asset_flight.LostLock`, `lost_lock_offset`):
  - The boresight swings off the aircraft between the frame before the run and its first frame
    (`LOST_RAMP_FRAMES` = 1, a smoothstep in between for sub-frame poses), and back after it. A
    set's frames are 5.6 s apart, so that is a 16° swing in 5 s, which an operator makes. A first
    try over 3 frames left the frame either side of each run empty too (a 4-frame run gave 6
    empty frames), and its pace was 16 s.
  - It holds for the run's frames, then swings back.
  - Held, its larger component is `1 + aim_jitter + LOST_CLEARANCE` half-fields. That is past the
    frame edge by more than the wander can bring back, plus half a half-field for the drone's own
    size (the real boxes are 34–83 px of 640).
  - The direction is never down. A frame swung down would reach the horizon, which the aerial
    scenes do not author, and `horizon_safe_aim` would lift it back onto the aircraft.
- **The label** needs nothing new. The truth `part_id` plane holds no aircraft pixel, so the
  frame's YOLO file is empty, which is how YOLO marks a negative. Each row records `lost_lock`
  (0 on the drone, 1 lost), and the summary records `frames_without_aircraft`.

## Consequences

- The planner reaches the real share to the frame: 22 of 1,800 frames at 6 clips, 112 of 9,000
  at 30, 373 of 30,000 at 100. At 100 clips 18 have a run (the real share of clips is 16 %).
- **Run lengths are matched in frames, not seconds.** The real clips run at 20 fps; a set's
  frames are a time-lapse, about 5.6 s apart at the default 1,666 s over 300 frames. A detector
  sees frames, and a run of 14 frames is what it learns from. A 14-frame loss in a set's clip
  therefore spans about 78 s of mission time; the mount's wander makes that look like a
  search, not a freeze.
- Between runs the mount's wander still lets the drone touch or cross a frame edge, as before;
  the run's own frames are the ones it is held out for.
- Redrawing lengths that would overshoot makes a set's last runs a little shorter than the real
  ones on average. At a few hundred frames of target the bias is small.
- Measured on a render: a 24-frame Phantom 4 clip (telephoto, clear, 27–38 m) with runs of 4 and 2 frames has no aircraft pixel in exactly those 6 frames, empty labels for them, and a box in each of the other 18. Without the drone the display's AGC stretches the bare sky's gradient, so an empty frame reads brighter than its neighbours, as a real camera's would.
- The other half of the negatives, things in the sky that are not drones (birds first), is
  `EV.27`.

## Revisit when

- `EV.30` measures the real mount's shake: the swing's shape and speed should then come from it.
- A set is trained with and without the empty frames (`EV.19`'s protocol). That is the
  measurement of whether they matter for the benchmark score.
