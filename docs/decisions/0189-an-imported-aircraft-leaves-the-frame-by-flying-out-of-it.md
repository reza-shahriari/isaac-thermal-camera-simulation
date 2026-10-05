# ADR 0189 — An imported aircraft leaves the frame by flying out of it, not by a pan

**Status:** Accepted
**Date:** 2026-10-04
Roadmap: EV.20 (the clear-sky exit for the Phantom 4), after EV.16 and TC.18

## Context

`EV.16` filmed a generated heavy-lift quadrotor flying out from 12 m to 150 m in a cloudless sky
and then leaving the frame. The owner asked for the same clip with the DJI Phantom 4, the imported
asset, after the quad's physics had been corrected: the clip starts at cruise rather than on the pad
(EV.16 amendment), the bolometer smear is its membrane (ADR 0077 amendment), and each motor warms
its arm (TC.18, ADR 0187).

The Phantom 4 already had two of those corrections, by construction:

* **The motors conduct into the arms.** `phantom4_solved.yaml` (TC.13, ADR 0166) solves the whole
  aircraft as one network: each winding heats its bell through the stator press fit, each bell
  heats its mount through a 2 cm² boss, and each mount heats the arm member (`configs/assets/
  phantom4_parts.yaml`, `contacts:`). It is two-way, which ADR 0187's mount is not.
* **The arms are their real shape.** They are a mesh member of the archive's own geometry at 2 cm
  cells, not strips, so the side the camera sees is solved as that side.

What it lacked was the exit itself, and the cloudless sky.

`EV.16` left the frame by **panning the camera**: the quad's occluders are authored in the world
frame and the thermal core refuses to move them (ADR 0123), so that aircraft cannot fly. The pan
turns the sun's bearing in the frame by 16.7° and moves the whole sky behind the target, which is
not what a fixed observer sees.

## Options considered

1. **Pan, as EV.16.** One shared code path for both clips. But the Phantom 4 does not need it: its
   mesh field takes sky view analytically per cell (`self_occluding: false`, ADR 0104) and its scene
   authors no occluders, so it already flies (`StraightOutTrack`). A pan here would be a constraint
   copied from a scene that has it to one that does not.
2. **Fly out of a fixed field (chosen).** The mount follows the climbing run, stops where it ends,
   and the aircraft strafes sideways out of the stopped field. The background never moves after
   the mount stops, which is the observer's view.
3. **Turn and fly away.** A real drone leaving is as likely to turn as to strafe, but the aspect
   would change in the last second of the clip, from tail-on to broadside. The exit would then test
   two things at once.

## Decision

**`asset_flight.StraightOutExitTrack`**:
- `outbound_s` of `StraightOutTrack`'s run, unchanged;
- then `exit_s` of a strafe along −X (the camera's left) at `exit_speed_m_s`, with the nose
  unturned, because a multirotor translates without yawing;
- then `hold_s` with the aircraft still flying, so it gets further out of the field rather than
  parking on its edge.

`aim_m(phase)` is where the mount points, frozen once the run ends.

**`clear_exit_seconds` solves the strafe in closed form.** The strafe is perpendicular to a
boresight that has stopped, so the depth stays the slant range `R` and the near wingtip clears the
edge when `x − extent/2 = R tan(hfov/2 + 3 px)`. EV.16's pan needed a bisection because the camera
turned and the aircraft did not.

**`render_phantom4.py --clear-exit`** defaults to:
- `phantom4_clear_exit.yaml`, which is `phantom4_solved.yaml` under `cloudless_midlat_summer_48h.csv`;
- 5 → 60 m at 16° elevation, in 15 s, then a strafe at 10 m/s;
- 60 Hz;
- mission second **540**, 210 s into the forward cruise (T+330..570). The whole 18 s clip lies
  inside the cruise, so the motors are at cruise temperature throughout.

Every frame records the pinhole prediction for the aircraft's bounding-box centre beside the drawn
centroid. The summary reports the centre error, the last frame anything was drawn, the first frame
the prediction puts the whole aircraft past the edge, and whether anything was drawn after it.

**The readout margin is measured, not fixed** (`irsim_eval.video.readout_margin`). Nineteen solved
parts with names such as `phantom4.motor_mount_front_right` ran the gauges about 100 px into the
fixed 400 px margin, and the colour bar sat on the frame's right edge. A drone leaving on the left
disappeared under the readout before it reached the edge of the picture. The left margin is now the
gauges' measured width, and a right margin holds the colour bar. This applies to every Phantom 4
clip, not only this one.

## Consequences

| | |
|---|---|
| The exit is what a fixed observer sees | the sky behind the target does not move after the mount stops, and the sun stays where it was |
| One clip still tests one thing | the aspect stays tail-on through the exit; only position and range change |
| The quad's clip and this one leave differently | EV.16 pans and EV.20 flies, because the two scenes have different constraints, not by taste |
| The centre error is not sub-pixel up close | the drawn centroid of a silhouette is not its bounding-box centre. The landing gear pulls the box centre down, about 6 px at 5 m, shrinking as 1/R. It is reported as a fraction of the span too, and the horizontal error is a fraction of a pixel |

## Not modelled

- The strafe has no bank. A real multirotor rolls a few degrees to accelerate sideways, and the
  thermal state and the aspect are held through it.
- The thermal tick is the parent's 30 s, so the whole 18 s clip sees at most one change in the
  solved state. A clip at the camera's frame rate shows the aircraft's state at cruise, not its
  evolution.

## Revisit

- Revisit if the generated quad gains movable occluders. EV.16 could then fly out the same way and
  drop the pan.
