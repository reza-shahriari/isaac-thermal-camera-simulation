# 0167 — The visible companion is exposed over the infrared's window by sub-frame accumulation

Date: 2026-09-29

**Status:** Accepted (2026-09-29, IG.19).

## Context

The infrared frame smears a moving edge over the detector's integration window with the SC.28
kernel (ADR 0077): the whole frame period for a bolometer, `integration_time_ms` for a photon
FPA. The companion RGB frame (ADR 0073) was rendered sharp at one instant. The owner reads the
pair together, and a target blurred in one band and crisp in the other is a defect of the pair.
The roadmap asked for Isaac's motion blur to be driven from the sensor's integration time and
the frame period.

## Options

1. **RTX post-process motion blur** (`/rtx/post/motionblur/*`): a screen-space effect from
   motion vectors, with its own exposure fraction and blur-diameter cap. Rejected: it is a
   post-effect on the tone-mapped image, its parameters are not a detector's window, it caps
   long smears, and a rotor blade or a curved track it cannot follow; the infrared side would
   describe one exposure and the companion another approximation of it.
2. **Sub-frame accumulation** (chosen): the driver's own trajectory poses the moving geometry at
   `n` instants spread evenly across the detector's window, centred on the frame's instant, the
   renderer is stepped at each and the frames are averaged. That is what a camera's exposure
   does. It needs no motion vectors, no assumption of linear motion and no engine setting; the
   glue only loops, poses, renders and reads, and the window, the offsets and the average live
   in the engine-free core (`irsim.optics.exposure`), where the acceptance is tested on a
   synthetic edge against the SC.28 kernel itself. The infrared planes are still taken at the
   frame's own instant, where the kernel smears them over the same window, so the pair describes
   one exposure. Cost: `n` extra renders per frame, paid only when asked (`--rgb-subframes`).

## Consequences

* Engine-free, on a step edge at 3, 6 and 12 px per frame under a bolometer's whole-period
  window and a photon FPA's quarter of it: the accumulated companion's 10-90 % edge width equals
  the kernel's to under 0.5 px, and both edges are centred on the frame's instant to 0.25 px; a
  static scene comes back bit-identical from either path (`tests/unit/test_companion_exposure.py`).
* `IrCamera.get_outputs(rgb_subframes=, pose_at=)` and `render_phantom4.py --rgb-subframes`;
  with one sub-frame nothing changes. The in-engine measurement on a rendered pair is the
  render lane's, with the Phantom 4 on its track.
* The accumulation is of the renderer's tone-mapped 8-bit frames, in float64, rounded once:
  a display-space exposure, which is what the companion is (it carries no radiometry).
