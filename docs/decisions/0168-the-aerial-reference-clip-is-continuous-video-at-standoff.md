# 0168 — The aerial reference clip is continuous video at standoff

Date: 2026-09-29

**Status:** Accepted (2026-09-29, EV.14).

## Context

Every aerial clip this project had rendered was a time-lapse at close range: the Phantom 4 at
11.6 s of flight per frame and 3-4 m away, the outbound quads at 6 s per frame. Those clips exist
to show a thermal history, and they do. They are not what the public aerial sets look like: the
sets EV.3-EV.7 compare against film drones at tens to hundreds of metres, 5-40 px across, frame
after frame at the camera's own rate. A comparison between a time-lapse and a video measures the
difference in sampling, not in physics -- the bolometer's smear over its window, the chain's
frame-to-frame noise and the AGC's response are all defined per frame period, and at 6 s per
frame none of them runs at its real rate.

## Options

1. **A new scene and a new driver.** Rejected: the outbound stage (ADR 0123) already has a
   point-wise airframe whose patches are checked cell by cell against its prims, a slewing camera
   the motion tracker follows, the cloud deck and the dome. A second stage would duplicate that
   agreement and the test that holds it.
2. **A mode of the outbound driver with its own scene config** (chosen).
   `configs/scenes/aerial_reference_clip.yaml` is `quad_outbound_pointwise.yaml`'s aircraft and
   thermal block unchanged -- the test asserts they are equal -- under the fair-weather cumulus
   weather and preset, so there is cloud with a top behind the target on the rays the camera
   reads. `render_quad_outbound.py --reference` flies
   `irsim_isaac.quad_outbound.reference_track`: 50 m to 250 m in ten seconds, geometric, 8 degrees
   of slew; captures every `1 / frame_rate_hz` (600 frames on the Boson); encodes at that rate;
   and hangs a second drone 1.2 km beyond the aircraft, 6 degrees off the boresight, which is
   injected by the analytic sub-pixel path (ADR 0071) on every frame.

The far end is 250 m rather than the roadmap's 300 m because the size band is the acceptance:
the heavy-lift's 1.14 m tip to tip is 4.5 px at 300 m through the Boson's 14 mm lens and 5.3 px at
250 m. The companion carries the sub-pixel case instead of the aircraft, so the resolved and
the analytic paths both run in one clip and neither is asked to do the other's job.

## Consequences

* Engine-free (`tests/unit/test_aerial_reference_clip.py`): the target is 26.7 -> 5.3 px across
  the clip; ten seconds is 600 frames at the Boson's 60 Hz; the slew is under a pixel a frame;
  the horizon never enters the frame; the companion fills under a tenth of a pixel, stays at
  least two degrees inside the frame and behind the aircraft on every frame; the deck's base is
  above the aircraft's highest point over the camera.
* The camera's AGC is damped at the Boson's factory 0.85 per frame (SC.10, ADR 0170), and at
  this clip's 60 Hz that is live: the clip records the display's frame-to-frame drift
  (`display8_mean_step_*` in `summary.json`), which is the number a comparison with a real
  set's lag-1 band reads.
* The sky and the cloud are analytic background (ADR 0060) and carry no `motion_px`, so the
  slew smears the rendered aircraft but not the cloud behind it. At under a pixel a frame of
  slew the difference is below the bolometer's own window at this clip's rate.
* The companion is infrared-only: it has no prim, so the RGB frame does not show it. At 1.2 km
  it is sub-pixel in the visible frame too.
