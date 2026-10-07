# ADR 0208 — A low drone is framed low, and the flat real boxes are other airframes

**Status:** Accepted (EV.10 part 1)
**Date:** 2026-10-07

## Context

`EV.10` exists because the largest single ablation in the drone sim-to-real literature is the
viewing geometry: mAP@0.5 of 0.464 with a fixed camera pitch, 0.981 with a random one. Real
Anti-UAV RGBT sky-only frames read 0.2–10° on their burnt-in mount readout, by eye (EV.19
report). irsim's clips stopped at 6°.

The driver refused any elevation band whose worst jittered frame could reach the horizon. That
check assumed the target could sit at the top of a frame aimed low. The aerial scenes author no
terrain (ADR 0060), so the horizon must stay out of shot. But a real mount tracking a drone just
above the horizon does not keep it centred: the drone sits low in the frame.

## Decision

- **The aim is raised, not the band.**
  - `irsim_isaac.asset_flight.horizon_safe_aim` raises the boresight's elevation offset wherever
    the frame's bottom edge would fall below `HORIZON_MARGIN_DEG` = 0.3°. The target then sits
    lower in the frame.
  - It refuses a target too low to stay inside the frame. `lowest_framable_elevation_deg` is
    the margin plus a tenth of the half field, 0.65° for the 50 mm lens.
  - `render_phantom4.py` applies it to the scatter and wander aims and checks the band against
    that lowest elevation.
- **The planner's band is 1–12°** (`generate_aerial_dataset.py`), where it was 6–20°.

## Consequences

Measured on 24 scatter frames at 1–10°, 20–90 m (`scripts/box_convention.py`, boxes as drawn,
ADR 0207):

| Boxes as drawn | Aspect p10 / p50 / p90 | Width p10 / p50 / p90, px | Height p50, px |
|---|---|---|---|
| Real sky-only test | 1.21 / 1.52 / 3.14 | 34 / 51 / 83 | 30 |
| Old clip set, 6–20° | 1.17 / 1.47 / 1.86 | 36 / 54 / 83 | 37 |
| 1–10° | 1.31 / 1.54 / 1.83 | 27 / 59 / 88 | 36 |

- Lower views did **not** flatten the boxes: a Phantom 4 seen edge-on is still at most about 1.9
  wide per tall, because of its landing skids. **The real set's flat tail (p90 3.1) is other
  airframes**, wider and lower than a Phantom 4. That moves to `EV.25`.
- Near the horizon the warm air at the frame's bottom crosses the visible-extent threshold. That
  exposed a fault in `visible_extent`, fixed in the same step (ADR 0207's amendment).
- The clear sky below 15° is the layered model, extrapolated: the ICI comparison starts at 15°
  (ADR 0183). The near-horizon gradient this band now samples is the part ADR 0206 suspects.
- The real elevation band is read by eye. Reading the burnt-in readout (as `ici_met_from_video.py`
  reads the ICI's) would give the real distribution to draw from. That is EV.10's part 2.

## Revisit when

- The real set's readout is read per frame: draw elevation from its distribution, as the
  published ablation's best arm did.
