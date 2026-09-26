# ADR 0148 — The FFC is an offset snapshot on the shutter

**Status:** Accepted
**Date:** 2026-09-26
Roadmap: SC.18. Spec: `docs/physics-model.md` §11.2 ("The shutter is a radiance reference, not a
reset button", revised 2026-09-26). Builds on [ADR 0145](0145-the-housing-is-seen-through-the-field.md);
amends the residual of [ADR 0053](0053-fpa-temperature-split-between-detector-and-nuc.md) in one
place, its reference level.

## Context

After ADR 0145 a housing that drifts away from its calibration temperature shades a uniform scene
radially. A real camera does not show that shading forever: it closes its shutter every few
minutes and re-measures its offsets. What it shows *between* shutter events is the bowl a public
clear-sky frame has. The chain modelled the shutter only as a scheduler: `FfcController` froze the
frame, reset ΔT_FPA and redrew the white residual fields. Nothing looked at the shutter, so the
display flat field stayed the factory one whatever the housing did, and the residual's gain term
multiplied the whole signal -- including the pedestal a shutter measurement removes.

## Options considered

1. **A radial residual field with a fitted amplitude.** Adds the bowl without the mechanism; its
   sign and its dependence on scene temperature would both have to be authored.
2. **Re-run the factory two-point calibration at every FFC.** Wrong physics: a shutter gives one
   point, not two, so a camera cannot refresh its gains in the field.
3. **A one-point offset update on a computed shutter frame.** At each event, compute the noiseless
   frame the closed shutter produces (`shutter_flux`: the shutter fills each pixel's cone at
   L_B(T_shutter), the housing is seen out of cone as in ADR 0145) and move the display offset so
   that frame reads uniform (`TwoPointNuc.refreshed`). Keep the frame as the residual's reference,
   so a gain error acts on signal minus shutter.

## Decision

Option 3. The shutter is taken at T_FPA (it sits beside the focal plane; the Boson publishes no
separate shutter temperature) and against the frame's housing. A shuttered core also takes one at
power-up, since that is when a real camera first flat-fields; `FfcController.fires_on` still never
freezes frame 0. `SensorChain` owns the snapshot (`shutter_dn`, `display_nuc`), `run_frame` computes
it and displays through the refreshed offset. `NucResidual.apply` takes the shutter frame as
`reference_dn` and computes ref + g (x − ref) + o. The Warp residual kernel takes the same plane,
zeros by default.

## Consequences

- **The bowl is the §11.2 closed form.** After a shutter event a uniform scene is flat; as the
  housing drifts it grows as (1/τ)(1/RI − 1) ΔL_h toward the corners -- a housing that cooled
  since the event leaves a bright centre. On the Boson 640 corner:

  | housing since the FFC | 300 K scene | 230 K sky |
  |---|---|---|
  | −0.5 K | −142 mK | −361 mK |
  | −2 K | −563 mK | −1446 mK |

  `tests/unit/test_shutter_reference.py` pins the closed form to 5 % and monotonic in radius.
- **The factory calibration's housing mismatch is gone after the first frame.** A coupled housing
  far from `t_housing_cal_k` no longer shades the display from the start.
- **The gain residual is now zero at the shutter's own signal** and largest on a scene far from it:
  a 250 K sky carries more than ten times the gain residual of a 300 K scene with the shutter near
  300 K. The old form did the opposite. The noise-budget test now scales that term by
  rms(signal − shutter).
- **Display only for the offset.** The radiometric branch still inverts with the calibration
  housing (ADR 0021): a radiometric camera that uses its shutter for absolute calibration is a
  different model and is not built.
- **Error introduced.** The shutter frame is noiseless; a real camera averages a few frames, so
  its offset carries a small temporal-noise imprint this model does not. The frames held during
  the freeze are displayed through the new offset although they were captured under the old one,
  a difference of one housing drift over the few hundred milliseconds of the freeze. T_shutter =
  T_FPA exactly: a shutter that lags the FPA would add a uniform radiometric bias, invisible in the
  display.
- **Engine.** The Warp residual kernel changed; the device pipeline does not launch it yet, and its
  device test was not run.

## Revisit when

A camera config needs a shutter temperature apart from the FPA, or the radiometric branch is asked
to use the shutter for absolute calibration.
