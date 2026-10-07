# ADR 0204 — A drone clip is focused once, and the Isaac camera supplies the sky a blurred edge uncovers

**Status:** Accepted (EV.21)
**Date:** 2026-10-07

## Context

In the real Anti-UAV RGBT frames the drone is almost never sharp: a soft blob, arms merged into
the body. The renders were crisp. That difference is not one an eye can be trusted to size, so it
is measured. `irsim.validation.targets.edge_width_px` is the target's contrast over its steepest
gradient. Any linear display mapping cancels in that ratio. A step blurred by a Gaussian of σ
pixels reads `√(2πσ² + 4)`; a sharp edge reads 2.

| Set (`scripts/target_sharpness.py`) | Targets | Edge width, px: p10 / p50 / p90 | Blur σ at the median |
|---|---|---|---|
| Anti-UAV RGBT sky-only test | 1,500 | 2.90 / 4.10 / 6.66 | 1.43 px |
| irsim clip set (`EV.19`, `irsim_sky_v1`) | 1,500 | 1.56 / 1.92 / 2.14 | 0 |

The renders sat on the sharp floor.

The physics for this existed: focus modes and Hopkins defocus (`OC.4`–`OC.7`). No drone camera
used it, for two reasons.

- **No camera named a defocus model.** With `defocus_model: none` a lens is sharp at every range,
  whatever its focus says.
- **The Isaac camera never supplied `background_t_k`.** Layered defocus (`OC.6`) blurs each depth
  layer and composites them. Where a blurred edge uncovers sky, it needs the sky behind the target
  (`OC.7`). Without it the composite renormalises by the accumulated alpha, which hands the
  uncovered pixels back to the target, so the silhouette stays nearly sharp. On a synthetic target
  at 21 m through a 50 mm F/1 lens the edge read 3.1 px; with the background, 7.2 px, the
  global kernel's own answer.

## Options considered

1. **A post-hoc Gaussian on the 8-bit frame** with a σ drawn to match. It is cheap, but it blurs
   the sky's noise and the box's surroundings alike, and it ties the blur to nothing physical: not
   to range, not to the lens.
2. **Defocus through the existing optics, with a focus drawn per clip.** The blur then follows
   from range and lens: `c = f²|s − s_f| / (F s (s_f − f))`. A drone closing in softens, a lens
   focused near its range keeps it sharp, and a 14 mm lens stays sharp at any range, as a real one
   does.
3. **Autofocus (`OC.9`) per clip.** Against a sky the drone is the only edge, so the servo would
   mostly hold it. That is a different camera from the one the real frames suggest.

## Decision

Option 2.

- `irsim.config.loader.with_focus(config, distance_m | None)` focuses a camera. If the camera
  names no defocus model, it also sets Hopkins, layered. The config hash moves.
  `render_phantom4.py --focus-m <m | inf>` applies it, and each frame's metadata records the
  focus.
- `IrCamera` computes the whole-frame background (`AerialThermalBridge.background_temperature_k`)
  once whenever layered defocus is on. It passes it as `background_t_k`, and as the sky pixels'
  own temperature, so the sky is not marched twice.
- `generate_aerial_dataset.py` draws one focus per clip on its own random stream, so every
  earlier draw is unchanged. Half the clips are focused at infinity; the rest at a distance
  log-uniform from the near range to ten times the far. `--dry-run` prints the plan without
  rendering.

## Consequences

Measured on 8 scatter frames (telephoto, 20–90 m, 6–20°, clear, 14:00), focused at infinity:

| | Edge width, px: p10 / p50 / p90 |
|---|---|
| Unfocused model (as before) | 1.71 / 2.15 / 2.63 |
| Focused at infinity, with the background | 2.02 / **4.62** / 6.15 |
| Real | 2.90 / 4.10 / 6.66 |

The rendered median lies inside the real interquartile range (3.20–5.40). The tight box stays the
airframe's own pixels from the truth plane; a defocused target's glow spreads past it, as the
real one spreads past a human's box. That convention is `EV.22`'s question.

- Defocus about doubles a frame's cost: 8 frames took 180–190 s against 93 s.
- Every layered-defocus render from the Isaac camera before this was nearly sharp at its
  silhouettes. No committed render used a defocus model, so no published frame changes.
- The clip set planned with the drawn focus has not been rendered. Its edge-width distribution is
  to be measured on the first set (`scripts/target_sharpness.py`) against the real one, not
  assumed.
- The real camera's compression and ISP may soften edges too. Part of the measured 1.4 px may be
  theirs, not the lens's. The bar is on the frame, so it does not need to tell them apart.

## Revisit when

- The first set rendered with drawn focus lands outside the real interquartile range at its
  median. Then move `FOCUS_INFINITY_SHARE` or the distance band, and record the measurement.
- A real anti-UAV camera's lens is published. Its focal length and F-number set the blur, not the
  stand-in's.
