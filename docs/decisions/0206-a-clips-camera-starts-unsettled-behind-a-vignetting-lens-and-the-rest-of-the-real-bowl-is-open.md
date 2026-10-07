# ADR 0206 — A clip's camera starts unsettled behind a vignetting lens, and the rest of the real bowl is open

**Status:** Accepted (EV.23 part 1)
**Date:** 2026-10-07

## Context

Real Anti-UAV RGBT clear-sky frames carry a strong radial bowl that differs from clip to clip.
`scripts/sky_bowl.py` fits a plane plus a paraboloid (`irsim.validation.radial.radial_fit`) with
the target and the top 40 rows masked, and measures it:

| 400 frames each | Signed depth p10 / p50 / p90 | Dark-centred | Radial share | Bowl ÷ vertical slope |
|---|---|---|---|---|
| Real, sky-only test | −0.40 / 0.82 / 1.07 | 71.5 % | 0.85 | 2.37 |
| irsim clip set (`EV.19`) | 0.04 / 0.06 / 0.09 | 99.8 % | 0.20 | 0.07 |

The project already models where such a bowl comes from:
- An off-axis pixel sees `1 − τ·RI` of housing (ADR 0145).
- The shutter refreshes only the offset (ADR 0148), so a housing that drifts after the shutter
  leaves a bowl.

The clips showed none, for two reasons.

- **The housing started settled and the shutter fired at power-up only.** The FFC counts
  captured frames at 60 Hz, and a 300-frame time-lapse never reaches its 300 s. So the drift over
  a clip was the slow ambient change.
- **The 50 mm stand-in lens falls off only by cos⁴, 0.98 at the corner.** Even a large drift
  barely varies the housing's share of the field.

## Options considered

1. **An empirical shading term**, drawn to match the real bowl's statistics. It would match by
   construction, and would stand for no mechanism. Nobody knows what the real camera is, and a
   cooled camera's narcissus effect would also give a dark centre.
2. **The existing mechanism, made reachable.** Each clip's camera starts its housing away from
   settled, as a camera does when just switched on (it warms) or carried out of a warm vehicle
   (it cools). Its lens vignettes mechanically, as fast IR lenses do. Then measure what that
   reaches and record the gap.

## Decision

Option 2.

- `SensorChain.build(..., housing_start_offset_k=)` and `attach_sensor_chain` start a coupled
  housing that far from steady state. `render_phantom4.py --housing-start-k` applies it.
- `irsim.optics.vignetting.radial_vignetting_map(w, h, corner)` gives `1 − (1 − corner) ρ²`
  (ESTIMATED; a quadratic is the shape the radial fit measures). `with_vignetting_map` points a
  config at it, and its content enters the config hash. `render_phantom4.py
  --corner-illumination` writes the map beside the run.
- `generate_aerial_dataset.py` draws both per clip on their own stream:
  - the housing start uniform on −8…+3 K, so three quarters are warming. −4 K is this camera's
    whole self-heating; a housing in sun runs further.
  - the corner illumination uniform on 0.6…0.9.
  Both bands are ESTIMATED.

## Consequences

Measured on 8 telephoto frames over 1,666 s, clear, starting −8 K with a 0.6 corner. The bowl
grows as the housing warms through the clip:

| | Signed depth p10 / p50 / p90 | Radial share | Bowl ÷ vertical slope |
|---|---|---|---|
| Settled camera, cos⁴ only | 0.02 / 0.05 / 0.07 | 0.14 | 0.05 |
| −8 K start, 0.6 corner | 0.11 / 0.21 / 0.49 | 0.67 | 0.26 |
| Real | −0.40 / 0.82 / 1.07 | 0.85 | 2.37 |

The engine-free bench (`validate_sky_flat.render_pair`) agrees. At 15° elevation, a 0.6 corner
and 6 K of drift give a depth of 0.62, a radial share of 0.87 and a bowl/slope of 0.89. At 8°
they give 0.29.

- **The shape is now the real one's kind** (radial share 0.67 against 0.20 before). **The depth
  is not.** The real median is four times this run's, and the real bowl is 2.4 times the sky's
  vertical gradient where this reaches 0.26. Two candidates, neither measured:
  - the real camera's shading mechanism is stronger or different (narcissus, a lens far warmer
    than its housing);
  - our near-horizon sky gradient is steeper than the real one. The clear sky is validated from
    15° up (ADR 0183), and these frames sit at 6–20°.
- The bright-centred real frames (28.5 %) need cooling housings. The draw gives a quarter of the
  clips a cooling start of up to 3 K, smaller than the warming side by design.
- Default renders are unchanged: offset 0 and no map unless asked.

## Revisit when

- A real set with a stated camera and lens is read (among `XD.11`'s sources), or a real clip with readable
  housing or time-since-FFC telemetry. Then fit the corner and drift bands to it.
- `AT.38` or a sounding pins the near-horizon clear-sky gradient. Then the remaining ratio is
  the camera's alone.
