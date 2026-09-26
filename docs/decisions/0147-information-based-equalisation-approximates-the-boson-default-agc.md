# ADR 0147 — Information-based equalisation approximates the Boson's default AGC

**Status:** Accepted
**Date:** 2026-09-26

## Context

Every camera config selects global plateau equalisation (ADR 0028), and §11.3 used to call it "what
most thermal cores actually use". On the `phantom4_perpart` clip it gave the drone **3 of 256** grey
levels for parts spanning 15 → 38 °C (spec issue S52). The camera's own `display8` shows the same thing,
so this is not only a problem in the demo script.

That is correct plateau behaviour. The plateau clips a bin only when one DN value holds more than
`plateau · N` of the frame. A cloud-clutter sky spreads over thousands of DN, never reaches the
plateau, and the operator becomes full histogram equalisation: each object gets grey shades in
proportion to its area. FLIR's application note says the same ("an image with 60 % sky will devote
60 % of the available 8 bit shades to the sky") and names the **Information-Based Equalization** mode,
blended by **Linear Percent**, as the Boson's factory default [R51]. FLIR describes that operator only
qualitatively and publishes no equations and no default for its weighting.

## Options considered

1. **Lower the plateau.** This does nothing on a cluttered sky. With 55 000 occupied bins the
   plateau only engages below about 2·10⁻⁵, and it then becomes a min–max linear stretch, which
   gives the drone no more than its share of the temperature range.
2. **Local (tiled) plateau, `plateau_local`.** It already exists and gives the drone more shades,
   but it is not what a Boson does by default. Making it the default would tune the simulator
   for a nice-looking picture instead of for the camera.
3. **An approximation of FLIR's documented operator** (chosen). Split the frame with a 5×5
   bilateral filter whose range sigma stands for the Smoothing Factor. Plateau-clip the low-pass
   histogram. Add a second histogram, each bin's summed |high-pass|, scaled by `info_weight`.
   Blend the result with the linear map by `linear_percent`. Add the high-pass back at the
   transfer's local slope, with `detail_headroom` reserved at each end.

## Decision

Option 3 goes in as `agc: information_based` in `irsim.isp.information`, plus `linear_percent` on
`plateau_equalization`. Both are **opt-in**. No config changes its default in this step, so no golden
array or ISP hash moves: a new field at its default value is left out of the hash. `SC.22` moves the
Boson configs to the new mode and regenerates the goldens on purpose.

`dde_gain` keeps its meaning (the extra gain on the high-pass). Under `information_based` the
high-pass is added back inside the AGC at gain `1 + dde_gain`, and the separate R3 unsharp mask is
skipped so the detail is not boosted twice.

## Consequences

- On a synthetic frame shaped like `phantom4_perpart`, the target goes from **≤ 3** codes under
  plateau to **≥ 25** codes with `linear_percent = 0.3`, and its four parts come out in temperature
  order. On the real frame 96 it goes from 3 to 30.
- **Error that cannot be bounded yet.** `info_weight` (default 1.0), the reading of the Smoothing
  Factor as a range sigma in this core's DN, and the 5×5 window are guesses. The operator has the
  right structure, but it is not a fitted Boson. Until a Tier 4 fit against public footage exists,
  nothing may claim that an `information_based` image matches a real Boson frame for frame.
- A global operator's ceiling does not change. With a sky spanning −47 … +14 °C, a drone at
  15 … 38 °C cannot get most of the ramp from any global mapping. More than that needs ROI or local
  AGC, which a real camera also has to be configured for.
- The operator is spatial, so it cannot be one lookup table. The Warp display path
  (`build_agc_table`) keeps raising for it until a device port is written. The demo renders run
  the CPU branch.

## Revisit when

- A public Boson clip with a small target against the sky is available (the `XD` lane). Fit
  `info_weight` and the range sigma so the rendered and real code distributions on the target
  match.
- FLIR publishes the operator, or a Boson SDK dump shows its histogram.
