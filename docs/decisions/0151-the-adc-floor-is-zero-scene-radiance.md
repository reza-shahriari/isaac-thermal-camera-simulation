# ADR 0151 — The ADC floor is zero scene radiance

**Status:** Accepted
**Date:** 2026-09-27
Roadmap: SC.23. Spec: `docs/physics-model.md` §11.1 ("The raw DN must hold the coldest scene"),
spec issue S53. Amends the pipeline default range of [ADR 0021](0021-nuc-level-and-sitf-route.md)
at its lower end; the upper end, the housing-at-calibration inversion and the two routes stand.

## Context

ADR 0021 sized the bolometer's DN gain so that blackbodies at −40 °C and +200 °C
(`RADIOMETRIC_RANGE_K`) span the ADC, which put DN 0 at −40 °C. That is a datasheet's *specified*
range, not a converter floor: a real core references its DN to the shutter and adds a mid-scale
pedestal, so zero radiance is still on scale. The model's version clipped every scene colder than
−40 °C to one code. In the saved `phantom4_perpart` planes that is up to **66 %** of a frame
(frame 56), and the AGC then sees one bin holding most of the image.

The roadmap row proposed moving DN 0 to the band LUT's floor, 200 K. Measured before coding, that is
not low enough. On the Boson response the layered model's zenith sky falls below L_B(200 K) whenever
the air is at or below 0 °C, in every shipped preset. `us_standard_clear` at 15 °C and 20 % RH gives
210 K. The coldest case walked (haze preset, −40 °C air, 2 % RH) has L = 1.38 W m⁻² sr⁻¹ against
L_B(200 K) = 5.56. A dry winter zenith reading −80 °C or colder in 8–14 µm is also what field
measurements report, so the model is not the thing to fix.

## Options considered

1. **DN 0 at the LUT's 200 K.** Clears `phantom4_perpart` (coldest pixel 226 K) and keeps DN and LUT
   on one floor, but clips any winter sky. It fails §11.1 on the next cold scene.
2. **Extend the LUT below 200 K and put DN 0 there.** Moves the problem to a new number and
   regenerates every LUT bundle for it.
3. **DN 0 at zero scene radiance.** No scene can go below it. The DN of a sky colder than 200 K
   is still correct and distinct; only its *apparent temperature* clamps at the LUT floor, flagged
   by `radiance_out_of_range` as before (ADR 0011).
4. **A pedestal below zero radiance**, as real cores carry. Faithful to the hardware, but it spends
   codes on radiance no scene produces, and nothing downstream reads the offset.

## Decision

Option 3. `ADC_FLOOR_RADIANCE = 0.0` in `irsim.pipeline.core`. `PipelineConfig.from_sensor` builds a
bolometer's calibration with `RadiometricCalibration.from_radiance_range(sensor, lut, 0.0,
L_B(T_top), T_cal)`. `from_scene_range` now delegates to that and is otherwise unchanged. The pixel still
receives a positive flux at zero scene radiance, because the housing fills the rest of its view,
so the transfer's offset stays non-negative.

`RADIOMETRIC_RANGE_K` keeps its value and now means only what its name says: the span the radiometry
is specified over. Its top is still the ADC's top code, the gain states still replace that top
(ADR 0116), and the flat field's two bench blackbodies still sit at its ends. `adc_floor_radiance=None`
restores the pre-SC.23 transfer, so the old clip can be reproduced.

## Consequences

- Every bolometer DN moves. The gain falls by exactly (L_top − L_233)/L_top: the Boson's DN/K at
  300 K goes from **177.95 to 170.33** (−4.3 %), 5.9 mK a code on a 16-bit converter, still a ninth
  of a 50 mK NETD. No kelvin-denominated quantity moves: noise and the NUC residual are converted
  through the transfer (`dn_per_kelvin`).
- The DN16 route now round-trips inside half a code across the whole LUT: ≤ **9.5 mK** from 200.3 to
  470 K, where it was 33 K out at 200 K.
- Goldens regenerated deliberately. Every bolometer `dn16` moves (~2 500 DN, the offset) and
  `display8` by up to 9 codes. The float `radiance` and `apparent_t` planes move by at most
  **0.06 mK**, float32 round-off through a different gain. The one exception is a *late dead pixel* in
  the sensor-chain golden (SC.19 leaves it in the image). It read 233.12 K because DN 0 meant −40 °C,
  and now reads the LUT's 200 K clamp. A dead pixel is not a −40 °C scene, so the new value is the
  honest one. Photon cameras have no calibration of this kind and are bit-identical.
- `tests/unit/test_adc_floor.py` walks every atmosphere preset through a cold-to-humid weather grid
  and requires both that the coldest zenith sky sits more than 100 DN above zero and that at least
  one sky falls under L_B(200 K). Moving the floor back to 200 K fails the first clause. If the
  second one fails, this ADR's premise has changed.

## Revisit when

A camera's published DN-to-radiance calibration gives a real pedestal and scale (then option 4, with
its numbers), or a photon FPA gains an ADC transfer referenced to scene temperature.
