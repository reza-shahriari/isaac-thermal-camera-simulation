# ADR 0177 — The atmosphere is anchored once per band, not once per camera

**Status:** Accepted
**Date:** 2026-10-01
Amends [ADR 0113](0113-one-wavelength-ladder-for-spectral-classes.md) (one spectral ladder, classes derived from a
band's span) and the anchor solve of [ADR 0071](0071-layered-slant-path-atmosphere.md).
Roadmap: `AT.35`. Follows [ADR 0175](0175-a-material-authors-a-curve-a-band-table-or-a-grey-value-layered.md),
which let a camera's own wavelength range decide its material values.

## Context

`LayeredAtmosphere.exponential_sum` splits a band into spectral classes from one wavelength ladder
(window, lines, edges, opaque water and CO₂ bands, each with a fixed multiplier). It then scales
the non-opaque classes by one factor so that the band's τ(200 m) equals the preset's grey
per-band value. That factor was solved on **each camera's own** classes and weights.

So every camera in a band was forced to the same τ(200 m), however much absorbing edge its range
held. A camera with less edge bought a murkier window. Measured on `us_standard_clear`, an 8–12 µm
camera, which sits almost entirely in the atmospheric window, read τ(5 km) **0.311** against
**0.588** for 7.5–13.5 µm. On `midlat_summer_humid` it read **0.008** against **0.206**. The
answer was backwards, and the ladder's per-wavelength physics was being undone by the anchor.

Once ADR 0175 made cameras distinct by their wavelengths, this stopped being hypothetical.

## Decision

The scale is a property of the air. It is solved **once per band, on the band's nominal top-hat**
(`classes_for(band)`, `class_weights(band)`, the range the preset's per-band coefficients
describe), and every camera in the band applies the resulting γ per class to its own classes and
weights (`LayeredAtmosphere._anchor_scale`).

- The nominal band is still anchored to the preset exactly (τ(200 m) to 1e-9).
- Two cameras now share each class's γ. Their τ differs only as their spectral classes do: a
  narrower camera that sheds absorbing classes transmits more, a wider one less.

## Options considered

1. **Keep per-camera anchoring.** This is consistent with "the preset is the camera's τ(200 m)",
   but it makes the extinction of air depend on who looks through it. Refused.
2. **Anchor on the shipped camera of each band.** The Boson would be bit-identical, but a band
   would then be defined by whichever file happens to ship. The presets name bands, not cameras.
3. **Anchor on the nominal top-hat.** Chosen. It is the only reference every camera of the band
   shares and the only one the preset tables (§7.2) can be read as describing.

## Consequences — measured

Shipped cameras whose responses reach past their nominal range now keep the absorption their extra
classes carry, rather than having it cancelled by the anchor. On `us_standard_clear`:

| Camera | τ(200 m) | τ(5 km) |
|---|---|---|
| Boson LWIR | 0.93006 → 0.93017 | 0.59948 → 0.59975 (+0.05 %) |
| InSb MWIR | 0.7970 → 0.7905 | 0.3614 → 0.3460 (−4.3 %) |
| InGaAs SWIR | 0.8245 → 0.8262 | 0.3880 → 0.3938 (+1.5 %) |
| Si NIR | 0.9743 → 0.9762 | 0.5288 → 0.5492 (+3.9 %) |

The golden arrays were regenerated deliberately: the LWIR rows move ≤ 0.12 % (the sea's apparent
temperature ≤ 8.8 mK), while the MWIR sky radiance moves up to 4.6 %. The ladder and golden pins
were re-pinned with this ADR named.

For custom LWIR ranges, the 8–12 µm camera now reads τ(5 km) 0.743 against 0.588 for the nominal
band, and 6–13 µm reads 0.526. Its τ(200 m) drops by the opaque 6.3 µm class's share of the band
(0.103) and more (`tests/unit/test_atmosphere_custom_band.py`).

## Revisit when

A preset's per-band coefficients are refitted against a measured camera (they would then describe
that camera's response, and the anchor should use it). Or the grey per-band coefficients are
replaced by a spectral γ(λ) table, at which point there is no anchor to solve.
