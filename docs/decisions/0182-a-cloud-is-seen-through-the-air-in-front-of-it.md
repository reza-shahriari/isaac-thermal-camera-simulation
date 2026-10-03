# ADR 0182 — A cloud is seen through the air in front of it, and that air may be in its shadow

**Status:** Accepted
**Date:** 2026-10-03
Implements the fifth criterion of `docs/physics-model.md` §7.5 for the visible companion's cloud.
Roadmap: `WX.5`, which this ADR narrows to aerial perspective; the dome's resolution is split out as
`WX.21`. The change is in `isaac-weather-fx` (`582bf47`), and this repository takes it as a submodule
bump.

The infrared band already composes its cloud through the air:

* sky pixels through the slant path to the base (ADR 0126);
* geometry pixels through τ_air(R_c) (ADR 0162).

This change puts the visible band on the same footing.

## Context

The dome composed a cloud straight over the sky, `T_c · L_sky + added`, as if the cloud were at the
observer. Two things followed.

* **A cloud 30 km off was as crisp and as bright as one overhead.** Near the horizon the dome showed a
  white wall of distant cumulus. A real sky shows them hazed toward the horizon's colour.
* **Most of a haze's light comes from nearby air.** Aerosol sits in the lowest kilometre or so, so a
  haze's light comes mostly from the first kilometres of a path. In the sky view's own integral, the
  first 1.4 km of air toward a 62° sun gives 28 of the aureole's 44.5 kcd/m².

## Options considered

1. **Leave it.** This costs nothing, and the horizon stays wrong.
2. **A uniform haze by range**, `exp(−R/V)` toward the horizon colour. It is cheap, but it is not the
   atmosphere the sky is drawn from. The colour does not change with the sun's angle, and nothing
   depends on altitude.
3. **The sky's own scattering integral, kept at each distance (chosen).** This is Hillaire 2020's
   aerial-perspective volume (`docs/physics-model.md` [R116]) on the sky view's angles, at 16 slices up
   to 64 km. Far enough up a ray it is exactly the sky.

## Decision

Option 3.

* **The march reports an emission range**: the mean distance along each ray, weighted by what each
  step scatters in. The composite becomes `T_air(R) · added + (1 − T_c) · L_in(0, R)`. It is
  premultiplied, so upsampling and edge integration (`WX.4`) are untouched.
* **The air light is kept in two parts**: the direct beam's share and the multiply-scattered share.
  The direct share is scaled by the sun's transmittance through the cloud field, read from the field's
  own light map at four points along the path below the cloud base.

The shadowed direct share is the approximation of record. Without it, a cloud near the sun was veiled
by an aureole of air that is in fact in its shadow, and it read brighter than the clear sky.

## Consequences

* **Measured** (upstream `tests/test_cloud_light.py`):
  * At 64 km the air light is the sky view within 1.5 % from 30° up.
  * A black object's contrast follows Koschmieder's law at 5 km near the horizon: 1.02 of the
    transmittance. As the ray climbs out of the aerosol it stays above that law: 1.07 at 15 km and
    1.16 at 30 km.
  * A cloud 20 km off moves to within 0.6 of its old distance from the horizon's colour. A cloud
    overhead changes by under 15 %.
* **The roadmap's test was the answer for uniform air.** It asked for contrast within 5 % of the
  transmittance at 5, 15 and 30 km. That holds only where the air is uniform. A stratified atmosphere
  departs from it by 7 % at 15 km and 16 % at 30 km, as it should.
* **It costs about 2.7 s on the first 1024-row bake.** The tables are cached per sun elevation.
  `clouds.aerial_perspective` turns the feature off.
* **The shadowing is approximate:**
  * four points per ray;
  * the air above the cloud base taken as sunlit;
  * the multiply-scattered share never shadowed.

  The unshadowed case bounds the error: in the measured near-sun case, a 28 kcd/m² veil.
* **The clear sky is not shadowed.** A cloud's shadow on the sky behind it and around it is still
  missing.

## Revisit when

* Light shafts (crepuscular rays) are wanted. They are this same shadowing, integrated along the ray
  with more than four points.
* The dome's resolution rises (`WX.21`). The aerial-perspective cost is per marched texel.
