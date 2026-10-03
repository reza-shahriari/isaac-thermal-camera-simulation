# ADR 0184 — A cloud's shape is measured from above, and removing the tiling is its own step

**Status:** Accepted
**Date:** 2026-10-03
Implements the shape table of `docs/physics-model.md` §7.5. Roadmap: `WX.6`, which this ADR narrows to
the shape targets and the stratocumulus streaks; removing the horizontal tiling is split out as
`WX.22`. The change is in `isaac-weather-fx` (`7d15e58`), and this repository takes it as a submodule
bump. Both bands read the one field, so the change reaches the infrared march and the visible dome
alike.

## Context

§7.5 gives targets for the shape of real cumulus: an area–perimeter dimension $D$ of 1.3–1.4, and a
size exponent $b$ of 1.7–2.0 below a 0.5–1 km break. Until now nothing measured the field against
them.

Measured at 15 m, the resolution of the ASTER studies, over three seeds:

* **The outline was already right.** D was 1.31–1.34 for 35 % cumulus.
* **The sizes were not.** b was 2.58 for 35 % cumulus and 2.86 for congestus. The coverage map that
  places the clouds was cut at 800 m, and its thermals sat on one lattice. The field held clouds
  about a kilometre across, plus specks of edge detail.
* **Stratocumulus was a barcode.** It was stretched 6.25 : 1 along the field's z axis, whatever the
  wind. Its mask had 4.1 times the mean spectral power along that axis; an isotropic field gives
  0.84–1.23.
* **The field repeats every 15.36 km tile.** The autocorrelation at the tile period is 1 by
  construction.

## Options considered

For the sizes:

1. **Tune the threshold or the erosion.** Neither moves the clouds' sizes: the coverage map places
   them.
2. **Widen the coverage map's band and give the thermals several sizes (chosen).** The map carries
   features down to 200 m, and the cellular term has three octaves, each twice as fine and 0.6 as
   strong. The decay was chosen between two measured failures: at 0.5, b was 2.0 at 35 % cover; at
   0.7, b was 1.67 at 15 %.

For the streaks:

1. **A milder stretch.** At 1.5 : 1 the stripe ratio is still 2.1.
2. **Isotropic stratocumulus (chosen).** Broken stratocumulus organises as mesoscale cells (Wood
   2012). The stretch was also never tied to the wind's direction. Rolls are a distinct regime, and
   one this field does not model.

For the tiling:

1. **A larger tile.** It halves the repetition at four times the memory, and still repeats.
2. **A field that does not repeat** — a warp or a coverage map with an incommensurate period. Every
   consumer that assumes exact periodicity then has to change:
   * the sun light maps, built over one tile;
   * the path tracer's volumes, one baked tile placed several times;
   * the cover and extinction calibration.

   That is a size-L step of its own: `WX.22`.

## Decision

* **Sizes:** option 2.
* **Streaks:** isotropic stratocumulus.
* **Tiling:** split out as `WX.22`.

The measurement lives upstream as `weather_fx.core.morphology` (plain numpy, periodic labelling), with
`tools/measure_cloud_shape.py` to print it.

## Consequences

* **Measured after**, at 15 m over three seeds:

  | case | D | b | stripe ratio |
  |---|---|---|---|
  | cumulus, 35 % | 1.31–1.34 | 1.91 | 1.11–1.23 |
  | cumulus, 15 % | 1.30–1.33 | 1.73 | 1.11–1.16 |
  | congestus, 35 % | 1.32–1.35 | 1.90 | 0.91–1.05 |
  | stratocumulus, 60 % | 1.26–1.29 | 1.97 | 1.15–1.19 |

  Enclosed clear area stayed at 2–6 % of cloud area, as before: the extra small clouds did not
  perforate the large ones.
* **Two upstream tests are restated:**
  * The thin-edge test now excludes 20° around the sun rather than 15°. The new field put one wisp
    at 17.5° in its own cloud's shadow, at 0.993 of the unshadowed aureole behind it. That is the
    unshadowed-aureole limitation ADR 0182 records, not a new dark rim; the other 388 thin edges are
    above 1.
  * The finest-pitch test marches 2000 rays, not 200. Its 99th percentile had been the
    second-worst ray. On 2000 rays the claim stands as before: the coarse march's error is 3.38
    times the fine one's, against 3.43 before.
* **Cirrus stays streaked** (stripe ratio 6.7), as cirrus fibratus is. Its streaks still lie along
  the field's axis rather than the wind's.
* **Stratocumulus D is 1.26–1.29,** just under cumulus' range. The targets are for cumulus, and no
  stratocumulus outline dimension is pinned.

## Revisit when

* `WX.22` makes the field aperiodic. The measurement must then run over more than one tile.
* Streaks are tied to the wind. Cirrus and roll clouds would then take a stretch along the wind,
  which needs the light maps rotated with it.
