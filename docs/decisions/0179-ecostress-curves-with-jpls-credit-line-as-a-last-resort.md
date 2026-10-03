# ADR 0179 — ECOSTRESS curves, with JPL's credit line, where no freer source exists

**Status:** Accepted (owner decision, 2026-10-01)
**Date:** 2026-10-01
Amends [ADR 0041](0041-data-provenance-and-licences.md) (only public-domain
or CC0 optical data is committed). Uses [ADR 0175](0175-a-material-authors-a-curve-a-band-table-or-a-grey-value-layered.md)'s
curve forms. Roadmap: `XD.14`.

## Context

A camera whose range is not its band's nominal one reads a material off its spectral curve
(ADR 0175). By `XD.14`'s second part, 62 of the 82 library materials had a curve. The licence-clean
sources were exhausted:

- **RefractiveIndex.INFO** (CC0) has infrared n, k only for simple substances. It gave glass, dry
  soil, snow and polycarbonate their curves.
- **USGS splib07a** (public domain) has thermal-infrared spectra of minerals and a few chemicals,
  but none of man-made materials. Its man-made spectra stop at 2.5 µm.
- **SLUM** (MPL-2.0) is urban surfaces, already imported.

What remained (rubber, road asphalt, construction concrete, car and aircraft paint) is measured
from the visible to 14 µm in the **ECOSTRESS spectral library** (NASA JPL; formerly the ASTER
library, with JHU and USGS contributions). ECOSTRESS states no licence. It asks users to cite two
papers and gives a credit line that begins "Reproduced from the ECOSTRESS Spectral Library through
the courtesy of the Jet Propulsion Laboratory" and ends "ALL RIGHTS RESERVED". Other projects
redistribute its spectra under that line; one is the R package photobiologyFilters. ADR 0041,
read strictly, refuses it, which is why `XD.14` stopped.

## Decision

The owner accepted ECOSTRESS with its credit line, **for materials no other source covers**
("use that where we don't have materials from other sources").

1. **Last rung.** A material takes an ECOSTRESS curve only when it has no CC0 n, k table, no
   SLUM or USGS curve of its own class, and no curve derived from one. `scripts/import_ecostress.py`
   lists each such material and the sample chosen for it, with the reason.
2. **Shape from the sample, level from the library.** These materials are §16.2's literature rows,
   which `test_library_v0.py` pins on purpose. One sample of a class (roofing rubber for a tyre,
   an enamel for a car) is not a better *level* for the class than the literature value. What
   the materials lacked was a *curve*. Each therefore becomes ε(λ) = 1 − s·R(λ) over 5.02–14.1 µm,
   with R the sample's measured reflectance and s holding the nominal LWIR average at the authored
   value. This is the rule `derive_proxy_shape_curves.py` applies to optical-constant shapes.
   NIR, SWIR and MWIR stay typed.
3. **All paints share one shape.** The four library paints take one gloss enamel sample (JPL 1247).
   Thermal emissivity is set by the binder, not the colour (`test_colour_does_not_set_emissivity.py`),
   so paints with the same authored level read the same value to every camera.
4. **The credit travels with the data.** Every committed file carries the credit line verbatim,
   both citations, the sample's ECOSTRESS name, number and description, and how it is used;
   `docs/materials.md` repeats the credit line. A class proxy says `CLASS PROXY`.

A first attempt adopted the samples' *levels* too. It broke the paint rule (the black and the
white paints read different samples), moved three golden frames, and replaced a tyre's 0.95 with a
roofing rubber's 0.914. It was withdrawn before commit.

## Consequences

- Eight materials gained a long-wave curve: `concrete`, `asphalt_dry`, `rubber_tyre`,
  `propeller_rubber` and the four paints (`car_paint_black`, `car_paint_white`,
  `aircraft_aluminium_painted`, `painted_composite`). `car_paint_black`'s 7-row estimated sketch
  is replaced, and its other bands keep the sketch's own band values. No standard-band value
  moved, so no render at a nominal range changes. A camera with its own range now sees each
  sample's structure: concrete reads 0.909 / 0.920 / 0.943 to 8–12 / 7.5–13.5 / 10–13 µm cameras.
- If JPL ever states terms that forbid this, the eight curve files and their material links go
  with one `git revert`; every one is listed in the importer.
- ADR 0041's rule still holds for everything else: a new material is sought in CC0 and
  public-domain sources first.

## Revisit when

A CC0 or public-domain measurement of one of these classes appears; it then replaces the
ECOSTRESS curve. Also revisit if JPL publishes an explicit licence for the library.
