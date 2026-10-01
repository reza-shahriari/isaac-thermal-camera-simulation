# ADR 0175 — A material authors a curve, a band table or a grey value, layered under the camera's response

**Status:** Accepted
**Date:** 2026-10-01
Extends [ADR 0010](0010-grey-within-band-emissivity.md) (Planck-weighted band averaging, still the
only sanctioned reduction) and [ADR 0040](0040-material-file-layout-and-closure-reading.md) (one
file per material, one authored quantity). Roadmap: `AT.32`; followed by `AT.33` (tables built
from the camera's response) and `AT.34` (the imported curves wired in).

## Context

A camera is configured by its wavelengths: `band.lambda_min_um`, `band.lambda_max_um` and a
response curve R(λ). The radiometry follows them exactly. The materials did not: 75 of the 83
library materials carried one number per band *name*, so a 6–13 µm camera and a 7.5–13.5 µm one
read the same LWIR ε. For most surfaces the error is small. For silicates (sand, glass, concrete)
it is not: their emissivity dips sharply at 8–10 µm, so how much of the dip a camera sees decides
the value.

The schema allowed exactly one optical form: a spectral file *or* a per-band table. That
blocked the obvious fix, because the measured curves in the repository do not span the bands:

- SLUM measures reflectance at 0.35–2.5 µm and emission at 8–14 µm. That is two files, one of
  them the complement quantity, with no MWIR between them and no 7.25–8 µm at the start of a
  Boson's response.
- USGS covers only 0.35–2.5 µm.

Under "exactly one form", a material could keep its curve only by losing its MWIR number. So
every importer reduced its curves to four numbers, and the curves were never used at render time.

The owner also asked for the simplest form, a single value for all bands. Until now that had to
be typed four times.

## Decision

1. **Three forms of the one authored quantity (ε or ρ), in any non-empty combination:**
   `spectral_emissivity` (a curve), `emissivity_per_band` (a table) and `emissivity` (one grey
   value), or the same three spelled `reflectance`. Authoring both quantities is still refused
   (CLAUDE.md #4).
2. **Layered wavelength by wavelength, under the camera's own response.** At each wavelength the
   value is the curve if a segment has data there, else the camera band's per-band value, else
   the grey value. The assembled spectrum is reduced by ADR 0010's Planck × response average.
   `BandProperties.curve_fraction` reports the curve's share of that weight.
3. **A curve may be a list of disjoint segments.** A segment may tabulate the opaque complement
   (`{file: sw.csv, quantity: reflectance}` inside `spectral_emissivity`), read as ε = 1 − ρ.
   This is not a second authoring: at every wavelength exactly one quantity is given. The library
   refuses such a segment over any band where the material authors τ > 0, because there the
   complement is 1 − ρ − τ.
4. **Each number is authored once.** A per-band value for a registry band whose whole nominal
   range the curve covers is refused at load, because the curve would silently overrule it. To
   fill outside a covered band for an unusually wide camera, author the grey value.
5. **Gaps are never extrapolated.** Where the curve has no data and neither fallback exists,
   the camera's band is refused, as before.

## Options considered

- **Keep "exactly one form" and require full-range curves.** This would throw away the measured
  curves we have, since none of them spans 0.3–14 µm.
- **Hold the curve's end values across gaps** (what `tabulated` does for its own padding).
  Across the 2.5–8 µm hole this invents a whole band. Refused, for the same reason ADR 0010
  refuses extrapolation at band edges.
- **Let the curve silently overrule a band value it covers.** Two numbers for one band, one of
  them dead, is the kind of contradiction AT.17 found in `bare_aluminium`. Refused at load.
- **Per-pixel spectral rendering** (ADR 0010 option 4). Still not needed. The reduction happens
  once per (material, camera) pair, and the kernels receive one value per material, as before.

## Consequences

- A camera's value is now its own, as long as its response is passed (`AT.33` makes every table
  builder do so). Materials without a curve read exactly what they read before, whatever the
  camera, which is the honest answer when nothing more is known.
- The boundary where the fallback takes over is a step in ε. Simpson on the 0.01 µm grid errs by
  O(jump × dλ) there: 2.2e-4 in ε for a 0.45 jump, about 15 mK apparent
  (`tests/unit/test_material_optical_forms.py`).
- A filled value assumes the material is flat across the gap. `curve_fraction` says how much of
  the band rests on that assumption.
- `Material.spectrum` remains for the one-plain-file case. The general form is `Material.curve`,
  a `SpectralCurve`.
- The Blender add-on's new-material form can still author only the per-band table. Adding the
  other two forms is its own step (`blender_addon/PLAN.md` B11; roadmap `AT.36` for what the
  project must expose).

## Revisit when

A material with in-band structure needs a value that depends on scene temperature (ADR 0010's
own revisit condition), or a measured library arrives whose segments overlap and need merging
rather than refusing.
