# ADR 0104 — A per-cell sky view factor from a sub-sampled 145-patch dome, scaling diffuse solar and longwave alike

**Status:** Accepted
**Date:** 2026-09-21
Roadmap: PT.21 (§5.4, §6.1; spec issue S44)

## Context

PT.18 (ADR 0095) gave every cell of a world-frame patch its own direct beam through the scene's
occluders. The diffuse solar and the longwave down still used the tilt's isotropic sky view
`V_s = (1 + cos β)/2` on every cell: a cell at the foot of a wall or under an overhang lost the
beam and nothing else, so it cooled to the sky all night as if the wall were not there. Measured
street canyons say otherwise — a hot-dry-climate study found passages at SVF ≈ 0.2 swinging 1.97 °C
where open walls swung 4.21 °C — and the nocturnal cooling of a surface goes with the sky it sees.

## Options considered

1. **One ray per Tregenza patch, visibility 0 or 1.** The roadmap's literal reading. A patch that
   straddles a wall's edge is neither seen nor hidden; measured, this put the foot of an infinite
   wall at 0.536 where the analytic answer is 0.5, outside the row's 0.01.
2. **Analytic view factors for rectangles.** Exact for one rectangle against one cell; the sum
   over several occluders over-counts where they overlap in the cell's view, which is every
   building corner. Rejected for the same reason as PT.18's shadow: the ray test composes,
   formulas do not.
3. **The 145 patches, each sub-sampled 3 × 4 in its own altitude–azimuth extent with the exact
   solid angle of each sub-cell,** visibility from the same `cell_shadow` ray–rectangle test the
   beam uses, cosine-weighted and normalised to the *unobstructed* quadrature of the same dome so
   an open cell's factor is `V_s` to the bit. 1740 rays per cell, once (the occluders do not
   move). The azimuth offsets `(2m + 1)/8` of a patch never land on a cardinal direction for any
   band count, so an N–S or E–W wall never sees a ray lying exactly in its own plane, which the
   ray test must let pass. Chosen.

## Decision

Option 3, `irsim.thermal.skyview`. The scene computes the factor for every world-frame patch
under occluders and hands it to `CellForcing`, which scales **both** the diffuse solar
(`SVF · DHI`) and the longwave down (`longwave_down` with the cell's factor, the rest of the
hemisphere at the air temperature, as the per-prim balance already does). `sky_view_longwave`
exists only as the roadmap's negative control. A surface without a patch, or a patch without
occluders, is unchanged, and a surface's own `shaded` flag now gates the beam under a supplied
sky view the way it always did without one.

The Perez split of the diffuse into dome, circumsolar and horizon bands (EnergyPlus's) is **not**
implemented: the dome is isotropic here. Deferred rather than skipped — it changes a wall's
diffuse by tens of W/m² near the sun and needs Perez's coefficient tables, which are a data
question for the day the aerial or maritime lane asks for a sky radiance distribution.

## Consequences

**Measured:** open sky exactly 1.000 (and exactly `V_s` on a tilt); the foot of an infinite
wall and the edge of an infinite overhang 0.5 within 0.01; the sub-rays tile each patch's solid
angle exactly and sum to 2π. Two shaded asphalt cells at SVF 0.2 and 0.9 under the facet
scene's weather: the enclosed cell swings 11.9 K against 14.9 K (ratio 1.25) and sits 3.2 K
warmer at its night minimum; with the factor on solar only the two minima are within 0.05 K.
On the R1 wall scene the concrete terminator drops from 10.3 K to 9.7 K and the render's from
7.4 K to 6.3 K, because the shaded half stands beside the neighbour and sees its roof instead of
a slice of cold sky; the post-cap scene's never-shaded cells sit within 20 mK of the per-prim
solve rather than on it.

**Not reached:** the roadmap's ratio near 2. With one air temperature for the whole scene a
cell's swing cannot fall below the air's own 12 K, whatever its sky view; the passage in the
study has its own cooler, stiller air. That is a microclimate the weather file does not carry,
not a term this factor could supply — recorded as a deviation, the night-minimum sign kept as
the criterion the control cannot pass.

**What it costs:** 1740 ray–rectangle passes per patch at build, about a second for the wall
scene's 48 × 48 ground; nothing per tick. PT.18's "unshaded cells stay bit-identical to the
prim" is now "cells with an unobstructed dome stay bit-identical": under any occluder every cell
sees a little less sky, and the identity moved to where the dome is open.

## Revisit when

A scene wants the Perez distribution (a wall's circumsolar diffuse), a moving occluder (PT.9:
the factor is computed once), or the hidden hemisphere at a surface temperature rather than the
air's (a canyon's walls at night) — the last is what the ratio-near-2 anchor actually measures.
