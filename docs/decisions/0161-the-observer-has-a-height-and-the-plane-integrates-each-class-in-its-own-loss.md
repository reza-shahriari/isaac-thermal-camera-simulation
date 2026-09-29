# 0161 — The observer has a height; the per-pixel plane integrates each class in its own loss

Date: 2026-09-28
**Status:** Accepted
Roadmap: AT.28 (§7.1); builds on ADR 0071 (the layered model), AT.1 (per-pixel slant rays), PT.28 (targets have an altitude)

## Context

`column_length` integrated the density from the surface, so a camera on a mast, a drone or an
aircraft saw surface-density air along its whole ray, and every pixel of an air-to-air frame at
3 km carried the extinction and emission of a sea-level path. PT.28 gave targets an altitude; the
camera needed one too. The transmittance is a closed form either way. The emission is not: the
per-pixel path-radiance plane is a table over elevation built once per frame at the surface
(ADR 0071's slant tables), and a table per observer height per frame would rebuild it every time
a drone climbs a metre. From height the plane needs a per-pixel integral that is cheap enough
to run every frame and accurate enough that the sub-pixel target path beside it (exact
quadrature) and the pixel it sits in agree.

## Options considered

1. **Ignore the camera's height.** The status quo: 37 %-class errors on every pixel of an
   airborne frame, the same order AT.1 fixed for slant rays.
2. **Exact per-pixel quadrature from height.** Correct, and a 4000-step quadrature per pixel
   per frame: the cost AT.1's tables exist to avoid (minutes per frame).
3. **Exact transmittance from height; emission isothermal at the ray's mean height.** One LUT
   lookup per pixel. Tried first and measured against option 2: 1–4 % high looking down and
   4 % low looking up from 3 km, because absorption weights the emission toward the camera's
   end of the ray, not its middle. Rejected on that measurement.
4. **Eight pieces of equal distance per ray.** Within 0.1 % where the ray is short, but a 5°
   descent from 8 km read 3 % low: the dense far end of that 90 km ray fell into one 45 km
   piece. Rejected.
5. **Each class on its own grid of equal transmittance loss, midpoint rule in the loss
   coordinate.** Every class of the exponential sum is cut into eight pieces that each hold
   1/8 of that class's molecular loss `1 − e^{−γ_k C_k}` -- a strong absorber's pieces crowd
   the first few hundred metres, a weak one's follow the air's density to the far end -- and
   each piece emits its **exact** loss (closed-form column, aerosol included) at the band
   radiance of the air where half of that piece's loss has happened. Both the piece edges and
   the half-loss points come from inverting the closed-form slant column, so nothing is
   iterated. The choice.

## Decision

Option 5 for the plane, exact quadrature for the point-target path, and both read the same
column: `column_length(d, θ, H, z₀)` is `e^{−z₀/H} · (H/sinθ)(1 − e^{−d sinθ/H})` for either sign
of sinθ, evaluated no further than the ground for a descending ray, and bit for bit the old
function at `z₀ = 0`. `optical_depths`, `transmittance`, `path_radiance_per_class`,
`class_transmittances`, `sky_beyond_per_class` and `apply_layered_gbuffer` take
`observer_height_m`; the frame reads it from the plane dict (`observer_height_m`, a scalar a
driver sets from its camera), defaulting to the surface.

**Why the loss coordinate.** The path radiance of one class is `∫ L_B(T(h(s))) d(1 − e^{−τ_k(s)})`:
an integral of the air's radiance against the class's transmittance loss. The midpoint rule in
that coordinate weights every piece exactly and puts its evaluation point where the piece's
emission actually comes from; its error is second order in the change of `L_B(T(h))` across a
piece, a few kelvin at most, and it needs no knowledge of where along the ray the air is dense
because the grid already follows it. Measured on the Boson LWIR band over `us_standard_clear`,
`midlat_summer_humid`, `haze` and `fog_light_200m`, thirteen geometries from a 1 m mast to a
10 km airliner, rays from 100 m to infinite, elevations from −80° to +90°: the plane sits within
**0.5 %** of the 4000-step quadrature everywhere (worst: an infinite ray rising 30° from 3 km),
0.3 % at 45° down over 5 km, exact on a horizontal ray. Twelve pieces bring the worst case to
0.3 % for 1.5× the time; eight is the choice at 0.7 s per 640 × 512 airborne frame, a surface
frame paying nothing.

## Consequences

* `z₀ = 0` scenes are bit-identical, including every slant-table pin and golden.
* An airborne frame's atmosphere is now its own: thinner going up, denser going down.
* The plane's emission from height is an eight-piece quadrature, ESTIMATED at 0.5 % worst case
  over the geometries in `tests/unit/test_observer_height.py`; `PLANE_SEGMENTS` is the knob.
* An airborne frame's path-radiance plane costs 0.7 s per 640 × 512 frame on the CPU
  reference; a surface frame still reads the slant tables.
* The sky (`SkyModel`) still describes the column above the *surface*; a sky seen from 3 km is
  a thinner column, and that is a later row.
