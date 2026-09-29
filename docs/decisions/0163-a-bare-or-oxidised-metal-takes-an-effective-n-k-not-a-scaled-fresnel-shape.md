# 0163 — A bare or oxidised metal takes an effective n, k, not a scaled Fresnel shape

Date: 2026-09-29
**Status:** Accepted
Roadmap: AT.25 (§4.2); builds on ADR 0038 (Level A: the n/k table supplies the angular shape, the authored band value the magnitude)

## Context

Level A took a clean metal's Fresnel curve from its published optical constants, divided by its
value at normal incidence and multiplied by the authored band emissivity, then clipped to 1.
For a dielectric the clean and the real surface have nearly the same normal emissivity and the
scaling is a few percent. For `bare_aluminium` the authored LWIR value is 0.09 -- the oxidised
alloy an airframe reaches within days -- while the clean metal's Drude curve gives 0.012 at
normal incidence and rises toward grazing: multiplied by 7.5 the curve reached 1 at 70° and
stayed there. A 280 K skin against a 230 K sky, which reads 236 K near normal, brightened to
280 K at the limb: a bright ring around every unpainted airframe and ship's hull, pinned as
an artefact by `test_material_table_angular.py:155`.

## Options considered

1. **Keep the scaled shape, cap it.** Hides the ring, keeps the wrong physics: the clip level
   would be a number nobody can defend.
2. **A thin dielectric film on a conductor.** The real structure -- a few tens of nanometres of
   oxide over the metal -- solved with the film's own optical constants and thickness. Correct,
   and it needs the oxide's n, k over four bands and a thickness per surface state, neither of
   which the library has.
3. **An effective conductor.** Scale the table's n and k by one factor until the band's
   normal-incidence Fresnel emissivity equals the authored ε_B, and take that conductor's own
   angular law. For a conductor ε(0) ≈ 4n/(n² + k²), so the factor is about ε_clean/ε_B < 1:
   a duller metal. The curve keeps a conductor's signature -- flat to ~60°, a peak short of
   the limb, zero at 90° -- and never clips. One line in the material file, no new data.
   The choice.

## Decision

`FresnelAngular` gains `fit: magnitude | effective_nk`, default `magnitude` (every existing
material bit for bit). With `effective_nk`, `directional_emissivity` calls
`effective_nk_table`, which solves the scale factor with Brent's method to 1e-10 in ε (cached
per table, response, target and reference temperature). ε(0) is unimodal in the factor: it
rises as n and k shrink, peaks where |ñ| ~ 1 and falls again as the metal becomes a weak
dielectric, so the root is taken on the branch continuous with the clean metal. Aluminium's
k/n caps that peak at 0.43 in LWIR, 0.24 in MWIR, 0.09 in SWIR and 0.06 in NIR; an authored
value beyond it (bare aluminium's NIR 0.08) takes the most emissive conductor and the rest as a
magnitude on its shape -- still 0 at grazing and never clipped, and returned as a second value
so the caller knows. The two aluminium entries that carry an authored ε_B far above the clean
metal's (`bare_aluminium`, `aluminium_polished`) declare the fit; water, glass, gold, copper and
steel keep the magnitude scaling.

**What the effective conductor gets right and wrong.** Right: the shape of a metal's angular
emissivity, which is what the limb needed -- peak below 0.3, zero at grazing, a hemispherical
emissivity 1.1-1.4 × the normal value where a dielectric's is below it. Wrong, and accepted:
the true oxide-on-metal curve peaks a little earlier and its hemispherical ratio is nearer
the low end of that range; the effective n and k are not the optical constants of anything
and must not be read back as material data. ESTIMATED, and named so in the material file.

## Consequences

* The bright limb on unpainted metal is gone; a bare aluminium skin at 89° now reads near
  the sky it reflects, as a mirror should.
* Every dielectric and every material without `fit` is unchanged, including all goldens.
* A material author who measures an oxidised surface's ε_B has one switch to make its angular
  behaviour a conductor's; a thin-film model (option 2) can replace `effective_nk` later
  under the same key.
