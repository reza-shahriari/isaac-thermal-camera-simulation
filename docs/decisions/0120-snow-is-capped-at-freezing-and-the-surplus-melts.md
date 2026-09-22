# ADR 0120 — Snow is capped at freezing, and the surplus melts isothermally

**Status:** Accepted. Adds a phase change to the §6.1 balance that
[ADR 0036](0036-rk2-for-the-surface-balance.md) chose RK2 for; the cap is applied *instead of* the
RK2 step in the melting regime rather than after it, for the reason measured below.
**Date:** 2026-09-22
Roadmap: PH.10 (§6.1)

## Context

Every surface in this project answers `C dT/dt = net_flux`. Snow does not. Under 200 W/m² of net
gain any other material climbs tens of kelvin above air by mid-afternoon; a melting snowfield stays
at **exactly 273.15 K** for as long as there is snow, and the gain goes into fusion.

Without the cap a spring snowfield renders as a hot patch. That is not a small radiometric error —
it is the opposite sign of what a thermal camera sees, and it would be the most visible thing in
any winter scene.

## Decision

Cap the surface at 273.15 K and route the surplus into melt at L_f = 334 kJ/kg, in
`irsim.thermal.snow.melt_capped_step`, with two regimes.

**A cell already at the cap melts isothermally, and its flux is read at the melt point.** This is
the sub-decision worth recording, because the obvious implementation is to step with RK2 and clamp
afterwards, and it is wrong in a way that passes review.

RK2 evaluates the balance at a midpoint half a step *above* the cap. That temperature does not
exist — the surface is melting and isothermal — and both emission and convection are larger there,
so the surplus comes out too small. Measured on a 63 kJ/m²K snowpack with a 60 s step: the naive
form under-reports the melt by between 0.1 % and 1 %, **every step, always in the same direction**.
Reading the flux at 273.15 K instead gives exactly `Q Δt / L_f`, and the test asserts the correct
answer against that closed form rather than against a remembered number.

**A cell below the cap that a step carries through it splits its step.** Its surplus enthalpy
`C (T_stepped − T_melt)` becomes melt and the temperature lands on the cap, so energy is conserved
exactly across the transition. Discarding the overshoot instead would lose energy at every thaw,
twice a day in a scene that crosses freezing twice a day.

**A finite pack hands back what it cannot melt.** With `available_kg_m2` given, melt is limited to
the snow present and the leftover energy warms the surface — bare ground is free to rise. Without
it the pack is unlimited, which is the right default for a scene showing a snowfield and the wrong
one for a patch about to disappear. Omitting the limit entirely would have let the last micron of
snow pin a surface at freezing forever, which is exactly the artefact a melt cap exists to avoid.

## Consequences

**The two phenomenology numbers.** +200 W/m² holds the cell at 273.15 K bit-exactly and sheds
**2.1557 mm of water equivalent per hour**, which is `200 × 3600 / 334000` and not a fit.

The night-time behaviour needs **no special case at all** and is the better evidence that the
emissivity chain is right. With snow's ε_hemi of 0.9874 (the library's 0.99 LWIR band value pulled
down by its own angular fall-off, which is what the row asked for), eight hours of night gives:

| sky | wind | T_s − T_air |
|---|---|---|
| clear | 0.5 m/s | **−12.73 K** |
| clear | 4 m/s | −4.75 K |
| overcast | 0.5 m/s | **+0.00 K** |

The overcast figure is *identically* zero, not merely small: a fully overcast sky radiates as a
blackbody at air temperature, so absorbed and emitted longwave cancel at `T_s = T_air` and there is
no imbalance left for wind or heat capacity to act on. Wind matters only in the clear case, which
is what the breezy row shows.

**The Tier 4 bar is recorded, not run.** `ALPINE_TIER4_MAE_K = (0.7, 1.3)` carries the alpine
ESSD 16 (2024) series' acceptance band for snow-surface temperature. The series is **not fetched**
and no check in this repo evaluates against it. It is stored now for the same reason SE.1 stored
the sea's published envelope: so that when a comparison is written the bar is the published
instrument's rather than one invented at the time to fit whatever the model produces.

**No scene declares snow.** There is no `snow:` block in the scene schema and no shipped scene uses
the melt cap, exactly as `PH.7` shipped fire with no `fire:` block. `melt_capped_step` has the same
shape as `rk2_step`, so the seam for a solver to swap it in exists; wiring it would mean a schema
version bump and touching every scene YAML, which is a step of its own and not this one.

**What is not modelled, in order of how much it would matter.** (1) No cold content through the
pack depth — one surface, one temperature, so a deep pack responds as fast as a thin one. (2) No
liquid water retained and refrozen, which is why `melt_rate_kg_m2_s` reports zero rather than a
negative rate for a cooling surface: there is no store to refreeze, and a negative melt would
create snow. (3) No densification and no albedo ageing — the material's 0.15 solar absorptivity is
fresh snow and does not decay toward the 0.4–0.5 of old melting snow, so a modelled spring
snowfield absorbs too little sun and under-melts. (4) Sublimation only insofar as a caller supplies
a latent term through the existing `wet_fraction` path, which is a liquid-water formulation and
uses L_v, not L_s.
