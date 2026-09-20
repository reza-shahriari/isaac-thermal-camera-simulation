# ADR 0101 — The latent-heat term, and a water film the solver keeps per cell

**Status:** Accepted
**Date:** 2026-09-20
Roadmap: PH.1 (§6.1, §15 Tier 3; spec issue S45)

## Context

§6.1's balance has solar, longwave, emission, convection and an internal term, and no latent
one. §15 Tier 3 asks that "wet asphalt reads colder than dry" emerge unscripted, which that
balance cannot do; `sea_skin.py` says in its own docstring that at sea the latent flux is
usually the largest of the turbulent terms and omits it; nothing consumes the weather's
`precip_mm_h`. The owner's 2026-09-18 requirement names water beside fire. The survey's anchors:
Hendel 2014's watered Paris pavements read 6–13 K colder than dry in sun (FLIR B400) and
reconverge in 15–120 minutes as the film goes; COARE 3.6's Dalton number sits at 1.1–1.2e-3 in
the 5–10 m/s range.

The term depends on the surface's own temperature through `q_sat(T_s)`, like emission, so it
cannot be a forcing term computed once per tick from the weather; and a film is a **state**
beside the temperature — rain fills it, evaporation empties it, and a cell is wet exactly while
it holds water.

## Options considered

1. **A fixed cooling offset for "wet" surfaces.** The shortcut every dataset renderer takes;
   independent of humidity, wind and time, so the drying spike the camera saw cannot happen.
2. **Latent flux as a forcing term at the air temperature** (`q_sat(T_a) − q_a`). Cheap and
   wrong in the way that matters: a wet surface in sun is warmer than the air and evaporates
   faster than this says; a surface below the dew point would evaporate instead of collecting
   dew.
3. **The bulk-aerodynamic term in the balance, with a film state in the solver.**
   `Q_L = L_v ρ_a (q_sat(T_s) − q_a) / (r_a + r_s)`, `r_a = 1/(C_E U)`, evaluated on the
   solver's own state at every stage of the step; the forcing carries what the weather knows
   (`q_a`, `g_e = ρ_a C_E U`, the rain rate); the solver keeps `film_kg_m2` per facet. Chosen.
4. **Penman–Monteith with a full canopy model.** The right tool for vegetation (PH.11) and far
   beyond what a road or a sea needs; `r_s` is kept as the hook it plugs into.

## Decision

Option 3, `irsim.thermal.latent` and the fields it adds:

* `SurfaceForcing` / `FacetForcing` gain `q_air_kg_kg`, `g_e_kg_m2_s`, `wet_fraction`,
  `r_s_s_m` and (`FacetForcing` only) `precip_kg_m2_s`. The balance subtracts
  `L_v E(T_s)` over the wet fraction; with no wetness declared and no film the expression is
  never evaluated, so a dry scene is **bit-identical** to before.
* `FacetSolver(film_kg_m2=)` keeps a film per facet. In each step the film changes by
  `(P − E(T_half)) dt`, clamped at zero, with `E` at the midpoint temperature the balance used;
  a cell with water is wet (`wet = max(declared, film > 0)`). `ThermalField` and
  `PlanarThermalField` pass the film through and expose it.
* `SceneSurfaceForcing` and `CellForcing` supply `q_a`, `g_e` and the rain rate from the
  scene's one weather series on every call, so declaring a film (PH.2) is all a scene needs.
* Saturation uses the same Magnus/Bolton fit `irsim.atmosphere.humidity` uses, vectorised;
  `q = ε e / (p − (1 − ε) e)` at standard pressure. Constants with sources in
  `irsim.radiometry.constants`: `L_V_WATER_J_KG` (2.45 MJ/kg at 20 °C), `C_P_AIR_J_KGK`,
  `RHO_AIR_STD_KG_M3`, `P_STD_HPA`, `EPSILON_WATER_AIR`, `C_E_BULK` (1.15e-3, ESTIMATED as a
  constant where COARE has a function).
* The sea's cool skin now conducts the net longwave **plus** the latent flux from the bulk
  SST; the sea got colder, which is the correction the sea-skin module's own docstring asked for.
* Dew: over a wet cell the term goes negative when the air is wetter than the surface is warm,
  and the film grows. Dew onto a *dry* surface is not modelled (a dry cell evaluates no term).

## Consequences

**Easy now:** PH.2 puts a 0.2 mm film on half a road patch from the scene config; PH.3's still
water and PH.11's vegetation take `r_s`; snow (PH.10) is the same bookkeeping with the latent
heat of fusion.

**Error introduced:** a constant Dalton number where COARE's varies with stability and
gustiness (±15 %, the tolerance PH.1's test carries); the wind floor of 0.5 m/s standing in for
free convection; the film as a sheet of uniform depth (no ponding, no runoff, no soil uptake);
saturation at the surface assumed over the whole wet fraction.

**Measured:** a saturated surface with no radiation relaxes to the psychrometric wet bulb of
(T_air, RH) within 0.1 K at RH 1.0, 0.7 and 0.4 and cools as RH falls; the flux is exactly zero
for saturated air at the surface's temperature; a dry cell under moist weather is bit-identical
to the plain balance over 200 ticks, with or without an empty film; the bulk flux at 5 m/s,
293 K, RH 0.7 is 75 W m⁻², within 15 % of COARE 3.6's 76; the film budget closes to 1e-6
against the solver's own evaporation; the sea skin runs colder with the term than without.

## Revisit when

A validation needs COARE's stability dependence (replace `C_E_BULK` with a function of the
bulk Richardson number), dew on dry surfaces is wanted for a night scene (the sign already
works; it needs a film to land in), or a soil model needs the film to infiltrate.
