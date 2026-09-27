# ADR 0153 — A reflective band sees the sun off the ground and in the cloud

**Status:** Accepted
**Date:** 2026-09-27
Roadmap: AT.20. Spec: `docs/physics-model.md` §5.3 ("In a reflective band the ground and the cloud
are lit"), spec issue S55. Extends [ADR 0086](0086-scattered-sunlight-sky.md) (the isotropic
skylight) and [ADR 0045](0045-reflected-environment-term.md) (the environment term). Mirrors the
visible dome's cloud base and terrain ([ADR 0073](0073-visible-companion-environment-dome.md)).

## Context

The four-band Phantom 4 render exposed two gaps in NIR and SWIR.

1. **The underside was black.** L_env = V_s L_sky,eff + (1 − V_s) L_B(T_ground). The ground half
   is thermal emission only, ~1e-15 of the sunlit value in NIR. So a downward-facing surface
   reflected nothing, and the white ABS shell (ρ_SWIR 0.78) rendered black beside a white RGB
   companion.
2. **The sky had no clouds.** ADR 0086's skylight is added to the clear column, and the deck blend
   swaps only the column's *thermal* emission for the cloud's. Both are ~0 in SWIR, so a cloud
   pixel read the clear sky's value to 1.0000 of the median.

## Options considered

1. **Per-band albedo authored in each environment preset.** Simple, but it duplicates numbers the
   material library already holds, and a new band means editing every preset.
2. **The ground names a library material, and its albedo is that material's Kirchhoff-closed band
   reflectance** (chosen). Bands stay data (CLAUDE.md), closure is guaranteed (#4), and a surface of
   the same material elsewhere in the scene reflects the same number.
3. **A diffuse-transmission cloud model** (T_total − e^(−τ/μ0) from the two-stream). This is more
   correct from below: thick cores come out darker than their edges. But it disagrees with the
   visible companion, which draws the base as R(τ, μ0) × GHI/π. The owner checks each IR frame
   against its RGB pair and treats disagreement as a defect, so the bands must share one model
   until both move together.

## Decision

- `environment.ground.material` (optional) names the ground's library material. `SkyModel` takes
  `site=` and `ground_albedo=`, and `Scene.from_config` supplies them for every band that has a
  skylight. ρ_B comes from `load_material(...).band_properties(band).reflectance`.
- `DiffuseSkylight` gains `per_dni`: the band's share of the ground-level direct spectrum
  (unweighted), beside ADR 0086's Rayleigh-weighted `per_dhi`. The band's horizontal irradiance
  is `E_B = per_dni · DNI · sin h + π · per_dhi · DHI`, using the weather's own DNI and DHI and the
  site's sun (`solar_terms_at`, the function every other solar term uses).
- **Ground:** L_ground = L_B(T_ground) + ρ_B E_B/π.
- **Cloud:** the cloud's source radiance gains R(τ_col, μ0) E_B/π, with the same `cloud_reflectance`
  the dome uses. A marched deck uses each ray's own depth brought to the vertical (τ_ray sin el); the
  uniform blend uses the preset's `optical_depth`, or `CLOUD_BASE_ALBEDO` = 0.55 (the dome's value,
  now one constant) for a `tau` preset.
- Every term needs the skylight, and an emissive band has none (`skylight_for_sensor` returns
  `None`), so **LWIR is bit-identical**. MWIR (`mixed`) gains the terms, as it physically should.
- The shipped presets: `clear_dry` and `overcast` use `soil_dry`, `humid` and `scattered_cumulus`
  use `vegetation_leaf`, and `sea_clear_day` uses `water`.

## Consequences

Measured at 45 N at midsummer noon, DNI 650 / DHI 140 W m⁻², the `clear_dry` ground
(`test_sunlit_environment.py`, 11 cases):

- **V_s = 0 environment:** NIR 7.8e-16 → **8.55** W m⁻² sr⁻¹, SWIR 9.2e-8 → **13.4**. The
  underside now reads 0.99× (NIR) and 0.91× (SWIR) of the topside instead of zero. A 0.78 panel reads
  ρ · albedo · E_B/π against a numpy-only oracle, well inside the 5 % the roadmap asked for.
- **Deck clouds against clear sky, median:** NIR 1.00 → **6.7×**, SWIR 1.00 → **13.2×**. A thin
  edge is dimmer than a deep core, which is the dome's texture.
- A real night (sun down, DHI 0) is exactly 0.0. Below the horizon the beam is cut and the
  weather's own DHI is kept, because twilight is real and the weather is the authority on it.

Known and flagged:

- R ignores droplet absorption, so a thick SWIR cloud is somewhat too bright (single-scattering
  albedo ~0.99 near 1.6 µm).
- A base drawn as a reflector of R is the dome's convention, not diffuse transmission (option 3).
- τ_ray sin el underestimates the column where a ray leaves a tower through its side.
- The ground is Lambertian, flat and unshadowed, with no terrain geometry. A sea ground is a
  diffuse reflector here and its glint is not modelled.

## Revisit when

The visible dome moves to diffuse transmission (option 3), after which both bands move together;
a measured SWIR cloud reflectance or single-scattering albedo is available; or the ground gains
geometry (`AT.8`, a second-hit ray).
