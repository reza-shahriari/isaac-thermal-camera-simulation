# 0162 — The cloud is marched to the hit, for every pixel

Date: 2026-09-29
**Status:** Accepted
Roadmap: AT.14 (rewritten); amends ADR 0146 (the sky march's dropout); builds on ADR 0144 (the
volume renders, and writes no depth) and ADR 0126 (the cloud at a range).
Sources: `docs/clouds-in-the-infrared.md`.

## Context

The owner, working on the weather-fx extension, found that "the clouds are not being detected
with rays at all", and that "if we read the clouds, we may lose object in between the clouds".
Both are true of the code as it stood:

* The visible companion draws the shared `CloudField` as a path-traced OpenVDB volume (ADR 0144),
  and that volume writes **no depth and no instance id**. The G-buffer's `sky_mask` is therefore
  "hit no surface", never "looked through cloud".
* The infrared band marches the same density array itself — but only where `sky_mask` is set
  (`AerialThermalBridge.sky_temperature_field`). A pixel that hit a drone was composed as the
  drone through *clear* air by `apply_layered_gbuffer`, whatever cloud lay in front of it or
  around it. A target behind an opaque cumulus was drawn in full; the visible frame hid it.
* Nothing in the frame recorded that a cloud had been crossed, so any label or viewer plane
  called a cloud pixel unknown or background.

The original AT.14 wanted "the renderer's own integration". Searched on 2026-09-29: Omniverse
RTX documents exactly one volume output, the `Volumes` shading AOV (`PtVolumes`), and no volume
transmittance, depth or id; NVIDIA state that RTX sensors have no physical connection to
rendered volumes. The engine gives the surface hit's range and nothing about the cloud along
the way. That decides the design: the per-pixel cloud transmittance to the hit has to come from
marching the shared array ourselves, ended at the G-buffer range — which is what the sky-only
march already did for a range of infinity. DIRSIG's CloudVDB plugin (density and temperature
grids, emitted and transmitted radiance solved along the same ray that finds the surface) is
the published precedent.

## Decision

1. **A finite-range march.** `WeatherFxDeck.march_to(el, az, range_m, radiance_at_height)`
   integrates the field from the camera to each ray's own range (`inf` for a sky pixel) with
   the same stratified, hashed quadrature as ADR 0146's `march`, and carries the band along:
   each sample absorbs `1 − e^{−od_ratio σ Δs}` of what lies behind it and emits that fraction
   of `radiance_at_height(h)` toward the camera through the cloud already crossed. That is the
   Schwarzschild emission–absorption integral for a non-scattering medium, discretised per
   sample. It answers `OccludedMarch(optical_depth, transmittance, path_radiance,
   emission_range_m)`.
2. **The same temperature law as the sky pixels.** `SkyModel.cloud_occlusion` supplies the
   emitter: the deck's own base, lapsed dry-adiabatically to it (ADR 0146) and at the preset's
   lapse inside the cloud, looked up per sample at the sample's height; a reflective band adds
   the preset's cloud shine. Sky and geometry pixels cannot carry two clouds.
3. **Stage 2 composes it.** The camera hands three planes beside the G-buffer —
   `cloud_transmittance`, `cloud_radiance`, `cloud_range_m` — outside the M0.6 contract, the way
   `radiance_behind` travels. `apply_layered_gbuffer` and `apply_atmosphere_gbuffer` then apply
   `L = τ_c (τ_air(R) L_hit + L_path,air(R)) + τ_air(R_c) L_cloud` to every non-sky pixel, the
   cloud's emission attenuated by the air over the range it came from. Sky pixels pass through:
   their cloud is in the temperature the bridge marched.
4. **The visible-to-infrared conversion is unchanged, and now cited.** `τ_IR = 0.5 τ_vis` is
   the literature's LWIR water-cloud number (Platt 1980, Minnis 1990, DeSlover 1999; Shaw &
   Nugent 2013; Stephens 1978's 0.13 m²/g ÷ 1.66 diffusivity), applied per ray without a
   diffusivity factor. It remains per-band, per-genus data (`od_ratio`), so ice (Hong et al.
   2009) and MWIR (no citable number; ESTIMATED) can differ without touching the march.
5. **Both marches drop a ray at the band's opacity.** `march` stopped when the *visible*
   transmittance fell below e⁻¹², i.e. at τ_vis = 12, which capped an opaque cloud's LWIR
   emissivity at `1 − e⁻⁶ = 0.9975` — a quarter of a percent of the −40 °C sky leaking through
   every opaque cumulus. It now stops at `od_ratio · τ_vis = 12`, and at infinite range the two
   marches agree to 1e-9 (a test says so).

## What was measured

* A hit 50 m short of a 600 m base: transmittance 1, radiance 0, range 0 (the target through
  clear air, whatever the sky behind it holds).
* Infinite range against the sky march, 432 rays, cover 0.6: transmittance and `1 − ε` agree
  to 8e-16 where the cloud is not opaque and to 1e-9 everywhere after decision 5.
* An isothermal cloud closes the integral, `L_path = (1 − τ_c) L_B`, to 1e-9.
* Through the sky model each ray's emission lies between the black base and the black top of
  the cloud, scaled by its own emissivity.
* Stage 2: a 40 W/m²/sr target at 2 km behind an opaque cloud whose emission is centred 300 m
  away reads as `9 W/m²/sr × τ_air(300 m)`, not as the target; τ_c = 1 with nothing emitted is
  bit-for-bit the clear result; a half-transparent cloud lands between the two; the sky pixel
  beside them is untouched; the grey Beer–Lambert path composes the same way.
* In the engine (A6000, `outputs/phantom4_cloud_occlusion`, on the site): a Phantom 4 at 53 to
  121 m through a stratus fractus layer (base 15 m, 80 m thick, 60 % cover) has parts solved at
  14 to 18 °C on the first frame and motors at 36 to 38 °C by the eighth, and *reads* 4 °C at
  59 m and −0.5 °C at 97 m; the layer itself reads −12 to −2 °C. The target is 6 to 14 pixels
  across; a large target behind a cumulus at its natural base needs an aircraft-sized asset.
* `tests/unit/test_cloud_occlusion.py`, 8 cases; the ADR 0146 bridge and base-law tests still
  pass with the new dropout.

## Consequences

* A resolved target behind or inside a weather-fx cloud is attenuated by it and has the
  cloud's emission in front of it, in every band and in both atmosphere models. The march
  runs a second time per frame, over the geometry pixels at the sky's stride.
* **Not yet covered:** the analytic point-target path (`inject_point_targets`) runs after
  stage 2 with its own atmosphere and does not read the cloud planes, so a sub-pixel target
  behind cloud is still unoccluded; the native `CloudDeck` has no `march_to` and answers
  `None`; the camera's own altitude and the wind drift of the volumes are not applied to the
  march (AT.30). A label plane still cannot name a cloud pixel until the three planes are
  written to disk (AT.29).
* The pinned weather-fx commit predates the path-traced volumes, so until AT.30 the visible
  companion draws the cloud on a dome and does not occlude the target the infrared band now
  occludes — the disagreement is now the other way round, and is the reason AT.30 is phase A.

## Alternatives rejected

**A second render product with an emissive infrared volume material.** The `PtVolumes` AOV
would give the cloud's *emission* in the band, never its transmittance to the surface behind
it, and only under the path tracer. It cannot answer the question the owner asked.

**Waiting for a volume-depth AOV.** None is documented, and the RTX sensor team's statement
is that volumes are render effects with no sensor coupling. A march of an array we already
hold costs seconds a frame and is testable without the engine.
