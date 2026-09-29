# Clouds in the infrared: how a visible cloud volume becomes an infrared one

Research note for the aerial lane, written 2026-09-29 after the owner found that the infrared
band never occludes a target with cloud. It records what is wrong today, what the literature
says the physical conversion is, and what the engine can and cannot supply. The roadmap steps
that act on it are `AT.14`, `AT.29` and `AT.30`.

## What is wrong today

The visible companion draws `isaac-weather-fx`'s cloud field as a path-traced OpenVDB volume
(ADR 0144). That volume writes **no depth and no instance id**, so the G-buffer's `sky_mask`
cannot tell a cloud pixel from a clear one. The infrared band therefore marches the *same*
density array itself, but only on pixels that hit no geometry
(`aerial_bridge.sky_temperature_field`). A pixel that hit a drone gets the drone's radiance
through clear air (`apply_layered_gbuffer`), whatever cloud lies between the camera and the
drone or around it. Two consequences:

1. **A target behind or inside a cloud is never attenuated by it** and never has cloud
   emission in front of it, while the visible frame does occlude it. The owner's rule that the
   two bands may not disagree (ADR 0073) is broken exactly where a detector would be tested.
2. **Any label plane calls a cloud pixel unknown or background**, because nothing in the frame
   knows a cloud was crossed. The infrared march has the optical depth of every ray and throws
   it away after turning it into a temperature.

The wind drift and the camera-following tiles that upstream added after the pinned commit are
not applied on the infrared side either: `WeatherFxDeck.march` samples the field at the camera
as origin with no drift, so once the submodule is bumped the two bands would place the cloud
differently. `AT.30` covers that.

## The physics: absorption, emission, and a finite range

The right along-ray model is the Schwarzschild emission-absorption integral, the same equation
`docs/physics-model.md` §6 already uses for the clear atmosphere, applied to the cloud's own
absorption coefficient and local temperature (Petty, *A First Course in Atmospheric Radiation*,
ch. 8; Liou 2002 §1.4):

```
L(0) = L_hit · exp(−τ_c(R)) + ∫₀^R β_a(s) L_B(T(s)) exp(−τ_c(s)) ds
```

`L_hit` is the surface (or clear-sky) radiance at range `R`, `β_a` the cloud's infrared
*absorption* coefficient per metre, `τ_c(s)` the cloud absorption depth from the camera to `s`.
Discretised per sample of width `Δs`, marched from the far end toward the camera:

```
L ← L · exp(−β_a Δs) + L_B(T_i) · (1 − exp(−β_a Δs))
```

This is what DIRSIG's `CloudVDB` plugin solves (a density grid plus a temperature grid; emitted,
scattered and transmitted radiance along the same ray that finds the surface hit), and it is the
published precedent closest to this project's situation
(<https://dirsig.cis.rit.edu/docs/new/vdb_plugins.html>). MODTRAN does the same in layers with a
finite path termination (Berk et al., MODTRAN4 User's Manual, 2003, §8 and Table 7).

### Visible extinction to infrared absorption

The cloud field's density is authored in visible extinction per metre. The literature gives the
conversion in three steps:

* **Visible optical depth from liquid water path.** τ_vis = 3 LWP / (2 ρ_w r_e) with Q_ext ≈ 2
  (Brenguier et al., *ACP* 11, 2011, eq. 1–3). At r_e = 10 µm that is 0.15 m²/g.
* **LWIR absorption is about half the visible extinction.** The large-particle limit is exactly
  Q_ext,vis / Q_abs,IR = 2 (Platt et al. 1980; Minnis et al. 1990 measured 2.13; DeSlover et
  al., *JAOT* 16, 1999, found 2–3 by cloud type). Shaw & Nugent, *Eur. J. Phys.* 34 (2013)
  S111, use "LWIR OD = half the visible OD" and ε = 1 − exp(−0.79 τ_vis) for a flux; AT.11
  already carries that pair (`CLOUD_OD_RATIO = 0.5`, diffusivity 1.58). For small droplets the
  extinction ratio is nearer 0.74 (r_w = 8 µm, 10.2–12.5 µm band; Chou, Lee, Tsay & Fu,
  *J. Climate* 12, 1999, §4) with single-scattering albedo ω ≈ 0.5 in the window, so the
  *absorption* ratio lands back near 0.4–0.5. A pencil-beam ray takes **no** diffusivity factor.
* **The same thing as a mass coefficient.** Stephens (*J. Atmos. Sci.* 35, 1978) chose 0.130
  (down) and 0.158 (up) m²/g in the flux form ε = 1 − exp(−1.66 k LWP); Platt 1976, Bonnel et
  al. 1980 and Schmetz et al. 1981 measured 0.11–0.15 m²/g (Stephens, AT622 notes §16, Table
  16.1). Dividing out the diffusivity gives 0.07–0.08 m²/g for a ray, again half of the
  0.15 m²/g visible value at r_e = 10 µm. A cloud with LWP ≥ 100 g/m² is black in LWIR (Shaw &
  Fedor 1993); a 0.3 g/m³ stratus is opaque within a few tens of metres.

So the conversion irsim already applies to the sky pixels, `τ_IR = 0.5 τ_vis`, is the
literature's number for LWIR water cloud, and the step is not to change it but to **apply it
along the finite ray to every pixel**. What should change:

* **Scattering, cheaply.** Dropping scattering entirely overstates the emission of thin cloud;
  Chou et al. 1999 keep the non-scattering solver and scale the depth, dτ̃ = (1 − ω f) dτ with
  f the forward-scattered fraction, holding the error under 2 % of the flux (Hogan & Bozzo,
  *JAMES* 10, 2018 §4.1 quote the 10 % effect of neglecting it; Jin et al., *JQSRT* 2019, put
  it in the 800–1250 cm⁻¹ window at τ ≈ 3). The ratio 0.5 is that scaling for large droplets;
  it stays a per-band constant, data not code.
* **Local temperature along the ray, not one base temperature.** ADR 0146 reads the base
  temperature lapsed to a transmittance-weighted emission height. For a target *inside* a
  kilometre of cumulus the emitting skin nearest the camera can sit several hundred metres above
  the base, and the moist-adiabatic rate inside cloud is 5–6 K/km (ICAO range 3.6–9.2), so each
  sample should carry `T(z) = T_base − Γ_m (z − z_base)` and the march should accumulate
  `L_B(T_i)` per sample rather than one height. The existing march already visits every sample;
  it only has to look the temperature up there.
* **Ice cloud is a different coefficient set.** Hong et al., *J. Climate* 22, 2009, Tables A1–A2
  give mass absorption and extinction as polynomials in 1/D_e per band (8–12.5 µm absorption at
  D_e = 50 µm ≈ 0.037 m²/g, ω ≈ 0.47). Not needed while the field is cumulus; the ratio must be
  a per-genus, per-band datum so it can be.
* **MWIR is not settled.** No accessible source gives a 3–5 µm water-cloud absorption number.
  Water's imaginary index is low there, so the cloud scatters more than it absorbs and daytime
  sunlight scattered off cloud is a first-order term; treat MWIR transmittance as visible
  extinction (Q_ext ≈ 2) with an emissivity ratio well below LWIR's and mark it ESTIMATED.

## What the engine can supply

Searched: Omniverse RTX path-tracer docs, Replicator annotator lists, Isaac Sim 6.0 RTX sensor
docs and the NVIDIA forums.

* The only volume output RTX documents is the **`Volumes` AOV** (`/rtx/pathtracing/volumesAOV`,
  Replicator `PtVolumes`): the volume's *shading*, a radiance pass. There is no documented
  volume transmittance, volume depth or volume hit output, and the surface annotators
  (`distance_to_camera`, `instance_id_segmentation`) are documented as surface-only.
* RTX sensors (lidar, radar) have "no physical simulation connection" to rendered volumes
  (NVIDIA staff, forum thread 328503, April 2025); their atmospheric model is a parametric rain
  attenuation, not the scene's VDB.

So the engine gives the **surface hit distance** and nothing about the cloud along the way. The
per-pixel cloud transmittance to that hit has to come from marching the shared density array
ourselves, terminated at the G-buffer depth, which is what the sky-only march already does for
`R = ∞`. AT.14's original wording ("the renderer's own integration") is therefore not
achievable on this engine and the step is rewritten.

## The plan

1. **`AT.14` — march to the hit.** `WeatherFxDeck.march` takes a per-ray `range_m` (the
   G-buffer distance; `inf` for sky) and returns, beside the optical depth, the cloud's
   transmittance and its path radiance for the band, accumulating `L_B(T_i)` per sample on the
   moist adiabat. `apply_layered_gbuffer` composes `τ_c · (τ_air L_hit + L_path,air) + L_path,c`
   for every pixel with the camera's world position and an `azimuth_rad` plane beside
   `elevation_rad`. The sky pixels reproduce ADR 0146's numbers to the tolerance the ADR states.
   Test: a target behind an opaque cloud reads as the cloud, in front of it unchanged, inside it
   between the two.
2. **`AT.29` — the frame knows its clouds.** Write `cloud_optical_depth` and `cloud_transmittance`
   planes, expose them in the viewer, and let the label writer mark a pixel `cloud` when the
   transmittance to its hit falls below a threshold. This is what makes "unknown" go away.
3. **`AT.30` — bump the submodule and align the field's frame.** Upstream now drifts the volume
   with the wind and re-tiles it around the camera; the infrared march must apply the same
   `volume_offset_m` so the two bands keep one cloud. The shim context gains `cloud_drift_m`,
   `anchor` and `sky_horizon_rgb`; two bridge tests are re-pinned to the smoothed field.
