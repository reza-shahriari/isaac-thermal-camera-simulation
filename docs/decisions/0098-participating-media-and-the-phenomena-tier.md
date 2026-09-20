# ADR 0098 — Participating media: a per-band slab in radiance space, and the phenomena tier

**Status:** Accepted
**Date:** 2026-09-20
Roadmap: PH.13, recording PH.4 (§2, §8.1, §8.3; spec issue S46)

## Context

docs/physics-model.md §2 is a surface equation: emitted, reflected and path terms for an opaque
facet. A flame, an exhaust plume, steam or any hot gas emits and absorbs **along the ray**, and
until PH.4 nothing in the code could say so. The owner's 2026-09-18 requirement names fire and
"something like these" beside water; §6.6's own plume row was deferred in revision 5 behind a
trigger (an MWIR Tier 3 bench) that had already fired. A scene author who needed a flame would
have painted it as a hot prim, and got a grey emitter that is wrong in every band.

The survey's anchors set both the operator and its fidelity:

* FDS (NIST) renders flame radiation with a gray-gas radiative transfer equation whose absorption
  coefficient comes from RadCal, "because in real fires, soot is the dominant source and sink of
  thermal radiation and is not particularly sensitive to wavelength"; its eq. C.3, read per band,
  is `L_b = τ_b L_behind + (1 − τ_b) B_b(T_g)`.
* RadCal's stated accuracy against line-by-line models is about **8 %** for 20–200 cm paths at
  800–1800 K. Nothing built on band coefficients can claim better.
* HITRAN is established at 296 K and "usually deficient" for hot gas; HITEMP-2010 is the
  line list for flame and plume work. Hot-gas coefficients therefore cannot be scaled from the
  ambient atmosphere model; they are a separate offline product.
* NIRATAM's ship-plume fits: MWIR transmission 65 % near the exit rising to 100 %, LWIR ≥ 0.90 —
  the same plume is a dominant MWIR feature and a few-K LWIR one. A camera-band simulator has
  to reproduce that split or it has no plume.
* DIRSIG and ThermoAnalytics both keep **transport and radiometry apart**: the plume's
  temperature and species field comes from a flow model (Blackadar puffs, CFD voxels) and the
  renderer only integrates emission and absorption through it. Neither simulates buoyancy inside
  the renderer.

## Options considered

1. **A grey emissivity knob on a hot prim.** One number for all bands. It cannot produce the
   MWIR/LWIR split above (PH.4's test: a CO₂/H₂O slab reads more than 900 K apart between the two
   cameras; a grey ε reads within 150 K), so it renders the plume's most visible property away.
   Rejected for the image path.
2. **Hottel / Leckner total emissivity charts** `ε_g(T, pL)`. Integrated over all wavelengths, so
   again one number per gas, not one per band. Rejected for the image path; kept as the right
   tool for **heating** — the radiative flux a flame puts on a nearby surface is a total, and
   PH.7's surface emissive power is that quantity.
3. **A Planck-mean absorption coefficient in the camera band.** Correct for the total loss of an
   optically thin gas (the TNF workshop model) and wrong for a band camera, which weights by its
   own R(λ), not by the whole Planck curve. Rejected; the coefficient must be band-integrated
   against R(λ), which is why the tables are per band and bands stay data (AT.4).
4. **An emissive dome or volume prim on the Isaac side**, through the renderer's own emission.
   Every colour AOV on this build is fp16 and exposure-scaled (ADR 0014); a 1300 K source through
   it would lose the radiometry the whole project exists to keep. Rejected.
5. **Per-band slab in radiance space, in the engine-free pipeline.** FDS's operator per band, with
   band-mean coefficients: soot analytic from the Rayleigh form over the camera's own R(λ), gas
   species from offline HITEMP/RadCal tables, applied at the target's range and attenuated by
   the rest of the path like MS.6's point target. Chosen.

## Decision

Option 5, shipped as `irsim.pipeline.gas_slab` (PH.4):

* A slab is authored as **(T_gas, L, p_CO₂, p_H₂O, f_soot)** and never as an emissivity; the
  band emissivity `1 − τ_b` is an output that differs per band, which is the phenomenon.
* `κ_b = κ_CO₂,b(T) p_CO₂ + κ_H₂O,b(T) p_H₂O + κ_soot,b(T)`, with the soot term
  `C0 f_v ⟨1/λ⟩_b` weighted by R(λ) B(λ, T_g) (`SOOT_RAYLEIGH_C0` = 7.0, Widmann 2003) and the gas
  terms from `SpeciesAbsorption` tables in 1/(m·atm) that refuse to extrapolate. A species with
  no table raises naming PH.5; no ambient coefficient is ever scaled up.
* `B_b(T_g)` by quadrature, not the 200–1000 K LUT; T_gas guarded to 300–2500 K.
* At range: `ΔL = Σ_k w_k τ_k(R) (1 − τ_b)(B_b − L_behind,k)` per class, the point target's
  convention, so a plume and a point target on the same ray attenuate identically.
* **The phenomena tier.** This feature claims the RadCal envelope, ~8 % in radiance, not the
  10 mK the surface path claims. It exists so a fire scene *can be generated* and shows the right
  band behaviour; it is not a calibration path. Tests hold it to identities (limits, range,
  band split, sign), not to a radiometric tolerance it cannot meet.
* **Transport stays outside.** The slab takes T_g, species and path length as inputs; where a
  plume's field comes from (an exhaust node, an authored cone, a puff model) is PH.6's and
  PH.7's business, and buoyancy is not simulated.

## Consequences

**Easy now:** PH.5 fills the tables; PH.6 places a slab per pixel over a plume cone with T and
species from the exhaust node; PH.7 places a soot slab at 1150–1300 K over a fire and reuses
`RadiantRectangle` with an authored surface emissive power for what the fire heats; PH.8 feeds a
1273 K patch into the ISP to test gain state and AGC.

**Error introduced, bounded where it can be:** a band-mean coefficient inside one exponential is
exact in the optically thin and opaque limits and approximate between them — the true band
transmittance is the R·B-weighted mean of `exp(−κ_λ L)`, which a narrow-band code evaluates
and this does not. For soot (κ_λ ∝ 1/λ, smooth) the error is small; for the CO₂ 4.3 µm band
head it is the band-model error RadCal's 8 % already includes. Recorded, not tuned away.

**Deferred, deliberately:** scattering (smoke and steam droplets, which is why LWIR sees through
what MWIR does not); a temperature gradient along the ray; buoyancy and plume dynamics; flicker
(a temporal model of the flame's radiance); a volume on the Isaac side. Each is a separate
decision when a scene needs it, and none changes the operator above.

## Revisit when

A validation needs the band transmittance of a structured band (CO₂ 4.3 µm) to better than
RadCal's envelope, which means a narrow-band or correlated-k integration inside the pipeline
rather than a band-mean κ; or scattering becomes the dominant term for a smoke scene, which is
a two-stream or Monte Carlo addition the slab's interface (T, species, L) already accommodates.
