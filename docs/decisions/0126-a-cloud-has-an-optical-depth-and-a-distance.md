# ADR 0126 — A cloud has an optical depth and a distance

**Status:** Accepted. The environment preset stops authoring a cloud's transmittance and authors
its **visible optical depth** instead; the emissivity is derived from it per ray, and the cloud is
placed at the lifting condensation level rather than at zero range.
**Date:** 2026-09-23
Roadmap: AT.11 (§5.3 a)

## Context

ADR 0125 gave a cloud a soft edge and it was still wrong. The owner's words, on the rendered
clip: *"the IR is too noisy, always white (which is not in real data)"*. Measured on
`outputs/phantom3_outbound/frame_000120`, a 640×512 LWIR frame of a Phantom 3 against scattered
cumulus:

* **53.9 % of the frame** sat within 0.25 K of one apparent temperature, 14.55 °C.
* Inside that region the *true* spread was **0.073 K** — the sensor's own NETD and nothing else.
* The camera's plateau-equalising AGC, with DDE at 0.35, rendered those 73 mK as display codes
  **106 to 255**, a standard deviation of 38.8 out of 255.

So both halves of the complaint had the same cause. The cloud was "always white" because every
full-depth ray carried *one identical radiance*; and it was "too noisy" because an AGC handed a
frame with no structure in it will find some, and what it finds is noise.

Why one identical radiance: `CloudSpec` authored a transmittance `tau`, every shipped preset
authored `tau: 0.0`, so `ε = 1 − τ = 1` everywhere, and `SkyModel.radiance_field` evaluated
`L_B(T_base)` as though the cloud were at the sensor. A cloud is not at the sensor. It is a
kilometre away and the air in front of it is warmer than it is.

Public anchor, since there is no cloud section in `docs/physics-model.md` (§5.3(a) says only that
T_sky → T_air under overcast) and no IR camera to measure with: Shaw & Nugent, *Physics
principles in radiometric infrared imaging of clouds in the atmosphere*, Eur. J. Phys. **34**
(2013) S111, a calibrated ground-based 8–14 µm all-sky study. Three of its results bear directly:

1. LWIR emissivity of a cloud is `1 − exp(−0.79 τ)` in τ, the **visible** optical depth, and the
   LWIR optical depth is about half the visible one. Cloud depth can be retrieved up to τ ≈ 4;
   beyond that a cloud emits as a blackbody.
2. Cloud temperature is the air temperature at the *base altitude*: they model cirrus at 10 km at
   216.8 K and an opaque altostratus at 2.4 km at 271.6 K.
3. In a raw calibrated sky image *"the overall spatial pattern … is dominated by the variation of
   atmospheric path length with angle from the zenith"* — the clouds are a residual on top of the
   limb gradient, not the dominant feature. The clean cloud maps in that paper's later figures are
   a *processed* product, obtained by subtracting a modelled clear sky.

## Decision

**1. A preset authors `clouds.optical_depth`, the visible OD at full cloud depth.** `clouds.tau`
stays, so every scene authored before this ADR is bit-identical, but the two are mutually
exclusive: authoring both is refused rather than silently resolved, because they are two answers
to one question. `CloudSpec.emissivity` *raises* under an `optical_depth` preset rather than
returning `1 − 0 = 1` and letting a caller render a thin cirrus as an opaque deck — the same shape
as `Scene.atmosphere` after AT.5.

**2. The emissivity is derived, and it is derived once for two cases.**
`cloud_emissivity(τ_vis, path_factor) = 1 − exp(−path_factor · 0.5 · τ_vis)`. With the default
`path_factor = DIFFUSIVITY_FACTOR = 1.58` this *is* the published `1 − exp(−0.79 τ)`, because
0.79 = 1.58 × 0.5 exactly. That decomposition is the whole design: the published number is quoted
for a hemispherically integrated **flux**, so a **ray** substitutes its own airmass `1/sin θ`
(`cloud_airmass`) and one relation serves the tilt-integrated sky and the image.

**3. The cloud sits at the LCL and the air in front of it is modelled.**

```
L(θ) = L_clear(θ) + τ(R, θ) · ε · [L_B(T_base) − L_beyond(R, θ)],   R = z_base / sin θ
```

exact rather than a blend: inserting an emitter of emissivity ε at range R gives
`L_path(R) + τ(R)[ε L_base + (1−ε) L_beyond]`, and `L_clear = L_path(R) + τ(R) L_beyond` holds
**identically** for the layered model, because its band transmittance and its `sky_beyond` are
weighted with the same spectral-class weights. So ε = 0 returns the clear sky to the bit and ε = 1
is exact too; the k-distribution approximation only enters in between.

**4. The clip gains a third display span, `sky`.** Neither existing span reaches the sky: `ir`
starts at the coolest airframe node and `skin` is tighter still, so cloud and clear zenith both
land on display code 0 and the only picture carrying the sky was the camera's own AGC — the one
measured above. The `sky` span is linear over the scene's own sky-to-target range, 63 K across
256 codes at 0.25 K each, five times the sensor's NETD. The AGC video is still written beside it.

## Consequences

Measured, on the Phantom scene's own frame geometry (24.75° vertical field aimed 20° up) with the
same cloud field and seed, the two models differing only in the preset:

| | before | after |
|---|---|---|
| apparent T across the cloud **core** (full-depth rays) | 0.000 K | **1.25 K** |
| transmittance to the base, 32° → 8° elevation | not modelled | 0.68 → 0.49 |
| cloud apparent T, high in frame → low in frame | one value | 16.3 → 17.4 °C |
| clear-sky limb gradient across the frame | −27.3 → −1.5 °C | unchanged (it was always right) |
| display: 73 mK of NETD inside the core | 149 display codes | 0.3 of one code |

The cloud base comes out at 1071 m and 14.6 °C on this scene's weather. As a public check, the
paper's modelled altostratus base at 2.4 km reads 271.6 K and this model's constant environmental
lapse rate gives 272.55 K from the standard atmosphere's surface — within a kelvin.

**What is still wrong, and named rather than hidden.**

* **A plane-parallel deck has no sides.** The emissivity now varies along the ray and the range
  term varies across the frame, but an optically thick cumulus core still reads one level to
  within a kelvin — and at an *oblique* aim like this clip's 20° that is not what a real cumulus
  field looks like, because a real one is a field of towers and the camera sees their flanks and
  the gaps between them. That structure needs a cloud with a third dimension, which is AT.12's
  deck, and the emission level along a ray through it, which is AT.13/AT.14.
* **High cloud needs an authored base.** The LCL is the cumulus base and is right for it. A
  constant lapse rate to 10 km gives 223.15 K against the paper's 216.8 K, 6.3 K out, because the
  real profile bends toward the tropopause. No scene here authors cirrus; one that did would need
  a base height and a profile, not this.
* **The flux path does not carry the range.** `radiance` and `effective_radiance` take the
  diffusivity-weighted emissivity, which is correct for them, but no transmittance to the base —
  a tilt integral has no single ray to take one along. The reflected-environment term therefore
  still sees a cloud at zero range. Small next to the change above, and stated here so it is not
  mistaken for the image path.
* **The optical depth is ESTIMATED.** 12 for fair-weather cumulus sits mid-range in the published
  5–20; past τ ≈ 4 the value is unrecoverable from a radiance measurement anyway, so it only
  bites near the fringe.

## Alternatives rejected

**Keep `tau` and just author a non-zero value.** It would lift the cloud off the blackbody level
but keep every covered pixel identical, which is the defect. An opacity that does not vary with
the ray is not an optical depth.

**Fix the AGC instead.** The speckle is the AGC amplifying 73 mK because nothing else was there;
that is what a plateau-equalising AGC with DDE is *for*, and a real Boson does it. Detuning it to
hide a flat scene would have made the rendered camera less like the real one to cover for the
scene model. The fixed span is an operator action a real user takes, and the AGC output is still
filmed beside it.

**Ray-march a 3-D cloud now.** The right answer for the oblique aim, and it is AT.12: it needs a
deck with a top, which is also the object the visible band's VDB volume is built from. Shipping it
inside this step would have merged the radiometry fix with the geometry one, and the radiometry
fix stands on its own — it is what makes a gap in the cloud read the clear sky exactly.
