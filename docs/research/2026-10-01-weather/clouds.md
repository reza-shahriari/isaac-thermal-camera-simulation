# Volumetric clouds research note (2026-10-01)

Scope: technique, morphology statistics, Isaac Sim/RTX volume capability, public data. Items marked UNVERIFIED were not confirmed from a primary source. All four sections are now sourced.

## 1. Production technique for detailed volumetric clouds

Primary sources read in full text for this section: the HZD 2015 slide notes [R1], Nubis 2017 [R2],
Nubis Evolved 2022 [R3], Nubis Cubed 2023 [R4], Hillaire 2016 course notes [R6], Wrenninge et al.
2013 [R7] and Wrenninge 2015 [R8], the UE5 component reference [R9]. The GPU Pro 7 chapter [R5] is
paywalled; its code is quoted here from a verbatim transcription [R5b]. That makes it secondary for
the exact constants, which are marked as such below.

### 1(a) Envelope + detail structure (the "modelling" half)

The core idea, the same in every Nubis generation: **a low-resolution field fixes where cloud may
exist (the envelope or "dimensional profile"). A tiling high-frequency noise is then applied at sample
time as an erosion of that envelope's edges**, never as a multiplication of the core. Detail costs
instructions, not memory. This is why it gets sub-metre detail out of 8–60 m data.

**HZD 2015 / GPU Pro 7 (2.5-D layer model)** [R1][R5b]

- Textures. All are tiling and stored as small 3-D textures [R1, slides 30–33]:
  - Base shape: **128³, RGBA**. R = Perlin–Worley. G, B, A = Worley at increasing frequency.
  - Detail: **32³, RGB**. Worley at increasing frequency.
  - Curl noise: **128², RGB 2-D**. It is non-divergent and is used to distort the detail-noise lookup.
- Weather map. R = coverage, G = precipitation, B = cloud type (0 = stratus, 0.5 = stratocumulus,
  1 = cumulus) [R1, slide 40][R5b]. Precipitation pushes the map to cumulonimbus at 70 % coverage.
  The layer is 1500–4000 m. Clouds are drawn within a 35 km radius, and from 15 km out the map moves
  towards cumulus at 50 % coverage [R1, slides 42, 45, 73].
- `remap(v, l0, h0, l1, h1) = l1 + (v − l0)·(h1 − l1)/(h0 − l0)` [R2, slide 28].
- Height-gradient (density-over-height) profiles. These are products of two remaps of the normalised
  height h in the layer. The stratus example is given verbatim [R2, slide 28]:
  `stratus = remap(h, 0.0, 0.1, 0.0, 1.0) · remap(h, 0.2, 0.3, 1.0, 0.0)`.
  Stratocumulus and cumulus use the same form with later fall-off points, and *type* interpolates the
  in/out points. **UNVERIFIED:** the exact stratocumulus and cumulus breakpoints are not printed in
  the slides I could read.
- Density sampler, GPU Pro 7 code (verbatim via [R5b]):
  ```
  low_fbm   = 0.625·N.g + 0.25·N.b + 0.125·N.a                  // N = 128³ texture
  base      = remap(N.r, −(1 − low_fbm), 1, 0, 1)                // dilate Perlin–Worley by Worley fBm
  base     *= density_height_gradient(h, type)
  base_cov  = remap(base, coverage, 1, 0, 1) · coverage          // coverage acts as an erosion
  p.xy     += curl.xy · (1 − h)                                  // turbulence, strongest at the base
  high_fbm  = 0.625·D.r + 0.25·D.g + 0.125·D.b                   // D = 32³ texture sampled at p·0.1
  mod       = lerp(high_fbm, 1 − high_fbm, saturate(h·10))       // wispy (inverted) at base → billowy above
  density   = remap(base_cov, 0.2·mod, 1, 0, 1)                  // detail erosion, max 20 %
  ```
  The Perlin–Worley channel itself is `perlin_worley = remap(perlin, 1 − worley, 1, 0, 1)` [R2, slide 34].
  Remapping is chosen over multiplying because it "prevents a loss of too much density at the core" [R2, slide 36].
- Wind and shear: `p += h · wind_dir · 500` (top offset) and `p += wind_dir · t · speed` [R2, slide 45][R5b].
- Anvils: `coverage = pow(coverage, remap(h, 0.7, 0.8, 1, lerp(1, 0.5, anvil_bias)))` [R2, slide 46].

**Nubis Evolved 2022 (vertical-profile, envelope and VFX models)** [R3]

- The envelope is now a "dimensional profile" built from 2-D NDF (Nubis Data Field) maps:
  `dimensional_profile = vertical_profile · cloud_coverage`.
- The noise is a 4-channel 128³ composite.
- The erosion became a subtraction:
  **`cloud_density = saturate(noise_composite − (1 − dimensional_profile))`**.
- Envelope model profile:
  `top = (1 − h)^1.5`, `bottom = h^2`, `edge = remap(dist_to_edge, 0, 35 m, 1, 0)`,
  `profile = bottom · top · edge`.
  Density is then `h · saturate(noise − (1 − profile))^0.27`.
- Wispy and billowy detail are blended over height by cloud type:
  `lerp(wispy, billowy, remap(h, type + 0.1, type − 0.1))`.
- Step size: `step = 3 m + 60 m · distance / 16384 m`.

**Nubis³ 2023 (voxel clouds, *Horizon Forbidden West: Burning Shores*)** [R4]. This is the generation
most relevant to an engine with a baked grid.

- Clouds are grown with a custom Houdini fluid solver ("Aero"), composited into
  "frankencloudscapes", then stored as NVDF voxel fields.
- Memory sizing for a ~4 × 4 km × 500 m region:
  - 2048×2048×256 at 2 m is 2.1 GB uncompressed or 537 MB in BC4.
  - Sparse formats were rejected because of the indirection cost per sample.
  - They chose **512×512×64 NVDFs at 8 m voxels (BC6, about 16.8 MB)**.
- The **dimensional profile is generated from a signed-distance field of the cloud** to get an
  outside-to-inside gradient. Extra channels: detail type (wispy↔billowy) and density scale.
- Up-rez noise is a **4-channel 128³** texture (16-bit, 4.2 MB): low- and high-frequency
  "Curly-Alligator" (inverted alligator noise distorted by curl noise) and low- and high-frequency
  alligator noise. Pseudocode (verbatim) [R4, slides 101–122]:
  ```
  wispy   = lerp(n.r, n.g, profile)                      // low→high freq toward the core
  billowy = lerp(0.3·n.b, 0.3·n.a, profile^0.25)
  comp    = lerp(wispy, billowy, detail_type)
  // within 50–150 m of camera: "twice-folded" noise for extra octave
  hhf     = lerp(1 − |(|2·n.g − 1|)·2 − 1|^4, |(|2·n.a − 1|)·2 − 1|^2, type)
  density = ValueErosion(profile, comp) · density_scale^4
  density = pow(density, lerp(0.3, 0.6, density_scale^4))   // sharpen low-density regions
  ```
- Result: **0.5 m effective precision from 8 m voxels**. Cost is 10 ms at 960×540 on PS5, against
  30+ ms for dense-grid sampling [R4, slide 121].
- A 512×512×64 SDF NVDF (range −256 to 4096 m) drives sphere-tracing for empty-space skipping.

**Hillaire 2016 (Frostbite)** [R6 §5.3–5.4] uses the same envelope + erosion approach with a smaller
data set:

- Weather texture: R = 2-D cloud density (coverage), G = cloud type index.
- A small **cloud-type texture** indexed by (type, normalised height): R = density profile over
  height, G = **erosion amount** (0 = smooth, 1 = fully eroded).
- Low-frequency noise is Perlin–Worley + Perlin octaves. High-frequency noise is Worley octaves. The
  generator is open source, https://github.com/sebh/TileableVolumeNoise.

### 1(b) Lighting

**Beer, powder, phase (HZD/GPU Pro 7)** [R1 slides 51–68][R5b]

- Transmittance toward the sun: `E = exp(−d_l)`, where `d_l` is the density summed over 6 cone samples
  toward the sun, the last one far away to catch distant-cloud shadows [R1, slide 83].
- "Powder" (missing in-scatter at depth zero): `P = 1 − exp(−2·d_l)`.
- **Beer–Powder: `E = 2·exp(−d_l)·(1 − exp(−2·d_l))`**. The powder term is view-dependent and is
  ramped toward the anti-solar direction [R1, slide 66].
- Rain clouds raise the absorption: `d_l` is scaled by precipitation.
- HG phase: HZD used **g = 0.2** in GPU Pro 7 [R5b]. Nubis 2017 used two lobes combined with `max`
  [R2, slides 80–82]:
  `Energy = max(HG(cosθ, 0.6), silver_intensity · HG(cosθ, 0.99 − silver_spread))`.
- Note a code bug. The HG code in the 2015/2017 slides ends in `/ 4.0 * PI`, which *multiplies* by π
  instead of dividing by 4π. Nubis Evolved fixes it with `· 1/(4π)` [R3]. Check any port.
- Nubis 2017 multiple-scattering stand-in (Wrenninge-style) [R2, slide 85]:
  `E = max(exp(−d_l), 0.7·exp(−0.25·d_l))`.
- Nubis 2017 in-scatter probability [R2, slides 89–92]. This replaced powder:
  ```
  depth_probability    = 0.05 + pow(lodded_density, remap(h, 0.3, 0.85, 0.5, 2.0))
  vertical_probability = pow(remap(h, 0.07, 0.14, 0.1, 1.0), 0.8)
  in_scatter           = depth_probability · vertical_probability
  ```
  Darker bases and edges therefore come from a **probability of in-scatter**, not from extra
  extinction.

**Nubis Evolved / Nubis³ structure** [R3][R4]

- `Light = Direct + Ambient (+ Secondary, e.g. lightning)`.
- `Direct = T·p_primary(θ) + MS·p_secondary(θ)`, with `T = exp(−Σ density)`.
- Multiple scattering is a **probability field derived from the dimensional profile**:
  ```
  ms = remap(profile·step, 0.1, 1, 0, 1)·(coverage·type)^0.25 · T_light^k_depth · h^k_height
  ```
- In Nubis³, `ms = profile · exp(−Σρ_sun · remap(sun_dot, 0, 0.9, 0.25, f(sdf)))`. The extinction is
  relaxed near the sun and deep inside, which gives the "inner glow".
- Sun optical depth is **precomputed into a 256×256×32 voxel grid**, amortised over 8 frames. That
  gave 40 % faster rendering and long-distance shadows.

**Ambient / sky term by height**

- HZD: "ambient sky contribution increases over height" [R1, slide 88].
- Hillaire: only the SH-probe DC term, × a linear gradient from 0 at the layer base to 1 at the top,
  biased to [a, 1] to include ground bounce [R6 §5.5.1].
- Nubis: `ambient = (1 − profile)^0.5` [R3]. The envelope variant is `(1 − coarse_density)^0.25 · h`.
- Nubis³ adds directional occlusion: `· exp(−Σρ toward the sky)` from the precomputed grid [R4, slide 147].

**Dual-lobe HG (Frostbite)** [R6 §5.7, eq. 18]
`p_dual(θ) = lerp(p_HG(θ, g0), p_HG(θ, g1), w)`. A strong forward lobe is kept, plus a backward lobe
so that clouds seen with the sun behind the camera are not flat.

- UE5 exposes Phase G, Phase G2 and Phase Blend, all defaulting to 0 [R9]. Caution: the UE page
  describes g < 0 as forward, which is the opposite of the usual convention and is probably a doc error.
- Typical community values are g0 ≈ 0.8 and g1 ≈ −0.2…−0.5, w ≈ 0.5 (**UNVERIFIED**: these are not
  given numerically in R6).
- Physically, water-cloud asymmetry at 550 nm is g ≈ 0.85–0.87 (see §2 and [R12]).
- A better analytic fit is the **HG + Draine blend** of Jendersie & d'Eon 2023 [R10], for droplet
  diameter d in µm, valid for 5 < d < 50:
  ```
  p(θ)  = (1 − w_D)·HG(θ; g_HG) + w_D·Draine(θ; g_D, α)
  Draine(u; g, α) = (1 − g²)(1 + α u²) / (4π (1 + α(1 + 2g²)/3) (1 + g² − 2 g u)^{3/2})
  g_HG  = exp(−0.0990567/(d − 1.67154))
  g_D   = exp(−2.20679/(d + 3.91029) − 0.428934)
  α     = exp(3.62489 − 8.29288/(d + 5.52825))
  w_D   = exp(−0.599085/(d − 0.641583) − 0.665888)
  ```
  At d = 20 µm this gives g_HG ≈ 0.995, g_D ≈ 0.59, α ≈ 27 and w_D ≈ 0.50. The paper claims a match
  to 95 % of the Mie phase function. The HLSL eval/sample code is MIT-licensed [R10].

**Multiple-scattering octaves (Wrenninge et al. 2013, "Oz")** [R7, eq. 1–2]
```
L = Σ_{i=0}^{N−1} L_i ,   L_i = σ_s · b^i · L_light(ω_i) · p(ω_i, ω_o, c^i·g) · exp(−a^i ∫_0^t σ_t(s) ds)
```
- a = "attenuation" (extinction scale), b = "contribution", c = "eccentricity attenuation".
- **Defaults are N = 8 and a = b = c = 1/2** (a common mis-quote is "a = b = c = 1").
- It was applied on shadow rays in a path tracer, choosing one octave stochastically with probability ∝ b^i.
- Hillaire writes the same scheme as `σ_s' = σ_s·a^n, σ_e' = σ_e·b^n, p'(θ) = p(θ·c^n)` and notes
  that energy conservation needs **a ≤ b in his notation**, i.e. the scattering scale must not exceed
  the extinction scale [R6 §5.8, eq. 19–20].
  - His a and b are swapped relative to Wrenninge's. In Wrenninge's notation the condition is
    **b ≤ a** (contribution ≤ attenuation).
  - Frostbite used N = 2–3.
- UE5 exposes the same three parameters per material: Multi Scattering Contribution (b), Occlusion
  (a) and Eccentricity (c). **Each defaults to 0.5**, with up to 2 octaves [R9][R9b].
- Wrenninge 2015 (Pixar) notes that physically accurate thick media need "**upwards of 100
  bounces**" [R8]. That is the reason the octave trick exists. It is also the number to compare with a
  path tracer's volume collision/bounce cap (§3).

### 1(c) Energy-conserving per-step integration (Hillaire 2015/2016)

For one ray-march step of length d, with constant extinction σ_t and constant source term
S = σ_s·L_in (with L_in = Σ lights · phase · shadow + ambient), the exact in-scattered radiance that
leaves the front of the step is [R6 §5.6.3, eq. 17; originally Hillaire 2015, R6b]:
```
∫_0^d exp(−σ_t x)·S dx = (S − S·exp(−σ_t d)) / σ_t = S·(1 − T_step)/σ_t
L_acc += T_acc · (S − S·T_step)/max(σ_t, ε)      // then
T_acc *= T_step,   T_step = exp(−σ_t d)
```
- With σ_s = ϖσ_t the step contribution is simply **ϖ·L_in·(1 − T_step)**. A step can therefore never
  emit more than its opacity times its albedo times the incident radiance, whatever its size.
- The naive forms are both wrong for dense steps [R6 §5.6.1, Fig. 35]:
  - `L += σ_s·L_in·T_acc·d` then `T *= T_step` (scatter first): over-bright, not energy-conserving.
  - The reverse order (transmit first) is over-dark.
- With this formula, 21 samples gave the result that needed 512 samples before [R6, Fig. 37].
- Reference shader: Hillaire's Shadertoy "VolumetricIntegration", https://www.shadertoy.com/view/XlBSRz.
- **Same formula for LWIR emission.** With S = σ_a·B_λ(T):
  `ΔL = (1 − ϖ)·B_λ(T)·(1 − T_step)`, i.e. emissivity of the slab = (1 − ϖ)(1 − T_step) at no scattering.
  It is an exact closed form per step, so the IR ray march should use it too.

### 1(d) Why dark halos/rims appear, and how production avoids them

Real physics first. Edges facing the sun *are* darker than the cloud body: the "powdered-sugar" effect,
where less in-scatter happens at small depth [R1, slides 57–66][R2, slide 87]. That makes them darker
**than the cloud core**, never darker than the sky directly behind a thin, sunlit or skylit, non-absorbing
cloud. A halo that is darker than the sky behind is an artefact. The causes in the literature, plus one
derivation of my own:

1. **Extinction applied without the matching in-scatter (missing ambient / sky in-scatter).**
   Derivation from the RTE, not from a cited source:
   - Take a medium with ϖ = 1 in a uniform radiance field L₀. Then
     `dL/ds = −σ_t L + σ_s ∫p L dω = −σ_t L₀ + σ_t L₀ = 0`, so the medium is **invisible**.
   - At a thin edge (τ ≪ 1) the observed radiance is approximately
     `L_sky,behind·(1 − τ) + τ·[ϖ·p(θ)·E_sun·T_sun + ϖ·⟨L_sky⟩_p]`.
     Here `⟨L_sky⟩_p` is the phase-weighted sky radiance *including the sky directly behind the cloud*,
     which a g ≈ 0.85 forward lobe sends straight on to the viewer.
   - Drop `⟨L_sky⟩_p`, or replace it with an "ambient from above × height gradient" term, and every
     thin edge subtracts τ·L_sky and adds back only τ·p·E_sun.
   - Away from the sun, HG(g = 0.85) at 90° is ≈ 0.0098 sr⁻¹, so `p·E_sun` can be smaller than
     `L_sky`. Result: a rim darker than the sky.
   - Fix: include a sky/ambient in-scatter term that does not vanish at low density. Nubis
     `(1 − profile)^0.5` is *largest* at the edge [R3]. Hillaire uses the SH-probe ambient [R6 §5.5.1].
     In a path tracer, make sure the dome/sky light actually scatters into the volume.
   - Validation: a **volume "white furnace" test**. Use a uniform dome of radiance L₀, no sun, ϖ = 1.
     The cloud must vanish to within noise.
2. **Non-energy-conserving step integration.**
   - Transmit-before-scatter ordering over-darkens [R6 §5.6.1].
   - With **60 m voxels and cumulus σ_t ≈ 0.05–0.12 m⁻¹** (Hess et al. 1998, quoted in [R6 §5.2]), a
     single step or voxel has τ ≈ 3–7. The whole optical "skin" of the cloud, where τ = 1 is about
     8–20 m in, sits *inside one voxel*.
   - Any order error is then maximal exactly at the edge. Use eq. 17 and steps ≲ 1/σ_t near edges.
3. **Wrong alpha convention when compositing over the sky.**
   - A ray march naturally produces *premultiplied* (associated) colour `C = Σ T·ΔL` and
     `α = 1 − T_total`. The correct composite is **`out = C + (1 − α)·L_sky`**.
   - Using `lerp(L_sky, C, α)` (i.e. `α·C + (1 − α)·L_sky`) multiplies the edge colour by α a second
     time. At thin edges (α ≪ 1) that is a dark fringe that scales as α².
   - The same happens when a non-premultiplied (C, α) buffer is bilinearly up-sampled, filtered, or
     painted into a lower-resolution dome and then composited [R11a][R11b].
   - In HZD/Nubis the march outputs direct, ambient, haze-blend and α in separate channels, and the
     post pass composites them over the sky [R2, slide 102].
4. **Ambient term missing or scaled to zero in thin regions.** This is the same physics as (1) but in
   art-directed form. Any ambient multiplied by density, by `(1 − exp(−2d))`, or by a two-stream
   *reflectance* `R(τ)` vanishes as τ → 0.
   - Two-stream conservative scattering gives `R ≈ (1 − g)τ/(2 + (1 − g)τ)`, which is → 0 at thin edges
     (Bohren 1987 [R13], formula from memory, UNVERIFIED exact form).
   - Meanwhile the direct transmittance `e^{−τ}` removes sky.
   - A two-stream "albedo envelope" must therefore be paired with its **diffuse transmittance**
     `1 − R` for the light from behind, not with `e^{−τ}`.
5. **Grid sampling.** Nearest-neighbour sampling of a thresholded 60 m grid gives stair-steps. Trilinear
   sampling of the same grid gives a 60 m-wide smooth ramp, which reads as "soft blobby".
   - Nubis³ shows the production answer: keep the voxel field as a smooth *SDF-derived profile*
     (8 m voxels) and create the edge at sample time by noise erosion, reaching 0.5 m precision [R4].

**Grey bodies.** These come from truncated multiple scattering: too few octaves or too few
path-tracer volume bounces. Thick cumulus needs ~10²-bounce paths [R8]. Frostbite's N = 1 vs N = 2–3
comparison is in [R6, Fig. 40].

## 2. Real cloud morphology statistics to validate against

Notation: l = √(projected area). n(x) is number density per unit x. Numbers marked "[calc]" are
derived, not quoted. Full DOIs are in the §2 reference list at the end of this note.

**Area–perimeter fractal dimension** (P ∝ (√A)^D)

- Lovejoy 1982 [M1]: radar and satellite rain and cloud areas, **1 to 1.2×10⁶ km²**, no characteristic
  length between 1 and 1000 km. **D = 1.35 ± 0.05.** The ± is quoted via [M3] and [M5].
- Cahalan & Joseph 1989 [M2]: Landsat at 30–80 m. A double power law fits best.
  - D ≈ 1.27 for small clouds and ≈ 1.56 for large, with a break near A ≈ 4 km² (secondary, via [M3][M4]).
  - D increases with brightness threshold.
- Other measured values:
  - Trade cumulus (ASTER, 15 m): **D = 1.28** [M7].
  - Cirrus 1.37 ± 0.02 and cumulonimbus 1.18 ± 0.05 [M8].
  - Deep convection (Himawari): D = 1.36 [M5]. Method: holes filled, clouds under 24 px dropped,
    edge-touching clouds dropped.
- Rees et al. 2024 [M3] argue the reported size dependence of D (Gifford's 1.35 → 1.77) is a
  hole-counting artefact.
- LES surface dimension ≈ 7/3, from Siebesma & Jonker 2000 [M9] but quoted second-hand (UNVERIFIED).
- **Target for the generator:** projected outlines with **D ≈ 1.3–1.4**. Measure with holes filled
  and edge-touching clouds excluded.

**Cumulus size distributions**

| Source | Data | Definition | Exponent / break |
|---|---|---|---|
| Neggers et al. 2003 [M11] | LES, 3 cases | n(l) ∝ l^−b | **b = 1.70** over ~1 decade below a case-dependent break; curves collapse when scaled by the break |
| Dawe & Austin 2012 [M12] | LES BOMEX, 100–1000 m | n(l) | 1.88 (snapshots) / 1.96 (tracked); break ≈ 1 km |
| Zhao & Di Girolamo 2007 [M7] | ASTER 15 m, 152 scenes | n(D_eq) | 2.19; cloud fraction 0.086, half of it from clouds under 2 km |
| Koren et al. 2008 [M13] | Landsat 30 m | n(a) ∝ a^−m | m = 1.3 ± 0.1, no break [calc: n(l) ∝ l^−1.6] |
| Wood & Field 2011 [M14] | MODIS + aircraft | 1-D chords n(L) ∝ L^−β | **β = 1.66 ± 0.04**, 0.1 to ~1500 km |
| DeWitt et al. 2024 [M4] | many satellites | dn/d ln a ∝ a^−α | α = 0.95 ± 0.08 from 3 to 3×10⁵ km², edge truncation handled |
| Mieslinger et al. 2019 [M17] | ASTER, 1158 scenes | — | double power law, break 590 m |
| Plank 1969 [M16] | Florida photographs | — | ≈ exponential |

- Conversion between definitions: n(A) ∝ A^−k is the same as n(l) ∝ l^−(2k−1). A 1-D chord exponent
  is about the area-based n(l) exponent minus 1 [calc].
- Breaks:
  - Benner & Curry: l ≈ 0.5–0.8 km. Neggers LES: l ≈ 0.4–1.3 km.
  - No source was found tying the break to boundary-layer depth (UNVERIFIED).
- **Target:** n(l) ∝ l^−(1.7–2.0) for ~50 m < l < 0.5–1 km, steeper above.

**Edge sharpness and the transition zone**

- Gerber et al. 2008 [M19], RICO, PVM probe at 10 cm resolution:
  - LWC gradients at the edge are **sharp at the 30 cm scale**.
  - Entrained parcels are a geometric mean of 1.6 m long and penetrate a geometric mean of 8.9 m
    inside the edge.
- Beals et al. 2015 [M24]: cloud/clear transitions are sharp **below 1 cm**.
- Wang, Geerts & French 2009 [M20], 1624 penetrations: the cloud margin, where LWC, droplet number
  and droplet size decline, extends **~200 m or ~10 % of the cloud diameter**, falling slowly at first
  and steeply near the edge.
- LES at 10 m (Eytan et al. 2022 [M26]): edge transition zone **~20–40 m**.
- Subsiding shell:
  - **~50–100 m** wide (Heus & Jonker 2008 [M21], via [M29]).
  - Doppler lidar typically ~100 m [M27].
  - DNS reaches ~100 m with w_min ≈ −2.3 m s⁻¹ [M28].
  - Asymmetric over land [M29].
- Twilight zone: hydrated aerosol and cloud fragments extend tens of km from clouds [M25].
- **Implication [calc].** Use cumulus β ≈ 0.05–0.12 m⁻¹ (§1, OPAC; cross-checked below as 75 km⁻¹
  at 0.5 g m⁻³ and 10 µm).
  - The visible τ = 1 skin is then 8–20 m deep, which is about the size of the LES edge zone.
  - The bulk of the cloud is a sharp-edged, nearly binary field at the metre scale, with a
    ~10 %-of-diameter interior decline of LWC.
  - A 60 m voxel grid with a 400 m minimum feature misses both.

**Adiabatic liquid water profile**

- Model: q_L(z) = f_ad · Γ_ad(T, p) · z (Albrecht et al. 1990 [M30]; Merk et al. 2016 [M31]).
- Values of Γ_ad in use: **1.44–2.0 g m⁻³ km⁻¹** [M31, Table 1].
- RICO adiabatic LWC: 0.605 → 2.46 g m⁻³ between 252 and 1074 m above the LCL, at 923–838 hPa [M19]
  [calc: ≈ 2.3 g m⁻³ km⁻¹].
- Sanity table [calc, Curry & Webster §6.5 + Bolton e_s], in g m⁻³ km⁻¹:

  | p \ T | 0 °C | 10 °C | 20 °C |
  |---|---|---|---|
  | 1000 hPa | 1.68 | 2.22 | 2.63 |
  | 900 hPa | 1.61 | 2.09 | 2.44 |
  | 800 hPa | — | 1.94 | 2.24 |

- Adiabatic fraction in RICO cumulus: 0.47 → 0.13 from 252 to 1074 m above base [calc from M19, Table 3].
  Near-adiabatic cores exist [M26]. Stratiform clouds: 0.3–0.9 [M31].
- Microphysics:
  - RICO cumulus: N = 95–116 cm⁻³, mean volume radius 9–12 µm [M19].
  - Stratocumulus: N ranges from under 10 to over 500 cm⁻³. Remote oceans are typically ≤ 50;
    downwind of industrial continents, over 200 [M32].
- Visible/NIR extinction: **β_ext ≈ 3·LWC/(2 ρ_w r_e)** (Stephens 1978 [M33]; Wood 2012 [M32], eq. 1).
  0.5 g m⁻³ at 10 µm gives **0.075 m⁻¹** [calc], consistent with OPAC.
- **LWIR is different.** It is absorption-dominated: β_abs ≈ κ_l · LWC with
  **κ_l ≈ 100–160 m² kg⁻¹**, decreasing with r_e [M32].
  - Longwave penetration is a few to tens of metres once q_L > 0.2 g kg⁻¹.
  - [calc] At 0.5 g m⁻³: β_abs ≈ 0.05–0.08 m⁻¹, giving an IR skin of ~12–20 m. The same edge-resolution
    requirement applies to the infrared march.

**Aspect ratio**

- Ghate et al. 2016 [M35], ARM radar/lidar:
  - Manus: thickness 267 ± 95 m, chord 308 ± 207 m.
  - Azores: thickness 179 ± 33 m, chord 411 ± 207 m.
  - [calc] Depth/chord ≈ 0.9 and 0.4.
- RICO turrets: width ~550 m, depth above the LCL 439–1350 m [calc: depth/width ≈ 0.8–2.1] [M19].
- BOMEX LES: only 15 % of clouds reach 1 km depth [M12].
- Chords understate diameter [M15], so depth/chord overstates depth/diameter.
- Lamer & Kollias 2015 aspect ratios ≈ 0.2–0.6 (snippet only, UNVERIFIED).
- **Target:** depth/width ≈ 0.4–1 for shallow cumulus, larger for growing turrets.

**UNVERIFIED in §2**
- Lovejoy's ± 0.05 and the details of his data.
- Cahalan & Joseph and Benner & Curry D values (second-hand).
- Siebesma & Jonker 7/3.
- The Neggers break–boundary-layer link.
- Wood & Field area exponent 1.87, which is from the submitted manuscript only.
- Continental N of 300–1000 cm⁻³.
- The claim that fair-weather cumulus have aspect ratio ≈ 1.

**References for §2** (resolve at https://doi.org/<DOI>)
- [M1] Lovejoy 1982, Science 216, 185, 10.1126/science.216.4542.185
- [M2] Cahalan & Joseph 1989, MWR 117, 261, 10.1175/1520-0493(1989)117<0261:FSOCF>2.0.CO;2
- [M3] Rees et al. 2024, NPG 31, 497, 10.5194/npg-31-497-2024
- [M4] DeWitt et al. 2024, ACP 24, 109, 10.5194/acp-24-109-2024
- [M5] Christensen & Driver 2021, GRL 48, e2021GL095746, 10.1029/2021GL095746
- [M6] Benner & Curry 1998, JGR 103, 28753, 10.1029/98JD02579
- [M7] Zhao & Di Girolamo 2007, JGR 112, 10.1029/2006JD007371
- [M8] Batista-Tomás et al. 2016, QJRMS 142, 983, 10.1002/qj.2699
- [M9] Siebesma & Jonker 2000, PRL 85, 214, 10.1103/PhysRevLett.85.214
- [M11] Neggers, Jonker & Siebesma 2003, JAS 60, 1060, 10.1175/1520-0469(2003)60<1060:SSOCCP>2.0.CO;2
- [M12] Dawe & Austin 2012, ACP 12, 1101, 10.5194/acp-12-1101-2012
- [M13] Koren et al. 2008, ACP 8, 3855, 10.5194/acp-8-3855-2008
- [M14] Wood & Field 2011, J. Climate 24, 4800, 10.1175/2011JCLI4056.1
- [M15] Rodts, Duynkerke & Jonker 2003, JAS 60, 1895, 10.1175/1520-0469(2003)060<1895:SDADPO>2.0.CO;2
- [M16] Plank 1969, JAM 8, 46, 10.1175/1520-0450(1969)008<0046:TSDOCC>2.0.CO;2
- [M17] Mieslinger et al. 2019, JGR 124, 11477, 10.1029/2019JD030768
- [M19] Gerber et al. 2008, JMSJ 86A, 87, 10.2151/jmsj.86A.87
- [M20] Wang, Geerts & French 2009, JAS 66, 3660, 10.1175/2009JAS3129.1
- [M21] Heus & Jonker 2008, JAS 65, 1003, 10.1175/2007JAS2322.1
- [M22] Katzwinkel et al. 2014, JAS 71, 2810, 10.1175/JAS-D-13-0222.1
- [M24] Beals et al. 2015, Science 350, 87, 10.1126/science.aab0751
- [M25] Koren et al. 2007, GRL 34, 10.1029/2007GL029253
- [M26] Eytan et al. 2022, JAS 79, 409, 10.1175/JAS-D-21-0201.1
- [M27] McMichael et al. 2020, GRL 47, 10.1029/2020GL089699
- [M28] Nair, Heus & van Reeuwijk 2020, JAS 77, 1353, 10.1175/JAS-D-19-0018.1
- [M29] Mallaun et al. 2019, ACP 19, 9769, 10.5194/acp-19-9769-2019
- [M30] Albrecht et al. 1990, GRL 17, 89, 10.1029/GL017i001p00089
- [M31] Merk et al. 2016, ACP 16, 933, 10.5194/acp-16-933-2016
- [M32] Wood 2012, MWR 140, 2373, 10.1175/MWR-D-11-00121.1
- [M33] Stephens 1978, JAS 35, 2123, 10.1175/1520-0469(1978)035<2123:RPIEWC>2.0.CO;2
- [M35] Ghate, Miller & Zhu 2016, MWR 144, 681, 10.1175/MWR-D-15-0110.1

## 3. Isaac Sim 6.0 / Omniverse RTX volume rendering

Sources: NVIDIA docs, forums and release notes, plus NVIDIA's shipped files read on this machine.
Items not confirmed by NVIDIA are marked UNVERIFIED.

**Formats and authoring**

- The path tracer renders VDB files, which are "internally converted to NanoVDB", and procedural MDL
  volumes. The grid may be density or SDF.
  - **"The VDB volume material can only be applied to a cube mesh."** At most 4 volumes may overlap.
    Source: RTX PT doc, https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html
  - Accepted extensions are `.vdb`, `.nvdb` and `.mvdb` (NeuralVDB):
    https://docs.omniverse.nvidia.com/kit/docs/omni.vdb/latest/source/extensions/omni.vdb.neuralvdb/docs/tutorials/basics.html
- The recipe is NVIDIA's own. Their sample scene `omni.vdb_timesample_editor-0.2.3/data/mdlanimation.usda`
  authors a `Mesh` cube with an identity xform and `primvars:isVolume = 1`, plus `OmniVolumeDensity`
  wired to `mdl:surface`, `mdl:displacement` and `mdl:volume`.
  - `inputs:volume_density_texture.timeSamples` drives VDB sequences.
  - It sets `rtx:pathtracing:ptvol:enabled = 1`.
  - This matches the project's ADR 0144.
- **`UsdVol.Volume` + `OpenVDBAsset` is not supported by RTX shading.**
  - "Currently we don't support OpenVDBAsset" (NVIDIA staff, 2024):
    https://forums.developer.nvidia.com/t/materialx-volume-shader-provides-unexpected-result/274278
  - https://docs.omniverse.nvidia.com/connect/latest/paraview/manual.html
  - That schema is for RTX Scientific (IndeX), which uses transfer functions, not lighting:
    https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx_scientific_index.html
- MDL object space equals VDB world space. Grids are chosen by a `selector`. MDL spec:
  https://raytracing-docs.nvidia.com/mdl/specification/MDL_spec_1.11.1_06May2026.pdf
  This explains ADR 0144's scaled-cube failure.
- `OmniVolumeDensity.mdl` (Kit 110.3, read locally) has only 4 inputs:
  - `volume_density_texture`
  - `volume_albedo` (scattering = albedo, absorption = 1 − albedo)
  - `volume_density_scale`
  - `directional_bias` ∈ [−1, 1], which feeds `df::anisotropic_vdf`. That this is HG is UNVERIFIED.
- **Emission is not supported**: "We don't support emissive volumes" (NVIDIA staff, 2023, thread
  274278 above). MDL `emission_intensity` is uniform only.
  - **Consequence: the LWIR cloud must stay on the in-house ray march.** RTX can only provide the
    visible-band scattering view.

**Path-tracer settings** (RTX PT doc above)

| Setting | Default | Meaning |
|---|---|---|
| `/rtx/pathtracing/ptvol/enabled` | False | Non-uniform volumes on |
| `/rtx/pathtracing/ptvol/transmittanceMethod` | 0 | 0 = biased ray marching, 1 = ratio tracking |
| `/rtx/pathtracing/ptvol/maxCollisionCount` | 1024 | Delta-tracking iterations (">32 for clouds"; too low and parts of the volume disappear) |
| `/rtx/pathtracing/ptvol/maxLightCollisionCount` | 32 | Ratio-tracking iterations on light rays |
| **`/rtx/pathtracing/ptvol/maxBounces`** | **2** | Bounces in non-uniform volumes (1 = single scattering) |
| `/rtx/pathtracing/maxBounces` | 4 | Bounces for any ray type |
| `/rtx/pathtracing/maxVolumeBounces` | 64 | Subsurface scattering only, not VDB |
| `/rtx/pathtracing/volumesAOV` | False | `PtVolumes` AOV |

- **Analysis.** ADR 0144 raised the global `maxBounces` to 32, but `ptvol/maxBounces` defaults to 2.
- If that default was left in place, the cloud gets about 2 scattering orders where thick cumulus
  needs ~10² [R8]. Expected result: grey bodies, plus dark thin edges where sky light is not
  re-scattered.
- **Check this setting first.** How the two caps interact is UNVERIFIED.

**Real-time (RT 2.0, the default mode in Isaac Sim 6.x)**

- No non-uniform/VDB volume settings are documented. The docs only say "volume rendering may diverge"
  from the path tracer: https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_rt.html
- What real-time does have is "Global Volumetric Effects": a froxel fog under
  `/rtx/raytracing/inscattering/*`. It can take density from homogeneous fog, height fog, noise, or
  sampled from a Flow grid; it cannot read a VDB.
  https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_common.html
- A user reported in March 2026 that textured-mesh volumes are visible only in Path Tracing:
  https://forums.developer.nvidia.com/t/rtx-scientific-colormap-texture-allocation-error-usd-composer/362540
- That RT 2.0 renders `isVolume` VDBs at all remains UNVERIFIED, and probably no.

**Built-in sky and clouds**

- **Dynamic Sky** is a DomeLight with a procedural sky MDL (CumulusLight.mdl). Its clouds are
  ray-marched inside the dome shader (`CloudRaySteps = 13`) and baked into a lat-long texture.
  - They therefore sit at infinity: no parallax and no occlusion of scene objects.
  - https://omniverse-content-production.s3.us-west-2.amazonaws.com/Environments/2023_1/DomeLights/Dynamic/CumulusLight.usd
  - NVIDIA staff call them "emulated volumetric clouds":
    https://forums.developer.nvidia.com/t/softer-shadow-for-dome-light/313734
  - `omni.kit.environment.core` 1.4.4 dates from Sep 2026:
    https://docs.omniverse.nvidia.com/kit/docs/omni.kit.environment.core/latest/CHANGELOG.html
- **Flow** (`omni.flowusd`) is maintained. It renders in real time (`/rtx/flow/*`) and has a cloud
  ray-march mode.
  - A `FlowEmitterNanoVdb` can ingest a `.vdb`/`.nvdb` grid.
  - Caution: `enableLowPrecisionDensity` stores density in 8 bits, which violates the project's
    float32 rule.
  - https://docs.omniverse.nvidia.com/extensions/latest/ext_fluid-dynamics/settings.html
  - Isaac Sim 6.0 added a FlowUSD writer:
    https://forums.developer.nvidia.com/t/announcement-isaac-sim-6-0-early-developer-release-for-gtc26/363709
  - Using Flow as a static km-scale real-time cloud is UNTESTED.

**Limits**

- No NVIDIA doc gives a VDB resolution or memory limit.
  - One 2023 user found a cloud VDB "terribly slow" on an A4000:
    https://forums.developer.nvidia.com/t/support-for-volumetric-cloud/243724
- No LOD or multi-resolution support is documented.
- Which grid is read from a multi-grid file is undocumented.
- Sequences are supported by time-sampling the texture input.
- Volumes write no depth (project measurement, ADR 0144).
- RTX 107.0 fixed volumes not contributing to alpha:
  https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer-release-notes/107_0.html

**Versions**

| Isaac Sim release | Kit |
|---|---|
| 6.0.0 GA (Jun 2026) | 110.1.1 |
| 6.0.1 | 110.1.2 |
| 6.1.0 (10 Sep 2026) | 110.3.0 (matches this machine) |

- Sources: https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/release_notes.html and
  https://docs.isaacsim.omniverse.nvidia.com/6.1.0/overview/release_notes.html
- The 6.0 and 6.1 release notes contain no volume items.

## 4. High-resolution cloud data for "hero" assets

Sources are the dataset landing pages, licences, papers and technical reports. Numbers marked
"derived" are arithmetic, not quoted.

**(a) Walt Disney Animation Studios Cloud Data Set**
- Landing page: https://www.disneyanimation.com/resources/clouds/
- Download: https://assets.disneyanimation.com/wdas_cloud.zip (≈3 GB).
- Readme: https://media.disneyanimation.com/uploads/production/data_set_asset/1/asset/Cloud_Readme.pdf
- Contents: `wdas_cloud.vdb`, "a heterogeneous volumetric cloud model containing density data", plus
  half, quarter, eighth and sixteenth LODs.
- Size:
  - Full grid 1987×1351×2449 voxels, 26.3 GB if stored dense: https://arxiv.org/html/2504.04564v1
  - Full VDB file ≈2.9 GB: https://blog.yiningkarlli.com/2018/07/disney-animation-datasets.html
- **Licence, verbatim**: "licensed under the Creative Commons Attribution-ShareAlike 3.0 Unported
  License" (Copyright 2017 Disney Enterprises):
  https://media.disneyanimation.com/uploads/production/data_set_asset/6/asset/License_Cloud.pdf
  - **This confirms the user's belief: CC BY-SA 3.0.**
  - ShareAlike applies to redistributed derivatives.
- Density is **arbitrary**: artist-made, with no units.

**(b) ARM LASSO**

LASSO-ShCu at the SGP site:
- Primary documents:
  - Gustafson et al. 2020, *BAMS*, doi:10.1175/BAMS-D-19-0065.1
  - DOE/SC-ARM-TR-216, doi:10.2172/1469590:
    https://armgov.svcs.arm.gov/publications/tech_reports/doe-sc-arm-tr-216.pdf
- Model: WRF 3.8.1 (FASTER modifications, Thompson microphysics). SAM runs were also included in
  2015–16.
- Grid:
  - **Δx = 100 m on a 25 km doubly-periodic domain (250×250).** A 14.4 km domain was tested and rejected.
  - **226 levels to 14.7 km: Δz = 30 m below 5 km**, stretching to 300 m at the top.
- 3-D `wrfout` fields:
  - **QCLOUD, QRAIN, QICE, QSNOW and QGRAUP in kg kg⁻¹**, plus QVAPOR.
  - T is the perturbation potential temperature; real temperature also needs P and PB.
  - Also U, V, W and CLDFRA.
- Output every 10 min. The raw tar is ≈30 GB per simulation. One float32 3-D variable is ≈56.5 MB (derived).
- 95 case days from 2015–2019.
- Access: Bundle Browser https://adc.arm.gov/lassobrowser with a free ARM login; delivery by FTP or Globus.
- **Licence: CC BY 4.0** (ARM public data): https://www.arm.gov/working-with-arm/acknowledging-arm
- Cite the BAMS paper, TR-216 and the data DOI 10.5439/1342961.

LASSO-CACTI (deep convection):
- D4 nest at 100 m: 2146×2776 points × 149 levels, output every 5 min. The cloud subset includes the
  Q* fields. https://lasso-cacti-doc.arm.gov/latest/modeling_config.html
- Raw data DOI: 10.5439/1905810

**(c) Other LES fields**

Intercomparison cases:
- BOMEX: 6.4 km, 100 m / 40 m, 75 levels. Secondary source:
  https://acp.copernicus.org/articles/21/14039/2021/
- RICO: 12.8 km, 128×128×100, Δz = 40 m.
- No official 3-D archive exists for either intercomparison.
- These grids are UNVERIFIED from the primary papers.

**htrdr Atmosphere Starter Pack** (Meso-NH). This is the finest physical grid found:
https://www.meso-star.com/projects/htrdr/htrdr-atmosphere-spk.html
- Cases:
  - **ARMCu**: SGP shallow cumulus, 6.4 km, **25 m isotropic**, 256×256×160.
  - **BOMEX**: 12.8 km, 25 m, 512×512×160.
  - FIRE stratocumulus: 100 m × 10 m.
  - Grids: https://arxiv.org/pdf/1902.01137
- Fields: liquid water (RCT), vapour (RVT), pressure (PABST) and **T in K**, as float64 in the `.htcp`
  format.
- Size: 1.8 GB tarball; the ARMCu NetCDF is 193 MB.
- **Licence: GPLv3+.**

**I3RC**: https://earth.gsfc.nasa.gov/climate/model/i3rc/testcases
- Case 4 is an LES cumulus field, 100×100×36 at **66.7 m × 40 m**.
- It carries **LWC in g m⁻³**, r_eff, and extinction/ϖ at 0.67, 2.13 and **11 µm**.
- The original host is offline.
- The Case 5 stratocumulus field (55 m) ships in https://github.com/RobertPincus/i3rc-monte-carlo-model
  under GPL-2.

**Cloud Botany** (DALES, EUREC4A): https://howto.eurec4a.eu/botany_dales.html
- 103 members on a 153.6 km domain.
- Δx = 100 m; Δz = 20 m at the first level, stretched 1 % per level.
- Hourly 3-D `ql` / `qt` / `qr` in kg/kg plus `thl`. Zarr on DKRZ Swift.
- One `ql` snapshot ≈1.4 GB (derived). The full output is 40 TB.
- Data licence UNVERIFIED.

**Zenodo, CC BY 4.0**
- WRF-LES RICO: 50 m / 40 m, 12.8 km, LWC in kg/kg, one 2.4 GB `.mat`: https://zenodo.org/records/10816160
- AT3D (GPL-3.0) ships a RICO cut at 20 m × 40 m with lwc and reff: https://github.com/CloudTomography/AT3D

**(d) Generators and samples.** All give **arbitrary density**.
- OpenVDB samples (https://www.openvdb.org/download/):
  - bunny_cloud 577×572×438, plus smoke, explosion and fire.
  - The website repo is CC BY 4.0, but bunny_cloud derives from the Stanford bunny, which has
    non-commercial terms.
- JangaFX free EmberGen VDBs, including a 10-cloud pack, are **CC0**:
  https://jangafx.com/software/embergen/download/free-vdb-animations
- Blender output is the user's property. Mantaflow / geometry-nodes VDB export is UNVERIFIED in the
  docs.
- Houdini has a Cloud Rig shelf tool.
- No official NVIDIA sample cloud VDB was found.

**Summary**

| Source | Physical LWC? | Temperature | Resolution | Licence |
|---|---|---|---|---|
| Disney cloud | no (arbitrary) | no | 1987×1351×2449 vox | CC BY-SA 3.0 |
| LASSO-ShCu | QCLOUD kg/kg | θ′ + P | 100 m / 30 m, 25 km | CC BY 4.0 |
| htrdr ARMCu / BOMEX | RCT | T (K) | 25 m isotropic | GPLv3+ |
| Cloud Botany | ql kg/kg | thl | 100 m / ≥20 m, 153.6 km | UNVERIFIED |
| I3RC Cu | LWC g/m³ + 11 µm optics | profile | 66.7 m / 40 m | not stated |
| RICO WRF (Zenodo) | kg/kg | no | 50 m / 40 m | CC BY 4.0 |
| OpenVDB / EmberGen / Blender | no | EmberGen temperature grid possible | 10²–10³ vox | mixed / CC0 / own |

**Recommendation (analysis).**
- Use LES (LASSO-ShCu for licence; htrdr ARMCu at 25 m for resolution) as the physically correct
  *envelope*. It supplies the LWC profile, the adiabatic slope, size statistics and IR temperature
  consistency.
- Add Nubis-style sample-time erosion for the edge detail below the grid scale. Even 25 m only
  marginally resolves the 8–20 m optical skin.
- Use the Disney cloud as a visual target for edge morphology, with an assumed LWC scaling.

**UNVERIFIED in this section**
- BOMEX and RICO primary grids.
- Disney voxel size in metres and its grid names.
- htrdr RCT units.
- Cloud Botany licence.
- Whether the RICO Zenodo file is full 3-D.
- I3RC availability.
- The scope of the OpenVDB licence over its model files.
- Blender and Mantaflow VDB export documentation.

## References

### References — §1 (rendering technique)

- [R1] A. Schneider, N. Vos, "The Real-time Volumetric Cloudscapes of Horizon: Zero Dawn", SIGGRAPH 2015 Advances in Real-Time Rendering in Games course. PDF with speaker notes: https://advances.realtimerendering.com/s2015/The%20Real-time%20Volumetric%20Cloudscapes%20of%20Horizon%20-%20Zero%20Dawn%20-%20ARTR.pdf
- [R2] A. Schneider, "Nubis: Authoring Real-Time Volumetric Cloudscapes with the Decima Engine", SIGGRAPH 2017 Advances course. Landing page https://sites.google.com/view/vonschneidz/publications/nubis-2017 ; PDF https://drive.google.com/file/d/0B-D275g6LH7LOE1RcVFERGpkS28/view ; course page https://advances.realtimerendering.com/s2017/
- [R3] A. Schneider, "Nubis, Evolved: Real-Time Volumetric Clouds for Skies, Environments, and VFX", SIGGRAPH 2022 Advances course. https://www.guerrilla-games.com/read/nubis-evolved ; PDF https://d3d3g8mu99pzk9.cloudfront.net/AndrewSchneider/NubisEvolved/NubisEvolved-NoVideos.pdf
- [R4] A. Schneider, "Nubis³: Methods (and Madness) to Model and Render Immersive Real-Time Voxel-Based Clouds", SIGGRAPH 2023 Advances course. PDF https://d3d3g8mu99pzk9.cloudfront.net/AndrewSchneider/Nubis%20Cubed.pdf
- [R5] A. Schneider, "Real-Time Volumetric Cloudscapes", in W. Engel (ed.), GPU Pro 7: Advanced Rendering Techniques, CRC Press 2016, pp. 97–127 (paywalled): https://www.oreilly.com/library/view/gpu-pro-7/9781498742542/chapter-30.html
- [R5b] Verbatim code transcription of [R5] (secondary source, Chinese-language notes): "【GPU Pro 7】Real-Time Volumetric Cloudscapes", https://www.jianshu.com/p/ae1d13bb0d86
- [R6] S. Hillaire, "Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite", SIGGRAPH 2016 Physically Based Shading course, course notes (62 pp.): https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/s2016-pbs-frostbite-sky-clouds-new.pdf ; index https://sebh.github.io/publications/
- [R6b] S. Hillaire, "Physically-based & Unified Volumetric Rendering in Frostbite", SIGGRAPH 2015 Advances course: https://www.ea.com/frostbite/news/physically-based-unified-volumetric-rendering-in-frostbite ; slides https://www.slideshare.net/slideshow/physically-based-and-unified-volumetric-rendering-in-frostbite/51840934 ; companion Shadertoy https://www.shadertoy.com/view/XlBSRz
- [R7] M. Wrenninge, C. Kulla, V. Lundqvist, "Oz: The Great and Volumetric", SIGGRAPH 2013 Talks, Article 46. PDF https://fpsunflower.github.io/ckulla/data/oz_volumes.pdf
- [R8] M. Wrenninge, "Art-Directable Multiple Volumetric Scattering", SIGGRAPH 2015 Talks, doi:10.1145/2775280.2792512. PDF https://history.siggraph.org/wp-content/uploads/2022/10/2015-Talks-Wrenninge_Art-Directable-Multiple-Volumetric-Scattering.pdf
- [R9] Epic Games, "Volumetric Cloud Component Properties in Unreal Engine": https://dev.epicgames.com/documentation/en-us/unreal-engine/volumetric-cloud-component-properties-in-unreal-engine
- [R9b] Epic Games, "Volumetric Cloud Component in Unreal Engine": https://dev.epicgames.com/documentation/en-us/unreal-engine/volumetric-cloud-component-in-unreal-engine
- [R10] J. Jendersie, E. d'Eon, "An Approximate Mie Scattering Function for Fog and Cloud Rendering", SIGGRAPH 2023 Talks, doi:10.1145/3587421.3595409. Project page https://research.nvidia.com/labs/rtr/approximate-mie/ ; PDF https://research.nvidia.com/labs/rtr/approximate-mie/publications/approximate-mie.pdf ; HLSL (MIT) https://research.nvidia.com/labs/rtr/approximate-mie/assets/draine.hlsl
- [R11a] Real-Time Rendering blog, "GPUs prefer premultiplication": https://www.realtimerendering.com/blog/gpus-prefer-premultiplication/
- [R11b] B. Ciechanowski, "Alpha Compositing": https://ciechanow.ski/alpha-compositing/
- [R12] Hess, Koepke, Schult 1998, "Optical Properties of Aerosols and Clouds: The Software Package OPAC", BAMS 79, 831–844 (quoted via [R6 §5.2]; σ_t at 550 nm: stratus 0.04–0.06 m⁻¹, cumulus 0.05–0.12 m⁻¹).
- [R13] C. F. Bohren, "Multiple scattering of light and some of its observable consequences", Am. J. Phys. 55, 524 (1987): https://pubs.aip.org/aapt/ajp/article/55/6/524/1053003
- [R14] B. Smits, H. W. Jensen, "Global Illumination Test Scenes" (furnace test): http://graphics.ucsd.edu/~henrik/papers/gitest/gitest.pdf
- [R15] S. Hillaire, TileableVolumeNoise (Perlin–Worley/Worley generator): https://github.com/sebh/TileableVolumeNoise
