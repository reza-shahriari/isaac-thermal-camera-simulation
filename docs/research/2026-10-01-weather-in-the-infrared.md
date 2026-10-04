# Weather in the infrared: clouds, fog, haze, rain, snow and wet surfaces — survey, 2026-10-01

**What is missing from the physics for clouds that look right, and for fog, haze, rain and snow in the
infrared frame?**

Produced in response to the owner, 2026-10-01: *"i dont like the weather management at all, we create
very bad clouds, even working hard on it the output is still not well designed, also i dont know how we
can simulate the impact of haze, fog, rain and snow in IR viewport"* — followed by *"add things if
anything missing inside the physics first, then go for clouds, focusing on RGB then IR … after these are
done move to rain snow etc"*.

A repository audit plus three parallel literature searches. This is a **snapshot of what was readable on
2026-10-01**, not a maintained document. The full search notes, with every claim tagged and their own
reference lists, are beside this file in [`2026-10-01-weather/`](2026-10-01-weather/), with the scratch
Mie scripts that produced the numbers marked *computed*:

| note | what it covers |
|---|---|
| [`clouds.md`](2026-10-01-weather/clouds.md) | production cloud technique (Nubis, Frostbite, Oz), why rims go dark, real cumulus statistics, Isaac Sim / RTX volume capabilities, public cloud data |
| [`fog-and-wet-surfaces.md`](2026-10-01-weather/fog-and-wet-surfaces.md) | fog droplet sizes and band ratios, fog and haze vertical structure, WMO thresholds, wet surfaces, dew, snow and ice emissivity |
| [`precipitation.md`](2026-10-01-weather/precipitation.md) | rain and snow extinction by band, what a pixel loses, drop temperature, fall speeds, what single drops look like to a bolometer, window drops |

The decisions taken from it are in `docs/physics-model.md` §4.6, §6.7, §7.2 (note), §7.5–7.8, §9.6,
Tier 3 and Appendix A item 12, as spec issues `S59`–`S65`, and as roadmap lane `WX`. References below
use the physics model's numbers, [R62]–[R115].

---

## 1. The short version

1. **The clouds look wrong because of the field, not the radiometry.** A dozen ADRs since 0070 made the
   cloud *consistent* across the two bands. The look is the shared field, which is coarser than a cloud's
   own optical skin: 60 m cells against an 8–20 m visible and 15–40 m LWIR skin [R70][R71]. ADR 0146
   placed that outside this repository's remit; it was the limiting factor all along.
2. **Production clouds keep the field coarse and make the edge at sample time.** An envelope says where
   cloud may be, and a tiling noise erodes its edge when it is sampled — 0.5 m effective precision from 8 m
   voxels [R72]–[R74]. Upstream `isaac-weather-fx` has moved this way since our pin (Worley coverage,
   Perlin–Worley billows, 15 m detail at sample time; commits `9469420`…`0596f9c`).
3. **What remains wrong on upstream main**, measured by a CPU march at one ray per pixel: static toward the
   horizon (detail marched at ~125 m steps with no footprint filter), flat lighting (brightness from a
   two-stream envelope, so no dark bases), no aerial perspective, striped stratocumulus, and a real-time
   dome at ~9 camera pixels per texel.
4. **The dark rims have a physical cause with a test.** A thin edge removes $\tau L_{\text{sky}}$ and must
   scatter back about as much; a two-stream *reflectance* vanishes as $\tau \to 0$ and does not. A
   non-absorbing cloud under a uniform sky must vanish: the white furnace [R78].
5. **RTX's volume scattering cap defaults to 2 bounces** (`ptvol/maxBounces`) where thick cumulus needs
   ~100 [R81][R83]. RTX volumes cannot emit, and Real-Time mode has no documented VDB support, so the
   infrared cloud is always our march and the real-time visible cloud is always the dome.
6. **Fog: the band ratios in the code describe haze.** ADR 0049's `droplet` set matches 0.5–1.5 µm droplets;
   real fog (3–20 µm) puts MWIR, SWIR and NIR extinction at 1.0–1.3× the visible. Only LWIR sees through
   fog, and only small-droplet fog [R84][R88].
7. **Visibility is defined at 5 %, not 2 %.** The code's 3.912/V reads every weather-fx visibility 31 % too
   opaque [R91].
8. **Rain has no LWIR advantage.** Extinction is flat from 0.55 to 12 µm (Q → 2). Half of it is absorption in
   the thermal bands and none in the visible, and in LWIR the diffracted half leaves the pixel as a blur
   halo, not a loss [R95][R101].
9. **An uncooled camera barely sees individual raindrops.** The bolometer's ~10 ms lag keeps 3 % of a
   falling drop's contrast and ~40 % of a snowflake's, independent of range [R108].
10. **Wet surfaces are a state the solver already carries and nothing reads.** A film of tens of µm is optically
   water in LWIR: specular, mirroring the cold sky at grazing angles [R100][R113].

---

## 2. Clouds

### 2.1 The technique [C: read in full — R72–R74, R78, R79, R81, R82]

- **Envelope plus erosion.** Horizon (2015): base noise 128³ (Perlin–Worley plus three Worley octaves),
  detail 32³ (Worley), curl-noise distortion, a weather map (coverage, precipitation, type), and the erosion
  `remap(v, l0, h0, l1, h1)`. Remapping rather than multiplying avoids hollowing the core. Nubis Evolved
  (2022): `density = saturate(noise − (1 − profile))`. Nubis³ (2023): Houdini-simulated clouds stored as
  512×512×64 NVDFs at 8 m, carrying a profile derived from the signed-distance field and up-rezzed by 128³
  noise to 0.5 m effective precision. Dense 2 m grids (2.1 GB) were rejected.
- **Lighting.** Per-step energy-conserving integration: each step adds $\varpi L_{\text{in}}(1-T_{\text{step}})$
  (Hillaire 2015, eq. 17); with it 21 samples matched a 512-sample result. Multiple-scattering octaves
  (Wrenninge 2013): defaults N = 8, a = b = c = ½, with contribution ≤ attenuation for energy
  conservation. A dual-lobe phase function, and an ambient term from the sky that rises with height. The
  HG code in the 2015/2017 slides multiplies by π instead of dividing by 4π — check any port.
- **Why rims go dark** (clouds.md §1(d)): sky in-scatter missing at thin edges; non-energy-conserving step
  ordering; `lerp(sky, C, α)` instead of `C + (1−α)·sky` when compositing (an α² fringe); a reflectance-only
  envelope paired with direct transmittance; trilinear reads of a thresholded grid.

### 2.2 Real cumulus statistics [C, except where marked in clouds.md §2]

| property | value | source |
|---|---|---|
| area–perimeter dimension | 1.35 (1–1.2 × 10⁶ km²); 1.28 trade cumulus at 15 m | [R62][R63] |
| size exponent, $n(l)\propto l^{-b}$ | 1.70 (LES); 1.88–1.96 (BOMEX); 2.19 (ASTER); chords 1.66 ± 0.04 | [R64][R66][R63][R65] |
| scale break | 0.4–1.3 km | [R64][R66] |
| edge | LWC steps over ~30 cm; decline over the outer ~10 %; subsiding shell 50–100 m | [R67][R68] |
| depth/chord | 0.9 (Manus), 0.4 (Azores) | [R69] |
| visible extinction | 0.05–0.12 m⁻¹ | [R70] |
| LWIR liquid-water absorption | 100–160 m² kg⁻¹ (flux) | [R71] |

### 2.3 Isaac Sim / RTX [C: NVIDIA docs and forum statements, clouds.md §3]

- A VDB volume material applies only to a cube mesh; at most 4 overlapping volumes. The project's recipe
  (ADR 0144) is NVIDIA's own sample's.
- `UsdVol.Volume` with `OpenVDBAsset` is for the IndeX renderer, not RTX shading.
- `OmniVolumeDensity` has density, albedo, density scale and `directional_bias`. NVIDIA staff: "We don't
  support emissive volumes".
- Path-tracer volume settings: `ptvol/enabled` (False), `transmittanceMethod` (0), `maxCollisionCount`
  (1024; ">32 for clouds"), `maxLightCollisionCount` (32), **`maxBounces` (2)**. `maxVolumeBounces` is
  for subsurface scattering, not VDBs.
- Real-Time 2.0 has no documented VDB support; its volumetric effects are frustum voxel fog.

### 2.4 Cloud data for the volume-asset source [C: landing pages, licences, papers — clouds.md §4]

| source | grid | contents | licence |
|---|---|---|---|
| LASSO shallow convection, ARM SGP [R75] | 100 m × 30 m, 25 km periodic, 10 min | QCLOUD/QRAIN/QICE kg kg⁻¹, θ′, p (physical) | CC BY 4.0 |
| htrdr starter pack, Meso-NH [R76] | 25 m isotropic (SGP cumulus 6.4 km; BOMEX 12.8 km) | liquid water, T in K | GPLv3+ |
| WRF RICO, Zenodo 10816160 | 50 m × 40 m | liquid water | CC BY 4.0 |
| Disney cloud [R77] | 1987 × 1351 × 2449 and four lower resolutions | density, arbitrary units | CC BY-SA 3.0 |
| EmberGen cloud pack | — | density, arbitrary | CC0 |

**Recommendation from the search:** large-eddy data for the envelope (physical liquid water and
temperature), Nubis-style erosion for detail below its grid, and the Disney cloud as a visual target. Even
25 m only marginally resolves the 8–20 m optical skin.

---

## 3. Fog, mist and haze

### 3.1 Band ratio versus droplet size [computed: `irsim.atmosphere.mie` on Segelstein water, modified gamma α = 6, γ = 1; within ~15 % of Shettle & Fenn's tabulated 10 µm values]

| $r_{\text{eff}}$ (µm) | 0.5 | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| LWIR 8–14 / 0.55 | 0.03 | 0.085 | 0.21 | 0.35 | 0.48 | 0.72 | 0.89 | 1.00 | 1.10 | 1.10 |
| MWIR 3–5 / 0.55 | 0.06 | 0.28 | 0.87 | 1.24 | 1.36 | 1.28 | 1.18 | 1.13 | 1.09 | 1.08 |
| SWIR 0.9–1.7 / 0.55 | 0.37 | 1.11 | 1.23 | 1.10 | 1.07 | 1.05 | 1.04 | 1.04 | 1.03 | 1.02 |
| NIR 0.7–1.0 / 0.55 | 0.71 | 1.23 | 1.05 | 1.04 | 1.03 | 1.02 | 1.02 | 1.02 | 1.01 | 1.01 |

- The closed form: $\beta_{11}/\beta_{\text{vis}} \approx \pi c_e r_{\text{eff}}/\lambda \approx 0.089\,r_{\text{eff}}$[µm],
  with $c_e(11\,\mu m) = 0.31$ [R87]. Extinction at 11 µm is ≈ 133 km⁻¹ per g m⁻³, within a factor of 2
  over 341 spectra [R86].
- Typical sizes: young radiation fog 2–4 µm; mature radiation fog 8–10 µm near the surface [R89] (mean
  diameter 15–20 µm, Price 2011); advection fog 16–20 µm [R84].
- Measured: 4 µm no better than 1.55 µm; 10 µm > 20 dB in continental fog [R88]. Heavy fog "neutral within
  a factor of about 2" [R85]. Detection-range table by ICAO category: MWIR no better than the eye at CAT II
  and below [R97].
- Single-scattering albedo: 0.33–0.55 in LWIR and 0.68–0.86 in MWIR. In MWIR a thick fog is a partly
  reflecting layer.

### 3.2 Structure, temperature, thresholds [C/V: fog-and-wet-surfaces.md §2–5]

- Radiation fog: usually below 200 m, measured tops 83–115 m [R89]. Optically thick above an LWP of
  30 g m⁻² [R90]. LWC peaks at ~80 % of depth [R89], and extinction rises with height [R85]. Patchiness
  is set by soil moisture and terrain at 100 m–1 km [R93].
- Temperature: a thin fog sits inside a surface inversion (12 °C at the ground, 14–16 °C above 40–60 m
  [R89]). A thick fog follows the saturated adiabat (5.3–6.5 K km⁻¹).
- Thresholds (WMO-No. 782): fog below 1 km; mist 1–5 km at RH > 95 %; haze ≤ 5 km, dry. MOR is defined at
  5 % [R91].
- Haze profile (LOWTRAN 7 source): 0.158 / 0.099 / 0.062 km⁻¹ at 0 / 1 / 2 km for 23 km visibility; uniform
  through the lowest 1 km at visibility ≤ 10 km [R94].

---

## 4. Rain and snow [C/computed: precipitation.md]

### 4.1 Rain, Marshall–Palmer, by Mie over 0.1–6 mm [computed; Chu & Hogg R95 read in full]

| λ (µm) | σ at 1 mm h⁻¹ (km⁻¹) | absorbed share | g |
|---|---|---|---|
| 0.55 | 0.3655 | 0.000 | 0.882 |
| 1.06 | 0.3663 | 0.005 | 0.888 |
| 1.55 | 0.3663 | 0.220 | 0.923 |
| 4.0 | 0.3700 | 0.460 | 0.969 |
| 10.0 | 0.3748 | 0.482 | 0.985 |

The closed form $\pi N_0/\Lambda^3 = 0.365\,R^{0.63}$ km⁻¹ agrees. The share of the diffraction lobe kept
inside a 1 mrad pixel at 10 mm h⁻¹: 70 % (0.55 µm), 48 % (1 µm), 30 % (1.55 µm), 7 % (4 µm), 1 % (10 µm).
The ITU visibility code implies an effective Q of 1.14–1.2 for a visual observation [R98].

### 4.2 Drop temperature, speed, appearance

- Drops relax to the wet bulb: e-folding fall 4.4 m (1 mm), 17 m (2 mm), 57 m (4 mm) at 10 °C [computed,
  after R102]. Convective rain averaged 3.8 °C below the wet bulb [R103].
- Water's k: 0.0508 at 10 µm (16 µm 1/e depth), 0.0046 at 4 µm (69 µm), 2.24 × 10⁻⁴ at 1.5 µm (0.53 mm)
  [R100]. Drop $Q_{\text{abs}}$: 0.97–1.0 at 10 µm.
- Rain speed: $v = 9.65 - 10.3e^{-0.6D}$ m s⁻¹ [R109]. Snow: Locatelli & Hobbs per habit [R110].
- A bolometer keeps $c = 1 - e^{-D/(v\tau_{\text{th}})}$ of a falling particle's contrast: 3 % for rain, 20–40 %
  for snow [R108].
- A drop on the window lowered LWIR responsivity to ~90 % without blur [R115]. No published thermal imagery
  of individual falling drops or flakes was found.

### 4.3 Snow

- ITU-R P.1817-1 Table 2 laws (dry $b$ = 1.38, wet $b$ = 0.72), fitted at 0.8–1.55 µm [R98][R99].
- IR/visible 1.05–1.43 measured [R104].
- Visibility versus snowfall scatters 3–10× at a given rate [R105].
- Snow LWIR emissivity 0.98–0.99, falling to 0.93 at 75° for coarse grains [R106].
- LOWTRAN 7 and MODTRAN have no snow model.

---

## 5. Wet surfaces [C/computed: fog-and-wet-surfaces.md §6]

- **Water reflectance in LWIR** [computed, Segelstein]:

  | incidence | 0° | 60° | 75° | 80° | 85° |
  |---|---|---|---|---|---|
  | reflectance | 0.013 | 0.05 | 0.20 | 0.34 | 0.57 |

  Emissivity is 0.987 at nadir and 0.66 at 80°.
- **Films:** a film over ~20–50 µm is optically water in LWIR. A night's dew, 0.14 ± 0.12 mm [R114], already
  qualifies.
- **Wetting raises emissivity:** asphalt, concrete and granite by more than 0.02 in 8–10 µm, 0.9–2.3 K
  [R112]. Puddles mirror the sky in LWIR [R113]. Surfaces darken in the visible and SWIR when wet [R111].
- **"Thermal washout"** is quantified only by vendors; no peer-reviewed number was found.

---

## 6. What remained unverified

Each note keeps its own list. The ones that matter to a decision:

- The fit wavelength of Carbonneau's law.
- Any 3–5 or 8–12 µm rain fit.
- Whether $S$ in the snow law is a water-equivalent rate, and whether its λ term holds past 1.55 µm.
- Bergot's ~500 m fog-roll scale.
- MWIR snow and ice emissivity.
- Whether RTX Real-Time 2.0 renders `isVolume` VDBs at all.
- Whether `directional_bias` is a Henyey–Greenstein $g$.
- Whether the dome light scatters into RTX volumes — the last bears directly on the dark rims in the
  path-traced tier.
