# Fog, haze and wet-surface phenomenology: literature note for irsim

Compiled 2026-10-01. Scope: what the current homogeneous Beer–Lambert haze/fog model (Koschmieder 3.912/V × per-band
ratio, path radiance (1−τ)B(T_air)) is missing.

**How to read the tags**

- **[V]** I read the number in the primary text (PDF, Fortran source, or official document).
- **[A]** The number comes from the abstract or the publisher landing page only.
- **[S]** The number comes from a secondary source or a search-engine summary. Treat it as weaker evidence.
- **[C]** I computed it myself in this session. The scripts are `fog_mie.py`, `fog_sweep.py` and `water_fresnel.py`
  in `scripts/` beside this note. They use the repo's `irsim.atmosphere.mie.mie_efficiencies` (a BHMIE port) and Segelstein (1981)
  water n/k from `data/nk/water.csv`. At 0.55 µm they use Hale & Querry n = 1.333, k = 1.96e-9.
- **UNVERIFIED**: I could not confirm the item against any source.

---

## 0. What this means for the current model

1. **The fog "droplet" ratios do not describe natural fog.** The current regime is LWIR 0.16, MWIR 0.24, SWIR 0.40,
   NIR 0.60. Those values match haze or very small droplets (r_eff ≈ 0.5–1.5 µm). That is also roughly what Cerema
   PAVIN's small-droplet artificial fog produces.
   - For the Shettle & Fenn fog models, the Mie band-mean ratios (β_band/β_0.55) are:
     - LWIR 8–14 µm: 0.35–0.72 (radiation fog) and 1.08–1.09 (advection fog) [C].
     - MWIR: 1.08–1.28 [C].
     - SWIR: 1.02–1.09 [C].
     - NIR: 1.01–1.03 [C].
   - **MWIR, SWIR and NIR do not see through real fog better than the eye.** Only LWIR does, and only when droplets are
     small (§1.3).
   - A single fixed "fog" ratio set cannot be right. The ratio is a function of droplet effective radius, so it should
     be driven by r_eff, or equivalently by fog type and life-cycle stage.
2. **Visibility convention.** WMO meteorological optical range (MOR) is defined at 5 % transmission, so
   β = −ln 0.05 / MOR = 2.996/MOR [V; Liandrat et al. 2022, quoting WMO]. The code uses 3.912/V, the 2 % Koschmieder
   threshold. If the input is a reported MOR (METAR, PAVIN), 3.912/V overstates β by 31 %.
3. **Fog is not vertically homogeneous** (§2). It is not isothermal either: it has an inversion at the top, and a thick
   fog has a saturated-adiabatic interior.
4. **Haze also has a vertical profile** (§4). LOWTRAN 7 and OPAC give ready-made ones.
5. **Scattering is not negligible in MWIR fog.** Single-scatter albedo is ω ≈ 0.68–0.86 in MWIR and 0.33–0.55 in LWIR
   [C]. Path radiance of (1−τ)B(T_air) is exact only when the in-fog radiance field is isotropic at B(T). That fails
   near the fog top and for paths that look up at the sky or down onto the fog (§3).
6. **Wet surfaces need a state variable.**
   - In LWIR, a water film of ≳ 20–50 µm makes the surface optically water: ε ≈ 0.987 at nadir, falling to 0.66 at 80°
     incidence [C].
   - Wetting raises soil and asphalt emissivity, mostly in the 8–10 µm band [V/A].
   - Puddles are near-specular mirrors at grazing angles (§6).

---

## 1. Fog droplet size distributions and the spectral extinction ratio

### 1.1 Shettle & Fenn (1979) fog models [V, scanned report]

The models use the modified gamma distribution n(r) = A r^α exp(−b r^γ), with r in µm and n in cm⁻³ µm⁻¹. Shettle &
Fenn take them from Silverman & Sprague, following Dyachenko. They describe models 1–2 as typical of **advection**
fogs and models 3–4 as typical of **radiation** fogs, and say they are "also characteristic of developing and mature
fogs, respectively." In each pair, models 1 and 3 are heavy fog and models 2 and 4 are moderate fog.

| Model | Type | A (as scanned) | α | b | γ | r_mode µm | N cm⁻³ | r_eff µm [C] | LWC g m⁻³ [C] | β₀.₅₅ km⁻¹ [C] | V = 3.912/β [C] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Advection, heavy | 0.06592 | 3 | 0.3 | 1 | 10.0 | 20 | 20 | 0.372 | 28.8 | 136 m |
| 2 | Advection, moderate | 0.027 | 3 | 0.375 | 1 | 8.0 | 20 | 16 | 0.191 | 18.5 | 212 m |
| 3 | Radiation, heavy | 2.37305 | 6 | 1.5 | 1 | 4.0 | 100 | 6 | 0.063 | 16.7 | 234 m |
| 4 | Radiation, moderate | 607.5 | 6 | 3.0 | 1 | 2.0 | 200 | 3 | 0.016 | 8.71 | 449 m |

Notes on the table:

- **The A column for models 1 and 2 looks swapped in the scan.** Normalising to N = 20 cm⁻³ needs A = 0.027 for
  b = 0.3 and A = 0.0659 for b = 0.375 [C]. Normalise A from N rather than copying it.
- Models 3 and 4 are self-consistent.
- Shettle & Fenn's own text gives meteorological ranges of "130 m for Fog Model 1 to 450 m for Fog Model 4." This
  agrees with the computed column.
- The models are scalable from fogs of V < 50 m to fogs of V > 1 km [V].

Shettle & Fenn's usage guidance (§4.3) [V]:

- Heavy fog with V < 200–250 m: extinction is "virtually independent of wavelength"; use Fog Model 1.
- Light to moderate fog with V > 350 m: "a noticeable difference between the extinction for visible wavelengths and
  in the 8–12 µm window"; use Fog Model 4.
- Very light fog with V ≈ 1 km: the 99 % RH aerosol models may represent the wavelength dependence as well as any fog
  model.

Shettle & Fenn also quote valley-fog measurements by Pilié et al. [V, OCR]:

- Developing fog: 100–200 droplets cm⁻³ in the 1–10 µm radius range, with mean radius 2–4 µm.
- As the fog thickens, concentration falls (the OCR'd value is illegible) and mean radius rises from 6 to 12 µm.
- Droplets with r < 3 µm were "seldom observed in fully developed fog".

**Spot-check of my Mie code against Shettle & Fenn Tables 44–47 (OCR'd).**

| Model | Shettle & Fenn, 0.55 µm | Mine, 0.55 µm | Shettle & Fenn, 10 µm | Mine, 10 µm |
|---|---|---|---|---|
| 1 | ≈ 28.7 km⁻¹ | 28.8 km⁻¹ | 35.25 km⁻¹ | 34.8 km⁻¹ |
| 3 | ≈ 16.6 km⁻¹ | 16.7 km⁻¹ | ≈ 11.3 km⁻¹ | 9.9 km⁻¹ |
| 4 | ≈ 8.67 km⁻¹ | 8.7 km⁻¹ | ≈ 2.0–2.1 km⁻¹ | 1.86 km⁻¹ |

The two agree within about 15 % at 10 µm. The remaining gap comes from the water n/k set and the OCR.

### 1.2 Other measured fog microphysics

- **Price (2011)**, Cardington, UK; droplets measured at 2 m [A]:
  - Initial phase: small drops, about ≤ 10 µm *diameter*.
  - Mature phase: mean *diameter* about 15–20 µm, i.e. radius 7.5–10 µm.
  - Fog becomes optically thick to longwave at about 100 m depth.
- **Nurowska et al. (2025)**, valley radiation fog, Poland; 74 balloon soundings [V]:
  - Near-surface r_eff 8–10 µm, decreasing by 3–4.6 µm over 100 m of height.
  - Larger droplets (≥ 18.5 µm) near the ground.
  - Mean LWC in the fog core 0.2–0.4 g m⁻³; individual maximum 0.97 g m⁻³.
  - N_c up to 300 cm⁻³ (peaks up to 410 cm⁻³).
  - LWP > 15 g m⁻².
  - **Implication: mature radiation fog can have r_eff as large as advection fog.** That erases the LWIR advantage
    (§1.3).
- **Pinnick et al. (1978)**, Grafenwöhr, Germany, February 1976; balloon soundings up to 250 m [V]:
  - Haze is lognormal with r_g = 0.2 µm, σ_g = 1.9.
  - Fog is bimodal. The large mode peaks at r = 4–6 µm (r_g = 5 µm, σ_g = 1.6). The small mode has r_g = 0.3–0.6 µm
    and σ_g = 1.8–2.5.
  - LWC of haze and fog spans about 10⁻⁴ to 0.45 g m⁻³.
- **Gultepe et al. (2006)**, warm-fog visibility parameterisation [V, via Gultepe et al. 2009]:
  - Vis[km] = 1.002 / (LWC[g m⁻³] · N_d[cm⁻³])^0.6473.
  - Fitted over LWC 0.005–0.5 g m⁻³ and N_d 1–400 cm⁻³.
  - N_d is "traditionally" set to 100 cm⁻³ (marine) or 200 cm⁻³ (continental).
- **Gultepe et al. (2007), review**: "most fogs have LWC 0.01–0.4 g m⁻³" [S]. I could not open the full text (paywall).
- **Natural fog D_mean ranges from 2 to 27 µm.** Advection fog has larger droplets than radiation fog, with diameters
  up to 30 µm [V; Duthon et al. 2024].
- **Cerema PAVIN artificial fog is small-droplet.** Keep this in mind when interpreting PAVIN results.
  - Two DSD classes: "unimodal … centred around 1 micron and bimodal … centred around 1.5 and 10 microns"
    [V; Pinchon et al. 2016/2018].
  - Bijelic et al. describe them as "radiation fog … mean diameter of 2 µm" and "advection fog … mean diameter of 6 µm"
    [V].
  - Duthon et al. (2020) give a small-droplet mode of 0.5–1 µm diameter and a medium-droplet fog with modes at
    0.5–1 µm and 7–10 µm [S].
  - The platform's maximum mean diameter is about 8 µm [V; Duthon et al. 2024].
- **Bibliographic only, parameters UNVERIFIED** (full text not reached):
  - Tampieri & Tomasi (1976), modified-gamma fog and cloud models.
  - Eldridge (1966): twelve averaged fog DSDs grouped into "two general types" plus an LWC–visual-range relation [A].
  - Eldridge (1971): the visibility–LWC relation depends on DSD width [A].
  - Kunkel (1984), β = 144.7·LWC^0.88: UNVERIFIED (from memory, not confirmed).

### 1.3 How strongly the LWIR/visible ratio depends on droplet size

**Physics: the linear IR regime.**

- For r ≲ λ, Q_ext ≈ c_e·2πr/λ (Chylek 1978). This makes infrared extinction proportional to LWC and independent of
  the DSD.
- Pinnick et al. (1979) confirmed it at 11 µm "within a factor 2 for 341 different fog and haze droplet distributions."
  The relation "works better for radiation fogs than advection fogs" [V].
- "There exists no size distribution independent relation between extinction in the visible (0.55 µm) and fog liquid
  water content" [V].
- Klein & Dabas (2014) re-validated it at SIRTA and give c_e(11 µm) = 0.31 and c_e(4 µm) = 0.64. The 4 µm relation does
  *not* hold [V].
- From c_e = 0.31, with W the liquid water content (LWC) in g m⁻³ [C]:
  - β₁₁ [km⁻¹] ≈ 133·W.
  - β_vis [km⁻¹] ≈ 1500·W / r_eff[µm].
  - Therefore **β₁₁/β_vis ≈ π c_e r_eff/λ ≈ 0.089 × r_eff[µm]**. This holds while most of the water is in droplets with
    r ≲ 7–11 µm. Beyond that the ratio saturates at about 1.0–1.1.
- Pinnick et al. (1979) predict absorption at 11 µm of σ_a = 93 W km⁻¹ [V].

**Mie band-mean ratio β_band/β₀.₅₅ [C].** Values are unweighted spectral means of β over the band.

Shettle & Fenn models:

| Model (r_eff) | NIR 0.7–1.0 | SWIR 0.9–1.7 | MWIR 3–5 | LWIR 8–14 | at 10 µm |
|---|---|---|---|---|---|
| Advection 1 (20 µm) | 1.01 | 1.02 | 1.08 | 1.09 | 1.21 |
| Advection 2 (16 µm) | 1.01 | 1.03 | 1.10 | 1.08 | 1.19 |
| Radiation 3 (6 µm) | 1.02 | 1.05 | 1.28 | **0.72** | 0.59 |
| Radiation 4 (3 µm) | 1.03 | 1.09 | 1.22 | **0.35** | 0.21 |

Sweep over r_eff, using a modified gamma with α = 6, γ = 1 (the shape of the radiation-fog models):

| r_eff µm | 0.5 | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 15 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| LWIR/vis | 0.03 | 0.085 | 0.21 | 0.35 | 0.48 | 0.72 | 0.89 | 1.00 | 1.10 | 1.10 |
| MWIR/vis | 0.06 | 0.28 | 0.87 | 1.24 | 1.36 | 1.28 | 1.18 | 1.13 | 1.09 | 1.08 |
| SWIR/vis | 0.37 | 1.11 | 1.23 | 1.10 | 1.07 | 1.05 | 1.04 | 1.04 | 1.03 | 1.02 |
| NIR/vis | 0.71 | 1.23 | 1.05 | 1.04 | 1.03 | 1.02 | 1.02 | 1.02 | 1.01 | 1.01 |

How to use these numbers:

- A path transmittance needs the spectral exp(−βL) integrated over the band, not exp of the band-mean β.
- The current "droplet" set (LWIR 0.16 / MWIR 0.24 / SWIR 0.40 / NIR 0.60) is not consistent with any single r_eff in
  this sweep. SWIR 0.40 needs r_eff ≈ 0.5 µm, while LWIR 0.16 needs r_eff ≈ 1.5 µm.
- For comparison, Pinnick's measured haze lognormal (r_g 0.2 µm, σ_g 1.9, r_eff 0.56 µm), treated as wet haze with the
  water index, gives NIR 0.78, SWIR 0.50, MWIR 0.11, LWIR 0.045 [C]. Real dry rural aerosol absorbs more in LWIR. The
  current rural-haze set (0.70 / 0.43 / 0.19 / 0.10) is within a factor of about 2 of this.

Single-scatter albedo, band-mean ω [C]:

| Model | MWIR | LWIR |
|---|---|---|
| Advection 1 | 0.68 | 0.54 |
| Advection 2 | 0.70 | 0.55 |
| Radiation 3 | 0.82 | 0.50 |
| Radiation 4 | 0.86 | 0.33 |

**Measured and modelled evidence, by source:**

- **Arnulf et al. (1957)**, spectrophotometry from 0.35 to 10 µm [A]: "the transmission of haze increased markedly with
  increasing wavelength, from the visible to 10 microns, but this marked increase was not found for fogs."
- **Pinnick et al. (1978)** [V]:
  - Haze extinction goes roughly as 1/λ.
  - Heavy fog is "neutral (wavelength independent) within a factor of about 2; the extinction at 4 µm being slightly
    greater than that at 0.55 µm and 10 µm."
- **Pinnick et al. (1979)** [V]:
  - On English fogs (Garland; Roach et al.): "extinction in advection fogs tends to be more neutral (wavelength
    independent) than in radiation fogs … because advection fogs generally have a broader distribution of droplet
    sizes."
  - Power-law fits σ_e(λ) = a·σ_e(0.55)^b are given for each fog type, but the OCR of a and b is illegible —
    UNVERIFIED.
- **Nebuloni (2005)** [A]:
  - When V is less than a few hundred metres, extinction at 0.55, 1.2 and 3.7 µm is "comparable to and roughly twice as
    much as that in the far IR (10.6 µm)."
  - The far-IR advantage "grows as visibility exceeds 500 m."
- **Breton et al. (2026)**, Optics Letters [A]:
  - Measured transmission at 0.532, 1.55, 4 and 10 µm through real controlled fog, with continental and maritime DSDs
    measured at the same time.
  - 10 µm gives a ">20 dB" transmission gain in continental fog.
  - **4 µm shows "no meaningful advantage over 1.55 µm".**
  - Visibility-based models "fail to predict transmission performance beyond the near-infrared."
- **Beier & Gemperlein (2004)**, MODTRAN 4 plus TTIM, via FLIR technical note TN_0001 [V for the note]. Detection range
  for ΔT = 10 K:

  | ICAO category | Visual | MWIR | LWIR |
  |---|---|---|---|
  | CAT I | 1.22 km | 3.0–9.8 km | 5.9–10.1 km |
  | CAT II | 0.61 km | 0.54 km | 2.4 km |
  | CAT IIIa | 0.305 km | 0.294 km | 0.293 km |
  | CAT IIIc | 0.092 km | 0.089 km | 0.087 km |

  The note adds that maritime aerosols give the lowest detection range in every climate, because maritime particles
  are larger.
- **Pinchon et al. (2016/2018)**, PAVIN tunnel [V]:
  - Fog density at which a pedestrian 25 m away becomes visible: visible RGB MOR 47 ± 10 m, NIR 28 ± 7 m,
    SWIR 25 ± 3 m, LWIR 15 ± 4 m.
  - This was measured in **small-droplet artificial fog**.
- **Rivera Velázquez et al. (2022)**, PAVIN, LWIR 7.5–13.5 µm, 50 °C target at 10 m [A, full text read via summary]:
  - Median attenuation below 10 % for MOR > 20 m.
  - 10–20 % for MOR 16–20 m.
  - Above 50 % for MOR 10–15 m.
  - If "attenuation" means 1−τ over 10 m, this implies β_LWIR/β_vis ≲ 0.05 at MOR = 20 m and ≳ 0.18–0.27 at
    MOR = 10–15 m [C]. That is consistent with r_eff ≈ 1–2 µm.
- **Rivera Velázquez et al. (2022), extended conclusion** [A]: 18° and 30° FOV cameras still detect objects down to a
  13 m MOR.

---

## 2. Vertical structure of fog

### Depth and how the top is set

- Radiation fog "develops from the ground upwards, usually not exceeding 200 m in height" and is tied to the
  temperature inversion [V; Nurowska et al. 2025, introduction].
- Measured tops in Nurowska et al. (2025) [V]:
  - Mostly 85 m, maximum 115 m.
  - Nightly maxima of 102, 87 and 83 m.
  - The fog layers sat inside the temperature inversion.
- **Transition from thin to optically thick fog:**
  - LWP > 30 g m⁻² [V; Wærsted et al. 2017; Toledo et al. 2021].
  - Alternative criteria from Costabloz et al. (2025), as quoted by Nurowska et al.: fog top > 110 m, or
    LWP > 15 g m⁻² [V].
  - Price (2011) puts the transition at about 100 m depth [A].
- **Above that threshold, longwave cooling moves to the fog top** [V, Wærsted 2017]:
  - Cooling produces 40–70 g m⁻² h⁻¹ of condensate, roughly 28–49 W m⁻² of latent-heat equivalent [C].
  - A cloud layer above the fog cuts this by up to 100 %.
  - Winter-midday shortwave absorption evaporates 10–15 g m⁻² h⁻¹.
- **Model defaults in the LOWTRAN 7 Army Vertical Structure Algorithm (VSA)** [V]:
  - Fog thickness 0.2 km.
  - Inversion height 2 km, or 0.2 km for fog.
  - Case 1 (cloud or fog at the surface): extinction *increases* with height from base to top.
  - Case 3 (radiation fog or boundary-layer inversion, no ceiling): extinction *decreases* with height to the fog or
    layer top.
  - Case 2/2′ (haze or light fog under a ceiling): extinction increases with height up to the cloud base.
- **Deep basin and valley fog:**
  - Cuxart & Jiménez (2012): a 60-hour deep radiation fog in the Ebro basin [A]. The ~300 m depth is UNVERIFIED.
  - Swiss Plateau fog and low stratus is "several hundred metres" thick [S].
  - California tule fog reached about 300–335 m in GOES cloud-thickness retrievals [S, CIMSS blog].
  - Treat these depths as indicative only (UNVERIFIED).

### Visibility versus height inside the fog

- **Pinnick et al. (1978)**, radiation fogs (winds < 2 m s⁻¹) [V]:
  - Concentration of droplets with r > 4 µm increases strongly with height over the first 150 m.
  - In one fog, extinction and LWC changed by **2–3 orders of magnitude over 150 m**.
  - In heavy fog they increased by a **factor of 2–4** with height.
  - Haze showed "little vertical variation."
  - These trends contradict the Pilié et al. valley-fog measurements, which found the opposite.
- **Nurowska et al. (2025)** [V]: in the mature stage, LWC and N_c peak at about 80 % of fog depth.
- **Toledo et al. (2021)**, well-mixed "adiabatic" fog [V]:
  - LWC increases with height, then decreases near the top because of entrainment.
  - Adiabaticity converges toward 0.7 once LWP exceeds about 30 g m⁻².
- **Shallow fog**, WMO-No. 782 [V]:
  - MIFG: reported when horizontal visibility is ≥ 1000 m but a layer between the ground and 2 m (eye level) has
    visibility < 1000 m.
  - BCFG: fog patches randomly covering the aerodrome.
  - PRFG: a substantial part of the aerodrome is covered by fog.

### Fog-top temperature

- **Thin, stable fog sits inside a surface-based inversion**, so temperature rises with height. In Nurowska et al.
  (2025) the ground was 12.0–12.8 °C, with the inversion starting at 40–60 m and reaching 14–15.8 °C [V]. In Pinnick
  et al. (1978), haze and fog traverses were "stable", for example 1.1 °C at the ground and −0.1 °C at 155 m [V].
- **Thick, well-mixed fog follows the saturated adiabat** and is capped by an inversion [V; Price 2011 [A];
  Toledo et al. 2021].
  - At 1000 hPa the saturated-adiabatic lapse rate is about 5.3 K km⁻¹ at 10 °C and about 6.5 K km⁻¹ at 0 °C [C].
  - So the top of a 200 m fog is about 1.1–1.3 K colder than its base, with warmer air above.
- **Fog-top cooling rates of about 1–4 K h⁻¹ are UNVERIFIED** (search summary only).

### Horizontal patchiness

- **Fog location is set by mesoscale conditions** (10–200 km) [A; Lin et al. 2023].
- **Soil-moisture heterogeneity changes fog duration** by more than 50 minutes, at microscales of 100 m to 1 km
  [A; Lin et al. 2023].
- **Horizontal roll structures of about 500 m wavelength** appear in a radiation-fog LES (Bergot 2013) — UNVERIFIED,
  search summary only.
- WMO keeps separate "patches" (BCFG) and "partial" (PRFG) descriptors [V].

---

## 3. Fog temperature and path radiance

- **Fog air is saturated.** Mist needs RH > 95 % (WMO-No. 782) [V]. Thick fog follows the saturated adiabat
  [V, Toledo 2021].
- **Droplets are at air temperature to first order.** Taking T_fog ≈ T_air is standard. I found no source quantifying
  the droplet–air offset (Roach 1976 treats radiative effects on droplet growth: bibliographic only).
- **The real caveats are the vertical temperature profile (§2) and scattering.** Using (1−τ)B(T_air) with T_air at the
  camera is wrong in three cases:
  1. The path crosses the inversion. Fog in a thin fog is colder than the air above it.
  2. The camera looks *at* a fog bank from outside, e.g. a drone above the fog top. Then the emitting layer is at the
     fog-top temperature, which is the coldest point.
  3. In MWIR, where ω ≈ 0.7–0.86 [C]. An optically thick fog seen against the cold sky (looking up) or from above has an
     effective emissivity below 1. It reflects or scatters part of the sky and ground radiance.
- **Satellite night-fog detection relies on point 3.** It uses the brightness-temperature difference between 3.7–3.9 µm
  and 11 µm (Ellrod 1995) [A, bibliographic]. The water-cloud emissivity at 3.7–3.9 µm of roughly 0.8–0.9 is
  UNVERIFIED.
- **Forward scattering:** the asymmetry parameter at 10 µm is g ≈ 0.94 for advection fog 1, about 0.92–0.93 for
  advection fog 2, about 0.86 for radiation fog 3 and 0.68 for radiation fog 4 [V, Shettle & Fenn Tables 44–47, OCR].
  Diffraction-peak light partly counts as "transmitted" for a wide-FOV imager. Pinnick et al. (1979) explicitly neglect "multiple
  scatter effects and forward scatter corrections."

---

## 4. Haze vertical profile

- **Shettle & Fenn (1979)** [V]: in the boundary layer, "up to 1–2 km above the surface, the composition of the aerosol
  particles is primarily controlled by sources … at the earth's surface."
- **LOWTRAN 7 user guide** (Kneizys et al. 1988) [V]:
  - Boundary-layer models (rural, urban, maritime, tropospheric, advection fog, radiation fog, desert) apply over
    0–2 km.
  - The tropospheric profile applies over 2–10 km, and stratospheric profiles over 10–30 km.
- **LOWTRAN 7 Fortran BLOCK DATA PRFDTA.** Aerosol extinction at 0.55 µm in km⁻¹ [V, source code]:

  | Altitude | VIS 50 km | VIS 23 km | VIS 10 km | VIS 5 km | VIS 2 km |
  |---|---|---|---|---|---|
  | 0 km | 0.0662 | 0.158 | 0.379 | 0.770 | 1.94 |
  | 1 km | 0.0415 | 0.0991 | 0.379 | 0.770 | 1.94 |
  | 2 km | 0.0260 | 0.0621 | 0.0621 | 0.0621 | 0.0621 |

  - Above 2 km the profile is seasonal:
    - 3 km: 0.0346 km⁻¹ (spring–summer, VIS 23) and 0.0272 km⁻¹ (fall–winter, VIS 23).
    - 4 km: 0.0185 km⁻¹ and 0.0120 km⁻¹ respectively.
  - Interpretation [C]:
    - Clear cases (VIS 50 and 23 km) decay exponentially with a scale height of 2.14 km.
    - Hazy cases (VIS ≤ 10 km) are uniformly mixed through the lowest 1 km and fall steeply to 0.062 km⁻¹ at 2 km,
      a scale height of about 0.4 km between 1 and 2 km for VIS = 5 km.
- **OPAC** (Hess et al. 1998), Table 5 [V]: N(h) = N(0)·exp(−h/Z).

  | Aerosol type | Layer thickness H | Scale height Z |
  |---|---|---|
  | Continental clean, average and polluted; urban | 2 km | 8 km (≈ well-mixed) |
  | Maritime (all) | 2 km | 1 km |
  | Desert | 6 km | 2 km |
  | Arctic | 2 km | 99 km (uniform) |
  | Antarctic | 10 km | 8 km |
  | Free troposphere above the layer, to 12 km | — | 8 km |

- **CALIOP mean extinction height** (Koffi et al. 2016) [A/S]: Z_α over 0–6 km is about 1.5–3.5 km, depending on region
  and season.
- **Typical boundary-layer heights** (daytime convective ~1–2 km, nocturnal stable ~0.1–0.3 km): textbook values, not
  verified this session — UNVERIFIED.

---

## 5. Mist, fog and haze thresholds

- **WMO-No. 782 (2019)** [V]:
  - **Fog (FG):** "visibility … less than 1 000 m."
  - **Mist (BR):** "microscopic water droplets or wet hygroscopic particles … reducing horizontal visibility to 1 000 to
    5 000 m. The relative humidity is greater than 95 per cent."
  - **Haze (HZ):** "extremely small dry particles … opalescent appearance with a reduction in horizontal visibility to
    5 000 m or less."
  - **Smoke (FU):** can be reported with visibility < 1000 m "if there are no suspended water droplets and the relative
    humidity is not greater than about 90 per cent."
- **WMO International Cloud Atlas** [V]:
  - Fog: the droplets reduce horizontal visibility to less than 1 km.
  - Mist: the droplets "do not reduce horizontal visibility to less than 1 km"; mist is "light fog" in practice.
  - Haze: dry particles; the Atlas gives no numeric threshold.
- **WMO-No. 49 Vol. II**: BR at 1000–5000 m, HZ at ≤ 5000 m [S].
- **Older haze RH thresholds** [S, ACPD supplement table — UNVERIFIED]:
  - WMO-No. 266: RH < about 80 %.
  - WMO-No. 8: e.g. RH < 80 %.
  - UK Met Office aviation handbook: RH < 95 %.

---

## 6. Wet surfaces, puddles, rain, dew and frost

### Optics of a water film [C, Segelstein water]

- **1/e absorption depth in water:**
  - LWIR: 3–19 µm (median 9 µm).
  - MWIR: 0.9–89 µm (median 30 µm). It is 69 µm at 4.0 µm and 29 µm at 3.5 and 4.5 µm.
  - SWIR and NIR: 0.3 mm to more than 100 mm.
- **What that means:**
  - A film thicker than about 50 µm is opaque in LWIR. A film thicker than about 0.3 mm is opaque across MWIR.
  - In SWIR and NIR the film is transparent, and wet surfaces darken instead. The cause is internal reflection and a
    lower relative index (Twomey et al. 1986; Lekner & Dorf 1988) [A].
- **Fresnel reflectance of smooth water**, band-mean, Planck-weighted at 293 K:

  | Incidence | 0° | 30° | 45° | 60° | 70° | 75° | 80° | 85° | 88° |
  |---|---|---|---|---|---|---|---|---|---|
  | LWIR R | 0.013 | 0.014 | 0.020 | 0.050 | 0.122 | 0.200 | 0.336 | 0.574 | 0.799 |
  | MWIR R | 0.019 | 0.020 | 0.027 | 0.058 | 0.130 | 0.209 | 0.344 | 0.580 | 0.802 |

  - ε = 1 − R, so LWIR ε is 0.987 at nadir and 0.66 at 80°.
  - Masuda et al. (1988) is the standard reference for rough water; bibliographic only.

### Measured emissivity changes on wetting

- **Mira et al. (2007)**, lab study [S, abstract as indexed]:
  - Emissivity rises by 1.7–16 % as water content increases, most in sandy soils at 8.2–9.2 µm (the quartz
    reststrahlen band).
  - Up to +0.16 for sand.
  - The effect is negligible above field capacity.
- **Mira et al. (2010)** [A]:
  - Soil-specific relations, R² up to 0.99, error < ±0.014.
  - The 10–12 µm channels show little variation with soil type or soil moisture [V, quoted in Sánchez 2011].
- **Sánchez et al. (2011)**, field measurements [V]:
  - Emissivity increases with soil moisture in all CE312 channels.
  - Agreement with the lab models is within 2 %, or up to 5 % at 8–9 µm when the soil is cracked.
  - Hysteresis: the dried-out soil ends with higher emissivity than it started.
- **Hulley et al. (2010)** [S]: wetted sands returned to dry emissivity within about 1 hour.
- **Zhong et al. (2022)**, ten urban surfaces [V, abstract]:
  - Emissivity rises with moisture, mostly at 8.0–10.0 µm.
  - Concrete, granite, **asphalt** and soil change by more than 0.02, which is 0.9–2.3 K of surface-temperature error.
  - Permeable brick, gravel, clay brick, grass and shrubs change minimally.

### Puddles and sky reflection

- **Rankin et al. (2011)**, JPL unmanned ground vehicles [V]:
  - In thermal IR, water bodies reflect background objects such as trees.
  - They approach sky intensity at long range.
  - Their intensity falls with incidence angle faster than that of the surrounding terrain.
  - In 24-hour MWIR data, the pond was warmer than soil and vegetation from 23:00 to 04:00, because water has a high
    heat capacity. By day it was similar to vegetation.
- **Worked geometry** [C]:
  - A ground camera 1.5 m high looking at a puddle 10 m away sees it at 81.5° incidence. LWIR R ≈ 0.39, reflecting sky
    at about 8.5° elevation, which in LWIR is close to air temperature.
  - A drone looking down at 30–45° incidence sees R ≈ 0.014–0.020. The puddle then shows its own temperature at
    ε ≈ 0.98.

### "Thermal washout" after prolonged rain

- **No peer-reviewed quantification was found.** The term appears only in vendor and hunting-optics material —
  UNVERIFIED as a named effect.
- **The physical basis is documented:**
  - Raindrops in equilibrium are at the wet-bulb temperature (Kinzer & Gunn 1951) [V, as quoted by Zhang et al. 2019].
  - In convective rain, rainwater was on average 3.8 °C below wet-bulb, 5.1 °C below air temperature and 11.1 °C below
    the soil at 5 cm.
  - The soil at 5 cm can drop by 6.5 °C in 12 minutes [V, Zhang et al. 2019].
  - Rain sensible-heat flux reaches −200 W m⁻² over the ocean [S, Gosnell et al. 1995].
- **Implication:** under sustained rain, wetted surfaces relax toward T_wb (or below) with time constants set by their
  thermal inertia. In LWIR their emissivities converge to about 0.98–0.99, so emissivity contrast is lost as well.

### Dew and frost

- **Dew amounts** [V, Ritter et al. 2019]:
  - 0.14 ± 0.12 mm per night on grassland; 0.13 ± 0.10 mm in passive condensers.
  - Dew forms on 15–95 % of nights in grassland and 25 ± 14 % in forest.
  - The grassland RH threshold is about 75 %.
- **Theoretical maximum** of about 0.8 mm per night [S] (Beysens 1995, bibliographic).
- **Implication** [C]: 0.1 mm = 100 µm, which is more than 5× the LWIR absorption depth. A dewed surface is optically
  water in LWIR (ε ≈ 0.98–0.99, no quartz features) but not in SWIR or NIR.
- **Frost** [A; Salisbury, D'Aria & Wald 1994]: frost and snow emissivity "departs significantly from blackbody
  behavior in the 8–14 µm region." It decreases with particle size and with density (packing or welding), and
  increases with meltwater. I found no number for thin frost films — UNVERIFIED.

---

## 7. Snow and ice

- **Warren (2019)**, citing Hori et al. (2006) [V]:
  - "The emissivity of snow is independent of snow depth and impurities. It is very high for fine-grained snow,
    0.98–0.99 across the infrared 'window' from 8 to 13 µm."
  - "The emissivity of coarse-grained snow becomes sensitive to viewing angle; it is as low as 0.93 for a viewing zenith
    angle of 75° at λ = 13 µm."
  - Emissivity *decreases* with grain size, as Wald's model predicts, which corrected the earlier Dozier–Warren result.
- **Dozier & Warren (1982)** [S]: emissivity 0.985–0.990 for all grain sizes, decreasing above 45°, especially at
  12–14 µm.
- **Hori et al. (2006)**: in-situ measurements of fine dendrite, medium granular and coarse-grained snow, sun crust,
  and smooth bare ice, at six exitance angles [S]. Sun crust ≈ 0.935–0.965 — UNVERIFIED.
- **Hori et al. (2013)** [A]: bulk snow emissivity is modelled as a weighted sum of a specular (Fresnel) component and a
  blackbody component, with a weight per snow type. Bare ice is the specular limit.
- **Salisbury et al. (1994)** [A]: meltwater raises snow emissivity.
- **MWIR snow and ice emissivity: no source found — UNVERIFIED.** Ice absorbs more weakly near 3.7–4 µm, so expect lower
  emissivity and a strong solar component by day.

---

## References

- Arnulf, A., Bricard, J., Curé, E., Véret, C. (1957). Transmission by haze and fog in the spectral region 0.35 to 10
  microns. *J. Opt. Soc. Am.* 47(6), 491–498. https://opg.optica.org/josa/abstract.cfm?uri=josa-47-6-491
- Beier, K., Gemperlein, H. (2004). Simulation of infrared detection range at fog conditions for Enhanced Vision Systems
  in civil aviation. *Aerospace Sci. Technol.* 8, 63–71. https://elib.dlr.de/8112/ — via FLIR Technical Note TN_0001,
  "Seeing through fog and rain with a thermal imaging camera".
  http://www.flirmedia.com/MMC/CVS/Tech_Notes/TN_0001_EN.pdf
- Bergot, T. (2013). Small-scale structure of radiation fog: a large-eddy simulation study. *Q. J. R. Meteorol. Soc.*
  139, 1099–1112. https://doi.org/10.1002/qj.2051 (scale value UNVERIFIED)
- Bijelic, M., Gruber, T., Ritter, W. (2018). Benchmarking image sensors under adverse weather conditions for autonomous
  driving. IEEE IV 2018; arXiv:1912.03238. https://arxiv.org/abs/1912.03238
- Breton, A., Poletti, T., Roux, A., Sorrente, B., da Silva, A., Fade, J., Grillot, F. (2026). Free-space optical
  transmission measurements from 0.532 to 10 µm in real controlled fog. *Opt. Lett.* 51(16), 4729–4732.
  https://doi.org/10.1364/OL.609352
- Chylek, P. (1978). Extinction and liquid water content of fogs and clouds. *J. Atmos. Sci.* 35, 296–300.
- Cuxart, J., Jiménez, M. A. (2012). Deep radiation fog in a wide closed valley: study by numerical modeling and remote
  sensing. *Pure Appl. Geophys.* 169, 911–926. https://doi.org/10.1007/s00024-011-0365-4
- Dozier, J., Warren, S. G. (1982). Effect of viewing angle on the infrared brightness temperature of snow. *Water
  Resour. Res.* 18(5), 1424–1434. https://doi.org/10.1029/WR018i005p01424
- Duthon, P., Colomb, M., Bernardin, F. (2019). Light transmission in fog: the influence of wavelength on the extinction
  coefficient. *Appl. Sci.* 9(14), 2843. https://doi.org/10.3390/app9142843 (full text blocked; not used for numbers)
- Duthon, P., Colomb, M., Bernardin, F. (2020). Fog classification by their droplet size distributions: application to
  the characterization of Cerema's platform. *Atmosphere* 11, 596. https://doi.org/10.3390/atmos11060596
- Duthon, P., Ferreira Fernandes, M., Liandrat, S. (2024). Production of artificial fog in the PAVIN Fog and Rain
  platform: in search of big droplets fog. *Atmos. Clim. Sci.* 14, 42–61. https://doi.org/10.4236/acs.2024.141003
- Eldridge, R. G. (1966). Haze and fog aerosol distributions. *J. Atmos. Sci.* 23(5), 605–613.
  https://doi.org/10.1175/1520-0469(1966)023<0605:HAFAD>2.0.CO;2
- Eldridge, R. G. (1971). The relationship between visibility and liquid water content in fog. *J. Atmos. Sci.* 28(7),
  1183–1186. https://doi.org/10.1175/1520-0469(1971)028<1183:TRBVAL>2.0.CO;2
- Ellrod, G. P. (1995). Advances in the detection and analysis of fog at night using GOES multispectral infrared imagery.
  *Wea. Forecasting* 10, 606–619.
- Fernando, H. J. S., Gultepe, I., Dorman, C., et al. (2021). C-FOG: Life of coastal fog. *Bull. Amer. Meteor. Soc.*
  102(2). https://www.osti.gov/servlets/purl/1779886
- Gosnell, R., Fairall, C. W., Webster, P. J. (1995). The sensible heat of rainfall in the tropical ocean. *J. Geophys.
  Res.* 100(C9), 18437–18442.
- Gultepe, I., Müller, M. D., Boybeyi, Z. (2006). A new visibility parameterization for warm-fog applications in
  numerical weather prediction models. *J. Appl. Meteor. Climatol.* 45, 1469–1480. Equation verified in Gultepe et al.
  (2009), AMS Annual Meeting P1.22. https://www-pm.larc.nasa.gov/icing/pub/conf/Gultepe.etal.eabs.AMS.09.pdf
- Gultepe, I., Tardif, R., Michaelides, S. C., Cermak, J., Bott, A., et al. (2007). Fog research: a review of past
  achievements and future perspectives. *Pure Appl. Geophys.* 164, 1121–1159. https://doi.org/10.1007/s00024-007-0211-x
- Hess, M., Koepke, P., Schult, I. (1998). Optical properties of aerosols and clouds: the software package OPAC. *Bull.
  Amer. Meteor. Soc.* 79(5), 831–844. https://pure.mpg.de/rest/items/item_3365752/component/file_3365754/content
- Hori, M., Aoki, T., Tanikawa, T., Motoyoshi, H., Hachikubo, A., Sugiura, K., et al. (2006). In-situ measured spectral
  directional emissivity of snow and ice in the 8–14 µm atmospheric window. *Remote Sens. Environ.* 100, 486–502.
  https://doi.org/10.1016/j.rse.2005.11.001
- Hori, M., Aoki, T., Tanikawa, T., Hachikubo, A., Sugiura, K., Kuchiki, K., Niwano, M. (2013). Modeling
  angular-dependent spectral emissivity of snow and ice in the thermal infrared atmospheric window. *Appl. Opt.* 52(30),
  7243–7255. https://doi.org/10.1364/AO.52.007243
- Hulley, G. C., Hook, S. J., Baldridge, A. M. (2010). Investigating the effects of soil moisture on thermal infrared
  land surface temperature and emissivity using satellite retrievals and laboratory measurements. *Remote Sens.
  Environ.* 114, 1480–1493.
- Kinzer, G. D., Gunn, R. (1951). The evaporation, temperature and thermal relaxation-time of freely falling waterdrops.
  *J. Meteor.* 8, 71–83.
- Klein, C., Dabas, A. (2014). Relationship between optical extinction and liquid water content in fogs. *Atmos. Meas.
  Tech.* 7, 1277–1287. https://doi.org/10.5194/amt-7-1277-2014
- Kneizys, F. X., Shettle, E. P., Abreu, L. W., Chetwynd, J. H., Anderson, G. P., Gallery, W. O., Selby, J. E. A.,
  Clough, S. A. (1988). Users Guide to LOWTRAN 7. AFGL-TR-88-0177.
  https://archive.org/details/DTIC_ADA206773
  - Fortran source: https://github.com/space-physics/lowtran (`src/lowtran/fortran/lowtran7.f`, BLOCK DATA PRFDTA)
- Koffi, B., Schulz, M., Bréon, F.-M., et al. (2016). Evaluation of the aerosol vertical distribution in global aerosol
  models through comparison against CALIOP measurements: AeroCom phase II results. *J. Geophys. Res. Atmos.* 121,
  7254–7283. https://doi.org/10.1002/2015JD024639
- Lekner, J., Dorf, M. C. (1988). Why some things are darker when wet. *Appl. Opt.* 27(7), 1278–1280.
  https://opg.optica.org/ao/abstract.cfm?uri=ao-27-7-1278
- Liandrat, S., Duthon, P., Bernardin, F., Ben-Daoued, A., Bicard, J.-L. (2022). A review of Cerema PAVIN fog & rain
  platform: from past and back to the future. ITS World Congress 2022.
  https://award-h2020.eu/wp-content/uploads/2023/04/WC22_LIANDRAT_PAVIN_back_to_the_future.pdf
- Lin, D., Katurji, M., Revell, L. E., Khan, B., Sturman, A. (2023). Investigating multiscale meteorological controls
  and impact of soil moisture heterogeneity on radiation fog in complex terrain using semi-idealised simulations.
  *Atmos. Chem. Phys.* 23, 14451–14479. https://doi.org/10.5194/acp-23-14451-2023
- Masuda, K., Takashima, T., Takayama, Y. (1988). Emissivity of pure and sea waters for the model sea surface in the
  infrared window regions. *Remote Sens. Environ.* 24, 313–329.
- Mira, M., Valor, E., Boluda, R., Caselles, V., Coll, C. (2007). Influence of soil water content on the thermal
  infrared emissivity of bare soils: implication for land surface temperature determination. *J. Geophys. Res.* 112,
  F04003. https://doi.org/10.1029/2007JF000749
- Mira, M., Valor, E., Caselles, V., Rubio, E., et al. (2010). Soil moisture effect on thermal infrared (8–13 µm)
  emissivity. *IEEE Trans. Geosci. Remote Sens.* 48(5), 2251–2260.
- Nebuloni, R. (2005). Empirical relationships between extinction coefficient and visibility in fog. *Appl. Opt.*
  44(18), 3795–3804. https://doi.org/10.1364/AO.44.003795
- Nurowska, K., Makuch, P., Markowicz, K. M. (2025). Measurement report: Microphysical and optical characteristics of
  radiation fog – a study using in situ, remote sensing, and balloon techniques. *Atmos. Chem. Phys.* 25, 13493–13525.
  https://doi.org/10.5194/acp-25-13493-2025
- Pinchon, N., Ibn-Khedher, M., Cassignol, O., Nicolas, A., Bernardin, F., Leduc, P., Tarel, J.-P., Brémond, R.,
  Bercier, E., Julien, G. (2016/2018). All-weather vision for automotive safety: which spectral band?
  HAL hal-01406023, https://hal.science/hal-01406023
  - Also published in *Advanced Microsystems for Automotive Applications 2018*, Springer, pp. 3–15.
    https://doi.org/10.1007/978-3-319-99762-9_1
- Pinnick, R. G., Hoihjelle, D. L., Fernandez, G., Stenmark, E. B., Lindberg, J. D., Jennings, S. G., Hoidale, G. B.
  (1978). Vertical structure in atmospheric fog and haze and its effects on visible and infrared extinction. *J. Atmos.
  Sci.* 35, 2020–2032.
  - Also ASL-TR-0010, https://archive.org/details/DTIC_ADA058127
- Pinnick, R. G., Jennings, S. G., Chylek, P., Auvermann, H. J. (1979). Verification of a linear relation between IR
  extinction, absorption and liquid water content of fogs. *J. Atmos. Sci.* 36, 1577–1586.
  - Also DTIC ADA074395, https://archive.org/details/DTIC_ADA074395
- Price, J. (2011). Radiation fog. Part I: observations of stability and drop size distributions. *Boundary-Layer
  Meteorol.* 139, 167–191. https://doi.org/10.1007/s10546-010-9580-2
- Rankin, A., Huertas, A., Matthies, L., Bajracharya, M., Assad, C., Brennan, S., Bellutta, P., Sherwin, G. W. (2011).
  Unmanned ground vehicle perception using thermal infrared cameras. *Proc. SPIE* 8045 (Unmanned Systems Technology
  XIII). https://robotics.jpl.nasa.gov/media/documents/spie-2011-rankin-final.pdf
- Ritter, F., Berkelhammer, M., Beysens, D. (2019). Dew frequency across the US from a network of in situ radiometers.
  *Hydrol. Earth Syst. Sci.* 23, 1179–1197. https://doi.org/10.5194/hess-23-1179-2019
- Rivera Velázquez, J. M., Khoudour, L., Saint Pierre, G., Duthon, P., Liandrat, S., Bernardin, F., Fiss, S., Ivanov,
  I., Peleg, R. (2022). Analysis of thermal imaging performance under extreme foggy conditions: applications to
  autonomous driving. *J. Imaging* 8(11), 306. https://doi.org/10.3390/jimaging8110306
- Roach, W. T., Brown, R., Caughey, S. J., Garland, J. A., Readings, C. J. (1976). The physics of radiation fog: I – a
  field study. *Q. J. R. Meteorol. Soc.* 102, 313–333.
- Salisbury, J. W., D'Aria, D. M. (1992). Emissivity of terrestrial materials in the 8–14 µm atmospheric window.
  *Remote Sens. Environ.* 42, 83–106. https://doi.org/10.1016/0034-4257(92)90092-X
- Salisbury, J. W., D'Aria, D. M., Wald, A. (1994). Measurements of thermal infrared spectral reflectance of frost,
  snow, and ice. *J. Geophys. Res.* 99(B12), 24235–24240. https://ntrs.nasa.gov/citations/19950048358
- Sánchez, J. M., French, A. N., Mira, M., Hunsaker, D. J., Thorp, K. R., Valor, E., Caselles, V. (2011). Thermal
  infrared emissivity dependence on soil moisture in field conditions. *IEEE Trans. Geosci. Remote Sens.* 49(11),
  4652–4659. https://pages.uv.es/gtt/NOVA/CAS/articulos/Sanchez_et_al_2011.pdf
- Segelstein, D. J. (1981). The complex refractive index of water. M.S. thesis, Univ. Missouri–Kansas City (repo
  `data/nk/water.csv`).
- Shettle, E. P., Fenn, R. W. (1979). Models for the aerosols of the lower atmosphere and the effects of humidity
  variations on their optical properties. AFGL-TR-79-0214, Environmental Research Papers No. 676.
  https://web.gps.caltech.edu/~vijay/Papers/Aerosol/SF79-Aerosol-Models-part1of4.PDF (parts 1–4)
- Tampieri, F., Tomasi, C. (1976). Size distribution models of fog and cloud droplets in terms of the modified gamma
  function. *Tellus* 28(4), 333–347. https://doi.org/10.3402/tellusa.v28i4.10300
- Toledo, F., Haeffelin, M., Wærsted, E., Dupont, J.-C. (2021). A new conceptual model for adiabatic fog. *Atmos. Chem.
  Phys.* 21, 13099–13117. https://doi.org/10.5194/acp-21-13099-2021
- Twomey, S. A., Bohren, C. F., Mergenthaler, J. L. (1986). Reflectance and albedo differences between wet and dry
  surfaces. *Appl. Opt.* 25(3), 431–437. https://doi.org/10.1364/AO.25.000431
- Wærsted, E. G., Haeffelin, M., Dupont, J.-C., Delanoë, J., Dubuisson, P. (2017). Radiation in fog: quantification of
  the impact on fog liquid water based on ground-based remote sensing. *Atmos. Chem. Phys.* 17, 10811–10835.
  https://doi.org/10.5194/acp-17-10811-2017
- Warren, S. G. (1982). Optical properties of snow. *Rev. Geophys.* 20(1), 67–89.
  https://doi.org/10.1029/RG020i001p00067
- Warren, S. G. (2019). Optical properties of ice and snow. *Phil. Trans. R. Soc. A* 377, 20180161.
  https://doi.org/10.1098/rsta.2018.0161
- WMO (2019). *Aerodrome Reports and Forecasts: A Users' Handbook to the Codes*, WMO-No. 782.
  https://www.condor-va.de/training/metar_taf_wmo.pdf (2019 edition copy)
- WMO International Cloud Atlas: "Fog compared with mist". https://cloudatlas.wmo.int/en/fog-compared-with-mist.html
- Zhang, S., Meurey, C., Calvet, J.-C. (2019). Identification of soil-cooling rains in southern France from soil
  temperature and soil moisture observations. *Atmos. Chem. Phys.* 19, 5005–5020.
  https://doi.org/10.5194/acp-19-5005-2019
- Zhong, X., Zhao, L., Zhang, X., Yan, J., Ren, P. (2022). Investigating the effects of surface moisture content on
  thermal infrared emissivity of urban underlying surfaces. *Constr. Build. Mater.* 327, 127023.
  https://doi.org/10.1016/j.conbuildmat.2022.127023

## UNVERIFIED list

1. **Fog microphysics values and fits:**
   - Tampieri & Tomasi (1976) parameters.
   - Eldridge (1966) LWC–V relation coefficients.
   - Kunkel (1984), β = 144.7·LWC^0.88.
   - Gultepe (2007) "LWC 0.01–0.4 g m⁻³" (secondary).
   - Pinnick et al. (1979) power-law coefficients (illegible OCR).
   - Pilié mature-fog concentration (illegible OCR).
2. **Fog depth and dynamics:**
   - Fog-top cooling rate of 1–4 K h⁻¹.
   - Bergot (2013) ~500 m roll wavelength.
   - Ebro (~300 m), Swiss Plateau and tule fog depths.
3. **Boundary layer:** typical boundary-layer heights (textbook values only).
4. **Haze thresholds:** haze RH thresholds of 80 % (WMO-No. 266 / No. 8) and 95 % (UK Met Office).
5. **Fog emissivity:** water-cloud/fog emissivity at 3.7–3.9 µm.
6. **"Thermal washout" as a named, quantified effect.**
7. **Dew:** the theoretical maximum dew of 0.8 mm per night.
8. **Frost:** frost-film emissivity numbers.
9. **Snow and ice:**
   - Hori (2006) sun-crust 0.935–0.965 and bare-ice values.
   - MWIR snow and ice emissivity.
10. **Mira (2007) abstract numbers:** read only from an index snippet, not the publisher page.
