# Rain and snow along the line of sight: literature notes for the irsim physics spec

Scope: falling precipitation between the camera and the scene, in LWIR (8–14 µm), MWIR (3–5 µm),
SWIR (0.9–1.7 µm), NIR (0.7–1.0 µm) and visible RGB. Compiled 2026-10-01.

How the evidence is marked:
- **[n]** is a citation to the References list at the end. Each reference says what was actually read
  (full text, abstract only, or a secondary source).
- **DERIVED** means I calculated it in this session. The scripts are in `scripts/` beside this note: `mie_rain.py`,
  `fresnel_eps.py`, `drop_relax.py`. Those numbers are reproducible but not peer-reviewed.
- **UNVERIFIED** means I could not confirm the number from a primary source in this session.

Water and ice optical constants come from the refractiveindex.info database copies of Hale & Querry
1973 [15], Segelstein 1981 [16], Warren & Brandt 2008 [17] and Rowe et al. 2020 [18].

---

## 1. Rain extinction vs wavelength; absorption vs scattering; Q_ext = 2 or about 1 for an imager

### 1.1 Measurements and classical calculations

**Chu & Hogg 1968 [1] (full text read).** Laser transmission at 0.63, 3.5 and 10.6 µm over a 2.6 km path.
- For drops of about 1 mm radius, Q_ext → 2. At 3.5 and 10.6 µm, Q_abs → 1. At 0.63 µm, Q_abs ≈ 0.
- "Notice that the total extinction is the same for the visible and near-infrared wavelengths. Half of
  the extinction coefficient at 3.5 µ or 10.6 µ is contributed by the absorptive loss." Elsewhere: "At
  least half of the attenuation resulting from rain at 3.5 µ and 10.6 µ is absorptive loss while at
  0.63 µ it is caused almost entirely by scattering."
- A worked number: 1 g/m³ of liquid water (about 25 mm/h) with drops of about 1 mm radius gives an
  extinction of about 6.5 dB/km, the same in the optical and near-IR. The specific extinction is
  0.0065 (dB/km)/(mg/m³).
- The measured attenuation at 3.5 µm was "consistently about 20 per cent greater" than at 0.63 µm. In a
  heavy storm (10–60 mm/h), attenuation at 10.6 and 3.5 µm was "consistently greater than that at
  0.63 µ". The authors attribute the lower visible value to forward scattering into the receiver,
  which is strongest when the rain is near the receiver.
- Receiver acceptance angles were about 10 arcmin (≈2.9 mrad) at 0.63 µm and about 1 arcmin
  (≈0.29 mrad) at 3.5 µm. A 600 cm² receiver did not measure significantly less attenuation than a
  5 cm one.
- Refractive indices used: n = 1.33 at 0.63 µm; 1.42 − 0.013i at 3.5 µm; 1.18 − 0.08i at 10.6 µm.

**Other measurements.**
- Rensch & Long 1970 [2] (abstract only): computed extinction for 0.34–10.6 µm and confirmed it with
  outdoor transmission at 10.6 and 0.63 µm.
- Chimelis 1982 [3] (abstract only): 10.6 µm extinction in rain of 1.7–17 dB/km.
- ITU-R P.1814-1 [4] Table 1 says rain attenuation goes as λ⁰ ("optical scattering"), for drop radii
  0.1–4 mm. It states that rain attenuation "is flat from the visible to the near-IR region and beyond".
- Vasseur & Gibbins 1996 [40] (abstract only): geometric optics gives Q_ext = 2 and a
  wavelength-independent coefficient, but scattered power reaching the receiver lowers the *apparent*
  extinction. They model this with Fraunhofer diffraction plus Twersky multiple scattering.

### 1.2 Mie calculation over a Marshall–Palmer DSD (DERIVED)

**Method.**
- Exact Mie theory, BHMIE-style recurrences (Bohren & Huffman [44]), Segelstein 1981 water constants.
- Drop diameters 0.1–6 mm, as spheres.
- Marshall–Palmer distribution N(D) = 8000·exp(−ΛD) m⁻³ mm⁻¹, with Λ = 4.1·R^−0.21 mm⁻¹ [11].
- Code check: Q_ext(m = 1.5, x = 10) = 2.8820 matches the commonly quoted benchmark (I did not
  re-confirm that benchmark against a source this session). Q_ext(m = 1.33, x = 1000) = 2.016.

**Results** (R in mm/h; σ_ext in km⁻¹; "abs frac" = σ_abs/σ_ext; g is the Mie asymmetry factor):

| λ (µm) | σ_ext at R=1 | R=10 | R=25 | R=100 | abs frac (R=10) | abs frac range (R 1→100) | g |
|---|---|---|---|---|---|---|---|
| 0.55 | 0.366 | 1.560 | 2.777 | 6.62 | 0.000 | 0.000 | 0.88 |
| 0.85 | 0.366 | 1.562 | 2.780 | 6.63 | 0.002 | 0.001–0.004 | 0.89 |
| 1.06 | 0.366 | 1.563 | 2.781 | 6.63 | 0.008 | 0.005–0.012 | 0.89 |
| 1.55 | 0.366 | 1.564 | 2.784 | 6.64 | 0.289 | 0.22–0.36 | 0.92–0.95 |
| 2.2 | 0.368 | 1.568 | 2.790 | 6.65 | 0.367 | 0.30–0.42 | 0.94–0.96 |
| 3.5–4.6 | 0.370 | 1.574 | 2.80 | 6.67 | 0.46–0.47 | 0.46–0.47 | 0.97 |
| 8–12 | 0.373–0.375 | 1.585–1.590 | 2.82 | 6.70–6.71 | 0.47–0.48 | 0.47–0.49 | 0.98–0.99 |

**Conclusions from the table.**
- σ_ext varies by less than 2% between 0.55 and 12 µm. This confirms the claim that rain extinction is
  wavelength-independent.
- The geometric-optics closed form σ = π·N0/Λ³ = 0.365·R^0.63 km⁻¹ (1.584·R^0.63 dB/km) reproduces the
  Mie result within 2% for 1–100 mm/h.
- Absorption share of extinction: about 0% in the visible, about 1% at 1 µm, 22–36% at 1.55 µm,
  about 46% in MWIR and about 48% in LWIR.
- So in LWIR and MWIR, extinction is about one diffraction unit (scattered, Q ≈ 1) plus about one
  absorption unit (Q ≈ 0.93–0.97, see §3).

**Why SWIR is in between.** At 1.45–1.55 µm the absorption length of water (0.3–0.9 mm, §3) is
comparable to the drop size, so larger drops absorb more. At 1.55 µm, Q_abs is 0.36 for D = 0.5 mm and
0.80 for D = 2 mm. SWIR is therefore partly absorbing and the share rises with rain rate. The 0.97, 1.2
and 1.45 µm water bands make this strongly wavelength-dependent inside the SWIR band.

### 1.3 Does the forward-diffracted light stay inside a pixel? (DERIVED)

**Why it matters.** The diffraction half of Q_ext = 2 goes into an Airy-like lobe whose first zero is at
θ₁ ≈ 1.22 λ/D. For D = 1 mm that is:

| λ | θ₁ |
|---|---|
| 0.55 µm | 0.67 mrad |
| 1.55 µm | 1.9 mrad |
| 4 µm | 4.9 mrad |
| 10 µm | 12 mrad |

**What I computed.** The fraction of diffracted power inside a cone of half-angle θ, using the Airy
encircled energy 1 − J0²(x sinθ) − J1²(x sinθ), weighted by cross-section over the Marshall–Palmer
distribution at R = 10 mm/h:

| half-angle θ | 0.55 µm | 1.0 µm | 1.55 µm | 4 µm | 10 µm |
|---|---|---|---|---|---|
| 0.25 mrad | 0.44 | 0.21 | 0.10 | 0.02 | 0.00 |
| 0.5 mrad (≈1 mrad IFOV) | 0.70 | 0.48 | 0.30 | 0.07 | 0.01 |
| 1.0 mrad (≈2 mrad IFOV) | 0.86 | 0.73 | 0.59 | 0.21 | 0.04 |
| 5 mrad | 0.97 | 0.95 | 0.92 | 0.79 | 0.48 |

At R = 2 mm/h the fractions are smaller, because drops are smaller. At R = 50 mm/h they are larger. The
full tables are in `mie_out.txt`.

**Interpretation for an imaging sensor.** This is my synthesis. It is consistent with the
aerosol-MTF literature [41][42][43], but I have not seen it written out for rain.
- Light diffracted by angle θ lands θ away from its true image point, whatever the drop's range.
- **Visible/NIR, 0.5–2 mrad pixels.** 50–85% of the diffracted light stays in the pixel. The
  effective extinction for pixel-scale detail is roughly Q ≈ 1.1–1.5. The remaining loss is
  refraction and reflection over tens of degrees (g ≈ 0.88 overall), which is extinction plus
  "airlight".
- **LWIR.** Under 5% stays in a 1–2 mrad pixel. The diffracted light forms a halo of about 5–25 mrad
  (5–25 pixels). Pixel-scale contrast and small targets are attenuated with **Q = 2**. The mean level
  of large uniform regions is attenuated by roughly **absorption only** (Q ≈ 1), because the halo light
  is returned from the neighbours.
- **The right model is a spatial-frequency-dependent transmittance, not a single Q.** Classical aerosol
  MTF, attributed to Lutomirski 1978 [43]: T(f) = exp(−A·z − S·z·(f/f_c)²) for f ≤ f_c and
  exp(−(A+S)·z) for f > f_c, with f_c ≈ a/λ cycles/rad, A the absorption coefficient, S the scattering
  coefficient and a the particle radius. That exact form is UNVERIFIED (secondary summary only).
  Sadot & Kopeika 1994 [42] (abstract) state that absorption effects are also spatial-frequency
  dependent and "very significant" in thermal imaging.
- For raindrops with a = 0.5–1 mm: f_c is about 1000–2000 cycles/rad at 0.55 µm (period 0.5–1 mrad)
  and about 50–100 cycles/rad at 10 µm (period 10–20 mrad).

**Empirical support.**
- Chu & Hogg [1] (above): the visible attenuation is lower, depending on receiver geometry and on where
  the rain is.
- Korai, Luini & Nebuloni 2018 [6], the basis of P.1814-1: the forward and multiple-scattering gain
  G_ms was computed for a receiver with **90° field of view**, 10 cm radius and a collimated 2.5 cm beam.
  It is an open-detector correction and **does not transfer to a 1 mrad pixel**.
- Implied effective Q from the ITU visibility code (P.1817-1 [5]) (DERIVED): taking σ = 3.0/V for the
  5% threshold and comparing with Marshall–Palmer Q = 2, visual observation in rain implies:

  | R (mm/h) | Implied Q_eff |
  |---|---|
  | 2.5 | 1.56 |
  | 12.5 | 1.20 |
  | 25 | 1.14 |
  | 100 | 1.17 |

  This is consistent with the eye (about 0.3 mrad acuity) keeping most of the visible diffraction lobe.

---

## 2. Empirical rain laws, fit wavelengths, rain rate vs visibility

| Law | Form (A in dB/km, R in mm/h) | Wavelength / basis | Source status |
|---|---|---|---|
| Carbonneau | A = 1.076·R^0.67 | Fit wavelength **UNVERIFIED** (FSO context; original SPIE 3232 paper [7] not accessed) | Quoted as "the CARBONNEAU relation" by Al Naboulsi et al. 2005 [8] (read) |
| "Japan" | A = 1.58·R^0.63 | UNVERIFIED (search snippet only) | Equals Marshall–Palmer with Q = 2 exactly (DERIVED, §1.2) |
| LOWTRAN 7 (λ ≤ 28.6 µm) | σ = π·N0/Λ³ = 0.365·R^0.63 km⁻¹ (= 1.584·R^0.63 dB/km) | Wavelength-independent; Marshall–Palmer, Q = 2 | Read in the LOWTRAN 7 Fortran source [12] |
| ITU-R P.1814-1 (2025) | A = k·R^α. µ = −2: 2.2838, 0.4050; µ = −1: 1.5921, 0.5506; **µ = 0: 1.2924, 0.6436**; µ = 1: 1.1394, 0.7057; µ = 2: 1.0505, 0.7497 | "Classical scattering theory" over a gamma DSD with shape µ. Stated for 0.78–0.85 and 1.52–1.60 µm; the authors computed at λ = 1.55 µm [6] | Full text read [4][6]. Plus a path reduction factor F = 1/(1 + L(R − 6.2)/2623) and a multiple-scattering gain G_ms = a·L^b (90° FOV) |
| ITU-R P.1817-1 (2012) | A = k·R^α | Coefficients not printed in the text, only Fig. 9 | Full text read [5] |
| Atlas 1953 | σ ≈ 0.25·R^0.63 km⁻¹ (visible, "Bergeron-process" rain) | Visible | **UNVERIFIED**: from a secondary summary; AMS PDF blocked. Montero-Martínez & García-García 2025 [10] quote the literature range a = 0.15–0.38, b = 0.55–0.70, and measured σ = 0.11·R^0.88 and 0.12·R^0.81 km⁻¹ at 830 nm |

**Laws fitted in 8–12 µm or 3–5 µm.** None found that I could read.
- Chu & Hogg [1] drew least-squares lines at 0.63, 3.5 and 10.6 µm for 10–60 mm/h, but the coefficients
  are only in the figures (OCR text only).
- Chimelis 1982 [3] (10.6 µm) and Winchester, Gimmestad & Lee 1982 [32] (3–5 and 8–12 µm vs rain rate)
  are abstract-only. Their coefficients are **UNVERIFIED**.

**Side-by-side numbers (dB/km).**

| R (mm/h) | MP Q=2 | Carbonneau | P.1814 µ=0 | P.1814 µ=2 | MP Q=1 |
|---|---|---|---|---|---|
| 2.5 | 2.82 | 1.99 | 2.33 | 2.09 | 1.41 |
| 10 | 6.76 | 5.03 | 5.69 | 5.90 | 3.38 |
| 25 | 12.0 | 9.30 | 10.3 | 11.7 | 6.0 |
| 100 | 28.8 | 23.5 | 25.0 | 33.2 | 14.4 |

**Rain rate vs visibility, ITU-R P.1817-1 [5] "International visibility code" (read).**

| Condition | R (mm/h) | Visibility (m) | Attenuation (dB/km) |
|---|---|---|---|
| Storm | 100 | 770 | 18.3 |
| Strong rain | 25 | 1900 | 6.9 |
| Average rain | 12.5 | 2800 | 4.6 |
| Light rain | 2.5 | 5900 | 2 |
| Drizzle | 0.25 | 18100 | 0.6 |

- The attenuation column is consistent with γ = 13/V dB/km, which is the MOR (5% threshold).
- P.1814-1 [4] gives K = 13 for instrumental MOR, 11.3 for daytime visual observation and 9.6 for
  night-time lights, plus the conversion V(2%) = 1.31·V(5%).
- P.1817-1 [5] defines visibility at 550 nm with 3.912/γ (2% contrast).

---

## 3. Raindrop temperature and LWIR emissivity

### 3.1 Drop temperature

**Steady state.** For a ventilated, evaporating drop, the steady temperature solves the
heat-and-vapour balance (Loftus & Wordsworth 2021 [20], eqs. 10–11 and 17; after Rogers & Yau). When the
heat and vapour ventilation factors are equal, this is the psychrometric (wet-bulb) temperature.
- Zhang, Meurey & Calvet 2019 [22] take rain temperature equal to wet-bulb temperature, "valid for
  raindrops in thermal equilibrium with the ambient air (Kinzer and Gunn, 1951)" [19].
- They also cite Byers et al. 1949 [21]: measured rain temperatures were "close to air temperature,
  close to wet-bulb temperature, and lower than air temperature by as much as 10 °C", the last case
  attributed to melting hail.
- LOWTRAN 7 sets RH = 100% in rain layers and uses the layer air temperature as the rain temperature
  (source [12]). With RH = 100%, wet-bulb equals air temperature.
- MODTRAN 6 defaults to 100% RH at rain layer boundaries (CHUMID) [13].

**Relaxation time (DERIVED,** `drop_relax.py`**).**
- Linearised: τ = ρ_w·c_w·r² / (3·[k_a·f_h + L·D_v·f_v·dρ_vs/dT]).
- Ventilation f = 0.78 + 0.308·Re^0.5·X^(1/3), with X = Pr or Sc (Beard & Pruppacher 1971 /
  Pruppacher & Rasmussen 1979, as given in [20]).
- Fall speed from Atlas et al. 1973. Conditions 10 °C, 1013 hPa.
- Loftus & Wordsworth's simpler estimate, τ ≈ r²ρc/(3K_air), ignores ventilation and latent heat and is
  slower.

| D (mm) | v (m/s) | τ (s) | e-folding distance (m) | 3τ distance (95%) (m) |
|---|---|---|---|---|
| 0.5 | 2.0 | 0.47 | 1.0 | 2.9 |
| 1 | 4.0 | 1.1 | 4.4 | 13 |
| 2 | 6.6 | 2.6 | 17 | 51 |
| 3 | 8.0 | 4.4 | 35 | 105 |
| 4 | 8.7 | 6.6 | 57 | 172 |
| 5 | 9.1 | 9.0 | 83 | 248 |

So drops at camera height are at the local wet-bulb temperature to within a few % of (T_a − T_w) for
D ≤ 2 mm. Large drops lag by tens to hundreds of metres of fall. Kinzer & Gunn's own measured relaxation
times are **UNVERIFIED** (paper not accessible).

### 3.2 Water optical constants and absorption length δ = λ/(4πk)

| λ (µm) | k, Hale & Querry 1973 (25 °C) | k, Segelstein 1981 | δ (Segelstein) |
|---|---|---|---|
| 1.0 | 2.89e-6 | 3.0e-6 | 27 mm |
| 1.45 | — | 3.63e-4 | 0.32 mm |
| 1.5 | (not tabulated; 1.38e-4 at 1.4 µm) | 2.24e-4 | 0.53 mm |
| 1.55 | — | 1.34e-4 | 0.92 mm |
| 3.5 | 9.4e-3 | 9.36e-3 | 30 µm |
| 4.0 | 4.6e-3 | 4.61e-3 | 69 µm |
| 10.0 | **5.08e-2** | 5.08e-2 | **15.7 µm** |
| 10.6 | 7.1e-2 (interp.) | 7.1e-2 | 11.8 µm |
| 12.0 | 1.99e-1 | 2.0e-1 | 4.8 µm |

- Supercooled water at 0 °C (Rowe et al. 2020 [18]): k(10 µm) = 0.0531, δ = 15.0 µm.
- Ice at −7 °C (Warren & Brandt 2008 [17]): k(10 µm) = 0.050 (δ = 15.9 µm); k(11 µm) = 0.248 (δ = 3.5 µm);
  k(12 µm) = 0.41 (δ = 2.3 µm); k(4 µm) = 0.0119 (δ = 27 µm).
- Conclusion: water is optically thick in LWIR. A 2 mm drop is about 130 absorption lengths at 10 µm.
  This confirms the 10–20 µm figure in the brief.

### 3.3 Raindrop emissivity

By Kirchhoff's law for a particle, emission efficiency equals absorption efficiency (Bohren & Huffman
[44]). Per unit geometric cross-section, a drop's emissivity is its Q_abs (DERIVED, Mie, §1.2):

| Band | Q_abs (= disc emissivity) |
|---|---|
| 10 µm | 0.97–1.00 (D = 0.5 mm: 0.996; D = 2 mm: 0.967) |
| 8 µm | 0.95–0.98 |
| 3.5–4.6 µm | 0.93–0.96 |
| 1.55 µm | 0.36 (D = 0.5 mm) to 0.80 (D = 2 mm) |
| 1.06 µm | 0.007–0.026 |

Cross-check: smooth-water Fresnel hemispherical emissivity (DERIVED, `fresnel_eps.py`) is 0.955 at
10 µm (normal incidence 0.992), 0.94 at 8 µm and 0.93 at 4 µm.

---

## 4. Fall speeds

### 4.1 Rain

- **Gunn & Kinzer 1949 [23].** Laboratory terminal velocities for D = 0.078–5.8 mm at 20 °C and
  1013 hPa (range confirmed via [25] and search). Their maximum of about 9.2 m/s is **UNVERIFIED**
  (paper not read).
- **Atlas, Srivastava & Sekhon 1973 [24]**, as quoted by Lane et al. [25]:
  **v(D) = 9.65 − 10.3·exp(−0.6·D) m/s, D in mm.** Lane et al. print "1030", an evident typo for 10.3.
  - Fit to Gunn & Kinzer. It gives v < 0 for D < 0.11 mm and should not be used below about 0.5 mm.
  - Air-density scaling (ρ0/ρ)^0.4 is commonly applied (Foote & du Toit 1969); **UNVERIFIED**.
- **Atlas & Ulbrich 1977** power law: v = 3.78·D^0.67 m/s, D in mm [25].
- Raindrops reach 99% of terminal velocity within 1% of their fall distance [20].

### 4.2 Snow and graupel

Locatelli & Hobbs 1974 [26], Table 1 (full text read). V in m/s; D = maximum dimension in mm. Measured at
750–1500 m ASL.

| Particle | V(D) | D range (mm) |
|---|---|---|
| Aggregates of unrimed radiating assemblages of dendrites, or dendrites | 0.8·D^0.16 | 2–10 |
| Aggregates of densely rimed dendrites / radiating assemblages | 0.79·D^0.27 | 2–12 |
| Aggregates of unrimed radiating assemblages of plates, side planes, bullets and columns | 0.69·D^0.41 | 0.2–3 |
| Aggregates of unrimed side planes | 0.82·D^0.12 | 0.5–4 |
| Unrimed side planes | 0.81·D^0.99 | 0.4–1.2 |
| Densely rimed dendrites | 0.62·D^0.33 | 1.8–4 |
| Densely rimed radiating assemblages of dendrites | 1.1·D^0.12 | 0.8–2.8 |
| Densely rimed columns | 1.1·L^0.56 | 0.8–2 |
| Graupel-like snow, lump type | 1.1·D^0.28 | 0.5–2.2 |
| Graupel-like snow, hexagonal type | 0.86·D^0.25 | 0.8–2.8 |
| Hexagonal graupel | 1.1·D^0.57 | 0.8–3.2 |
| Conical graupel | 1.2·D^0.65 | 0.8–3 |
| Lump graupel, density 0.05–0.1 g/cm³ | 1.16·D^0.46 | 0.5–2 |
| Lump graupel, density 0.1–0.2 g/cm³ | 1.3·D^0.66 | 0.5–3 |
| Lump graupel, density 0.2–0.45 g/cm³ | 1.5·D^0.37 | 0.5–1 |

Measured speeds span 0.5–3.0 m/s. Rasmussen et al. 1999 [27] use 1 m/s for dry aggregates and 2 m/s for
wet or rimed ones ("typically fall twice as fast", after Magono & Nakamura 1965 and Zikmunda & Vali 1972).

---

## 5. Snow: extinction, visibility, temperature, emissivity

### 5.1 Empirical snow law (exact form confirmed)

ITU-R P.1817-1 [5], Table 2 (read): **γ_snow = a·S^b dB/km**, S = "snowfall rate (mm/h)", λ in nm.

| Snow | a | b |
|---|---|---|
| Wet | 0.0001023·λ_nm + 3.7855466 | 0.72 |
| Dry | 0.0000542·λ_nm + 5.4958776 | 1.38 |

- These are the user's numbers to all printed digits.
- The same table appears in Al Naboulsi, Sizun & de Fornel 2005 [8], with no original source given.
- Attribution to "Kim" or "Nebuloni" is **UNVERIFIED**. Nebuloni co-authored the *rain* model [6], not
  this snow law.
- **UNVERIFIED and important:** whether S is liquid-equivalent rate or depth rate, and whether the
  linear λ term is valid beyond 1.55 µm (P.1817 plots only 1.55 µm).

What the formula implies (DERIVED):
- Dry snow at S = 1 mm/h: 5.53 (550 nm), 5.58 (1550 nm), 5.71 (4 µm), 6.04 (10 µm) dB/km.
  LWIR/visible = 1.09.
- Wet snow at S = 1 mm/h: 3.84 (550 nm), 4.81 (10 µm) dB/km. LWIR/visible = 1.25.

### 5.2 Rasmussen et al. 1999 [27] (full text read)

- Koschmieder: Vis = 3.912/σ. The extinction uses Q = 2 ("the factor of 2 ... is due to diffraction").
- Exponential DSD gives Vis = 1.245·Λ²/N_t (cgs).
- With ρ_s·D = C3 (density inversely proportional to size): **S = 1.3·C3·V_t / Vis** (cgs: S in cm/s,
  Vis in cm, V_t in cm/s). Dry snow: C3 = 0.017 g/cm², V_t = 100 cm/s. Wet or rimed: C3 = 0.072 g/cm²,
  V_t = 200 cm/s.
- The paper states that at 2 mm/h liquid-equivalent, visibility ranges from 0.3 km (dry) to 2.6 km (wet).
  Evaluating eq. 13 myself gives 0.40 km and 3.4 km, so their figure presumably used slightly different
  inputs; this discrepancy is unresolved.
- Observed scatter: a factor of 3–10 in visibility for a given rate. At 2 mm/h, observed visibility was
  0.3–3.0 km. At 1 km visibility, the rate was 0.8–3 mm/h.
- Night-time visibility can be more than 2× the daytime value for the same σ.
- NWS intensity categories: light snow at ≥ 1.0 km, moderate at 0.5–1.0 km, heavy at ≤ 0.5 km.
- Implied σ (DERIVED): dry snow 4.9 km⁻¹ (21 dB/km) at 1 mm/h; wet snow 0.58 km⁻¹ (2.5 dB/km).
  The P.1817 formula sits between these (5.5 and 3.8 dB/km), so the two are not mutually consistent
  without knowing what S means.

### 5.3 Band comparisons in falling snow

**Seagraves 1986 [28]**, Table I (from the Optica abstract page). IR extinction is 1.05–1.43× visible:

| Investigators | Relation |
|---|---|
| Bisyarin et al. | σ(10.6) = 1.4·σ(0.63); 1.38·σ(0.63) |
| Duncan; Sola & Bergmann | σ(3–5) = 1.05·σ(0.55)^1.05; σ(8–12) = 1.30·σ(0.55)^0.993 |
| Gimmestad & Lee | σ(8–12) = 1.25·σ(vis)^0.901 |
| Lacombe & O'Brien | σ(10.37) = 1.42·σ(0.55); σ(3.0) = 1.43·σ(0.55); σ(1.06) = 1.06·σ(0.55) |
| Seagraves & Ebersole | σ(3.0) = 1.21·σ(0.55); σ(10.5) = 1.18·σ(0.55) |

The explanation offered is the sharply forward-peaked diffraction lobe. The longer the wavelength, the
wider the lobe and the less of it a receiver collects. This is the same mechanism as §1.3.

**Other studies.**
- Chu & Hogg [1]: 10.6 µm "often appears to suffer greater attenuation in snow ... however, no definite
  wavelength dependence is consistently observed". Snows above about 12.5 mm/h (accumulation) drove all
  three wavelengths below noise over 2.6 km. Snow forward-scatters less than rain: equal beam
  broadening was seen at 5.4 dB/km for rain and 15.6 dB/km for snow.
- Rahm et al. 2023 [30] (FOI, abstract): "In snow, the transmission is higher in the SWIR band as an
  effect of forward scattering", compared with LWIR.
- Border & McEwan 2020 [29] (DSTL, abstract): transmission spectra SWIR–LWIR in Älvdalen snow,
  compared with MODTRAN. Only weak positive correlations of extinction with precipitation rate, RH and
  wind. Numbers **UNVERIFIED** (paywalled).
- Winchester & Gimmestad 1982 [31] (KRC 82-10): 0.6328, 1.06 and 8–12 µm in fog, rain and snow.
  Abstract only.

### 5.4 Snow temperature

- Dry falling snow is at or below 0 °C.
- In sub-saturated air above 0 °C, flakes stay frozen because sublimation cools them (Matsuo & Sasyo
  1981 [34]).
- Heymsfield et al. 2021 [34] (via search): an ice-bulb temperature of 0 °C, together with air
  temperature and pressure, is a good first-order predictor of snow survival.
- Spec implication (DERIVED from those): T_flake ≈ min(T_ice-bulb, 0 °C). Melting flakes sit at 0 °C.

### 5.5 Ice and snow LWIR emissivity

- Hori et al. 2006 [33], via secondary sources: snow emissivity at 8–14 µm is **> 0.98** for fine,
  medium and most coarse snow, and **0.90–0.98** for some sun-crust snow (specular). It falls at view
  angles > 45°, especially at 12–14 µm.
- Smooth ice Fresnel (DERIVED): normal emissivity 0.992 at 10 µm, 0.954 at 12 µm. Hemispherical
  emissivity 0.955 at 10 µm, 0.89 at 12 µm.
- Single falling crystals may be semi-transparent at 8–10.5 µm: the ice absorption length there is
  16–19 µm, comparable to the thickness of plates and dendrite arms. They are opaque at 11–13 µm
  (δ = 2–3.5 µm). Single-flake emissivity is therefore < 1 and size-dependent at the short end of LWIR.
  This is DERIVED; I found no measurement.

---

## 6. Close-range appearance; drops on the window; bolometer smear

**Visible: Garg & Nayar 2007 [35] (read).**
- A drop refracts a 165° field of view toward the camera. The refracted light is attenuated by only 6%.
  A stationary drop is "much brighter than its background" and its brightness depends only weakly on
  the background.
- A drop's dwell on a pixel is τ < 1.18 ms, much shorter than a typical 30 ms exposure. A streak pixel's
  intensity is therefore a linear mix: I = ∫₀^τ E_r dt + ∫_τ^T E_b dt.

**LWIR difference.** In LWIR the drop is *opaque* (§3). It is a near-blackbody sphere at T_wb (ε ≈ 0.96)
reflecting about 4% of its surroundings, not a lens.

**Drops on the window: Jobert et al. 2024 [36] (Lynred / Saint-Gobain, read).**
- Setup: LWIR camera with a 12 µm pitch, EFL 13.6 mm, f/1.0, behind a windshield IR window.
- Statement: "In the thermal range, water is considered fully opaque ... the droplet only reduces the
  amount of thermal radiation that is captured ... as a function of the droplets cross-section".
- Their model: T(Ø_d) ≈ 1 − Ø_d²·sinγ / (EFL/f#)².
- Results: responsivity drops to about 90% for a droplet of about 6 mm. Thermal SFR showed no
  degradation and detection was unaffected. The visible SFR degraded strongly from about 2 mm.
- Their test (two blackbodies) cancels the droplet's own emission. A simulator should add it:
  an out-of-focus uniform veil of ε_water·B(T_drop) times the occluded aperture fraction. That
  addition is DERIVED, not measured.
- Vendor material: Ophir AquaShieldIR [47] claims a contact angle > 110° and "improved image quality",
  with no numbers.

**Uncooled bolometer smear: Oswald-Tranta 2018 [37] (read).**
- Microbolometer signal decays exponentially, "typical value of about 10 ms" (10–15 ms elsewhere in the
  paper). Cooled InSb with a 1–1.5 ms integration time showed negligible blur.
- Point-spread function along the motion: h(x) = (1/τ_im)·exp(−x/τ_im), with τ_im = τ_cam·v·r pixels.
- Peak apparent contrast of an object of size d: ΔT_blurred = ΔT_true·(1 − exp(−m)), m = d/(τ_cam·v).
  Measured check: at m = 0.67 the ratio was 0.48 predicted and 0.55 measured.

**Applied to precipitation (DERIVED, τ = 10 ms).**

| Object | m | 1 − e^−m | Object-space streak length v·τ |
|---|---|---|---|
| Raindrop 1 mm, 4 m/s | 0.025 | 0.025 | 40 mm |
| Raindrop 2 mm, 6.55 m/s | 0.031 | 0.030 | 66 mm |
| Raindrop 4 mm | 0.046 | 0.045 | 87 mm |
| Graupel 2 mm, 1.5 m/s | 0.13 | 0.13 | 15 mm |
| Wet snowflake 5 mm, 2 m/s | 0.25 | 0.22 | 20 mm |
| Dry snowflake 5 mm, 1 m/s | 0.5 | 0.39 | 10 mm |

Consequences:
- At 3 m, a 2 mm drop subtends 0.67 mrad, moves at 2.2 rad/s and leaves a 22 mrad tail. A drop–road
  contrast of 5 K gives a peak of about 0.1 K (near NETD). A drop seen against a cold sky (30–50 K
  contrast) gives a peak of about 1–1.5 K, which is visible.
- Snowflakes keep 20–40% of their contrast, so they show clearly as short streaks.
- Cooled MWIR/LWIR with 1–2 ms integration gives a box blur of v·t: about 6–13 mm for rain.
- No paper found that shows thermal imagery of individual raindrops or snowflakes; this is a **gap**.
- The FLIR tech note [38] says only: "LWIR and MWIR perform similarly in the presence of rain". Rain
  degradation is "very range sensitive ... dramatic drop off in the 100–500 meter range".

---

## 7. Path radiance of a rain-filled path

**What LOWTRAN 7 does** (source code read [12]):
- `TNRAIN`: Marshall–Palmer, N0 = 8000 m⁻³ mm⁻¹, Λ = 41·R^−0.21 cm⁻¹. For ν ≥ 350 cm⁻¹ (λ ≤ 28.6 µm),
  σ = π·N0/Λ³ (**Q = 2, geometric**). At lower frequencies it uses a microwave table (`GMRAIN`).
- For ν ≥ 250 cm⁻¹: rain absorption/extinction **CSSA = 0.5** and asymmetry **g = 0.85**, applied at
  every wavelength, including the visible. That is wrong in the visible by Mie (~0 absorption) but right
  in MWIR/LWIR (0.46–0.48).
- Rain temperature is the layer air temperature (DENSTY(61) = R^0.63·T). RH is forced to 100% in rain
  layers.
- With multiple scattering off (IMULT = 0), layer emission is SUMV += B(T_layer)·Δτ_total, where
  Δτ_total includes rain *scattering*. Effectively ε_path = 1 − τ_ext: in-scattered radiance is assumed
  to equal B(T).
- With IMULT = 1, a two-stream multiple-scattering scheme uses CSSA and g.
- No snow model.

**MODTRAN 6 manual [13].**
- Inputs: RAINRT (rain rate) and user rain profiles (CLDALT, RR). Built-in ICLD 6–10: drizzle 2.0,
  light 5.0, moderate 12.5, heavy 25 and extreme 75 mm/h at the ground, falling linearly to cloud top.
  RH defaults to 100% in rain.
- The rain optical model is not described; it cites Berk & Anderson SSI-SR-69. That MODTRAN 6 keeps the
  LOWTRAN 7 parameterisation is **UNVERIFIED**.
- No snow model.

**Shettle 1989 [14] (AGARD CP-454).** Abstract only. It reviews the AFGL aerosol, cloud and
precipitation models behind LOWTRAN and FASCODE. Its rain and snow equations are **UNVERIFIED** (DTIC
blocked).

**Recommended radiance form for an imaging pixel (DERIVED synthesis of §1–3).**

L_pix = τ_ext·L_scene
      + (τ_abs − τ_ext)·[L_scene ⊗ K_diff]
      + (1 − τ_abs)·[ε_d·B(T_d) + (1 − ε_d)·L_env]
      + L_wide

where:
- τ_ext = exp(−σ_ext·L) and τ_abs = exp(−σ_abs·L).
- K_diff is the normalised Airy-like halo of width about 1.22·λ/D̄.
- L_wide is the single-scattered wide-angle term (refraction/reflection lobes, g ≈ 0.88 in VIS); it
  matters in VIS/NIR and is small in LWIR.
- σ_abs/σ_ext comes from §1.2.

Worked numbers (DERIVED, LWIR at 10 µm):

| R (mm/h) | Path length | τ_ext | τ_abs | ε_path = 1 − τ_abs |
|---|---|---|---|---|
| 25 | 100 m | 0.754 | 0.873 | 0.127 |
| 25 | 500 m | 0.244 | 0.508 | 0.49 |
| 5 | 100 m | 0.902 | 0.952 | 0.048 |

---

## Things I could not verify

- Atlas 1953 coefficient 0.25·R^0.63 (secondary summary only).
- Carbonneau's fit wavelength, and the origin of the "Japan" 1.58·R^0.63 law.
- Any 3–5 or 8–12 µm rain fit coefficients (Chimelis 1982; Winchester et al. 1982).
- Kinzer & Gunn 1951 relaxation numbers; Gunn & Kinzer's maximum speed of 9.2 m/s.
- The air-density correction (ρ0/ρ)^0.4.
- Origin of the P.1817 snow law (Kim / Nebuloni attribution); the meaning of S; validity of the λ term
  beyond 1.55 µm.
- Shettle 1989 content; the MODTRAN 6 rain optics; Border & McEwan and Rahm numbers.
- Hori 2006 values (secondary sources only); the exact Lutomirski MTF form.
- The Mie benchmark value used for the code check.
- No primary source for thermal imagery of individual drops or flakes.

---

## References

1. Chu, T.S. & Hogg, D.C. (1968). Effects of precipitation on propagation at 0.63, 3.5, and 10.6
   microns. *Bell Syst. Tech. J.* 47(5), 723–759. https://archive.org/details/bstj47-5-723 — full OCR
   text read.
2. Rensch, D.B. & Long, R.K. (1970). Comparative studies of extinction and backscattering by aerosols,
   fog, and rain at 10.6 µ and 0.63 µ. *Appl. Opt.* 9(7), 1563–1573. doi:10.1364/AO.9.001563 — abstract.
3. Chimelis, V. (1982). Extinction of CO2 laser radiation by fog and rain. *Appl. Opt.* 21(18),
   3367–3372. doi:10.1364/AO.21.003367 — abstract.
4. ITU-R Rec. P.1814-1 (09/2025). Prediction methods required for the design of terrestrial free-space
   optical links. https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.1814-1-202509-I!!PDF-E.pdf — full
   text.
5. ITU-R Rec. P.1817-1 (02/2012). Propagation data required for the design of terrestrial free-space
   optical links. https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.1817-1-201202-I!!PDF-E.pdf — full
   text.
6. Korai, U.A., Luini, L. & Nebuloni, R. (2018). Model for the prediction of rain attenuation affecting
   free space optical links. *Electronics* 7(12), 407. doi:10.3390/electronics7120407 — full text
   (Strathprints).
7. Carbonneau, T.H. & Wisely, D.R. (1998). Opportunities and challenges for optical wireless...
   *Proc. SPIE* 3232 — not accessed.
8. Al Naboulsi, M., Sizun, H. & de Fornel, F. (2005). Propagation of optical and infrared waves in the
   atmosphere. URSI General Assembly, paper F01P.7.
   https://www.ursi.org/proceedings/procGA05/pdf/F01P.7(01729).pdf — read.
9. Atlas, D. (1953). Optical extinction by rainfall. *J. Meteor.* 10(6), 486–488 — not accessed (403).
10. Montero-Martínez, G. & García-García, F. (2025). The influence of rainfall on the extinction
    coefficient and the meteorological optical range. *Atmósfera* 39. doi:10.20937/ATM.53491 — read via
    fetch summary.
11. Marshall, J.S. & Palmer, W.McK. (1948). The distribution of raindrops with size. *J. Meteor.* 5,
    165–166 — not accessed; parameters confirmed as implemented in [12].
12. Kneizys, F.X. et al. (1988). Users Guide to LOWTRAN 7. AFGL-TR-88-0177 (DTIC ADA206773) — guide not
    accessed. LOWTRAN 7 Fortran source read:
    https://github.com/space-physics/lowtran (src/lowtran/fortran/lowtran7.f; TNRAIN, RNSCAT, TRANS).
13. Berk, A. et al. MODTRAN 6.0 User's Manual (Spectral Sciences / AFRL).
    https://web.meteo.mcgill.ca/~frederic/MODTRAN6_Manual.pdf — read (rain sections).
14. Shettle, E.P. (1989/1990). Models of aerosols, clouds and precipitation for atmospheric propagation
    studies. AGARD CP-454 (DTIC ADA221594) — abstract only.
15. Hale, G.M. & Querry, M.R. (1973). Optical constants of water in the 200-nm to 200-µm wavelength
    region. *Appl. Opt.* 12, 555–563. doi:10.1364/AO.12.000555 — data via refractiveindex.info.
16. Segelstein, D.J. (1981). The complex refractive index of water. M.S. thesis, Univ. Missouri–Kansas
    City. http://hdl.handle.net/10355/11599 — data via refractiveindex.info.
17. Warren, S.G. & Brandt, R.E. (2008). Optical constants of ice from the ultraviolet to the microwave.
    *J. Geophys. Res.* 113, D14220. doi:10.1029/2007JD009744 — data.
18. Rowe, P.M., Fergoda, M. & Neshyba, S. (2020). Temperature-dependent optical properties of liquid
    water from 240 to 298 K. *J. Geophys. Res. Atmos.* 125, e2020JD032624. doi:10.1029/2020JD032624 —
    data.
19. Kinzer, G.D. & Gunn, R. (1951). The evaporation, temperature and thermal relaxation-time of freely
    falling waterdrops. *J. Meteor.* 8, 71–83. doi:10.1175/1520-0469(1951)008<0071:TETATR>2.0.CO;2 —
    not accessed.
20. Loftus, K. & Wordsworth, R.D. (2021). The physics of falling raindrops in diverse planetary
    atmospheres. *J. Geophys. Res. Planets* 126, e2020JE006653. doi:10.1029/2020JE006653;
    arXiv:2102.09570 — read.
21. Byers, H.R., Moses, H. & Harney, P.J. (1949). Measurement of rain temperature. *J. Meteor.* 6,
    51–55 — via [22].
22. Zhang, S., Meurey, C. & Calvet, J.-C. (2019). Identification of soil-cooling rains in southern
    France... *Atmos. Chem. Phys.* 19, 5005–5020. doi:10.5194/acp-19-5005-2019 — read (summary).
23. Gunn, R. & Kinzer, G.D. (1949). The terminal velocity of fall for water droplets in stagnant air.
    *J. Meteor.* 6, 243–248 — not accessed.
24. Atlas, D., Srivastava, R.C. & Sekhon, R.S. (1973). Doppler radar characteristics of precipitation at
    vertical incidence. *Rev. Geophys.* 11(1), 1–35. doi:10.1029/RG011i001p00001 — formula via [25].
25. Lane, J., Kasparis, T., Michaelides, S. & Metzger, P. A phenomenological relationship between
    vertical air motion and disdrometer derived A-b coefficients. arXiv:1707.05225 — read.
26. Locatelli, J.D. & Hobbs, P.V. (1974). Fall speeds and masses of solid precipitation particles.
    *J. Geophys. Res.* 79(15), 2185–2197. doi:10.1029/JC079i015p02185 —
    https://www.atmos.albany.edu/facstaff/rfovell/ATM562/locatelli-hobbs-1974.pdf (read).
27. Rasmussen, R.M., Vivekanandan, J., Cole, J., Myers, B. & Masters, C. (1999). The estimation of
    snowfall rate using visibility. *J. Appl. Meteor.* 38, 1542–1563.
    doi:10.1175/1520-0450(1999)038<1542:TEOSRU>2.0.CO;2 —
    https://opensky.ucar.edu/system/files/2024-08/articles_15245.pdf (read).
28. Seagraves, M.A. (1986). Visible and infrared extinction in falling snow. *Appl. Opt.* 25(7),
    1166–1169. doi:10.1364/AO.25.001166 — abstract and Table I.
29. Border, T. & McEwan, K. (2020). Measurement of spectral transmission through snow from SWIR to LWIR.
    *Proc. SPIE* 11538, 115380A. doi:10.1117/12.2570930 — abstract.
30. Rahm, M., Kullander, F., Björck, M. & Sjöqvist, L. (2023). Turbulence and transmission effects on
    laser beam propagation in the SWIR and LWIR bands. *Proc. SPIE* 12731, 127310G.
    doi:10.1117/12.2678535 — abstract (via search).
31. Winchester, L.W. Jr. & Gimmestad, G.G. (1982). Propagation of visible and infrared radiation in fog,
    rain, and snow. Keweenaw Research Center, Michigan Tech, Rep. KRC 82-10 — abstract.
32. Winchester, L.W., Gimmestad, G.G. & Lee, S.M. (1982). Attenuation of visible and IR radiation by
    rain. CLEO 1982, paper FV4 — abstract.
33. Hori, M. et al. (2006). In-situ measured spectral directional emissivity of snow and ice in the
    8–14 µm atmospheric window. *Remote Sens. Environ.* 100, 486–502. doi:10.1016/j.rse.2005.11.001 —
    via secondary sources.
34. Matsuo, T. & Sasyo, Y. (1981). Melting of snowflakes below freezing level in the atmosphere.
    *J. Meteor. Soc. Japan* 59(1), 10–25 (and 26–32); Heymsfield, A. et al. (2021). Survival of snow in
    the melting layer: relative humidity influence. *J. Atmos. Sci.* 78(6) — via search summaries.
35. Garg, K. & Nayar, S.K. (2007). Vision and rain. *Int. J. Comput. Vis.* 75(1), 3–27.
    doi:10.1007/s11263-006-0028-6 —
    https://www.cs.columbia.edu/CAVE/publications/pdfs/Garg_IJCV07.pdf (read).
36. Jobert, G. et al. (2024). Windshield integration of thermal and color fusion for automatic emergency
    braking in low visibility conditions. arXiv:2410.04928 — read.
37. Oswald-Tranta, B. (2018). Temperature reconstruction of infrared images with motion deblurring.
    *J. Sens. Sens. Syst.* 7, 13–20. doi:10.5194/jsss-7-13-2018 — read.
38. FLIR Commercial Vision Systems. Seeing through fog and rain with a thermal imaging camera, tech note
    TN_0109_0001_EN. http://www.flirmedia.com/MMC/CVS/Tech_Notes/TN_0001_EN.pdf — read.
39. Deepak, A. & Box, M.A. (1978). Forwardscattering corrections for optical extinction measurements in
    aerosol media. 2: Polydispersions. *Appl. Opt.* 17(19), 3169–3176 — abstract.
40. Vasseur, H. & Gibbins, C.J. (1996). Prediction of apparent extinction for optical transmission
    through rain. *Appl. Opt.* 35(36), 7144–7150. doi:10.1364/AO.35.007144 — abstract.
41. Sadot, D. & Kopeika, N.S. (1993). Imaging through the atmosphere: practical instrumentation-based
    theory and verification of aerosol modulation transfer function. *JOSA A* 10(1), 172–179.
    doi:10.1364/JOSAA.10.000172 — abstract.
42. Sadot, D. & Kopeika, N.S. (1994). Effects of absorption on image quality through a particulate
    medium. *Appl. Opt.* 33(30), 7107–7111. doi:10.1364/AO.33.007107 — abstract.
43. Lutomirski, R.F. (1978). Atmospheric degradation of electrooptical system performance.
    *Appl. Opt.* 17, 3915–3921 — formula via secondary summary.
44. Bohren, C.F. & Huffman, D.R. (1983). *Absorption and Scattering of Light by Small Particles*. Wiley —
    algorithm basis; not re-read.
45. van de Hulst, H.C. (1957). *Light Scattering by Small Particles*. Wiley — cited via [1].
46. Beard, K.V. & Pruppacher, H.R. (1971). *J. Atmos. Sci.* 28, 1455–1464 — ventilation formula via [20].
47. Ophir Optronics. AquaShieldIR hydrophobic coating (vendor page).
    https://www.ophiropt.com/en/s/AquashieldIR-hydrophobic-coating — read.
