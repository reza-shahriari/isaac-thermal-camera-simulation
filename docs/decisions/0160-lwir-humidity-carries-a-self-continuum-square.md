# 0160 — The LWIR humidity term carries a self-continuum square, anchored on the clear-dry row

Date: 2026-09-28
**Status:** Accepted
Roadmap: AT.27 (§7.2, §7.3); spec issue S57; builds on ADR 0049 (the presets and their rows)

## Context

§7.3 writes the molecular extinction as `γ = γ₀ + β w`, linear in absolute humidity, and every
preset carries the same LWIR pair (`γ₀ = 7.5e-5`, `β = 4.61e-5` per metre) fitted so that the
clear-dry and humid rows of the §7.2 table land. The physics is not linear: the 8–12 µm
water-vapour continuum that dominates LWIR window absorption is *self-broadened*, its
coefficient going as the vapour's number density times its own partial pressure, `n_w · e ∝ w²`
at a given temperature (MT_CKD; Mlawer et al. 2012, Phil. Trans. R. Soc. A 370). A line fitted to
two rows is right at its anchors and wrong everywhere else, and anchored on the dry row the
physics says every wetter air absorbs *more* than the line: mid-latitude summer air a little,
a monsoon afternoon a lot.

## Options considered

1. **Leave it.** The two rows are reproduced. But every humidity between 6 and 24 g/m³ -- most
   summer days -- sits on the wrong side of the curve, and the roadmap's aerial validation
   against public thermal video will be fitting that error into other parameters.
2. **Refit β₁, β₂ to the two rows.** Two unknowns, two anchors that already lie on a line: the
   solution is the line. The rows alone cannot see the curvature.
3. **Size β₂ from the continuum's own magnitude, keep the clear-dry anchor exact, and let the
   humid rows fall where the physics puts them -- checked against the §7.2 band.** The choice.

## Decision

`gamma_molecular` takes a fourth coefficient, `beta2_per_m_per_g2_m6`, and every band's
preset coefficients gain it with a default of 0 -- the linear law bit for bit, which MWIR,
SWIR, NIR and visible keep. LWIR carries

    β₂ = 8.6e-7 m⁻¹ (g m⁻³)⁻²  (ESTIMATED),  β₁ = 4.61e-5 − β₂ · w_dry = 4.10e-5,

with `w_dry = 5.90 g/m³` the us_standard_clear check condition (288 K, RH 0.46), so the
clear-dry row -- and with it every pinned LWIR ladder value, golden and demo taken at that
condition -- is the number it was, to the three digits the coefficient is written to.

**Where β₂ comes from.** MT_CKD's self-continuum coefficient at 10 µm is about
2.2 × 10⁻²² cm² molecule⁻¹ atm⁻¹ at 296 K, falling roughly 1 % per kelvin; at the §7.2 humid
row (30 °C, 80 % RH, 24.3 g/m³: `n_w = 8.1e17 cm⁻³`, `e = 0.033 atm`) that is
`k ≈ 5.1e-4 m⁻¹`, i.e. `β₂ = k / w² ≈ 8.6e-7`. It is a band-window magnitude with the
continuum's strong wavelength dependence averaged by eye rather than by the camera's
response, hence ESTIMATED. What it does: at the humid row LWIR τ(200 m) goes from 0.787 to
0.73, still inside the table's 0.72–0.85; at the tropical check (19 g/m³) to 0.79; at
12 g/m³ the law is murkier than the line by 1 %, at 35 g/m³ by 16 % (τ 0.60 against 0.71).

**What is not decided.** The temperature dependence of the continuum (it weakens as the air
warms, which partly offsets the higher w of warm air) is folded into one β₂; the MWIR wing's
foreign-broadened continuum stays linear; and no MODTRAN-class third anchor exists in this
repository to pin the curvature independently -- the first such number (a public 8–12 µm
transmittance at a known humidity) should replace the estimate and re-check the rows.

## Consequences

* LWIR transmission falls for every air wetter than the clear-dry anchor, by 1 % at 12 g/m³
  and 7 % at the humid row over 200 m; the fog and haze regimes (aerosol-dominated, ADR 0049)
  keep their ordering.
* The clear-dry anchor is exact by construction; anything pinned there does not move.
* A preset written before the field existed loads unchanged (β₂ = 0), which is also the
  sensor-file rule of ADR 0159 applied to atmospheres.
