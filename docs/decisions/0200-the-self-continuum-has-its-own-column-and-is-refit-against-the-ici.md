# ADR 0200 — The self-continuum has its own column, and is refit against the ICI

**Status:** Accepted (AT.37). Amends ADR 0160; closes the humid half of ADR 0183.
**Date:** 2026-10-07

## Context

ADR 0183 held the layered clear sky (ADR 0071) to ARM's calibrated Infrared Cloud Imager (ICI) at
the Southern Great Plains site. The cold, dry December night agreed to 7 %. The humid August night
was 24–28 % too bright at every elevation, while the instrument's own clear-sky model was within
3 %. ADR 0183 named two suspects and declined to fit either:

- the vertical profile: the comparison invented a surface humidity to hold the measured PWV over
  the preset's fixed 2 km water scale height;
- AT.27's squared self-continuum term (ADR 0160).

Fitting the continuum against a column that was itself wrong would bake the profile error into the
coefficient. ADR 0160 had warned of exactly that. AT.37 therefore needed each night's measured
column first.

Two things changed that.

1. **The surface dew point is available.** ERA5's hourly 2 m dew point at the site comes from
   Open-Meteo's archive, with no account needed. With it the surface humidity is measured. The
   scale height follows from the imager's own PWV as `H_w = PWV / w0`. It is 1.5 km on the May
   night, 2.6 km in August and 2.8 km in December, not one 2 km.
2. **The square was carried on the wrong column.** The water classes' extinction is
   `γ₀ + β₁w + β₂w²`, evaluated at the surface. The whole sum was then spread up the column on
   `H_w`. The square goes as `w(h)² = w₀² e^{−2h/H_w}`, so its column is half the linear part's.
   Carrying it on `H_w` counted the self-continuum's column twice. A horizontal path never leaves
   the surface, so the τ(200 m) anchor never noticed.

## Options considered

1. **Two exponentials per water class.** This is exact. But every consumer of the exponential
   sum integrates one exponential analytically: the slant tables, the loss-share plane
   integrator, `column_length` and the per-class path radiance. All of them would change.
2. **One exponential per class, with the height that holds the column exact.**
   `H_eff = H_w (γ_lin + γ_sq/2) / (γ_lin + γ_sq)`. The surface value, and so every horizontal
   path and the anchor, is unchanged. Only the vertical shape is approximated: one exponential
   stands in for a sum of two with the same integral. A band with `β₂ = 0` is bit-identical.
3. **Refit β₂ alone on ADR 0183's comparison.** This is what ADR 0160 warned against.

## Decision

Option 2 (`LayeredAtmosphere.exponential_sum`), then refit β₂ on the pinned columns.

`irsim.validation.ici.pinned_column` gives each image its measured RH and `H_w`.
`scripts/ici_extract_profiles.py --era5` adds the hour's dew point to the committed CSV as
`td_era5_c`; every other value is unchanged. The fit holds the clear-dry anchor exactly, as
ADR 0160 did: `β₁ = 4.61e-5 − β₂ w_dry`. It minimises the RMS log ratio over the August and
December nights at every elevation.

| β₂ / 8.6e-7 | 0 | 0.25 | **0.35** | 0.5 | 1.0 |
|---|---|---|---|---|---|
| RMS log(model/measured), Aug + Dec | 0.0339 | 0.0306 | **0.0303** | 0.0310 | 0.0410 |

**β₂ = 3.0e-7** (from 8.6e-7) and **β₁ = 4.4332e-5**, in all seven presets that carry LWIR. Model
over measured, VOx stand-in band, pinned column:

| Night | Before (ADR 0183) | After |
|---|---|---|
| 11 Dec, 2.5 °C, 0.86 cm | 1.00–1.07 | 0.99–1.06 |
| 6 Aug, 27.0 °C, 3.96 cm | 1.24–1.28 | 0.97–1.08 |
| 21 May, 20.8 °C, 1.41 cm | 1.03–1.24 | 1.06–1.23 |

`test_ici_clear_sky.py` now holds August to 10 % and December to 7 % at every elevation. The strict
xfail on August is gone.

## Consequences

- Only LWIR moves; MWIR, SWIR and NIR are bit-identical (their β₂ is 0). Goldens regenerated
  deliberately:
  - the layered transmittance table: LWIR row up to 0.28 %;
  - the sky against elevation: LWIR up to 0.48 %, about 0.09 W/(m²·sr);
  - the sea against depression: up to 37 mK.
- At the §7.2 humid row's own condition, τ(200 m) is 0.77 (was 0.73). It is still inside the row's
  0.72–0.85. The continuum's share of the water term there falls from about a third to 14 %.
- **May is not fixed, and is not a continuum question.** Its water sits in a moist layer to
  about 1.3 km under very dry air: ERA5 gives 61 % RH at 770 m and 3 % at 2.9 km. One
  exponential holding the right total puts too much water at 2–4 km, and the zenith reads 22 %
  bright. That is a strict xfail naming `AT.38`, a tabulated humidity profile.
- R13's −40 °C Tucson sky at 15° was read off a figure at an RH the test assumes (0.2). Its band
  widens from ±5 to ±6 K. The column fix alone moves it from about −35.2 to −35.35 °C, and the
  refit to −34.9 °C. It was 0.2 K inside its edge before this change. The calibrated December
  night is now the stronger dry-sky check, and it agrees to 1–6 % in radiance.
- `LayeredAtmosphere.sky_beyond` now forms its weighted mean from `Σ w_k (L_sky,k − L_path,k)`, and
  the clip at zero applies only to the band's total. The old per-class clip dropped an opaque
  class's −1e-7 quadrature residue (at τ_k = 1e-8), which broke `L_path + τ L_beyond = L_sky` at
  1e-10. The new coefficients exposed it; it now holds by construction.
- The band is still the ICI stand-in (ADR 0183). It can scale the skies but not make a misfit that
  grows with humidity, and none remains between December and August.

## Revisit when

- `AT.38` gives the column a tabulated humidity profile (ERA5 or a sounding). May is the test.
- A spectral reference (ARM's AERI at the same minutes) is available. It would fit β₂ without the
  band stand-in.
