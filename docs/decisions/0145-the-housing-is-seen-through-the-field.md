# ADR 0145 — The housing is seen through the field, referenced to the optical axis

**Status:** Accepted
**Date:** 2026-09-26
Roadmap: SC.17. Spec: `docs/physics-model.md` §8.2 ("Where the housing radiation lands on the
array", revised 2026-09-26). Amends [ADR 0016](0016-optics-self-emission-and-housing-temperature.md)'s scalar
self-emission term; [ADR 0015](0015-distortion-in-engine-cos4-pinhole.md)'s cos⁴ stands and
now has its measured-map half wired.

## Context

A public clear-sky frame from an uncooled 640×512 core shows a smooth radially symmetric bowl over
the whole frame, stretched to full contrast by the AGC because the scene has none of its own. The
optics stage could not make one. It computed

    Φ_ij = A_d Ω_eff [τ RI_ij L_scene + (1 − τ) L_h]

-- cos⁴ on the scene, one self-emission number for the whole array. That form darkens the corners of
*every* uniform scene, including one at the housing temperature, which is an isothermal enclosure
and cannot be shaded. The physics it leaves out: a pixel off axis sees the aperture as the smaller
projected solid angle Ω_eff RI_ij, and what it loses from the scene it gains from the inside of the
camera. An uncooled detector measures the scene *relative to its housing* [R48, R49].

`optics.vignetting_map` had been in the schema since v2 and was read by nothing.

## Options considered

1. **The full hemisphere.** Φ_ij = A_d π L_h + A_d Ω_eff τ RI_ij (L_scene − L_h). Physically
   complete, but the pedestal A_d π L_h is five times the on-axis scene power at F/1.0. It would
   move every absolute power the project has anchored -- the bolometer transfer, the NETD anchor,
   the ADC range of ADR 0021 -- and in a microbolometer it is balanced by the detector's own
   emission at T_FPA, which the model does not carry either. Carrying one half of a balanced pair
   is worse than carrying neither.
2. **Referenced to the optical axis.** Drop the uniform A_d (π − Ω_eff) L_h pedestal, keep the
   field-dependent part:

       Φ_ij = A_d Ω_eff [L_h + τ RI_ij (L_scene − L_h)]

   On axis (RI = 1) this is the old single-lens form to the last bit, so every on-axis number
   stays; off axis the non-scene power is A_d Ω_eff (1 − τ RI_ij) L_h instead of
   A_d Ω_eff (1 − τ) L_h.
3. **A phenomenological radial field** with a fitted amplitude, as §8.2 suggests for narcissus.
   Cheap, but it would have to be tuned per scene temperature, and the physics that sets its sign
   is known.

## Decision

Option 2. `irsim.optics.self_emission.housing_power_field` is the non-scene power and
`housing_power_axis` its RI = 0 limit A_d Ω_eff L_h; `apply_optics` and `invert_optics` use them,
and the Warp twin takes L_h and A_d Ω_eff L_h as two host-computed scalars, computing
(L − L_h)·Ω τ RI A_d + A_d Ω L_h. RI_ij is cos⁴ times the measured map, which the loader now
resolves against the data root and hashes by content like the spectral response.

## Consequences

- **Sign.** A uniform scene colder than the housing is brightest in the corners before any
  correction, warmer darkest, equal flat. `tests/unit/test_housing_field.py` pins all three and the
  closed-form depth; `test_flat_field.py`'s premise moved from a 300 K scene (now correctly flat
  through a 300 K housing) to 400 K.
- **Nothing changes at the calibration housing.** The two-point flat field and the radiometric
  inverse are both per-pixel affine in L_scene, so with the housing at its calibration temperature
  both are still exact. Every existing render with a fixed housing is unchanged after correction.
- **A drifted housing now shades.** Measured on the Boson 640 (τ 0.92, corner RI 0.793), the
  apparent-temperature error a +1 K housing leaves in the radiometric branch:

  | scene | centre, before and after | corner, before | corner, after |
  |---|---|---|---|
  | 300 K | 87 mK | 110 mK | 372 mK |
  | 230 K (clear sky) | 222 mK | 279 mK | 938 mK |

  The centre is ADR 0016's 87 mK/K unchanged. The corner more than triples. This is the mechanism
  behind the bowl, and the reason real cores carry housing thermistors [R49]. It lands on any
  scene rendered with a `coupled` housing away from `t_housing_cal_k`.
- **Error introduced.** The dropped pedestal A_d (π − Ω_eff) L_h is uniform in space but not in
  time: a housing drift moves it, and only the part the detector's own emission cancels is
  justified. With T_FPA = T_housing (both Boson configs) the cancellation is exact to the extent
  the bolometer is a black absorber; with a separate FPA node it is not, and the residual is a
  uniform offset of order (π/Ω_eff − 1) times the in-cone drift. Unbounded until the detector's
  emission is modelled; SC.18's FFC snapshot absorbs it at each shutter event.
- **Cooled cameras.** The cold shield replaces the out-of-cone housing view (§9.1), so for a
  photon detector with `cold_shield_efficiency = 1` this form overstates the shading. No cooled
  camera renders a sky scene today; the cold-shield leakage path is where it belongs when one does.
- **Engine.** The Warp kernel changed by one subtraction and one scalar. Its contract test passes on
  the CPU; the device equivalence test in `tests/integration/test_kernels_vs_reference.py` was
  updated and **not run** (no GPU work in this session).

## Revisit when

A camera config sets `fpa_temp_mode` apart from the housing, or a cooled camera renders a scene
colder than its window: both break the cancellation option 2 relies on.
