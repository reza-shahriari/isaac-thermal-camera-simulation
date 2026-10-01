---
name: thermal-materials
description: Material modelling for the IR camera simulator — spectral emissivity, Kirchhoff closure, Fresnel angular emissivity from complex refractive index, per-band roughness, semi-transparent glass, and the mapping layer that assigns thermal/optical properties to imported USD assets. Use this skill whenever the task involves emissivity, reflectance, transmittance, material libraries, material YAML, BRDF or roughness in the IR, n/k optical constants, how a specific surface (paint, glass, metal, skin, asphalt, vegetation) should look in a given band, or importing a mesh that has no thermal properties. Also use it before adding any new material, since the closure and audit rules are easy to violate.
---

# Thermal materials

Materials are where an IR simulator most often stops being physical without anyone noticing. The rules
here are cheap to follow and expensive to retrofit.

Reference: `docs/physics-model.md` §4, §12.3, §16.2.

## Rule 1: author one property, derive the rest

Kirchhoff's law for an opaque surface in local thermodynamic equilibrium:

```
ε(λ,θ) = α(λ,θ) = 1 − ρ(λ,θ)
```

and in general `ε + ρ + τ = 1`.

Author **spectral reflectance or spectral emissivity, never both.** The other is computed. Authoring
both is how simulators silently violate energy conservation, and the resulting error looks like a
temperature error, so it gets "fixed" in the wrong place.

There is a test that walks every material in `configs/materials/` and asserts closure to 1e-6 in every
configured band. Keep it passing. When it fails, fix the material data — do not loosen the tolerance.

## Rule 2: the metal question is paint, not alloy

The instinct when modelling a vehicle is to distinguish steel from aluminium from magnesium. That is
mostly wasted effort. For thin panels under solar load the thermal answer is dominated by thickness and
by the coating, not by the substrate alloy.

What matters enormously is the **surface finish**:

| Finish | ε (LWIR) | Behaviour |
|---|---|---|
| Painted (any colour) | 0.88–0.93 | Emits its own temperature. Colour changes `solar_absorptivity`, not ε. |
| Bare / polished metal | 0.05–0.15 | **A mirror.** Shows the sky and surroundings, not itself. |
| Oxidised / corroded | 0.75–0.90 | Climbs back toward painted behaviour. |
| Coated exhaust / heat shield | 0.20–0.85 | Highly variable, measure it. |

So the material taxonomy is by finish, with alloy as a second-order correction to substrate heat
capacity. A bare aluminium trim strip at ε = 0.09 will dominate the visual character of a vehicle in
LWIR far more than the choice between steel and aluminium body panels.

## Rule 3: roughness is per-band

Surface roughness that scatters 0.5 µm light diffusely can be optically smooth at 10 µm (when
σ ≪ λ/8). IR reflection lobes are therefore **narrower and stronger** than the same material's visible
lobe. Never reuse visible-band roughness values.

```yaml
roughness_per_band: { nir: 0.35, swir: 0.30, mwir: 0.18, lwir: 0.12 }
```

The decreasing trend is the physics. This one parameter is what makes vehicles reflect the sky correctly
in LWIR, so it is worth getting approximately right rather than leaving at a default.

## Rule 4: bands change what a material *is*

Glass is opaque in LWIR (ε ≈ 0.88) and transparent in SWIR/NIR. A windshield in LWIR shows the
windshield's own temperature; in SWIR it shows the driver behind it. Water is emissive in LWIR
(ε ≈ 0.96) and strongly absorbing — dark — in SWIR. Vegetation is very bright in NIR.

The schema must express `transmittance_per_band`, and the shading path must handle `τ > 0` as a second
ray in the reflective bands. Build this in from the start; retrofitting transmission into a
surface-only model is painful.

## Angular emissivity

Three levels. Pick per material class, not globally.

**Level A — Fresnel from complex refractive index.** Physically correct. With `ñ = n + ik` and
`Z = ñ² − sin²θᵢ`, use the branch-safe reformulation:

```
A = sqrt((|Z| + Re(Z)) / 2)
B = sign(Im(Z)) * sqrt((|Z| − Re(Z)) / 2)
sqrt(Z) = A + jB
```

then the standard Fresnel amplitudes with `sqrt(Z)` replaced by `A + jB`, `R = (R⊥ + R∥)/2`,
`ε(λ,θ) = 1 − R(λ,θ)`.

The reformulation is not cosmetic. A naive complex square root has two algebraic branches; passivity
requires `Im(k_z) > 0`. Picking the wrong branch gives reflectance discontinuities across wavelength,
instability near grazing incidence, and reflectances exceeding unity. In the IR this bites hard because
`k` is large for many materials. Assert `0 ≤ R ≤ 1` in the implementation.

Sanity check: in strongly absorbing media there is no sharp total internal reflection — R stays below 1
at every angle. If your implementation shows a hard TIR knee for a lossy material, the branch is wrong.

n/k tables live in `data/nk/`. refractiveindex.info is the usual source; record the source in the
material YAML.

**Level B — empirical falloff.** Adequate for most of the scene:

```
ε(θ) = ε₀ · (1 − a·(1 − cos θ)^p)
```

`a ≈ 0.15–0.35`, `p ≈ 4–6` for painted metals and plastics; `a → 0` for rough dielectrics (asphalt,
vegetation, fabric). Fit `(a, p)` once per class against Level A and bake it. Two instructions in a
shader.

**Level C — constant ε.** Only for rough, high-emissivity surfaces (ε > 0.93) within ±50° of normal.
Vehicle bodies, glass and water all violate this — do not use Level C for them.

## The USD import problem

No asset in the wild ships with thermal properties. Every imported mesh has visible-band materials only.
This is the real per-object cost of the project, and it is automatable.

Build a **mapping layer**, not a manual process:

1. Resolve by explicit override (a `thermal:material` USD attribute) if present.
2. Else resolve by semantic class (`car_body`, `road`, `pedestrian`) from the semantics schema.
3. Else resolve by material-name pattern matching (`*paint*`, `*glass*`, `*chrome*`, `*rubber*`).
4. Else fall back to a loud default and **record the miss**.

Then provide two things the team can actually use:

- `scripts/audit_materials.py` — prints every prim that fell through to the default, grouped and counted,
  with a coverage percentage. Run it on every new asset. Coverage below a threshold fails CI.
- A debug AOV / render mode that colours unmapped surfaces magenta. Nobody can eyeball whether a thermal
  image is right, but everybody can spot magenta.

These two tools are what turn "adding an object takes ages and I'm never sure it worked" into a
30-second check. Build them early — before the material library is large, not after.

## Material schema

```yaml
car_paint_black:
  source: "measured|literature|estimated"     # be honest; it drives how much to trust results
  thermal:
    density_kg_m3: 7800
    specific_heat_j_kgk: 470
    conductivity_w_mk: 45
    thickness_m: 0.0012
    solar_absorptivity: 0.94
  optical:
    spectral_emissivity: "spectra/car_paint_black.csv"    # λ(µm), ε(λ), 0.3–15 µm
    roughness_per_band: { nir: 0.35, swir: 0.30, mwir: 0.18, lwir: 0.12 }
    angular_model: { type: fresnel, n_k_file: "nk/acrylic_paint.csv" }
    transmittance_per_band: { nir: 0.0, swir: 0.0, mwir: 0.0, lwir: 0.0 }
```

**Three forms of the one authored quantity (ADR 0175).** `spectral_emissivity` (a curve: one file, or
disjoint segments such as `[{file: sw.csv, quantity: reflectance}, lw.csv]` where the reflectance segment
is the opaque complement), `emissivity_per_band` and a grey `emissivity: 0.93` may be combined. The library
resolves each wavelength as curve → the camera band's value → grey, and averages under the **camera's own
response** (`MaterialTable.for_sensor`). So:

- Prefer a curve whenever measured data exists. It is the only form under which a 6–13 µm and a
  7.5–13.5 µm camera read different values.
- Do not write a per-band value for a band the curve already covers. The loader refuses it as a second
  authoring.
- Gaps are refused, never extrapolated. Fill one with a per-band or grey value and read
  `BandProperties.curve_fraction` to see how much of the band rests on the fill.

The `source` field matters more than it looks. When a validation comparison disagrees with reality, the
first question is always "which of these numbers did we actually measure?" — and without this field
nobody can answer it six months later.

Starting values for common materials are in `docs/physics-model.md` §16.2. They are literature
estimates, not measurements. Mark them as such and replace them for anything the project depends on.

## Tests to write

- Kirchhoff closure across the whole library, every band, 1e-6.
- Fresnel: `0 ≤ R ≤ 1` for a sweep over θ ∈ [0°, 89.9°] and a lossy/lossless material pair.
- Fresnel vs. a known reference: water at 10 µm (n = 1.218, k = 0.0508) against published curves.
- Level B fit vs. Level A: max ε error < 0.02 over 0–70°.
- Band resolution: glass returns τ > 0.5 in SWIR and τ ≈ 0 in LWIR.
- Import mapping: an asset with known material names resolves to the expected thermal materials, and an
  asset with unknown names is reported as unmapped rather than silently defaulted.
