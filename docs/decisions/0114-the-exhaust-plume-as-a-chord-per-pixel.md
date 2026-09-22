# ADR 0114 — The exhaust plume: a chord per pixel, and where its temperature comes from

**Status:** Accepted. Applies [ADR 0098](0098-participating-media-and-the-phenomena-tier.md)'s
operator to a frame; takes its gas state from [ADR 0105](0105-the-exhaust-line-as-a-gas-stream-in-a-wall.md)'s
solved line and its coefficients from the tables `PH.5` generates.
**Date:** 2026-09-22
Roadmap: PH.6 (§6.6, §8.1, §8.3, §13.4 stage 2)

## Context

ADR 0098 settled what hot gas does to one ray: `L = τ_b L_behind + (1 − τ_b) B_b(T_g)`, per band,
with `τ_b = exp(−κ_b L)`. `PH.5` filled the coefficient tables. What was still missing was the
step from one ray to a picture — and the several choices hiding inside it, each of which has a
plausible wrong answer that would produce a frame nobody could tell was wrong.

§6.6 deferred the exhaust plume "without a number". Its trigger — an MWIR Tier 3 bench — fired
with `test_tier3_multiband.py`, and the owner asked for fire and hot gas on 2026-09-18.

## Options considered

**A hot prim with an authored emissivity.** What a scene author reaches for, and what every
engine makes easy. It cannot express the phenomenon: a CO₂/H₂O plume absorbs a fifth of a 3–5 µm
band and almost none of a 7.5–13.5 µm one, and one ε cannot be two numbers. Measured here on the
committed tables: τ = 0.866 against 0.979 for the same gas. Rejected, and kept as the negative
control in `tests/unit/test_plume.py` so the test that passes is measuring the tables rather than
the plumbing.

**A volumetric pass in the renderer.** Right, expensive, engine-bound, and outside the
engine-free core by construction (CLAUDE.md #1). ADR 0098 already deferred it.

**Ray-marching the cone per pixel with the gas state sampled along the ray.** Tempting, because
it would lift ADR 0098's "no temperature gradient along the ray" deferral. Rejected for now: it
multiplies the cost by the sample count, and the deferral it lifts is not the one that matters —
the gradient a camera *sees* is along the plume, which a chord per pixel already produces.

## Decision

**A truncated cone in camera space, one analytic chord per pixel, one homogeneous slab on it.**

* **The composite is the blend, and it is exact.** Write the plane's value as
  `L_plane = τ_atm(R_p) L_behind + (1 − τ_atm(R_p)) L_air`. Putting the slab in the way and
  rearranging gives `L = τ_p L_plane + (1 − τ_p) B_gas,at-sensor` **provided** `B_gas,at-sensor`
  is the gas radiance run through stage 2's own path at the plume's range. It is: the injector
  calls the same `apply_layered_gbuffer` / `apply_atmosphere` the plane went through. So nothing
  has to decide what is behind the plume, and the plume and the pixels under it cannot disagree
  about the atmosphere. The *excess* form MS.6's point targets use would need that decision and
  would subtract a sky column that is not there wherever the plume crosses the car.

* **Occlusion is the depth plane.** Each chord is clipped to the `distance_m` the G-buffer
  already carries. One comparison per pixel, and it removes the class of bug where a plume paints
  over the bumper it passes behind.

* **The gradient is along the plume, not along the ray.** ADR 0098's deferral stands: each pixel
  gets one homogeneous slab at the axial station of its own chord's midpoint. A frame therefore
  still shows a plume that cools with distance from the tip, which is the feature, while every
  ray keeps the operator `PH.4` verified.

* **Entrainment dilutes temperature and species by one factor.** `φ = exp(−x / mixing_length)`,
  `T_g = T_air + φ (T_tip − T_air)`, `p_i = φ p_i,tip`. Both ride the same conserved scalar in a
  mixing jet, so one authored length sets how fast the plume cools *and* how fast it thins, and
  the two cannot be authored inconsistently.

* **The temperature is not authored at all.** Schema v15's `plume:` block carries geometry and
  chemistry and may sit only on an `exhaust` target. Its temperature is `ExhaustSolver`'s own
  outlet **gas** at that instant — not `temperature()`, which is a *skin*. The two differ by
  about 50 K at load on the shipped line, and taking the skin would be the easy mistake. The air
  it mixes into is the scene's weather, for the reason CLAUDE.md #6 gives.

* **World in the scene, camera in the pipeline.** `Scene.plumes_at` returns `WorldPlume`s in
  world coordinates; `WorldPlume.in_camera(rotation, eye)` produces the camera-space
  `ExhaustPlume` the injector wants. Nothing between the scene and the frame has to hold both
  frames at once, and the engine-free half is testable without a renderer.

* **A band the absorption model does not reach raises.** `PipelineConfig` carries `gas_tables`
  and leaves them `None` for SWIR and NIR (`PH.5`); a plume carrying gas there is an error naming
  the missing table, not a transparent plume.

## Consequences

**Easy now:** a scene file authors one plume and two cameras disagree about it by construction.
Measured on `configs/scenes/car_exhaust_plume.yaml` at twenty minutes of light cruise — tailpipe
outlet gas 594 K, skin 542 K — the MWIR camera gains **+72.7 K** of peak apparent temperature and
the LWIR bolometer **+7.1 K**, with τ = 0.866 and 0.979. `PH.7`'s fire reuses the same injector
with a soot slab; `PH.9`'s steam adds droplet extinction to the same coefficient.

**Error introduced, bounded where it can be:** one slab per ray rather than an integral along it,
so a chord that crosses a steep part of the plume reads its midpoint; the near-axial case, where
the ray lies within the cone's own half-angle of its axis, can produce two chord segments and
their lengths are added while the temperature is read at the longer one's midpoint; and the tail
of the plume reaches below the tables' 300 K floor, where the coefficient is read at the floor
while the radiance is computed at the true temperature — bounded by how little κ moves over the
last few kelvin of an already optically thin column, and happening where the plume has nearly
stopped absorbing.

**Not done, and why:** `PH.6`'s row says "on the car **and the vessel**", and only the car has
one. A plume may sit only on an `exhaust` target, because that is the only solver that produces a
gas temperature rather than a skin, and `stock_exhaust()` is a car's line —
`vessel_departure_clear_day.yaml` drives its stack from a prescribed schedule. Giving the vessel a
plume means either a marine stack in `exhaust_line.py` or a deliberate decision to let a
`prescribed` target declare that its schedule *is* a gas temperature. The second is one line of
schema and is exactly the inconsistency this ADR removed, so it is not taken quietly.

**Deferred, deliberately:** buoyancy and bend (the cone is straight, so a crosswind is not
modelled); puffing and flicker; scattering by soot or droplets (`PH.9`); a plume that shadows
itself; and the in-engine frame, which is `IG.2`'s.

## Revisit when

A validation needs the plume's structure rather than its magnitude — a bent or buoyant plume, or
one whose apparent temperature profile is compared against a measured one — which means sampling
the gas state along the ray rather than at one station, and a plume geometry that is not a cone.
