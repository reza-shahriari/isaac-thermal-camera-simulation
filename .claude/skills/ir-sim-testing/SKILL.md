---
name: ir-sim-testing
description: Testing strategy for the IR camera simulator — the five validation tiers, how to write tests that would actually fail if the physics were wrong, tolerance conventions in physical units, golden/regression fixtures, synthetic G-buffer fixtures for engine-free kernel testing, and the phenomenology checklist. Use this skill whenever writing or reviewing tests, choosing a tolerance, deciding whether something is adequately verified, setting up pytest markers or fixtures, adding regression data, or when a test is failing and the question is whether the test or the code is wrong. You cannot eyeball a thermal image for correctness, so tests carry more weight in this project than usual.
---

# Testing the IR simulator

Nobody has intuition for what a correct thermal image looks like. That is the central problem of this
project, and tests are the only way around it. A simulator you have not validated is a renderer with
physics-flavoured variable names.

Reference: `docs/physics-model.md` §15.

## The test that is worth writing

A test that calls a function and asserts it returned something is worthless here. Every physics test
should satisfy: **if the physics were wrong, would this fail?**

Three patterns that meet the bar:

1. **Analytic identity.** Two independent routes to the same number must agree. `∫L dλ == σT⁴/π`.
   `L_q·hc/λ == L`. `Lb⁻¹(Lb(T)) == T`. These catch wrong constants, dropped π, unit errors — the exact
   class of bug that is invisible in output imagery.

2. **Independent implementation.** Fast path vs. reference path. LUT vs. quadrature. CUDA kernel vs.
   NumPy. Neither is trusted; agreement between two independent derivations is the evidence.

3. **Known-answer / published reference.** Water at 10 µm against published Fresnel curves. Wien's
   displacement law. A datasheet NETD figure. These anchor the model to reality rather than to itself.

Everything else — smoke tests, shape assertions, does-not-crash — is fine to have, but it is not
verification and should not be counted as such in the TECHNICAL_REPORT.md status table.

## Tolerances in physical units

Never write `assert abs(a - b) < 1e-6` on a radiance value. Nobody reading it knows whether that is
tight or absurdly loose. Convert to a unit a human can judge:

```python
# BAD
assert abs(Lb_lut - Lb_ref) < 1e-4

# GOOD
err_mK = abs(Lb_lut - Lb_ref) / dLb_dT * 1000
assert err_mK < 5.0, f"LUT error {err_mK:.2f} mK exceeds 5 mK (NETD is 50 mK)"
```

Standard tolerances for this project:

| Comparison | Tolerance | Rationale |
|---|---|---|
| Analytic identities (Planck forms, Stefan-Boltzmann) | 1e-12 relative | Floating point only |
| LUT vs. quadrature | < 5 mK equivalent | 1/10 of the tightest NETD we model |
| Apparent-temperature round trip | < 1 mK | Must be negligible vs. everything else |
| GPU kernel vs. CPU reference | 1e-4 relative, or 5 mK | fp32 accumulation differences |
| Encode/decode through an AOV | < 10 mK | 1/5 of a 50 mK NETD |
| Kirchhoff closure | 1e-6 absolute | Data quality, not arithmetic |
| Thermal equilibrium vs. analytic | 0.1 K | Solver convergence |
| Validation vs. real imagery, per class | 2 K bias | Realistic given material uncertainty |

When a test fails, the default assumption is that **the code is wrong, not the tolerance**. Loosening a
tolerance requires a comment explaining why, and if it is a physics tolerance, an ADR.

## Test layout and markers

```
tests/unit/          # no GPU, no Isaac Sim, < 30 s total. The default gate.
tests/integration/   # needs Isaac Sim. @pytest.mark.isaac — deselected by default.
tests/golden/        # regression fixtures: reference arrays + the config that produced them
```

```python
# pytest.ini / pyproject markers
isaac: requires a running Isaac Sim environment
slow:  takes more than a second
gpu:   requires CUDA
```

`make test` runs unit only. `make test-all` adds the rest. Keep the unit suite genuinely fast — the
moment it takes minutes, people stop running it, and this project depends on people running it.

## Synthetic G-buffer fixtures

The most useful fixture in the repo. A function that produces a fake render output — flat or ramped
temperature field, known normals, known distances, known material IDs — so that **kernels can be tested
without launching Isaac Sim at all**.

```python
@pytest.fixture
def gbuffer_ramp():
    """256x256 G-buffer with a linear 250-450 K temperature ramp,
    normals facing camera, constant 50 m distance, single material."""
```

Most kernel bugs are findable this way in seconds instead of minutes. Provide at least: uniform field,
linear ramp, two-material split, grazing-angle sphere, and a step edge for MTF work.

## Golden / regression fixtures

For things with no analytic answer — a full pipeline render, an AGC output — store a reference array and
compare. Rules that keep golden tests from becoming a liability:

- Store the **config hash** alongside the array. A golden file whose config changed is stale, not
  failing; detect and say which.
- Compare with a stated tolerance, never exact equality.
- On failure, write the actual output next to the expected one and print both paths. A golden failure
  you cannot inspect wastes more time than it saves.
- Regenerate deliberately via `make golden-update`, never by editing files by hand, and never in the
  same commit as a behaviour change without saying so in the commit body.

## The five tiers

**Tier 1 — Unit.** Analytic identities. No renderer. Covered above.

**Tier 2 — Radiometric bench.** Reproduce standard lab characterisation inside the simulator:
SITF (DN vs. blackbody temperature), NETD, 3D noise decomposition, MTF from a slant edge. These are the
same measurements made on real cameras, so they are directly comparable.

**Tier 3 — Phenomenology.** Qualitative but decisive. Each should emerge without being scripted; treat
them as a checklist run against a demo scene, and automate the ones that can be automated:

- [ ] Diurnal cycle runs; contrast collapses near dawn and dusk (thermal crossover)
- [ ] Overcast flattens the image; clear night sky darkens vehicle roofs and glass
- [ ] Wet asphalt reads colder than dry; shaded ground colder than sunlit
- [ ] A departed vehicle leaves a warm tyre trace and a cool body shadow
- [ ] Fog kills visible and SWIR before LWIR
- [ ] Humid clear air degrades LWIR more than SWIR
- [ ] MWIR shows solar glint at midday and looks like LWIR at night
- [ ] SWIR at night is usable from airglow alone, with no modelled light source
- [ ] A hot exhaust entering frame collapses global AGC contrast
- [ ] FFC freezes the image and resets the fixed pattern
- [ ] Fast lateral motion smears edges in LWIR but not in cooled MWIR

Several of these are automatable as scalar assertions (contrast ratio at 0600 vs. 1200, roof apparent
temperature clear vs. overcast). Automate those; leave the rest as a documented manual pass.

**Tier 4 — Against real data.** Recreate a real dataset scene's geometry, materials, weather and time,
then compare:

- apparent-temperature histograms by semantic class (shape and separation matter more than absolute bias)
- radiometric contrast between object classes and local background
- **noise power spectral density of a flat region** — the single most diagnostic comparison, because it
  exposes wrong 3D-noise ratios immediately
- edge spread function on real edges

**Tier 5 — Task-level.** Train a detector on synthetic only, evaluate on real, and vice versa. Measure
both gaps. If Tier 4's noise PSD comparison fails, no amount of Tier 2 accuracy will save Tier 5.

## Reviewing a test before accepting it

- Would it fail if the physics were wrong, or only if the code crashed?
- Is the tolerance in a unit a human can judge?
- Does it depend on Isaac Sim when it could use a synthetic G-buffer instead?
- Is randomness seeded?
- Does the failure message say what went wrong, or just that something did?
- If it is a golden test, can the reviewer see the diff?
