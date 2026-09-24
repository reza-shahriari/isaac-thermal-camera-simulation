# CLAUDE.md

Project guidance for Claude Code. Read this fully before the first edit in a session.

---

## What this project is

A physically-based multi-band infrared camera simulator. It renders what a real LWIR / MWIR / SWIR / NIR
camera would see, with radiometry that closes in physical units, so the output can be used for sensor
trade studies, perception development, and eventually deployment on an autonomous vehicle.

Primary target: **NVIDIA Isaac Sim 6.0**. A port to Unreal Engine follows later, so the physics core must
not depend on either engine.

The full physics specification lives in **`docs/physics-model.md`**. It is the source of truth for every
equation in this repo. When you implement something from it, cite the section number in the docstring
(e.g. `# docs/physics-model.md §3.2`). When you disagree with it, say so before coding — do not silently
diverge.

---

## Non-negotiables

These are the mistakes that are expensive to discover late. They are not style preferences.

### 1. The physics core is engine-free

`src/irsim/` must never import `omni`, `pxr`, `isaacsim`, `warp`, `carb`, or any engine module, nor the glue
package `irsim_isaac`, nor the ML/imaging stack (`torch`, `cv2`, `PIL`, …, which lives in `src/irsim_eval/`). It is pure
Python + NumPy, importable and testable with plain `pytest` on any machine with no GPU and no Isaac Sim.

All engine glue lives in `src/irsim_isaac/`. This is enforced by a test — see `tests/unit/test_layering.py`.

Reason: the physics iteration loop must run in seconds. If verifying Planck's law requires launching a
simulator, nobody will verify Planck's law. It also makes the Unreal port a rewrite of the glue only.

### 2. float32 or better, everywhere temperature or radiance flows

At 300 K, float16 has a spacing of 0.25 K — five times coarser than a 50 mK NETD. A float16 buffer
anywhere in the temperature or radiance path destroys the sensor's entire sensitivity, silently, while
still producing an image that looks fine.

Applies to: NumPy dtypes, Isaac Sim AOV formats, CUDA texture formats, and any file you write to disk.
Assert dtype at every boundary. There is a test for the encode/decode round trip; keep it passing.

### 3. Noise is added in radiance or electron space, never in Kelvin

The blackbody thermal derivative `∂L/∂T` rises with temperature, so a fixed Kelvin-space sigma is wrong
everywhere except the one temperature it was tuned at. Convert to radiance, add noise, convert back.

### 4. Kirchhoff closure holds for every material in every band

`ε + ρ + τ = 1`, to 1e-6. Author one of the three and derive the others. Never author two independently.
There is a test that walks the whole material library; keep it passing.

### 5. The aperture factor is `π/(4F² + 1)`, not `π/(4F²)`

At F/1.0 the difference is 20%. Defined once in `irsim.optics`, used nowhere else by hand.

### 6. Weather feeds the thermal solver and the atmosphere from one object

Nothing in the code may let a scene run summer weather in the temperature model and winter weather in
the atmosphere model. One `WeatherSeries`, injected into both.

---

## Repository layout

```
src/irsim/              # engine-free physics core (pure Python + NumPy)
  radiometry/           # Planck, band integration, LUTs, apparent temperature
  materials/            # spectral material model, Kirchhoff, Fresnel, library + loader
  thermal/              # energy balance solvers, weather, vehicle regimes
  atmosphere/           # Beer-Lambert transmittance, path radiance
  optics/               # aperture, MTF cascade, distortion, self-emission
  detector/             # photon and bolometer models, NETD
  noise/                # 3D noise model, FPN, bad pixels, NUC residual
  isp/                  # NUC, AGC, DDE, palette
  config/               # pydantic schemas for sensor / material / atmosphere YAML

src/irsim_isaac/        # Isaac Sim glue — the ONLY place engine imports are allowed
  pipeline/             # AOV setup, annotators, Warp-based reference pipeline
  spg/                  # empty on purpose: the .cu / .cu.lua / .usda assets are blocked on one
                         # capability question, written down in its README (DC.1)

tests/
  conftest.py           # synthetic G-buffer fixtures (ramp, uniform, two-material, grazing sphere,
                         # supersampled step edge, moving edge) — the engine-free kernel test bed
  unit/                 # fast, no GPU, no Isaac Sim. Must run in < 30 s total.
  integration/          # requires Isaac Sim. Marked @pytest.mark.isaac, skipped by default.
  golden/               # regression fixtures (reference arrays + tolerances)

configs/sensors/        # one YAML per camera (see docs/physics-model.md §12.2)
configs/materials/      # material library
configs/atmospheres/    # atmosphere presets
configs/environments/   # illumination/weather-regime presets
data/                   # spectral response curves, n/k tables, generated LUTs, weather files
                         # ($IRSIM_DATA_DIR overrides the root)
docs/physics-model.md   # THE physics specification
docs/roadmap.md         # milestones, one-commit steps, risks, ADR backlog
docs/spec-issues.md     # contradictions found in physics-model.md and the resolution the code assumes
docs/decisions/         # ADRs — one file per significant decision
docs/maps/              # frozen 2026-09-10 subsystem snapshots, not maintained — history, not
                         # navigation; read the tree itself for the current shape
scripts/                # LUT generation, validation reports, dataset export
.github/workflows/      # CI: plain-CPython gate only, no GPU, no Isaac Sim (mirrors `make ci`)
```

---

## Commands

```bash
make install       # editable install + dev dependencies
make test          # unit + golden tests (fast, no GPU) — this is the default gate
make test-all      # includes integration tests (needs Isaac Sim)
make lint          # ruff check + ruff format --check
make fmt           # ruff format
make typecheck     # mypy on src/irsim and src/irsim_isaac
make check         # lint + typecheck + test  ← run this before every commit
make ci            # reproduces the GitHub Actions job locally: plain CPython 3.10 venv + make check
                    # (proves the engine-free core needs neither Isaac Sim nor CUDA)
make luts          # regenerate band LUTs from configs + spectral response data
make golden-update # regenerate golden reference arrays deliberately (never to silence a failure)
```

Every target honours `PYTHON=`. Locally this should point at the Isaac Sim interpreter (ADR 0002),
e.g. `make check PYTHON=/path/to/IsaacSim/_build/linux-x86_64/release/python.sh` — the system Python
and a bare conda base typically lack pytest/numpy/ruff/mypy. Any CPython ≥ 3.10 with `.[dev]` installed
also works for everything under `src/irsim/` and `tests/unit/`; only `test-all` needs the Isaac interpreter.

`make check` must pass before any commit. No exceptions, no `--no-verify`.

---

## Workflow: what "a step" means

Work in small, complete steps. A step is done only when **all six** of these are true:

1. **Code** implements one coherent piece, with a docstring citing the physics-model section.
2. **Tests** exist and pass. New physics needs a test that would fail if the physics were wrong —
   not a test that merely calls the function. See the `ir-sim-testing` skill.
3. **`make check` is green.**
4. **`README.md` is updated** — the status table, and anything the change makes untrue.
5. **`CHANGELOG.md` has an entry** under `## [Unreleased]`.
6. **The site shows it** — `make site` (ADR 0139). Modules, tests, ADRs, configs, scripts and
   documents are picked up automatically *from their docstrings*, so the work is writing those;
   a render or a clip has to be named in `site/gallery.yaml` by hand. Everything that can be
   presented belongs there — not debug output. See the `present-on-the-site` skill.

Then commit. One step, one commit.

If a step is turning out bigger than expected, stop and split it rather than committing something
half-tested. Say so in the conversation.

The `ship-step` skill has the exact checklist and commit format. Use it every time you finish a step —
do not reconstruct the process from memory.

### Commit format

Conventional Commits, with a scope from the module list:

```
feat(radiometry): add band-integrated Planck LUT with inverse lookup

Implements docs/physics-model.md §3.2 (b) and §3.3. LUT covers 200-1000 K
at 0.05 K spacing, float32. Inverse by binary search.

Round-trip error < 1 mK across the full range (tests/unit/test_planck.py).
```

Types: `feat`, `fix`, `test`, `docs`, `refactor`, `perf`, `chore`.
Scopes: `radiometry`, `materials`, `thermal`, `atmosphere`, `optics`, `detector`, `noise`, `isp`,
`config`, `isaac`, `spg`, `build`, `docs`.

### Decisions

When a choice has consequences that outlive the commit — a solver, an encoding, a fidelity level you
deliberately skipped — write an ADR in `docs/decisions/NNNN-short-title.md` using the template in
`docs/decisions/0001-record-architecture-decisions.md`. This project involves a lot of "we chose the
cheaper approximation on purpose" and future-you needs the reasoning, not just the result.

---

## Skills

Consult these. They contain project-specific detail that is not in your training data, particularly
around Isaac Sim 6.0, which is recent.

| Skill | Use when |
|---|---|
| `ir-radiometry` | Planck, band integration, LUTs, apparent temperature, any unit question |
| `thermal-materials` | Material properties, emissivity, Fresnel, importing USD assets |
| `thermal-solver` | Surface temperature, energy balance, weather, vehicle heat sources |
| `sensor-noise-chain` | Detector models, NETD, 3D noise, NUC/FFC, AGC and the ISP |
| `isaac-sim-spg` | Anything touching Isaac Sim, AOVs, SPG kernels, Warp |
| `ir-sim-testing` | Writing tests, tolerances, golden data, validation tiers |
| `ship-step` | Finishing any step — README, CHANGELOG, git |
| `present-on-the-site` | Anything worth showing reaching the project site — renders, tests, reports |

---

## Working style for this repo

**Physics before pixels.** The order in `docs/physics-model.md` §16.4 is deliberate. Do not build the
Isaac Sim integration before the radiometry core has passing unit tests. An image that looks thermal is
easy; one that is correct is not, and you cannot tell them apart by looking.

**Prefer a scalar reference implementation first.** Write the clear NumPy version, test it, then optimise.
Keep the reference implementation as the oracle the fast path is tested against.

**Bands are data, not code.** If adding SWIR requires editing the radiance kernel, the abstraction is
wrong — stop and fix the abstraction. This is the main scalability requirement of the project.

**Don't invent physical constants.** They live in `irsim.radiometry.constants`. If you need one that
isn't there, add it there with a source, not inline.

**Flag uncertainty rather than guessing.** Isaac Sim 6.0's SPG API is new and the public documentation
is incomplete. If something is ambiguous, say so and propose a way to verify it, instead of writing code
that assumes the ambiguity resolved in your favour.

**Team context.** Several people work on this. Write for a reader who was not in the conversation where
the decision was made — hence the ADRs and the physics-model citations.
