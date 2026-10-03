# irsim roadmap

**Revision 6, 2026-09-20.** Amends revision 5 (2026-09-15), which replaced revision 3 in full. The plan for
taking `irsim` from where it is today to an L2-fidelity multi-band IR camera simulator in Isaac Sim 6.1,
with a clean upgrade path to L3. Derived from `docs/physics-model.md` (cited as §N.M), the seven project
skills, subsystem audits run against the code and `git log`, and two surveys of production IR simulators,
published camera datasheets, measurement standards, public dataset primary sources and the sim-to-real
literature — [`docs/research/2026-09-15-field-survey.md`](research/2026-09-15-field-survey.md) and
[`docs/research/2026-09-18-thermal-coupling-survey.md`](research/2026-09-18-thermal-coupling-survey.md).

**What revision 6 changes.** On 2026-09-18 the owner restated the requirement that drove them off their
previous simulator and widened it: one building must show its sunlit and shaded parts at different
temperatures; a running engine must warm *the metal around it*, not only itself; and the same must hold
for water, fire and their kin. An audit that day found the point-wise machinery real but reachable from
no scene config, no mechanism anywhere for heat between parts, and no model or step for water or fire
(see *The owner's priorities*, items 1, 7 and 8). This revision adds phase **P**, two lanes (`TC`, `PH`),
six `PT` rows, and moves the mesh lane into P. Nothing shipped is re-described; the ledger is unchanged.

---

## How to read this

**Provenance of every number.** Every figure below marked *measured* was measured on 2026-09-15 on the
working tree at commit `ca5a663` **plus** the uncommitted work in it at that moment: modified
`CHANGELOG.md`, `README.md`, `docs/roadmap.md`, `src/irsim_isaac/maritime_demo.py`,
`tests/unit/test_maritime_stage.py`, and untracked `configs/scenes/vessel_departure_clear_day.yaml` and
`scripts/render_vessel_departure.py`. That is not a clean tree and the distinction matters: several
findings cite `render_vessel_departure.py` by line. Test collection moved 2746 → 2756 during the audits
because parallel sessions were adding files, so a test count is a timestamp, not a property.

**Picking a step.** Take the first step in phase order whose `deps` are all ticked. Ship it with the
`ship-step` skill. Tick it in the **Shipped ledger** in the same commit, with the commit hash.

**Why this document no longer contains 182 fat rows.** Revision 3 held 224,508 of its 298,272 characters
inside markdown table cells, averaging 1,234 characters per row with a 4,958-character maximum. Those rows
were a second changelog: unrenderable, unreviewable as a diff, and unmergeable between two sessions
without a manual rewrite. Shipped work now lives in a compact ledger; the narrative of what shipped lives
where it belongs — in `git log`, in the ADRs and in `CHANGELOG.md`. **Step rows are capped at 600
characters and ledger rows at 200**, enforced by a parser in `make check` (RP.9). A row that needs more
space needs an ADR instead.

**Row format.** `id · what · verification · deps · size · phase`. The **verification** cell states what
would fail if the physics were wrong, and why the tolerance is that number — not that a function was
called. Sizes: S (hours), M (a day or two), L (several days). A step that turns out bigger gets split,
per CLAUDE.md.

**Lane prefixes.**

| lane | subject |
|---|---|
| `RP` | Repair: damage already done to README, CHANGELOG, the roadmap, the ADR corpus and the spec-issues log |
| `PT` | Point-wise surface temperature — the owner's headline requirement, reachable from a scene config |
| `WM` | Warp mesh surface parameterisation — point-wise on curved geometry |
| `TC` | Thermal coupling — heat moving between cells and between parts (an engine warms the metal around it) |
| `PH` | Phenomena beyond opaque solids — water, fire and their kin |
| `AT` | Atmosphere, sky and materials physics |
| `SE` | Sea and maritime |
| `SC` | Sensor chain and published-reference anchoring |
| `EV` | Evaluation methodology and sim-to-real |
| `XD` | External data anchors: public datasets and public measurements |
| `AI` | Asset ingestion: third-party 3D models into a scene |
| `IG` | Isaac glue integrity |
| `GT` | Gates, tests and tooling |
| `DC` | Decisions, deferrals and probes |

Legacy ids (`M0`–`M12`, `ME`, `MS`, `MM`, `MP`, `IU`) are preserved in the ledger and in ADRs. They are
not reused for new work, and eleven of revision 3's `deps` cells named ids that existed only as
letter-suffixed rows (`ME.1`, `ME.2`, `ME.3`, `M10.7`, `M10.9a`), which broke the picking rule for eleven
rows including `M10.9b`. New ids are two letters and a number, and every `deps` cell below names a row
that exists in this document.

**ADR numbers are allocated at write time, never forward.** The highest existing ADR is 0089 and four
numbers below it are cited but were never written (0042, 0062, 0069, 0079). Revision 3 also instructed a
future author to "record as ADR 0073 when scoped" for the exhaust plume, and ADR 0073 is the
visible-companion environment dome, Accepted 2026-09-14 — following that instruction would overwrite a
live decision record. In a three-session shared tree a forward-assigned number is a scheduled collision.
**This document names ADR subjects, not ADR numbers.** RP.4 writes the two that shipped code cites; DC.4
records the rule.

**Commit scopes.** `pipeline`, `validation`, `eval` and `io` remain proposed additions to CLAUDE.md's
list (open question 9). Until approved, use `build` for infrastructure and the physics scope a step tests.

---

## Current state, measured 2026-09-15

Revision 3's "Current state (2026-09-10)" section is deleted. Every claim in it was false — it said
radiometry had "no LUT, no inverse, no R(λ) loader" (all shipped 2026-09-11), that every other
`src/irsim` package was a one-line stub, that `src/irsim_isaac` had no AOV code, and that configs and
data were empty. It was the second section of the document, so a session reading top-down formed a wrong
model of the project before reaching any table.

| thing | measured |
|---|---|
| Physics core | 145 modules under `src/irsim`, engine-free, enforced by an AST scan |
| Isaac glue | 30 modules under `src/irsim_isaac`; `src/irsim_isaac/spg/` holds one file, `README.md` |
| Tests | 2,756 collected across 163 unit files and 6 golden files; `2748 passed, 8 skipped` at the audit run |
| Of which structural | 540 of 2,756 come from two per-file AST scanners (`test_layering.py` 359, `test_aperture_guard.py` 181). They scale with file count, not with verified physics. Quote coverage, not the raw count |
| Coverage | `src/irsim` 93.9 %, `src/irsim_eval` 85.6 %, `src/irsim_isaac` 43.5 %. Six glue files at 0 %, totalling 1,101 statements; `warp_stages.py` 38 % |
| Gate time | ~130 s for `tests/unit tests/golden`, against CLAUDE.md's stated 30 s. Slowest single test 12.04 s, exercising a solver no scene can reach |
| Integration tests | 15 files, 4,480 lines, all auto-marked `isaac`, run in no automated job anywhere |
| Configs | 5 sensors, 19 materials plus a mapping, 7 atmospheres, 8 scenes |
| Demo lanes | Six render drivers produce frames from one command: aerial demo, quad flight, aircraft pass, maritime demo, vessel departure, car ignition |
| Point-wise temperature | Real, correct, end-to-end — and reaching **two prims in two of eight scenes**, both authored from Python rather than from the scene config, on the lane the owner ranked third |
| ADRs | 85 files, 0001–0089 with gaps at 0042, 0062, 0069, 0079. None is ever marked Superseded although the template offers it |
| Docs | `docs/roadmap.md` 888 lines / 304 KB; `CHANGELOG.md` 3,025 lines / 270 KB in one `[Unreleased]` section with 20+ repeated headings; `docs/maps/` 729 KB across 12 files with one commit ever |

**Measured again on 2026-09-18** (commit `7f0b372`, clean tree, `make check` green: lint, mypy over 187
files, 2 985 fast and 251 slow tests, 3 244 collected). All eight scene configs load; every step id claimed
in the last forty commit subjects is ticked and every ticked row has a commit; the CHANGELOG's newest
section matches `git log` entry for entry. Against the owner's requirements: no scene YAML carries a
`patch:` block, `cell_shadow` has no caller outside its tests, the only bound fields are hand-built in
`car_demo.py` with no solar term, `FacetSolver.net_flux` has no inter-cell term and no node-to-node
conductance exists anywhere, the engine is a scripted ΔT schedule, no balance has a latent-heat term,
and fire, flame and plume have no code. The full findings, with file and line, are in
[`docs/research/2026-09-18-thermal-coupling-survey.md`](research/2026-09-18-thermal-coupling-survey.md).

---

## The owner's priorities, restated as acceptance criteria

These are requirements, not preferences. Each has a lane that owns it and a criterion that can be read
off a rendered frame.

1. **Per-point surface temperature, not one value per object.** This is the defect that drove the owner
   off their previous simulator, restated on 2026-09-18 with its example: *a building, some parts in the
   sun and some in shadow*. A lane satisfies it when a rendered frame of that lane's target shows a
   temperature gradient *across one prim*, authored from a scene config — the config, not a Python
   driver, declares the patch, the occluders and the material. Today: the car's bonnet and the road,
   both built in `car_demo.py`; no config declares a patch. Owned by `PT` and `WM`; `PT.20` is the
   reference scene.
2. **Aerial first, then maritime, then ground.** A lane is done when **a scene config plus one command
   produces frames** — and, from this revision on, frames that carry float32 planes and a config-hash
   sidecar (IG.13), because three of the six drivers currently emit 8-bit PNGs only and two of those three
   are the aerial and maritime flight renders.
3. **No IR camera.** All validation against reality uses freely available public data. **No step in this
   document proposes a hardware purchase.** Note additionally that acquiring a camera would not rescue
   §15's radiometric target (see *Not adopted*), so the constraint costs less than it appears to.
4. **Isaac Sim is the only engine.** Engine portability stays a *property* — CLAUDE.md non-negotiable #1,
   enforced by `test_layering.py`, and cheaply extended by keeping the Warp kernels free of
   `omni`/`pxr`/`isaacsim` imports. No step is spent on an Unreal deliverable (open question 2).
5. **Grayscale, white-hot.** All five sensor configs carry `palette: gray` and `to_display8` defaults to
   it. SC.11 adds black-hot as a *config switch and an evaluation axis*, not as a default — real maritime
   sets carry both polarities and a detector trained on one fails on the other.
6. **Evaluation is a first-class deliverable.** The `EV` lane is not a reporting afterthought; it ends in
   EV.9, a measurement that can return a negative about the project's own headline feature.
7. **Parts interact through temperature.** In the owner's words (2026-09-18): *an engine will increase
   its temperature while being used — but not just itself, also the metals around it.* Today the only
   coupling between parts is radiation onto a parallel panel; nothing conducts through a mount, a bracket
   or a panel, and no balance has a neighbour term. Satisfied when a rendered frame shows a bracket, a
   wing and a bonnet warming *in the order of their conductance path* from a solved engine node, with
   energy conserved and the hot soak after key-off that every under-hood measurement records. Owned by
   `TC`; `TC.6` is the reference scene.
8. **Water, fire and their kin are modelled, not painted.** *This is correct about water, fire and
   something like these.* Today water is the open sea only, no balance has a latent-heat term, and fire,
   flame and plume have no code. Satisfied when a wet half of one road renders colder than its dry half
   and dries on the measured time scale, a puddle reflects the sky at grazing angles, an exhaust plume is
   bright in MWIR and nearly invisible in LWIR, and a flame rails the camera the way FLIR's own notes
   say it does — each from a scene config. Owned by `PH`.
9. **Realism is a switch, and the subject gets the real thing.** In the owner's words (2026-09-28): heat
   exchange between objects must be switchable *"with a boolean"*; the main object — a car, a drone —
   needs *"a real temperature estimation"* even when it is slow to solve; and a second switch decides
   whether an object's temperature changes at all, including freezing it at a chosen moment, *"say in
   the night when the engine is warm enough"*. Today which parts exchange heat is fixed by how a scene
   was hand-wired, the demo drone's parts exchange none, and nothing can be frozen. Satisfied when one
   scene config renders the same car with exchange off and on and the difference lands on the road and
   the wall it faces, the main object's parts warm through each other, and a frozen engine holds its
   temperature while the camera keeps filming. Owned by `TC`; `TC.9`–`TC.12`.

Items 1, 7, 8 and 9 are lane-independent physics and are scheduled first (phase **P**); items 2 and 3 order
the *scenes* that consume them. A reference scene in phase P is the smallest scene that can show the
requirement, not a lane deliverable: `PT.20` is a block on a ground patch, not a city.

---

## Do next

<!-- next:begin -->
<!-- generated by `python scripts/next_step.py --write`; do not edit by hand -->

### Start here → `WX.3` · Weather: clouds, fog, haze, rain and snow

> **The visible cloud to §7.5's five criteria** (`isaac-weather-fx`): energy-conserving in-scatter with a sky term that survives at low density, octaves, a premultiplied dome composite, `ptvol/maxBounces` raised from 2.

`WX.3` is phase A, size L, and unblocks 10 other step(s).

#### Then, in order — 83 open steps

| # | step | lane | phase | size | unblocks | waiting on |
|---|---|---|---|---|---|---|
| 1 | **`WX.3`** | WX | A | L | 10 | ready |
| 2 | **`WX.7`** | WX | A | L | 8 | `WX.3` |
| 3 | **`WX.8`** | WX | A | M | 7 | `WX.7` |
| 4 | **`WX.11`** | WX | A | S | 2 | ready |
| 5 | **`WX.4`** | WX | A | M | 2 | ready |
| 6 | **`WX.5`** | WX | A | M | 1 | `WX.3` |
| 7 | **`WX.6`** | WX | A | M | 1 | `WX.4` |
| 8 | **`WX.10`** | WX | A | M | — | ready |
| 9 | **`WX.9`** | WX | A | M | — | `WX.5`, `WX.6`, `WX.8` |
| 10 | **`XD.14`** | XD | A | M | — | ready |
| 11 | **`XD.6`** | XD | A | L | — | ready |
| 12 | **`AT.26`** | AT | B | S | — | ready |
| 13 | **`IG.16`** | IG | B | M | — | ready |
| 14 | **`WX.13`** | WX | C | M | 5 | `WX.8` |
| 15 | **`WX.15`** | WX | C | L | 3 | `WX.13` |

…and 68 more — `python scripts/next_step.py --queue 40`.

<!-- next:end -->

**How this is produced.**

```bash
python scripts/next_step.py            # the one step to start now
python scripts/next_step.py --queue 20 # the next twenty, in order
python scripts/next_step.py --lane PT  # restrict to one lane
```

The picking rule stated in *How to read this* — "the first step in phase order whose deps are all
ticked" — **does not pick a step**. Measured on this revision the day it landed: **51 of 109** open
steps satisfied it at once, thirteen of them in phase 0 alone, and **67 of 109** open steps have no
dependents at all, so the dependency graph cannot order the majority of the plan. What was actually
choosing the next step was its row's position in a table. That is not a rule, it changes whenever
anyone reorders a table, and it cannot be argued with.

`scripts/next_step.py` is the rule. It is a **topological sort**, so a step never comes before
something it depends on, with ties broken by a total order:

| | tiebreak | why |
|---|---|---|
| 1 | **Promoted** | The documented exceptions, and the only place judgement enters. The mechanical key counts dependents; it cannot see that a step prevents a recurring loss. Kept to three at most — a long list means the rule itself is wrong |
| 2 | **Phase**, 0 → P → A → B → C → X | Where the owner's ordering lives: repair, then the physics the owner's requirements need, then the scene lanes in application order. A dependency may still pull an X step forward; the sort does that on its own |
| 3 | **Dependents, descending** | Transitive, not immediate. A step unblocking ten outranks one unblocking two |
| 4 | **Size, ascending** | Between equal leverage, the small one first |
| 5 | **Step id** | A total order, so two sessions asking "what next" get the same answer |

**The queue is deliberately not written into this document.** A pasted list goes stale the moment a
step is ticked, and a stale queue is worse than none because it still answers. Ticking a step in the
table is all that is needed; the next call recomputes.

`tests/unit/test_roadmap_queue.py` is the guard, and it is what makes the answer safe to act on
without reading this document: the order is **total** (one head, two runs agree), **sound** (no step
before its dependencies), **complete** (every open step appears exactly once, so nothing is silently
dropped), and every pick is **the best available one at that moment** — so a disagreement with the
queue is a disagreement with the key above, which you can change.

---

## Phase plan

Ordered by the owner's application order, with one phase before all of them for damage that is live in
the working tree right now.

**CPU correctness before any acceleration.** The owner's rule, stated 2026-09-15: no GPU or device
work while the CPU reference still has known defects. The engine-free NumPy pipeline is the oracle
every fast path is tested against (ADR 0018), so a wrong oracle makes an accelerated path worth less
than nothing. Consequences, applied throughout this document:

* The **SPG/CUDA shader lane is not scheduled at all.** Legacy `M10.12` and `M10.13a`-`M10.13e` are
  dropped from the delivery phases; `DC.1` keeps only the *probe* that would tell us whether they are
  even possible on SPG 0.4.0, and `R2` stays open in the risk register. Nothing depends on them.
* **No step moves an existing stage onto a device.** Warp stage twins already exist and are already
  equivalence-tested; making more of the frame device-resident is a speed change and waits.
* Where a step needs a GPU-capable *library for a capability rather than for speed* — `WM` uses Warp's
  mesh closest-point query because it is the only exact per-pixel surface parameterisation available
  without a renderer change — it **runs on the Warp CPU backend by default**, keeps a brute-force NumPy
  closest point as its oracle (ADR 0018/0061), and is justified by what it makes possible, not by a
  timing figure. The RTX 5090 numbers quoted in `WM` are a headroom note, not a reason.
* `WM` was scheduled behind the phase-A CPU defects while `AT.1`, `SC.1`, `PT.1` and `PT.2` were open.
  All four shipped by 2026-09-17, so that reason is spent; revision 6 moves `WM` into phase **P** with
  the rest of the point-wise physics. It stays what it was — a capability on the Warp CPU backend with a
  NumPy oracle, never an acceleration — and `PT.22` gives it a CPU-only shadow and sky-view oracle that
  needs no Warp at all.

**Phase P, added in revision 6.** The owner's three physics requirements — per-point temperature from a
config, heat between parts, water and fire — are needed by every scene lane and owned by none, so
putting them under the ground lane (where the 2026-09-18 audit found them: `PT.11` in phase C, no row at
all for a thermal network or for fire) meant the headline requirement was scheduled last. Phase P sits
between repair and the aerial lane. It is **CPU only** like phase A: every step in it is engine-free
physics with a test that fails if the physics is wrong, and where a step also has an in-engine half the
row says so and names the step that closes it.

| phase | contents | exit |
|---|---|---|
| **0 — Repair** | `RP.1`–`RP.10`, `PT.3`, `PT.4`, `IG.1`, `IG.5`, `IG.8` | The three shared documents are true and mergeable; no shipped physics result rests on a measured error |
| **P — Point-wise and coupled physics** | `PT.6`–`PT.8`, `PT.11`, `PT.12`, `PT.14`, `PT.15`, `PT.17`–`PT.22`, `WM.1`–`WM.7`, `TC.1`–`TC.7`, `TC.9`–`TC.12`, `TC.16`, `PH.1`–`PH.8`, `PH.13`, `PT.25`, `PT.26`, `AT.24`, `IG.20`, `IG.21`, `IG.22` | **CPU only.** From a scene config plus one command: a wall half in sun (`PT.20`), an engine warming the metal around it with hot soak after key-off (`TC.6`), a road wet on one half and dry on the other (`PH.2`), and a plume bright in MWIR and faint in LWIR (`PH.6`) — each with its engine-free test green; the rendered frames are the in-engine half, staged by `IG.20` |
| **A — Aerial to the bar** | `AI.1`, `AI.2`, `AI.5`, `AI.7`–`AI.12`, `AI.14`–`AI.17`, `PT.1`, `PT.2`, `PT.5`, `PT.9`, `PT.23`, `TC.13`, `TC.17`, `AT.1`–`AT.5`, `AT.10`–`AT.12`, `AT.15`, `AT.16`, `AT.18`, `AT.19`, `AT.14`, `AT.29`, `AT.30`, `AT.31`, `AT.32`–`AT.36`, `XD.14`, `IG.23`, `IG.24`, `XD.6`, `AT.20`, `SC.1`–`SC.4`, `SC.17`–`SC.26`, `IG.2`, `IG.6`, `IG.13`, `IG.18`, `GT.1`, `GT.2`, `PT.24`, `PT.28`, `AT.21`–`AT.23`, `AT.25`, `AT.27`, `AT.28`, `SC.28`–`SC.34`, `IG.19`, `EV.14`, `EV.15`, `WX.1`–`WX.11` | **CPU only.** An aerial scene config plus one command produces float32 frames whose target carries a gradient across one prim, with a per-pixel slant path behind it; **clouds appear in the infrared in both render tiers and meet §7.5's realism criteria against calibrated full-sky LWIR (`XD.6`), and in the path-traced tier a target behind or inside a cloud is occluded in both bands while in the real-time tier it is occluded in neither (ADR 0169, checked in the engine)** |
| **B — Maritime to the same bar** | `AI.3`, `AI.4`, `AI.6`, `PT.10`, `SE.1`–`SE.3`, `OC.6`, `OC.7`, `XD.3`, `IG.16`, `AT.26` | A maritime scene config plus one command produces the same, with the sea model's angular envelope recorded |
| **C — Ground and automotive** | `AI.13`, `PT.13`, `PT.16`, `TC.8`, `TC.14`, `TC.15`, `PH.9`–`PH.12`, `PH.14`, `AT.6`–`AT.9`, `AT.17`, `OC.8`, `XD.10`, `GT.7`, `PT.27`, `WX.12`–`WX.20` | Deferred material breadth stays deferred (see *Deferred deliberately*); what lands is depth on surfaces already modelled, plus the phenomena rows no earlier scene needed |
| **X — Cross-cutting, continuous** | `SC.5`–`SC.16`, `SC.27`, `EV.1`–`EV.13`, `XD.1`, `XD.2`, `XD.4`, `XD.5`, `XD.7`–`XD.9`, `XD.11`–`XD.13`, `AT.13`, `IG.3`, `IG.4`, `IG.7`, `IG.9`–`IG.12`, `IG.14`, `IG.15`, `IG.17`, `GT.3`–`GT.6`, `GT.8`–`GT.11`, `OC.1`–`OC.5`, `OC.9`, `OC.10`–`OC.13`, `DC.1`–`DC.6` | Runs alongside; `EV` gates nothing but is gated by `PT.9`/`PT.10` for its headline measurement |

**Dependency shape.** Phase 0 blocks nothing technically but blocks *knowing what is true*, and three
sessions share this tree. `AT.1` and `SC.1` are the two critical-priority physics defects and are
independent of each other. `WM.1` is a probe and gates `WM.2`–`WM.4`; the lane is phase P, so `PT.9`'s
dependency on `WM.3` no longer crosses a phase boundary (revision 5 had phase A's exit waiting on a
phase-B step, which the queue ignored and the prose did not). `EV.9` — the per-point ablation —
depends on `PT.9` or `PT.10`, because a negative from a feature that is not switched on in the measured
targets is not a negative about the feature.

---

## Shipped ledger

Revision 3 carried 182 step rows averaging 1,234 characters, which is what made it unmergeable and what
let four subsystems be over-reported. Shipped work is now one line per milestone. **The narrative of what
shipped lives in `git log`, in the ADRs and in `CHANGELOG.md`** — those are per-commit and per-file, so a
whole-file overwrite cannot silently revert them the way it reverted the MP rows.

Rows are capped at 200 characters. Legacy step ids inside a milestone remain valid citations in ADRs
and docstrings.

**What the hash is** (`RP.6`, backfilled 2026-09-16). The last commit that *claimed* one of that
milestone's steps, by the three markers this project's commit messages actually use: `(M9.4)` closing
the subject, `Roadmap MP.1` in the body, or `Implements M12.2`. It is deliberately not "the last
commit that mentions the milestone" — mature milestones are cited by everything downstream, and that
rule hands `M9` the roadmap-rewrite commit. An `open` row has no hash and reads `—`.

The point is recoverability, not attribution: one hash per row is where to start a `git log` when this
file has been clobbered again, which is the failure the revision exists to fix. Two independent
cross-checks came out right — `MM` resolves to the same `ca5a663` lineage its own note already cited
for Tier 3, and `M7` resolves to `1a0f13c`, the commit `RP.7` separately identified as when
`data/nk/glass.csv` first appeared.

| milestone | state | hash | note |
|---|---|---|---|
| M0 build, layering, encoding, goldens | done | `e0c6249` | `test_layering.py` 359 cases, `test_temperature_encoding.py`, `GoldenStaleError` machinery |
| M1 radiometry: Planck, band integration, LUTs | done | `3b8172b` | Four-quantity float32 LUTs, 200–1000 K at 0.05 K, kernel-identical lookup, band-hash sidecar |
| M2 Isaac gate spike | done | `c440271` | ADR 0014 and four addenda; M2.4 corrected the position frame against an independent oracle |
| M3 optics | done | `ddad9d0` | Aperture factor defined once, AST guard, 181 cases. `SC.4` adds the missing aberration term |
| M4 detector | done | `7bd10d1` | NETD anchoring, NETD(373)/NETD(300) = 0.576. `SC.1` wires the photon-FPA electron budget |
| M5 ISP | done | `b373f02` | NUC, AGC, DDE, grayscale default. `SC.10` adds the temporal behaviour `display.py:286` requires |
| M6 thermal solvers and weather | done | `b870d57` | Two-node and cabin are reachable from a scene since `PT.15` (schema v10 `back:` and `cabin:`). Heat traces likewise: `PT.16` |
| M7 materials | done | `1a0f13c` | 19 materials, Kirchhoff walk, Fresnel, n/k. **M7.9's stated Beer-Lambert derivation never ran**: `RP.7` |
| M8 atmosphere, grey and layered | done | `e4caef9` | Layered exponential-sum slant path, R13-anchored. Per-pixel slant path was never in scope: `AT.1` |
| M9 sensor chain | done | `d38c3f0` | 3-D noise, FPN, bad pixels, NUC residual, budget test. M9.8's IIR wiring is recorded by ADR 0082 |
| M10 Isaac pipeline | partial | `b2f89f2` | AOVs, IrCamera, Warp stage twins, six demo stages. M10.12 and M10.13a/b/e blocked: see `DC.1` |
| AI asset ingestion | partial | `pending` | Per-asset material map + CPU Blender prep (ADR 0128), solvable geometry (ADR 0132), and the asset flies in both bands (ADR 0133). Phantom 4: 48.8% -> 100%. Functional part decomposition and a bound battery (ADR 0138, `AI.5`). Still one temperature per prim: `AI.2`'s remainder. A shared link now arrives through a licence gate with committed provenance (`AI.8`, ADR 0150); the `ingest-asset` checklist is `AI.9` |
| M11 multi-band and aerial extras | done | `3553b33` | `72e8142` shipped the NIR config and response. Specular lobe not wired per pixel (ADR 0067) |
| M12 Tier 4 acceptance | partial | `e22c010` | The run fails and says so. `EV.1`-`EV.4` redo it before its attribution is used |
| ME evaluation data lane | partial | `2cac77d` | ME.5 measured 365 clips on a hashed archive. ME.7 blocked on labels, not compute: `EV.11`, `XD.11` |
| MS sky targets (phase 1) | done | `f68076c` | Sky model, cloud clutter, point targets, aerial scene. Per-pixel slant path outstanding: `AT.1` |
| MM maritime (phase 1b) | done | `379fdef` | Analytic sea, Cox-Munk facets, sea skin `c6f98aa`, Tier 3 `ca5a663`, MM.7 vessel-departure film |
| MP point-wise temperature | partial | `a341162` | MP.1-MP.4b shipped; the working-tree roadmap reverted them to open. MP.5 is now `PT.7` |
| IU engine interface | open | — | `IU-29` is an Unreal deliverable and is open question 2, not a step |

**Corrections this ledger makes to the working-tree roadmap**, all verified against `git log`: MP.3,
MP.4a and MP.4b are shipped and were reverted to un-started prose by a whole-file write; MP.5's row was
deleted entirely; the NIR Si-CMOS deferral at line 536 is stale because `72e8142` shipped it; M9.8's
"wire the IIR into `run_frame`" claim is corrected by ADR 0082; M6.14's cabin node, M6.16's heat traces
and M6.8's two-node solver are shipped and unreachable; and M11.6's electron-budget claim is true of the
module and false of the render path.

---

## RP — Repair (phase 0)

Three sessions share this working tree, and the clobbering this document diagnoses is **live right now**:
`git diff --numstat` shows `CHANGELOG.md` at 22 insertions and 135 deletions, `README.md` at 5 deletions
and `docs/roadmap.md` at 3 insertions and 5 deletions, and the deleted lines are the MP.3, MP.4a, MP.4b
and ADR 0087/0088/0089 entries. Capping row length does not stop a whole-file write. `RP.3` does.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| RP.1 | ✅ **done.** README's four broken status cells restored — `materials` had 1,473 chars of changelog prose where the state token belongs, `detector` 978, `noise` 976, `isp` 669. Prose moved to Notes. MP.5's limitation paragraph was clobbered a **third** time; `RP.3` is the structural fix, not this row. | **Measured.** `test_readme_status_table.py` was red on 4 of 15 rows, now green, with a negative control that reintroduces the defect and fails. Pins shape not content: Notes may say anything, State holds one of four tokens, Tier a tier. Zero CHANGELOG deletions. | — | S | 0 |
| RP.2 | ✅ **done.** Five dated sub-sections replace one `[Unreleased]` of 3,300 lines with twenty repeated headings; seventeen former heading runs merged, one set each. Dates from `git blame` per block. | **Measured.** The regrouping moved whole bullet blocks and refused to write until the multiset of non-heading lines matched — **3,242 lines unchanged** (it caught its own preamble first run). `test_changelog_structure.py`: 8 cases; injecting a second `Added` turns it red. | — | M | 0 |
| RP.3 | ✅ **done.** `scripts/stage_own_hunk.sh` is the default path: a `check` subcommand, `.pre-commit-config.yaml` wiring it as a hook, `make stage`, and the `ship-step`/README instructions now teach it over `git add -A`. | **Measured.** `test_stage_own_hunk.py`'s synthetic two-author tree: `check` refuses when a plain `git add` would clobber a commit made since the snapshot, passes when `stage` produced the staged content. 5 new cases, 13 total, 0 red. | — | S | 0 |
| RP.4 | ✅ **done.** ADR 0042 (angular-emissivity levels, and the S12/S18 resolutions) and ADR 0079 (sea-water n/k + Cox–Munk), plus `test_adr_citations.py`. 0062 was written meanwhile; 0069 is a phantom inside a post-mortem of a rejected revision, allowlisted with its reason. | **Measured.** Every citation resolves; the parser reads compound forms (`ADR 0053/0054`) a leading-number reader would skip on 20 sites. 13 cases; hiding 0042 turns 2 red. 0079 supplies 0078's deferred salinity bound: Δε_B ≤ 3.0e-3 per 1 % in n, worst at 85°. | — | M | 0 |
| RP.5 | ✅ **done.** One `**Status:**` label across all 88 ADRs (three spellings collapsed), and the three live supersessions recorded by clause rather than by retiring the document: 0059→0077, 0006→0014, 0048→0071. | **Measured.** 0059 forbade a "second blur" that `stage.py:77` applies, because 0077 makes a bolometer's duty 1 — different mechanisms, both real. 11 cases: sanctioned state words, every supersession target resolves, no self-supersession, body struck through. Reverting 0059 turns 3 red. | — | M | 0 |
| RP.6 | ✅ **done.** All 17 shipped milestones carry the hash of the last commit that *claimed* one of their steps — `(M9.4)`, `Roadmap MP.1` or `Implements M12.2`, the three markers the messages actually use. `IU`, the one open row, carries none. | **Measured.** `test_shipped_ledger.py` resolves every hash against this repository, not just its shape; a well-formed `deadbee` turns it red. Cross-checked: `MM` lands on its note's own `ca5a663` lineage, `M7` on `1a0f13c` — the commit `RP.7` found independently. 7 cases. | RP.1 | M | 0 |
| RP.7 | ✅ **done.** All sixty rows carry a status (`ADR NNNN` / `code` / `open`); the "everything else is open" line is replaced by counts the tests recompute. **S13 reopened, and its resolution was wrong:** `glass.csv` is fused silica, supplying shape not magnitude, so τ cannot be derived from it. | **Measured.** Beer–Lambert over the authored 5 mm gives nir 0.935 / swir 0.936 / mwir 0.109 / lwir 0.000 against 0.77 / 0.70 / 0.02 / 0.0 — three bands off, not one; only LWIR agrees. Glass declares `authored: S13`. 23 cases, 2 negative controls. | — | M | 0 |
| RP.8 | ✅ **done.** `spg/` is described as empty on purpose (its README says so; `DC.1` blocks it) and `docs/maps/` as frozen 2026-09-10 history rather than navigation. Line 13, the scope list and the version string are untouched — open questions 2 and 9. | **Measured.** `test_claude_md_layout.py` parses the block: every path exists, a directory claiming a file type holds one, and a directory whose own README disclaims currency is not sold as live. Red on both before; restoring either sentence turns it red again. | — | S | 0 |
| RP.9 | ✅ **done.** `test_roadmap_row_length.py` owns the 600-character step-row cap; the ledger's 200 already has an owner (`test_shipped_ledger.py`, RP.6) and is not duplicated. | **Measured.** 12 cases. Scope is checked, not assumed: no row found above the first lane heading, no ledger row at the wrong cap, and the generated queue block excluded because its rows start with a position number. Self-tested on a synthetic row; fattening `IG.13` past the cap turns it red. A lint, not counted as a verification. | — | S | 0 |
| RP.10 | ✅ **done.** Ten verification cells rewritten with an assertion and a tolerance: `EV.6`, `EV.11`, `XD.4`, `XD.5`, `XD.7`, `XD.11`, `IG.3`, `IG.4`, `GT.6`, `DC.3`. The two in-engine ones (`IG.3`, `IG.4`) say what a CPU-only session ships. | **By inspection.** Each cell names what fails: a closed form (HTV, Sobel, GLCM), a band (0.41–0.54 mAP, 25 W/m², 10 mK), a resolvable hash, or a recorded refusal. Rows stay under the cap (`test_roadmap_row_length.py`). | — | S | 0 |
---

## PT — Point-wise surface temperature

The owner's headline requirement. What shipped (ADR 0087 `PlanarThermalField`, ADR 0088 spatial sources,
MP.3 `point_bridge`, ADR 0089 `VehicleSourceSolver`, MP.4b the car demo) is real, correct and end-to-end —
and reaches **two prims in one of eight scene configs**, on the lane ranked third, at night, authored from
hand-written Python. Six things stand between that and the requirement, and revision 3 carried none of
them: no per-cell solar or shadow, no patch in the scene schema, a world-frame-only bridge so nothing that
moves can carry a field, a nested-rectangle view factor summing to 1.400, an axis guard that is 4.9×
wrong on a rotated patch, and no binding on either priority lane.

This is also the single most externally supported item in the plan. The current state-of-the-art LWIR
drone-detection study ranks target-crop Sobel gradient variance as the **second** largest measured
sim-to-real gap driver (Cohen's d = 1.224), writing that simulated targets "act as uniform silhouettes
rather than noisy, physical heat sources"; DIRSIG's own manual names the bonnet-over-engine case verbatim
as its documented limitation. `EV.9` turns that from a conviction into a cited number.

**What the 2026-09-18 audit added.** `PT.1`, `PT.2` and `PT.5` shipped the per-cell shadow, the patch in
the schema and the moving frame — and none of it is reachable from a scene config: `_build_thermal_field`
never reads `SurfaceSpec.patch` (a patched surface is still one facet), the schema has no occluder field,
`cell_shadow` has no caller outside its tests, no shipped YAML carries a `patch:` block, and the only
bound fields are hand-built in `car_demo.py` with **no solar term at all**, so a noon run of the car
scene would render a sun-free bonnet over a solved sunlit road. Several docstrings and CHANGELOG entries
say otherwise; they are corrected by `PT.17`. Shadows also come only from authored rectangles (never from
geometry, and never from a neighbour), the sky view factor is the unoccluded form, and the one scene the
owner named — a building — has no row. `PT.17`–`PT.22` close those gaps in phase P; `PT.6`–`PT.8`,
`PT.11`, `PT.12`, `PT.14` and `PT.15` move from phase C to P because they are the same physics.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| PT.1 | ✅ **done.** `irsim.thermal.shadow`: `ShadowRectangle` occluders and an exact ray–rectangle `cell_shadow` per cell, feeding the `shadow` argument `solar_loading` always accepted and nothing varied. Gates the beam only; diffuse sky survives. | **Measured.** A 4 m concrete wall at 45° sun, half occluded: lit 305.1 K vs shaded 283.0 K — a **22.1 K step across one prim**. Every cell holds its own §6.1 root to <1 mK. The per-surface `shaded` bool is flat to 1e-6 and wrong by the full step. 10 cases. | — | M | A |
| PT.2 | ✅ **done.** Schema **v7**: `PatchSpec` carries origin, axes, extents, cell size, thickness, frame and prim binding; `Scene.patches` exposes them by surface name. Optional, so every v4–v6 scene is unchanged. | **Measured.** The car demo's bonnet and road patches reproduce from a declaration with **bit-identical** cell centres (array equality, not approx) and bit-identical field temperature after 1500 s. Non-perpendicular axes, zero thickness and zero extents raise at load. 16 cases. | — | M | A |
| PT.3 | ✅ **done.** Clamped, not partitioned (ADR 0090): `clamp_view_factor_sum` rescales `underbody`/`engine_bay`/`exhaust_pipe`'s view factors proportionally wherever their sum exceeds 1, and `build_ground_field` uses the clamped values. Warns loudly when triggered. | **Measured.** Σ view factors ≤ 1 + 1e-6 on every cell of both car scenes (`test_ground_radiator_view_factors_never_exceed_one`), reproducing the pre-fix peak 1.400 / 28 cells exactly as the warning message. 6 new cases in `test_spatial_sources.py`. | — | S | 0 |
| PT.4 | ✅ **done.** `axes=` is **removed**, not guarded: the configuration factor is Howell C-11 for a *differential element*, which has no in-plane orientation to depend on. A rotated patch now gets the right answer rather than raising — better than the acceptance asked for. | **Measured.** 0.152 vs 0.104 on one element before the fix. 6 cases: a world-space quadrature oracle that walks the real corners (so a frame mix-up fails it), one rectangle labelled two ways, and a whole-configuration rotation. Re-mixing turns 3 red. | — | S | 0 |
| PT.5 | ✅ **done.** `apply(world_from_local=...)` plus `local_frames`; `IrCamera` reads the matrices off the stage each frame. World-frame patches are bit-identical and never touch USD. | **Measured.** A prim translated 10 m and yawed 90°/215°: the same material point reads the same cell to **1e-6 K** across four poses, where a cell spans ~4 K. Identity-pose control disagrees by >5 K; the transposed matrix by >1 K, so USD's row-vector convention is pinned. 7 cases; in-sim read is `IG.2`. | PT.2 | M | A |
| PT.6 | ✅ **done.** `Scene.surface_properties(name)` and `Scene.surface_materials`; `build_bonnet_field` / `build_ground_field` read C, ε, α and k, δ from the scene's `bonnet` / `asphalt` surface; `build_car_demo` refuses an authored override, naming `configs/materials/`. | **Measured.** Both fields equal `from_material` bit for bit: asphalt C 101 200 (was 60 000), α 0.90 (0.88), ε 0.935 (0.95); paint C 4399 (8000), ε 0.853 (0.92), α 0.94 (0.90); the operators carry the materials' k, δ. The overcast bonnet is 17 K max–min at 1500 s, the road's engine patch 2.4 K. 2 cases. | — | S | P |
| PT.7 | ✅ **done.** `build_car_demo(spin_up=True)`: both fields spun up through `spin_up_hours` with the car present (occlusion, cold radiators, shadow), cached; `emission_factor` on the balance returns one reflection off a grey body (ADR 0088 addendum); `spin_up=False` is the old start. | **Measured.** Clear night frame 0: a **4.5 K** standing patch, 24 h and 48 h spin-ups identical to 0.1 mK; overcast +0.009 K (the missing reflection had made it −2.1 K); the engine's 30 min growth is 1.81 K either way; `spin_up=False` is the uniform start bit for bit. The old pin is inverted. | — | M | P |
| PT.8 | ✅ **done.** `ThermalField(keep_ticks=, on_tick=)`: a `deque` ring (`None` keeps all, the per-prim default), a running SHA-256 fed per tick, a hook that sees every tick in order. `PlanarThermalField` defaults to the bracketing pair; a query outside the window raises (ADR 0093). | **Measured.** A day of the 10 400-cell road holds **166 KB** of ticks against ~240 MB before; the ring's hash equals the full history's, rebuilt from the hook, byte for byte; 500 queries inside the window are bit-identical to the unbounded field's; a query before it names `keep_ticks`. 10 cases. | — | S | P |
| PT.9 | ✅ **done, engine-free and rendered, on two airframes.** `quad_outbound_pointwise.yaml` and `phantom3_outbound_pointwise.yaml` (a **DJI Phantom 3**, ADR 0124) film per-cell skins against sky, now with cloud in both bands. Camera moves, aircraft still (ADR 0123). | **Measured.** Sunlit skin over air **+24.4 K** carbon vs **+2.3 K** white ABS (α 0.90 vs 0.25, both ε≈0.9 in LWIR); deck − belly 29.3 K rendered = 29.3 K in the oracle. Corrections: `SPAN_M` overstated 41 %, 1204 arm pixels outside every patch. 32 cases. | PT.1, PT.2, PT.5, WM.3 | M | A |
| PT.10 | ✅ **done.** `vessel_pointwise.py` + `vessel_pointwise_clear_day.yaml`: a weather-deck field and two faces of the **one** deckhouse prim, vessel static and camera moving (ADR 0123). Boxes from `maritime_demo.vessel_boxes`. | **Measured.** Deck span **5.49 K**, deckhouse sunlit − shaded **5.21 K**. The shadow is where the sun puts it — 3.42 m forward at 44.6°, outboard strakes 4.85 K warmer at the *same* stations. Roofed deck **1.3 K** over its own cast shadow (sky view). Occluders off: uniform, **0.061 mK** from the scalar — the WM.2 bar. Frames need IG.2. | PT.9 | M | B |
| PT.11 | ✅ **done.** `lateral_operator(patch, k, δ)`: K = k δ · side/gap per four-neighbour edge as a `ConductionOperator` on the IMEX step; every patched surface builds it from its material's k and thickness (`lateral_conduction: false` opts out); the car bonnet takes steel's. ADR 0102. | **Measured.** A 20 K step on 5 mm aluminium cells matches the semi-infinite sheet's erf to 0.6 % at 60 s; k → 0 is bit-identical; at 5 cm the explicit limit is 6.4 s and a 60 s implicit tick holds the maximum principle where forward Euler explodes; the steel bonnet is 5 % smoother. 6 cases. | — | L | P |
| PT.12 | ✅ **done.** `irsim.thermal.layers`: `LayerStack` (§6.4's R = δ/2k + δ/2k between layers, optional deep node) and `layered_field`, each layer a `CoupledFields` member joined by a contactor at 1/R, stepped implicitly; `layers: N` on a patched surface (default 1). ADR 0103. | **Measured.** Two layers reproduce `LumpedTwoNodeSolver` to < 1 mK on both nodes over 6 h; a 1 mm steel skin peaks at 12:19 and a 6-layer 0.3 m asphalt surface at 13:45; the lumped 0.3 m slab is 5.6 K warmer than the 6-layer one at 04:00; `layers: 6` on a scene's road binds its surface view. 5 cases. | — | M | P |
| PT.13 | ✅ **done.** `irsim.thermal.maps` + schema v17 `temperature_map:` / `parameter_maps:`. A raster **is** the surface (DIRSIG's map solver) or varies one `FacetProperties` field across it (MappedTherm). Units are declared, never inferred. | **Measured.** A raster at the patch's shape round-trips **bit-identically**; a resampled ramp to **0.1 mK**. A deck mapped α 0.2 vs 0.8 differs **12.87 K**, and a half-and-half raster reproduces both uniform solves to **0.0000 mK**. A 20 °C raster declared K is refused against the LUT domain. 20 cases. | PT.2 | M | C |
| PT.14 | ✅ **done.** ADR 0111: four tiers (per prim, cells on a plane, cells on a mesh, and a network node, which is a mass and not a surface), what each costs, and the rule for picking one. No per-material tier, and none chosen by range. | **A record, with numbers.** Cells are nearly free per tick (8× the cells for 1.75× the time; the forcing is the fixed 0.8 ms) and cost at build. Cell size is set by `L = √(kδ/h)` from the committed library — 9 mm on carbon, 145 mm on an aircraft skin — not by the pixel count. | WM.5 | S | P |
| PT.15 | ✅ **done.** Both exported from `irsim.thermal`; the cabin is a `LumpedMember` of its panels' `CoupledFields` (one implicit operator over panels *and* air) and a layered surface takes §6.4's R₂d/T_deep from `back:`. Schema v10 `thermal.cabin:`; `parked_car_cabin.yaml` + its script (ADR 0106). | **Measured.** Reproduces `CabinNode`'s equilibrium to **0.021 K** at a 2 s tick, roof **+4.84 K** on ADR 0038's panels. Scene: cabin 68.7 °C, roof +1.6 K over an adiabatic bonnet (not 4.8: only the roof faces the sun), night roof −3.8 K vs air. Frame: IG.2. | PT.12, TC.2 | M | P |
| PT.17 | ✅ **done.** A `PlanarThermalField` per patched surface, on its library material and spun-up state, forced by `CellForcing`; `Scene.surface_fields`, `surface_bindings()`, `bindings_from_scene`. Both car scenes declare bonnet and road; `build_car_demo` reads and checks them. | **Measured.** Under uniform forcing every cell equals the per-prim value **bit for bit** (contract 1 mK); a bonnet grid 24 cm off the skin and a road grid short of the footprint are refused; no patch keeps the hand-built grids; an unknown material fails at load naming the surface. 14 cases. | — | M | P |
| PT.18 | ✅ **done.** Schema v8: `world_frame:` (ENU default) and `thermal.occluders:`. `CellForcing` gates the beam per cell via `cell_shadow`, spun up with the shadow; `car_demo` fields gain q_solar from `Scene.solar_terms`, shaded by the car's faces. `shaded: true` beside occluders refused. ADR 0095. | **Measured.** SW concrete wall + overhang from YAML: **9.9 K** lit/shaded at 16:00 (PT.1's 22.1 K was a fixed-beam equilibrium); unreached cells **bit-identical** to the per-prim solve, spin-up included; the noon car's shadow strip runs 12.7 K colder after 1500 s. 16 cases. | — | S | P |
| PT.19 | ✅ **done.** `PointwiseTemperature(bindings, known_paths=)`: a binding to a prim path the stage does not know raises at construction, naming it (`IrCamera` passes its prim map); frame-only, an absent path raises under `strict`, warns otherwise; `last_coverage` counts pixels per binding. | **Measured.** `/World/raod` bound against a stage of `/World/road` raises naming both; frame-only, strict raises and lenient warns with the plane untouched; a correct binding is bit-identical to before; a bound prim off screen is coverage 0, not an error. 2 cases. | — | S | P |
| PT.20 | ✅ **done.** `wall_half_in_sun.yaml`: a concrete building (4 walls + roof; the west wall half concrete, half the new `etics_render`), a neighbour block as occluders, asphalt ground, 18:00; `scripts/wall_half_in_sun.py` prints the faces and writes the west wall as a synthetic-G-buffer frame. | **Measured** (after PT.21). Terminator: concrete **9.7 K** (10.3 before), render **6.3 K**; roof − north 12.4 K, west − east 7.9 K, west − north 7.7 K (not > 10: a grazing beam); 30 min after the shadow lifts the concrete is 8.05 K cooler (83 %, not > 10). Rendered: IG.20. 6 cases. | — | M | P |
| PT.21 | ✅ **done.** `irsim.thermal.skyview`: a 145-patch Tregenza dome, each patch sub-sampled 3 × 4, gated by the beam's own `cell_shadow`, an open cell keeping `V_s` to the bit; patches under occluders get their factor and `CellForcing` scales diffuse solar and longwave down by it. Perez split deferred (ADR 0104). | **Measured.** Open sky 1.000; wall foot and overhang edge 0.5 within 0.01. Shaded asphalt, SVF 0.2 vs 0.9: swing ratio 1.25 (not ~2: one air temperature floors it), night minimum 3.2 K warmer; solar-only: minima within 0.05 K. R1 terminator 10.3 → 9.7 K. | PT.18 | M | P |
| PT.22 | ✅ **done.** `irsim.thermal.raycast`: an `(origins, directions) → hit` protocol over `RectangleOccluders` (the oracle), a NumPy Möller–Trumbore `TriangleSoup`/`MeshOccluders` and `AnyOccluders`; the 0.53° disc on 1/7/19/37 concentric rays, outermost on the limb, as a sunlit fraction. Schema v11 `penumbra_rays:` (ADR 0107). | **Measured.** Box: rectangles = mesh cell-for-cell at 10–85°, and = trimesh. A neighbour block shades a wall its own mesh cannot. Ramp within 10 % of d·tan 0.53° at 0.5/2/6 m (18.0 vs 18.6 mm); binary gives zero width. | PT.21 | M | P |
| PT.23 | ✅ **done.** The chain was right. `IG.2`'s two branches were built with `replace(base_state)`, which copies the *reference* to `PipelineState.buffers`, so both drove **one** membrane IIR with alternating inputs. `buffers` is now `init=False`, so `replace` cannot share it. | **Measured.** The shortfall is α/(2−α) = **0.778545** at 60 Hz and τ = 8 ms, against 0.778546 measured in sim — six figures. Frame 1 gives α itself. An engine-free twin carries the shared state as its negative control; the in-sim `xfail` is off and the law holds at all four ranges. | — | M | A |
| PT.24 | ✅ **done.** `SceneSurfaceForcing.ground_temperature_k` / `ground_albedo` from the preset's `ground:` (air, fixed, SST -- the render path's rule) and the ground material's 1 − α_sol, on the tilt's ground share (1 − cos β)/2 in every forcing. `sky_target_clear_day` keeps Newton (test_scene's probe). | **Measured.** Tilt 0 bit-identical. Tilt-180 plate over ground at T_air + 20 K: Q_LW↓ +134 W/m², +8.2 K by the hand balance to 0.2 K; albedo 0.25 adds exactly 0.25·GHI to a belly, nothing to a deck (`test_ground_view.py`). Quad belly on the pad: T_air → T_air + 8.5 K. | — | M | A |
| PT.25 | ✅ **done** (re-scoped). `FacetForcing.free_convection_c` + `effective_h`: the solver takes max(h_forced, c·ΔT^{1/3}) at its own state; the guard adds ∂Q_L/∂T. The 5 + 4v^0.8 default stays: convective-only correlations run 2.8 + 3.0v to 7.4 + 4.0v and bracket it, so S56 is recorded, not applied. | **Measured.** A facet 60 K over still air gets 5.88 not 5.0; below 37 K bit-identical, as is every producer without the field. Wet leaf at 10 m/s: bound 35 → 16 s, a 20 s step raises. `test_facet_convection_guard.py`, `test_convection.py`. | — | S | P |
| PT.26 | ✅ **done.** `ThermalSpec.conductivity_inplane_w_mk` (optional; `inplane_conductivity_w_mk` falls back to k) feeds `lateral_operator`, `mesh_lateral_operator` and `car_demo`'s surface helper; `carbon_fibre.yaml` authors 7.0 (ESTIMATED). | **Measured.** Link conductances scale by 7.0/0.8 to 1e-12; √(αt) at 300 s 12 → 35 mm on carbon; k_inplane unset is bit-identical (`test_inplane_conductivity.py`). Mesh arm crown−underside 10.9 → 3.3 K, span 20.8 → 13.0 K (fin length 26 mm vs 24 mm quarter-tube); outbound arm span 26.2 → 20.3 K. | — | S | P |
| PT.27 | **Ground defaults to a layered slab with a deep boundary.** Ground materials run as one lumped node with an adiabatic back; damping depths (0.08–0.14 m) exceed the authored thicknesses, so the diurnal wave reflects. Default ≥ 3 layers with `back: {deep_temperature_k: ambient}` for soil, asphalt, concrete. | Concrete 0.10 m: amplitude 11.7 K, lag 1.3 h against the semi-infinite solution (5 %), not 15.1 K / 2.5 h; the 47 K swing pinned in `test_tier3_thermal.py:171` falls into band. | — | S | C |
| PT.28 | ✅ **done.** `TargetSpec.altitude_agl_m`; `aerial.air_temperature_at_altitude_k`, `air_density_ratio`; the airframe, heat-source and ram-skin solvers take `altitude_agl_m` + Γ (preset, else ISA); `SceneSurfaceForcing.air_density_ratio` from `site.altitude_m`, `forced_convection(..., density_ratio)`. | **Measured.** −3.25 K at 500 m, −19.5 K at 3 km; ρ ratio = ISA to 1 % (0.742 at 3 km); isothermal column = barometric to 1e-6; z = 0 bit-identical; ram skin recovers at a(T(z)); the 520 m sky-target site convects at 0.95 of sea level. | — | S | A |
| PT.16 | **Make `HeatTraceLayer` reachable** (ADR 0039, M6.16). §6.6 calls heat traces a signature phenomenon of the band, and the sim-to-real literature says detectors trained on synthetic data lacking them are confused by them — so this is an evaluation deliverable, not a nicety. | A rendered frame shows the trace ghost at the authored offset and amplitude; the overlay is absent bit-identically when unbound. Red today: `irsim.thermal.traces` is imported only by its own unit test. | PT.15 | M | C |

---

## WM — Warp mesh surface parameterisation

**This lane is the one genuinely new unlock in revision 5, and it deserves its own error budget, its own
oracle and a superseding ADR.** ADR 0087 records "there is no UV AOV, no per-triangle id" and concludes
that curved geometry is "a real limit, not a temporary one". That conclusion is true only if the surface
parameterisation has to arrive through an AOV. It does not.

Warp 1.16.0 — already installed in this build, verified at
`warp/_src/types.py:7579,7613` — returns `MeshQueryPoint{result, sign, face, u, v}` from
`wp.mesh_query_point_no_sign` and `MeshQueryRay{..., face, t, u, v, normal}` from `wp.mesh_query_ray`.
Seeded by the float32 `Camera3dPositionSD` that `point_bridge.world_positions` already decodes, a
closest-point query against a bound prim's own mesh yields exact `(face, u, v)` per pixel, with no
renderer capability, no asset change and no new dependency. Measured on this machine: 327,680 queries
against a 516,960-triangle mesh in **0.63 ms** on an RTX 5090, ray queries 0.51 ms, LBVH build ~0.9 ms —
three orders of magnitude under the radiance kernels.

Conditions this lane must carry, stated up front: the **3.4 mm measured position residual** is the error
budget for the closest-point tolerance; `mesh_query_point*` takes no `root` argument in 1.16, so it is
**one `wp.Mesh` per bound prim**, which also removes the thin-panel closest-point ambiguity structurally
instead of by a tuned `thickness_m`; a brute-force NumPy closest point is the oracle the Warp path is
tested against, per ADR 0018/0061; and if PT.11's lateral conduction follows onto a mesh it needs an
intrinsic-Delaunay-safe Laplacian (WM.6), because a plain cotan Laplacian on an imported obtuse triangle
gives negative weights and a cell can leave the physical range.

Leaving ADR 0087 standing as written will cost another session a week, so WM.5 is not optional.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| WM.1 | ✅ **done.** `probe_warp_mesh.py` (no Kit) and `probe_warp_prim.py` (a USD prim): `wp.Mesh` + `mesh_query_point_no_sign` recover face and barycentrics from a world position, against a NumPy oracle. ADR 0087 gains an addendum. | **Measured.** Round trip **0.13 µm** vs the 3.4 mm budget (0.73 µm at 4.3 m: float32 grows with origin distance). Warp's (u, v) weight **v0, v1**, 1−u−v on v2 — the obvious reading misses by **0.56 m** on a 0.4 m box. Under 3.4 mm of error the face flips on 1.9–56 % of queries but the point moves ≤ 3.41 mm. Quads need triangulating. | — | M | P |
| WM.2 | ✅ **done.** `irsim.thermal.mesh_field`: `TriangleMeshPatch` cuts each face into k² congruent cells on the barycentric grid (per-face level, Ptex style), `TriangleMeshField` composes `ThermalField`; `sphere_mesh` joins `raycast`. | **Measured.** A 0.25 m sphere under an overhead sun holds every face to its own root within **1 mK** across a **34.3 K** span; one facet lands at 295.0 K — 25 K under the cap, 9 K over the far side. Uniform forcing is **bit-identical in float32** to the scalar solve; the varying mean sits below it by Jensen, energy closing to 1e-6. | WM.1 | M | P |
| WM.3 | ✅ **done.** `mesh_bridge.MeshPointBridge`: instance id picks the prim's `wp.Mesh`, the closest-point query gives (face, u, v), the mesh field gives the temperature. Additive; Warp accelerates, `closest_point_on_mesh` is oracle **and** no-Warp fallback. | **Measured.** Warp and the brute-force oracle agree on face, cell and sampled temperature for **100 %** of 4 000 pixels on a sphere, and both routes render the same frame. An exhaust pipe goes from **0.000 K** across the prim to **34.1 K** (12.9→47.0 °C). A pixel 50 mm off raises; unbound prims stay bit-identical. | WM.2 | M | P |
| WM.4 | ✅ **done.** `irsim.thermal.mesh_geometry`: a ray-traced sky view per cell on `PT.21`'s dome and a ray-traced beam on `PT.22`'s disc, through one `Occluders` query over the scene's occluders and the mesh's own triangles. Convex meshes skip it, exactly. NumPy, not `mesh_query_ray`. ADR 0088 addendum. | **Measured.** Convex traced = `(1 + n·up)/2` bit for bit; a mesh cell and a patch cell agree to the bit under one wall through two ray tests; wall foot 0.5000, overhang edge 0.5042; the pod over an arm's outer 60 mm takes its beam and all but 0.10 of its sky. | WM.3, PT.22 | M | P |
| WM.5 | ✅ **done.** ADR 0110: cells per face on the mesh, located by a closest-point query that **derives** the parameterisation instead of asking the renderer to transport it. ADR 0087's curved-geometry limitation is superseded; its planar patch is not. | **Measured.** A record. The rejected routes are named with reasons: a UV atlas as the *solver* domain (metric distortion, seam severing, a conservative-rasterisation tax), CPM narrow bands (a grid finer than a 1 mm panel), transient surfels (no 48 h spin-up memory), and ADR 0087's two AOV routes. | WM.3 | S | P |
| WM.6 | ✅ **done.** `irsim.thermal.mesh_conduction`: two-point flux between a mesh's cells, within each face and across every shared edge (levels matched by overlap), as the same `ConductionOperator` a patch uses. **Monotone over consistent**: the cotan weight goes negative on an obtuse face. ADR 0112. | **Measured.** = the cotan weight bit for bit on an equilateral face; level-free; linear-exact to 7e-16 there, 1.5-4.5 % when skewed; no cell leaves its neighbours' range on an obtuse mesh, where cotan is refused; a tube matches the fin equation's 0.744 to 1 %. | PT.11, WM.3 | M | P |
| WM.7 | ✅ **done.** A surface's `mesh:` (schema **v14**): `MeshSpec` builds the primitive named, `Scene.mesh_fields` / `mesh_bindings()` carry it, `MeshCellForcing` gives each cell its own face's beam and sky view. `quad_flight_mesh.yaml` + its script. | **Measured.** The aerial mission with the arms as **tubes**: crown 52.8 °C over an underside on air at 26.2 °C — **26.6 K around one arm**, where the patched strip carries under 1 mK across its width. Spun up per cell, so it opens with the gradient grown. Cylinder faces wound outward — they were not, and the crown read cold. | WM.3 | M | P |
---

## TC — Thermal coupling

The owner's seventh requirement, in their words: *"engine will increase its temperature while being used —
but not just itself, also the metals around it."* The 2026-09-18 audit found nothing in the spec, the code
or revision 5 that could do this. `FacetSolver.net_flux` is a pure per-cell function of that cell's own
temperature; the only coupling between parts is ADR 0088's radiative view factor onto a parallel panel;
the two objects that carry a conductance — the two-node solver and the cabin node — reach no scene; and
the engine is a scripted ΔT schedule (ADR 0089) with no watts, no mass and no link to anything. `PT.11`
diffused heat *within* one patch and sat in phase C behind the lane ranked third.

The production answer is old and small: a lumped **network** — nodes with capacity, links with conductance,
a fluid node for the bay air, heat imposed in watts — is how MuSES, TAITherm and Thermal Desktop couple
parts, and engine warm-up models in the literature use ten to sixty nodes, not thousands. Measured joint
conductances exist (bolted ferrous automotive joints ~1e4 W m⁻² K⁻¹; rubber mounts are first-order lags),
and the phenomenon every under-hood measurement records — the **hot soak** after key-off, when skins keep
rising for one to two minutes because forced convection stopped while the core still feeds them — is
unrepresentable by any schedule. Sources are in the 2026-09-18 survey.

What makes this a lane rather than a row is the integrator. The fixed explicit tick that every field runs on
cannot host a conduction operator (`TC.1`); once it can, lateral conduction (`PT.11`), the network (`TC.2`)
and contactors between fields (`TC.3`) share one implicit solve. The engine-free half of every row here is
closed by an analytic check; the frames of `TC.6` are the in-engine half and wait on `IG.2`.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| TC.1 | ✅ **done.** `ConductionOperator` (symmetric W/K links + areas) and an IMEX step in `FacetSolver`: the surface balance keeps the midpoint rule, conduction is backward Euler, prefactored per tick size; §6.4's bound is checked every step (ADR 0094). | **Measured.** A zero operator is bit-identical to none; a ladder with τ ≈ 0.1 s at a 60 s tick settles to its mean within 1e-6 where forward Euler explodes; the slow mode's error halves with the tick (0.7 % at 60 s, τ = 1 h); energy closes to 1e-9 for 50 unequal cells; a leaf at h = 30 raises with its 36 s bound. | PT.8 | M | P |
| TC.2 | ✅ **done.** `irsim.thermal.network`: `Node` (J/K), `FixedNode` (T(t)), `ImposedHeat` (W), `Link` as G or h·A (callable h), `LinkNode` (2G each side), `RadiationLink`; backward Euler on `ConductionOperator`, fixed nodes eliminated at the tick's end, radiation linearised per tick. ADR 0096. | **Measured.** ΔT = Q/G and Q/(hA) to 1e-6; three link forms bit-identical; a mount's τ within 2 % of C/4G; T⁴ steady state to 1e-6; energy closes to 1e-6 per tick on a 7-node bay with moving ambient, switching h and a mount; a bracket hot-soaks after key-off. 13 cases. | — | M | P |
| TC.3 | ✅ **done.** `irsim.thermal.coupling`: contactor conductances by exact clipping of overlapping cells; `CoupledFields` steps members as one `ThermalField` with a block operator, a `PatchView` per member; `RadiationExchange` reads the view factors both ways. ADR 0099. | **Measured.** G = h_c·A_overlap to 1e-9, identical after 4× refinement; a 30° grid overlaps its own area; a half-overhanging plate gets 0.5 m² where per-node says 1.0; a τ = 0.8 s joint stands under 60 s ticks, energy conserved to 1e-9; body → road power equals the cells' to 1e-6, reciprocity 1 %. 8 cases. | — | M | P |
| TC.4 | ✅ **done.** Schema v9: `thermal.nodes:` (capacity, mass×c_p, fixed or `ambient`, link node), `links:` in exactly one form (G, joint+area, h_c+area, fastener×count, h+area, radiation), `sources:`; `configs/thermal/joints.yaml` with provenance; `Scene.network`. ADR 0097. | **Measured.** h_c 1e-3, 50 and 2e6 refused by the loader, 1e-3 by the schema; a bracket on 25 cm² of `bolted_ferrous_new` vs `dry_default` on a 400 °C block, from YAML, settles in the ratio G/(G+hA) to 1e-6; every form builds to its conductance; a v7 surface beside it is bit-identical. 12 cases. | — | S | P |
| TC.5 | ✅ **done.** `irsim.thermal.engine`: block + coolant, bay air, rubber mounts, subframe on the TC.2 network; heat = P_rated · load · bay_fraction; forced → natural convection and venting at key-off; `solver: engine` reports the bay air as the bonnet's cavity. ADR 0100 supersedes ADR 0089 for the bay. | **Measured.** From 93 °C at 27 °C: +32.3 K after 1 h (schedule: 9 K), +0.92 K after 7 h; full load +53.7 K; bay air overshoots +27 K peaking 145 s after key-off; the bonnet over the block keeps warming ~23 min (not 60–120 s, the manifold-skin figure: TC.7). 7 cases. | — | M | P |
| TC.6 | ✅ **done.** Both car scenes: `solver: engine` (coolant loop + proportional thermostat) plus `nodes:`/`links:` — block followed as a boundary, rubber mounts, subframe, a bolted bracket whose fan convection stops at key-off, the wing on two bolts; key on 30 s, off 20 min. ADR 0100 addendum. | **Measured.** A bracket on a fixed block reaches 1 − e⁻¹ at τ = C/(G+hA) and settles at G/(G+hA) to 1e-6; parts warm block → bracket → mounts → wing; the overcast bonnet is 18 K max–min at 1500 s and warms 6 K more after key-off. Rendered: IG.22. | — | M | P |
| TC.7 | ✅ **done.** `irsim.thermal.exhaust_line`: the gas marched segment by segment (exact `exp(−NTU)`, Dittus–Boelter h_i) as links into wall nodes of the S42 network; outside h forced/natural with motion, radiation to pan and road, hangers 1 W/K, an inner mass for catalyst and silencer, a heat shield node; `solver: exhaust` from YAML (ADR 0105). | **Measured.** March vs closed form 1e-6; gas heat = ṁc_p ΔT to 1e-9. 60 % load: manifold 557 → tail 368 °C, hangers 40–50 K cold. Key-off: shield 287 → 347 °C at +65 s, < 260 °C 5.3 min on; shell peaks +65 s (270 °C, not 400). | TC.6 | M | P |
| TC.8 | ✅ **done.** `irsim.thermal.drive_cycle` walks a `VehicleState` trace: the disc takes its axle's share of ½m(v₁²−v₂²) and cools at τ 300 s, the tyre relaxes toward §6.6's speed relation, and `car_demo` authors a tread and a disc rectangle per corner. `SourceHistory` gained `step_to_target` so §6.6's step still exists once. ADR 0089 addendum. | **Measured.** 40 min at 27 m/s puts the tyre **28.1 K** over ambient; the arch 0.12 m above it takes **285.9 W/m²** against the door's 0.11. A 1600 kg stop from 30 m/s deposits **162 K** into an 8 kg disc, 4× from 60. 15 tests. | TC.6 | S | C |
| TC.9 | ✅ **done.** `irsim.thermal.view_factors`: `view_factors(meshes)` traces `skyview`'s 1740-ray fan from `subdivide²` points per face, nearest hit wins, one-sided faces (`back` booked apart), sky share per face and body; `howell_*` closed forms as oracles. ADR 0156. | **Measured.** C-11 and C-14 to 0.3 %, C-41 to 1 % (equal-area 16-gon), body reciprocity 0.1 %, closed box rows = 1 to 1e-12, a full blocker gives F = 0 exactly; 30-face pair 1 s, 300-face pair 60 s. | — | L | P |
| TC.10 | ✅ **done.** `thermal.object_exchange: true` (schema v18): `irsim.thermal.object_exchange` joins every plain patch and mesh, TC.9's factors symmetrised, `q = ε[Σ G ε_j σ T̄_j⁴ − cover·q_lw]` per tick from one snapshot, members in lockstep behind `ExchangedField`. ADR 0157. | **Measured.** Off/absent: same state hash. Pan over road: closed form to 1 % (3 % of peak per cell); black bodies close to 1e-6; clear-night asphalt under a 310 K pan +0.25 K/h, no rectangle authored. Not yet: exchange in spin-up; layered/cabin/prescribed refused by name. | TC.9 | M | P |
| TC.11 | ✅ **done.** `thermal.objects: [{asset, solve: full, duty}]` (schema v20) → `<object>.<part>` mesh surfaces + `full_solve.build_full_solve`: mesh members, hidden parts as lumped members × duty, contacts by proximity / footprint / direct from the joints table, TC.9 exchange inside the solve. | **Measured.** Synthetic arm + can + 6 W winding: can before near arm before far arm; the can climbs (first tick < 1/5 of the rise), stays hot after the throttle stops (> half its rise at 1200 s); 3600 J in = stored to 1e-6. Inner/outer face: across the object. | TC.9, AI.11, TC.16 | L | P |
| TC.12 | ✅ **done.** `evolve:` / `freeze_at_s:` on surfaces, targets, nodes and `thermal:` (schema v19): `ThermalField.hold_from_s` (per field or facet), `hold.HeldSolver` for targets, `ThermalNetwork.hold` (infinite capacity in the solve, written back). ADR 0158. | **Measured.** Unset: hashes identical. Held field bit-identical to the hold, constant after; a 320 K pan held over asphalt keeps the road under it warmer than one left to cool; a frozen target stops on the tick; a held node is a boundary its neighbour moves toward; map, cabin panel, fixed node, two holds: refused. | — | M | P |
| TC.13 | ✅ **done.** `phantom4_solved.yaml` on `solve: full`: 19 coarse mesh parts (ADR 0165), windings + pack + ESCs + FC as hidden components, 19 contacts; `ObjectSpec.duties` / `speeds` per part; `FullObject.node_temperature_at`; the driver maps parts to `<object>.<part>`. ADR 0166. | **Measured.** Bells +1.4 K after the climb's first minute, +14 K at its end (lag); 42 °C hover, 60-66 °C orbits; rear > front by 10 K in cruise, outer > inner by 8 K in orbits; 43 °C ten minutes after landing; shell under the pack +2 K; adiabatic object closes energy to 1e-6. | TC.11, AI.12 | L | A |
| TC.14 | **The car's body is solved, not held at air temperature.** `shell` and `glass` are T_air targets — 5–8 K wrong on a clear night by the scene's own note — and TC.7's exhaust radiates into a floor pan held at body temperature, so the floor never warms. Solve the panels by §12.3's balance; make the pan a node. | Clear-night panels settle below air by the sky deficit (analytic facet, 0.5 K); the pan over the silencer warms after key-on and through the hot soak; overcast panels stay within 1 K of air. | — | M | C |
| TC.15 | **Underbody and exhaust follow the solved engine.** In both ignition scenes the underbody is §6.6's bay row at a third of its load (ESTIMATED) and the exhaust a scripted τ = 360 s row, so neither is heated by the TC.5 block solved beside them. Drive them from the block's network and TC.7's line. | The underbody lags the block along its conductance path (TC.6's ordering, extended); the road stripe under the pipe follows TC.7's wall to 0.1 K; no scripted row is left in either scene. | — | M | C |
| TC.16 | ✅ **done.** `coupling.FieldMember` takes a `TriangleMeshPatch` (per-cell areas); `ExplicitContactor` + `proximity_contactor` (cells within a gap, scaled to `h_c·A`); `LumpedLink.per_cell_w_k` + `footprint_conductances` (nearest cells to the footprint's area). Split out of TC.11. | **Measured.** Mesh member alone = `TriangleMeshField` bit for bit; plate-on-tube conserves energy to 1e-9, contact cells warm first; footprint sums to `h_c·A` to 1e-12; a 2 W ESC for 600 s stores 1200 J to 1e-6 and the arm keeps warming after it stops. | — | M | P |
| TC.17 | ✅ **done.** `mesh_coarse.split_long_edges`: `coarsen_mesh` bisects every edge longer than `cell_m` (conforming red-green, area-exact) before clustering, so a planar-dissolved tube gets cells along its length. ADR 0176. | **Measured.** Strip tube: mid-plane flux = k δ·2πr to 10 % (whole strips ~460× out); butt-jointed tubes couple at the joint only (whole: refused); closed mesh stays closed. Inspire 3 contacts 14/18 → 18/18. | TC.13, TC.16 | S | A |

---

## PH — Phenomena beyond opaque solids

The owner's eighth requirement: *"this is correct about water, fire and something like these."* The
2026-09-18 audit found water only as the open sea (an authored bulk SST under a skin model that omits the
latent flux by its own admission), no latent-heat term in any balance, nothing consuming the weather file's
`precip_mm_h`, wetness as a second material rather than a state, and — for fire, flame, hot gas and plume —
no code, no config surface and no step: §2 is a surface-only equation and the plume was "deferred without
a number" past a trigger that had already fired.

Two pieces of physics cover most of it. **A latent-heat term with a per-cell water film** (`PH.1`) gives wet
roads, drying spikes, lakes and puddles, and with a stomatal resistance, transpiring leaves. **A gas slab in
radiance space** (`PH.4`) — per band, L = τ L_behind + (1 − τ) B(T_gas), the one-line kernel FDS, DIRSIG and
NIRATAM all reduce to — gives exhaust plumes, flames and steam, with the band selectivity that makes a plume
dominant in MWIR and nearly invisible to a LWIR bolometer. The hot-gas absorption tables are generated
offline like the Planck LUTs, so bands stay data. Fire on the *camera* is its own row (`PH.8`): what a
microbolometer does with a 1000 °C object is a gain state, a rail and an AGC choice, and FLIR's own notes
quantify it. Snow, vegetation and people are one rule each on top of `PH.1` and the patch machinery; they
are phase C because no scene in the owner's order needs them yet, but they are rows, not deferrals.

Everything here is engine-free and CPU-only; the tolerance for gas radiometry is RadCal's 8 %, which makes
fire a phenomenology feature and not a 10 mK one, and `PH.13` records that so nobody tunes it to 10 mK.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| PH.1 | ✅ **done.** `irsim.thermal.latent`: Q_L = L_v ρ_a (q_sat(T_s) − q_a)/(r_a + r_s) in the balance; `FacetSolver(film_kg_m2=)` keeps a film per cell that rain fills and evaporation empties; scene forcings carry q_a, g_e and rain; the sea skin's net loss gains the latent flux. ADR 0101. | **Measured.** A saturated cell with no radiation lands on the wet bulb within 0.1 K at RH 1.0, 0.7, 0.4; Q_L = 0 at RH 1, T_s = T_a; a dry cell is bit-identical; 75 W/m² at 5 m/s, 293 K, RH 0.7 (COARE 76); the film budget closes to 1e-6; the sea skin runs 0.3 K colder. 11 cases. | — | M | P |
| PH.2 | ✅ **done.** `configs/scenes/wet_road_noon.yaml`: `film: {depth_mm, region_m}` on a patched surface (ADR 0101 addendum) waters the west half of a 12 m asphalt patch at 14:00, a wall shading the south rows; `evaporated_kg_m2` closes the budget. 0.5 mm, not 0.2: this road dries at 1.3 mm/h. | **Measured.** Sunlit wet − dry peaks at **9.9 K** at 27 min, shaded near 5 K later; the sunlit film is gone at 28 min and the halves reconverge below a quarter of the peak by 3 h; film₀ − film = evaporated and a quadrature of E(T) agree to 1e-6. Rendered: IG.20. 5 cases. | — | S | P |
| PH.3 | ✅ **done.** `irsim.thermal.still_water`: a fresh-water sublayer with a **signed** skin (the sea's clamp is wrong on land), the mixed layer's capacity, and the Fresnel/sky mix a camera reads; schema v12 `water:` makes a puddle a region of a road's patch (ADR 0108). | **Measured.** Sublayer 0.67–3.61 mm over 0.8–8.2 m/s. Clear calm night −0.38 K; cloudy humid **+0.21 K** (clamped: 0.00). ε+ρ=1 to 1e-12. Puddle −0.55 K nadir, −1.90 K at 60° (row: 0.5 / several — this sky is colder), −4.40 K at 70°. Road: −3.8 K at t0, −11 K at +1 h. | PH.1 | M | P |
| PH.4 | ✅ **done.** `irsim.pipeline.gas_slab`: `GasSlab(T_gas, L, p_CO₂, p_H₂O, f_soot)`, no emissivity field; L_b = τ_b L_behind + (1−τ_b) B_b(T_g); soot κ_b by quadrature, gas κ_b(T) from tables (PH.5); attenuated per class from range as MS.6; guard 300–2500 K. ADR: PH.13. | **Measured.** κL→0 returns the background bit-exactly, κL→∞ B_b(T_g); an opaque soot flame reads T_g in both bands to 1 mK; a CO₂/H₂O slab (synthetic table) reads > 900 K apart between bands, a grey ε within 150 K; steam before a hot wall is negative; range identities to 1e-12. 12 cases. | — | M | P |
| PH.5 | ✅ **done.** `data/gas/<key>_{co2,h2o}.npy`: float32 κ over 300–2500 K **and over column density**, from RadCal via `scripts/radcal.py`; the sidecar hash pins the database and `gas_tables` **refuses** a band the model cannot reach. RADIS/HITEMP not taken. Adds `insb_flame_window.csv`. ADR 0098 addendum. | **Measured.** The port reproduces both RadCal arrays **bit-exactly at every grid node**. CO₂ MWIR rises with T at every column; κ 35.8→0.61 over 0.005–0.5 atm·m. 0.25 m 600 K plume: τ MWIR **0.830**, LWIR **0.913**, window **1.000**. SWIR/NIR refused. 21 cases. | PH.4 | M | P |
| PH.6 | ✅ **done.** `irsim.pipeline.plume` (stage 2d): a cone in camera space, one analytic chord per pixel, `PH.4`'s slab on it; occlusion is the depth plane; entrainment dilutes T and species by one factor. Schema v15 `plume:` on an exhaust target takes `TC.7`'s outlet **gas**, not the skin. `car_exhaust_plume.yaml`. ADR 0114. | **Measured.** One scene, two cameras: τ **0.866** MWIR / **0.979** LWIR, peak ΔT_app **+72.7 K** / **+7.1 K**. A grey table makes them agree to 0.02 % — the control. τ monotone along it; chords exact. Rendered: IG.21. 21 cases. | PH.5, TC.7 | M | P |
| PH.7 | ✅ **done.** `irsim.thermal.fire`: `sep_w_m2` on `RadiantRectangle` is an authored emissive power that **refuses** a temperature; `q_int = F (α SEP − ε L_occ)`, α for the flame and ε for the sky. Heskestad's ΔT₀ replaces T_air, held at the tip inside the flame; `flame_plume` solves the slab's cooling from it. ADR 0115. | **Measured.** The tip is **452.7 K** above ambient for a 50 kW ring and a 40 MW pool alike — Q and D cancel. Continuous, monotone, → 0. F = 0.10 at 120 kW/m² lands at T_air + α·12 kW/h to 1e-9. Σ F ≤ 1. Watts refused. Soot 2.47× vs CO₂ 5.49×. | PH.5 | M | P |
| PH.8 | ✅ **done.** `irsim.detector.gain_state`: `fpa.gain_ceiling_k` is the state's intrascene ceiling (Boson 140/500 °C), clipped on the at-aperture **radiance** plane and used as the radiometric range's top, so a clipped pixel lands on the top code. ADR 0116. | **Measured.** A 400 °C object rails high gain at 65535 reading 413 K; low gain reads 668 K. Person/room **252 DN** calm → **1.0 DN** with fire under `agc_linear` (FLIR: 0.7 %), **90 DN** under plateau. A kelvin ceiling misses the rail at ε = 0.6 and is **119 K wrong** at ε = 0.4. | PH.7 | M | P |
| PH.9 | ✅ **done.** `irsim.atmosphere.mie` (Bohren-Huffman, SciPy-free, downward `Dₙ` recursion) and `irsim.atmosphere.droplets`; `GasSlab` gains `lwc_kg_m3` and `droplet_radius_um`. ADR 0135. | **Measured.** `cloud.py` has **no Mie tables** — the row's premise — so `Q_ext` is computed from `data/nk/water.csv`, oracle the exact Rayleigh limit to 2e-4. At 373 K over 1 m in LWIR: vapour **0.169**, 5 g/m³ of 5 µm droplets **1.02**, equal at **0.83 g/m³**. MWIR beats LWIR **4.1×** at 2 µm but **0.99×** at 20 µm — both geometric. 13 cases. | PH.6 | S | C |
| PH.10 | ✅ **done.** `snow.melt_capped_step` caps at 273.15 K and melts the surplus at L_f; the flux is read **at** the cap, not at an RK2 midpoint above it. A finite pack hands back what it cannot melt. No scene declares snow. ADR 0120. | **Measured.** +200 W/m² holds the cell **bit-exactly** at 273.15 K, shedding **2.1557 mm w.e./h**. Clamping after RK2 under-reports melt 0.1–1 % **every step, one-signed**. Night needs no special case: ε_hemi 0.9874 gives **−12.73 K** clear calm, −4.75 K breezy, **exactly 0.00 K** overcast. ESSD 16 bar 0.7–1.3 K recorded. | PH.1 | S | C |
| PH.11 | ✅ **done.** `vegetation.py` — a leaf **solved, not stepped** (15.4 s τ breaks a 60 s tick), with **its own** boundary layer, not the bulk one. Oracle is Campbell & Norman in molar units. No scene declares vegetation. ADR 0121. | **Measured.** Watered **−1.93 K** vs stressed **+7.00 K** at 3.18 kPa — nine kelvin from r_s alone. Idso slope **−1.84 °C/kPa**, inside [−3.8, −1.1]. The bulk boundary layer (435 vs **33 s/m**) gave **+6.4 K** — wrong *sign*. C&N agrees **0.10 K** near air, 0.37 K at 4.6 K out, **4e-10 K** degenerate. | PH.1, TC.1 | S | C |
| PH.12 | ✅ **done (3 of 4 criteria).** `thermal/human.py` — skin **authored** from the ISO 7730 set point, clothing **solved** by bisection (the textbook fixed point diverges in wind). Two patches per prim is scene authoring, not shipped. ADR 0122. | **Measured.** 0 °C/1 clo: skin **34.07**, coat **13.96 °C**, step **20.1 K** — the row's 10–15 K is the same equation at **10–15 °C air**, ISO 7730's validity floor. Wind *warms* the coat above **3 clo** under a −40 °C sky. ⚠️ `pythermalcomfort` check **not run** — not installed; `comfort` extra declares it, test skips loudly. | PT.17 | S | C |
| PH.13 | ✅ **done.** ADR 0098: the per-band slab in radiance space, the ~8 % RadCal envelope as the phenomena tier, transport kept outside the renderer (DIRSIG/FDS), Leckner/Hottel rejected for the image path and kept for heating, and the deferrals: scattering, gradients along the ray, buoyancy, flicker, an Isaac-side volume. | **Measured.** A record. The options list names the rejected routes: a grey emissivity knob (PH.4's test: > 900 K between bands vs 150 K), a Planck-mean coefficient in a band camera, Hottel/Leckner totals, an emissive prim through ADR 0014's fp16 path. | — | S | P |
| PH.14 | **Moving air that carries heat to what it passes.** The radiator fan and the underbody flow carry bay heat rearward and down; nothing models it, so no part is warmed by air heated elsewhere. An approximate model, flagged ESTIMATED: a mixed stream marched along a path of cells, losing heat to each (TC.7's march, in air). | What the stream loses the cells gain, to 1e-6; a downstream part warms after an upstream one, in that order; switched off, scenes are bit-identical. | TC.15 | M | C |

---

## AT — Atmosphere, sky and materials

`AT.1` is the audit's one critical-priority radiometry gap and it sits on the lane ranked first.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| AT.1 | ✅ **done (CPU).** Elevation broadcasts through `column_length`/`transmittance`; `path_radiance_plane` interpolates a per-elevation cumulative table, uniform in `w = 1-e^-u` and in sin θ with **θ = 0 as node zero**. `elevation_rad` is an optional, precision-critical G-buffer plane. | **Measured.** 5 km: τ 0.5995→0.7230, L_path 18.27→11.60. LUT within **2.4 mK** over 0.05–90°, 200 m–20 km; horizon join <0.1 mK (a clamp left 209 mK); isothermal exact to 2e-14. 22 cases. **Warp twin not updated — needs a GPU to verify.** | — | L | A |
| AT.2 | ✅ **done.** `Scene.from_config`/`from_file` take `responses=` and pass them to `LayeredAtmosphere`; `load_band_response_for_config` sits beside the LUT loader so both come from one config field. All six drivers pass it; a model built without one warns. | **Measured.** MWIR `h2o_wing` 0.0238 → 0.1303, **×5.46**. LWIR moves <5e-4, which is why it hid — the band rendered most is the one the top-hat suits. 18 cases with AT.3. | — | S | A |
| AT.3 | ✅ **done.** NIR `window` reaches 1.10 µm (where SWIR's first class begins), SWIR spans 0.80–1.80, MWIR `h2o_wing` reaches 6.0 — the 6.3 µm bend's onset, not a window. | **Measured.** NIR carried **0.188 %** of its Planck-weighted band outside every class and raised; SWIR and MWIR carried **0.000 %** and passed on luck. The guard walks `configs/sensors/*.yaml`, so a new camera cannot add a short band. Reverting NIR to 1.05 turns 3 red. | AT.2 | S | A |
| AT.4 | ✅ **done.** The guard scans **every** module under `src/irsim`; the carve-out is `BAND_AWARE`, a path → (ceiling, reason) constant, tested for existence, still-load-bearing and no growth. Names widen to `BAND_KEYS`. `WEIGHT_T_REF_K`, `VISIBLE_RANGE_UM` and the `band == "visible"` branch are gone. ADR 0092. | **Measured.** 140 of 146 modules guarded, 6 exempt (was 72, by omission). `materials`/`io`/`thermal`/`validation` had **0** offences all along. Gauge: ×3 on free multipliers moves τ ≤**3.2e-14**. `BAND_CLASSES` kept: `AT.10`. | — | M | A |
| AT.5 | ✅ **done.** The field is `grey_atmosphere`; `Scene.atmosphere` is a property that **raises** once a layered model stands beside it, naming the three ways out (`transfer_atmosphere`, `atmosphere_preset`, `grey_atmosphere`). Both live readers wanted the preset. | **Measured**, and larger than expected: at 5 km / 20° the grey model gives τ **0.057** against the layered **0.590**, L_path **45.1** against **18.7** W/m²/sr — the k-distribution, not a bug, but L1-only. **All eight shipped scenes carry a layered model**, so the primary name was wrong on every one. 8 cases. | — | S | A |
| AT.6 | **Spec issue S40: the Level B (a, p) table.** Twelve of sixteen Level B materials carry ESTIMATED (a, p); the four fitted use `paint_proxy.csv`, whose header says "PROXY: PMMA, not paint", giving a = 0.75 against §4.2's 0.15–0.35. Oblique surfaces are most of a maritime or urban frame. | Either a measured pigmented-paint n/k table in the fit, or the four painted materials' `a` moved inside §4.2's range against a public angular measurement. S40 closes, or is restated with what remains unexplained. | RP.7 | M | C |
| AT.7 | ✅ **done.** `Scene.emissivity_extrapolation()` + `extrapolation_breakdown()` report it and decompose it; `validate_thermal_diurnal.py` prints it under the temperatures it quotes. Which reports carry it is a registry with a reason per exemption. ADR 0119. | **Measured.** Two corrections: the fraction is **bit-identical across all 20 materials** (0.6067 at 300 K, ε 0.113–0.943), so there is no *worst material*; and the total hides the claim — 0.508 **red tail** at 300 K vs 0.394 **interior gaps** at 800 K, the regime `PH.6`/`PH.7` render in. Not monotone: 0.509 at 1200 K. | — | M | C |
| AT.8 | **Angle-dependent τ and the second-hit ray.** `directional_properties_for` moves ε(θ) and re-derives ρ but holds τ at its normal-incidence value, and `surface_radiance` defaults `L_behind = L_env`. The limb of every windscreen and shop window keeps full normal-incidence transmittance in NIR/SWIR. | The limb of a glass panel at 75° shows the transmittance falling toward total reflection, against a flat value today, with Fresnel as the oracle. Needs a second-hit AOV from the Isaac side before the behind-radiance half can improve; the angular half does not. | IG.4 | M | C |
| AT.9 | **Urban aerosol preset, and an honest bound on the seven that exist.** `AerosolRegime` already admits `urban` and nothing uses it; all seven presets declare `status: ESTIMATED` from §7.2 midpoints calibrated to one dry Tucson anchor, with `valid_range_m: 500`. | An urban preset exists and a test asserts its per-band extinction ratio differs from rural by more than the presets' own stated uncertainty. Any range claim beyond 500 m in a report names the calibration anchor. | — | M | C |
| AT.10 | ✅ **done.** `ATMOSPHERE_LADDER`: one wavelength-ordered, gap-free table 0.35-14.5 um, and a band's classes are derived by intersecting its own span (nominal **and** response). `BAND_CLASSES` is gone. ADR 0113. | **Measured.** LWIR and NIR bit-identical, MWIR within 1 ulp (wavelength order), visible within the anchor solver's 1.5e-14. SWIR gains `h2o_0p94`: **16.9 %** of the InGaAs band leaves `window`, tau **-6.5 % at 5 km**, 200 m exact. Holes at 1.8-2.0 / 6.0-7.0 um filled. The carve-out is 7 offences -> 2. | AT.4 | M | A |

| AT.11 | ✅ **done.** `clouds.optical_depth` replaces an authored transmittance: ε = 1 − exp(−0.5 m τ) per ray at airmass m = 1/sin θ, which at the diffusivity factor **is** Shaw & Nugent's 1 − exp(−0.79 τ). The cloud sits at the LCL, so the air in front attenuates its excess. A `sky` display span. ADR 0126. | **Measured.** **53.9 %** of a rendered LWIR frame lay within 0.25 K of one value; the spread inside it was **0.073 K**, drawn as codes **106–255**. After: core spans **1.25 K**, τ to the base 0.68 → 0.49 down the frame. 15 cases. | — | M | A |
| AT.12 | ✅ **done, one half in the engine.** `cloud_deck`: column depth at the LCL, each column given a top and the profile 6u(1−u), whose integral is the thickness — a vertical ray is ADR 0126 to the bit. LWIR marches it, the dome bakes it. The NanoVDB volume is written; **this build's IndeX plugin refuses it**. ADR 0127. | **Measured.** Cloud spans **19.7 K** against the sheet's 1.25 K; emission level **12–834 m** above base; largest bin 55 % → **40 %**. Three corrections: the angular field makes fins, 48 steps alias, every supersample costs 107 s/frame. 21 cases. | AT.11 | L | A |
| AT.15 | ✅ **done.** The deck is a field of *towers*: column depth rises with the field's excess over the condensation threshold, clipped by the inversion, where it was a soft threshold — a flat-topped mesa. Band-limited at 400 m, tiled, marched by the **dome** too and lit by a two-stream reflectance in the same τ. ADR 0130. | **Measured.** Largest 0.25 K bin **53.9 % → 39.6 % → 6.4 %**; in-cloud spread **28.8 K**; dome/LWIR coverage was 26 % / 57 % of the same pixels, now identical. Step size carries the error, not stride: 0.073 → 0.029 for 3x. 25 cases. | AT.12 | M | A |
| AT.16 | ✅ **done.** The thermal solver's weather is synthesised from the *same* weather-fx state the sky is drawn from: `diurnal_series` there, `weather_series_from_state` here, injected through `Scene.from_config(weather_override=)`. `--weather-csv` keeps the measured file. ADR 0136. | **Measured.** Irradiance from the real sun elevation with Kasten-Czeplak cloud attenuation: overcast keeps 25 % of the global and broken cloud *raises* diffuse above clear. Air anchored on the state's own hour, floored at the dew point. 36 + 13 tests. | AT.15 | M | A |
| AT.13 | ✅ **done — and the answer is yes** (reopened 2026-09-26; `write_openvdb` stands, ADR 0140). ADR 0140's verdict rested on a control containing the cube under test and on `OmniVolumeDensity` inputs the MDL lacks. **The path tracer renders the grid**: a mesh box at the grid's bounds, no transform, `isVolume`, the `.vdb` as its density texture. ADR 0144. | **Measured**, 11 runs in linear HDR: the box renders the cloud (p99.9 **2.12** vs 0.92); a scaled cube darkens the frame 40× at any density; `UsdVol.Volume` renders nothing; a volume writes **no depth**. | AT.12, IG.3 | S | X |
| AT.14 | ✅ **done** (ADR 0162). RTX gives no volume depth, so the band marches the shared field to each pixel's own hit: `WeatherFxDeck.march_to`, `SkyModel.cloud_occlusion`, three planes composed by stage 2 on every non-sky pixel. `τ_IR/τ_vis` stays cited data (`docs/clouds-in-the-infrared.md`). Not yet: point targets, the native deck, the drift. | **Measured.** Finite march = sky march at infinite range to 1e-9 once both drop out at the band's opacity (old cap ε ≤ 0.9975); isothermal cloud closes to 1e-9; a target behind an opaque cloud reads as the cloud; 8 cases. | AT.13, AT.19 | L | A |
| AT.17 | ✅ **done.** `surface_treatment` is required on every material, from §4.5's own vocabulary plus `as_manufactured`/`natural` for substances nobody treated, with **no default** — a default is a state nobody chose. Material schema v2. `aluminium_polished` and `aluminium_anodised` join the corrected `bare_aluminium`. ADR 0142. | **Measured.** 23 materials all name a state. One metal, three states: ρ/c_p/k identical, ε **0.04 → 0.09 → 0.845** [R39], a span of 0.805. The anodised one *falls* toward grazing where the two bare metals rise — its surface is oxide. 24 tests. | — | M | C |
| AT.18 | ✅ **done.** `*metal*`/`*alumin*` point at the matte entry and the `aircraft` class at painted skin; `*chrome*` is deleted, not redirected — `phantom4.yaml` settled its chrome from the shader's metallic 0.987, not the name. `audit(…, emissivity=)` fails a prim reaching ε < 0.2 by glob or class; only an asset map or an override may. | **Measured.** A 300 K housing under a 250 K sky reads **40.1 K** apart on a top-hat (38.0 K on the Boson response), **800 × NETD**, and the mirror reports the sky, not itself. Phantom 4 bare coverage 48.8 → **43.9 %**. 27 cases. | — | S | A |
| AT.19 | ✅ **done.** The infrared cloud: **one base** (the deck's own, for temperature and path), **dry-adiabatic** to it (g/c_p; the preset's 6.5 K/km overwarmed a base 3.3 K/km), and **the band's own march** over the shared array (uniform, two samples per pitch, hashed jitter). The stale two-weather `phantom4_weather` clip re-rendered. ADR 0146. | **Measured**: the clip's 17 °C cloud reproduced from the CSV weather; one weather 8.5 °C, dry lapse **7.1 °C**; residual lag-one correlation **0.4–0.56** vs 0.84–0.90; ε error p99 0.0018; 10 s a frame. | AT.16, AT.15 | M | A |
| AT.20 | ✅ **done** (ADR 0153). `ground.material` names a library material; in a sunlit band L_ground += ρ_B E_B/π, E_B = the weather's DNI·sin h + DHI through the band's two solar fractions (`per_dni`, `per_dhi`). Cloud bases gain R(τ, μ0) E_B/π, the dome's own form. In-engine render waits on a free A6000. | **Measured.** V_s = 0 panel = ρ·albedo·E_B/π vs a numpy oracle; SWIR env 9e-8 → 13.4, NIR 8e-16 → 8.55; deck clouds 1.0× → 13× (SWIR), 6.7× (NIR); night 0.0; LWIR bit-identical. `test_sunlit_environment.py`, 11 cases. | — | M | A |
| AT.21 | ✅ **done.** `run_frame` passes `planes["elevation_rad"]` to `apply_layered_gbuffer`, as `atmosphere_stage` already did. One argument: the entry point every driver and `IrCamera` call skipped it while the sky and point-target paths took the slant column. | **Measured.** A 310 K target at 5 km, 45° through `run_frame` read **1.14 K low** with path radiance 60 % high against the stage; now the two agree to < 10 mK on a 0–60° plane, rows attenuate less as they steepen, and a frame without the plane is bit-identical. `test_run_frame_slant_path.py`, red without the fix. | — | S | A |
| AT.22 | ✅ **done.** `excess_radiance` takes the pixel's post-stage-2 value and returns φ [τ_B(R) L_t + L_path(R) − L_pixel]; `inject_point_targets` reads it under every target. The grey L1 model keeps the clear-column form: its path and sky are not one column (S28, `AT.5`). | **Measured.** Over a cloud pixel 20 W/m²/sr above the clear sky a φ = 0.278 target at 700 m loses exactly φ·20 of excess; a target that looks like the cloud through its own air adds < 1e-9; over the clear column old and new agree to 1e-9. `test_point_target_background.py`, red without the fix. | — | S | A |
| AT.23 | ✅ **done.** `SceneTarget.tilt_deg/azimuth_deg`; `build_aerial_gbuffer` writes per-pixel `normal_dot_view`, tilt-derived `sky_view_factor`, and with `illumination=` plus `SkyModel.site` the `l_sun` / `sun_cos_incidence` / `shadow_mask` planes via `solar_terms(t_s)`. | **Measured.** Belly (tilt 180) sees no sky and reads 3.3 K warmer than the deck on paint in LWIR; the ground's n·v = |sin el|; a 1 Jan noon deck at 45 N is lit at cos 0.37, its belly not; the sunlit SWIR deck is 48× its emission; no site → no sun. Isaac-vs-CPU on the fixture awaits a render (`IG.19`). | — | M | A |
| AT.24 | ✅ **done.** `quadrature_grid` divides the support into an even number of intervals no wider than 0.01 µm nor `FINE_GRID_FRACTION` (0.5 %) of the short edge, ending on the last sample; the §3.2 sentence is spec issue S58. The ladder test's NIR and SWIR pins, produced by the padded grid, are re-pinned with the reason. | **Measured.** Odd-count 0.75–1.0 µm top-hat: +33 % → 3e-6 at 300 K, +54 % → 2e-5 at 200 K; vs a 20× finer quadrature NIR 28 → 2.6 mK, SWIR 4.7 → 0.21 mK, LWIR/MWIR bit-identical (0.53 / 0.11 mK). `test_band_integration.py`. | — | S | P |
| AT.25 | ✅ **done.** `FresnelAngular.fit: magnitude | effective_nk` (ADR 0163); `effective_nk_table` scales n, k by one factor until ε_B(0) is the authored value; `bare_aluminium` and `aluminium_polished` declare it. | **Measured.** Bare Al LWIR ε(θ) peaks 0.22 at 85°, 0 at 90°, hemispherical/normal 1.17 (a conductor's 1.1–1.4); a 280 K skin at 89° under a 230 K sky reads 237 K, not 280; ε(70°) 0.132 (was 0.146); water and every `magnitude` material bit-identical; the packed angle LUT follows the dispatch to 0.05. | — | M | A |
| AT.26 | **One emissivity for water.** `sea.py:334` uses the raw Fresnel band value (0.988 at nadir) while `water.yaml` gives 0.96 × shape: 0.9 K apart on the same sea. Decide (ADR) and make the sea model and the library consume the same number. | The sea model's nadir ε equals `MaterialTable`'s for `water` to 1e-6 in every band; a 290 K sea under a 250 K sky reads the same T_app by both routes to 1 mK. | — | S | B |
| AT.27 | ✅ **done.** `gamma_molecular(w, γ₀, β₁, β₂)`; `AtmosphereBandCoefficients.beta2_per_m_per_g2_m6` (default 0); LWIR in every preset: β₂ = 8.6e-7 from the MT_CKD self-continuum at 10 µm, β₁ = 4.1e-5 refit on the clear-dry anchor. ADR 0160; S57 closed. | **Measured.** Clear-dry anchor unchanged to 3 digits (ladder pins and goldens untouched); humid row at its own 24.3 g/m³: τ(200 m) 0.79 → 0.73, in band; tropical 0.79; 12 g/m³ −1 %, 35 g/m³ −16 %; continuum share 34 % at the humid row; β₂ = 0 bit-identical. MWIR wing left linear. | — | S | A |
| AT.28 | ✅ **done.** `column_length(d, θ, H, z₀)` (e^{−z₀/H}, descending rays to the ground); `observer_height_m` on the exponential sum, the layered wrappers, `path_radiance_plane` (per class, 8 equal-loss pieces, ADR 0161) and `apply_layered_gbuffer` via `planes["observer_height_m"]`. | **Measured.** z₀ = 0 bit-identical; air-to-air columns = the two-endpoint integral to 1e-6 (30 m and 3 km, eight elevations); down-to-ground denser than up; from 3 km τ higher and L_path lower; the plane within 0.5 % of the exact path over 13 geometries, 4 presets; exact when level. | PT.28 | S | A |
| AT.29 | ✅ **done.** `truth_planes` writes `cloud_transmittance`, `cloud_range_m` (NaN without cloud) and `cloud_id` (`clear`/`cloud`, transmittance under one half) from the AT.14 planes; the viewer lists them; `frame_labels(cloud_transmittance=)` gives every box the mean transmittance over its mask, into the COCO attributes. | **Measured.** A gradient plane recodes to `cloud` exactly where τ < 0.5, the range is NaN at τ = 1, nothing is faked when absent; a quad behind an opaque cloud labels 0.0, a hull in the clear 1.0, a shape mismatch is refused. 2 cases. | AT.14 | S | A |
| AT.30 | ✅ **done.** Submodule at upstream `544a7d4` (volumes, atmosphere, drift, smoothing). `StageOnlyContext` gains `anchor`, `cloud_drift_m` (zero: no manager clock), `sky_horizon_rgb`, `time`; `author_weather_fx_sky(anchor_m=)` starts the deck at the dome's own `stage_to_field(anchor − drift)`. | **Measured.** A test scans the submodule for every `context.<name>` the backends read and found `time` missing at once; a deck moved 2 km sees other cloud on the same ray; ε p99 0.01003 on the smoothed field, bound 0.012 (0.3 K); 16 cases; an in-engine pair renders. | AT.14 | M | A |
| AT.31 | ✅ **done.** `--cloud-tier`: `author_weather_fx_sky(tier=)` → `render_path` + `CloudVolumeEffect`; `IrCamera(cloud_occlusion=, companion_only_prim_paths=)` (volumes hidden from the G-buffer); both tiers bake synchronously. ADR 0171. | **Measured**: path-traced, LWIR τ 0.33–0.65 on the target and RGB hides it in cloud; real-time, dome carries 60 % overcast and the drone shows; marched sky identical across tiers. | AT.14, AT.30 | M | A |
| AT.32 | ✅ **done** (ADR 0175). One quantity as curve (segments; one may be the opaque complement), per-band table and/or grey `emissivity:`. Assembled per λ under the camera's response (curve → band → grey), Planck-averaged; `curve_fraction` reported; a band value the curve covers, or an unfilled gap, is refused. §4.1, §12.3. | **Measured.** A quartz-like dip reads apart to four camera ranges, each within 2e-4 of a dense-trapezoid oracle; table-only materials read one value for all; the fill step costs 2.2e-4 (~15 mK). `test_material_optical_forms.py`, 10 cases. | — | M | A |
| AT.33 | ✅ **done.** `MaterialTable.for_sensor(library, sensor)`: the camera's `Band.from_spec` response (the LUT's own edge check) and `weighting_for_fpa`; all 12 render/validation drivers use it. `ground_reflectance` takes the scene's response and weighting. | **Measured.** A silicate-like curve packs 0.8849 / 0.8916 / 0.9500 for 7.5–13.5 / 6–13 / 10–13 µm; the Boson vs its top-hat 1.6e-5; InSb packs the photon average; table-only and grey materials identical for every camera. `test_material_table_for_sensor.py`, 4 cases. | AT.32 | S | A |
| AT.34 | ✅ **done.** `import_slum.py --relink` / `import_paired_spectra.py --relink` rewrite each material from its committed curves, no fetch: 30 SLUM (SW 1 − R + LW emission) and 19 USGS (SW 1 − R) now author `spectral_emissivity`; only uncovered bands keep a number (MWIR; SLUM's LWIR as the 7.25–8 µm fill). 57 curve materials in all. | **Measured.** Nominal-band values move ≤ 5e-4 (the old 3-decimal rounding); Boson LWIR `curve_fraction` median 0.92; `docs/materials.md` regenerated (nine 0.01 flips). Slum/paired tests read the library. | AT.32 | M | A |
| AT.35 | ✅ **done** (ADR 0177). `LayeredAtmosphere._anchor_scale` solves the free classes' scale once per band on the nominal top-hat; every camera in the band applies the same γ per class, so its τ differs from the nominal band's only as its classes do. §7.2 says which range a preset describes. | **Measured.** 8–12 µm τ(5 km) 0.311 → 0.743 (≥ nominal at every range, five presets); 6–13 µm τ(200 m) down ≥ the 0.103 h2o_6p3 share; nominal anchor exact to 1e-9. Shipped τ(5 km): Boson +0.05 %, InSb −4.3 %, InGaAs +1.5 %, Si +3.9 %; pins and goldens re-pinned. 16 cases. | — | S | A |
| AT.36 | ✅ **done** (add-on B11). The bridge reports each material's forms (segments with spans, per-band, grey, a summary), a plot sample with gaps kept and per-band `curve_fraction`; `check-curve` reads a picked CSV through the library's loader; `plot-material` draws a PNG; `write-material` sorts segments, prunes band values a curve covers and copies the CSV into `data/`. Loads resolve against the repo served, not the package's root. | **Measured.** 77 bridge tests (5 new), 149 headless-Blender checks (7 new); the smoke test found the data-root bug. | AT.32 | S | A |
---

## WX — Weather: clouds, fog, haze, rain and snow

Opened 2026-10-01 at the owner's request: *"i dont like the weather management at all, we create very bad
clouds, even working hard on it the output is still not well designed, also i dont know how we can simulate
the impact of haze, fog, rain and snow in IR viewport."* The order is the owner's:

1. physics first (`WX.1`);
2. then clouds — the visible companion first, the infrared after it, with **both** cloud sources built to
   parity so one can be accepted later (`WX.2`–`WX.10`);
3. then fog, rain and snow (`WX.11`–`WX.20`).

The evidence is `docs/research/2026-10-01-weather-in-the-infrared.md`; the physics is
`docs/physics-model.md` §4.6, §6.7, §7.5–7.8 and §9.6; the gaps are spec issues `S59`–`S65`.

**Where the clouds stood.** About a dozen ADRs since 0070 made the cloud *consistent* — one field, one
base, both bands, occlusion to the hit. None could fix its *look*, because the look is the shared field and
the visible shading in `isaac-weather-fx`, which ADR 0146 placed outside this repository's remit.

Upstream has since moved (`9469420`…`0596f9c`, 29–30 September):

- a Worley coverage map with Perlin–Worley billows;
- 15 m edge detail at sample time;
- thin edges lit by transmission;
- path-tracer albedo and collision fixes;
- per-genus microphysics;
- a first hero-cloud source (the Disney cloud, path tracer only).

This repository then pinned `544a7d4` (`WX.2` moved it to `11bf843`). Measured on upstream main from the CPU march at one ray per pixel
(2026-10-01), what is still wrong:

- **static below ~10° elevation:** 15 m detail sampled at ~125 m steps, unfiltered;
- **flat lighting:** brightness from a two-stream envelope, so no dark bases or sunlit flanks;
- **no aerial perspective;**
- **striped stratocumulus;**
- a real-time dome that paints cloud at 256 rows over 180°, about 9 camera pixels per texel at 50° HFOV.

RTX's own volume scattering cap (`ptvol/maxBounces`) defaults to 2.

**Two cloud sources, one contract.**

- *Procedural:* upstream's `CloudField`, envelope plus detail.
- *Volume assets:* sculpted, simulated or large-eddy clouds (`core/hero.py`).

Both answer `density(x, y, z)` in the field frame and an extinction in m⁻¹, and both are eroded by the same
detail, so the dome, the volumes and the infrared march read either. Neither is removed until the owner
accepts one against the comparison clip `WX.9`.

**Where the code lives.** Cloud work in `isaac-weather-fx` is committed **there** and reaches this
repository as a submodule bump plus the adapter (the companion-project rule). Every step leaves this
repository green on `make check`.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| WX.1 | ✅ **done.** The physics before any code: §4.6 wet surfaces, §6.7 rain's heat and local shadows, §7.2 note (fog rows, visibility at 5 %), §7.5 cloud-shape targets, detail filtered to the pixel, two sources, the companion's five criteria, §7.6 one march for every medium, §7.7 fog layers, §7.8 rain and snow, §9.6 window drops; `S59`–`S65`; this lane. | **Sourced.** [R62]–[R115]; band-ratio and rain tables from this repository's Mie code; notes and scripts in `docs/research/2026-10-01-weather/`. | — | M | A |
| WX.2 | ✅ **done** (ADR 0178). Submodule `11bf843`: `CloudField`/`HeroClouds.finest_pitch_m`. `sample_pitch_m` reads it, three samples per pitch (`samples_per_pitch`), 512 cap. `path_traced_volume_settings()` reads upstream's tables (32 volume bounces). | p99 ε vs 4096 steps: bridge deck 0.024 → 0.007; production 20°–60° 0.018 → 0.009 (1.8× time), 5°–30° 0.012 → 0.005. Origin test on a fan of rays. | — | M | A |
| WX.3 | **The visible cloud to §7.5's five criteria** (`isaac-weather-fx`): energy-conserving in-scatter with a sky term that survives at low density, octaves, a premultiplied dome composite, `ptvol/maxBounces` raised from 2. | **White furnace:** a ϖ = 1 cloud under a uniform sky vanishes to < 1 % (fails today). A thin edge (τ < 0.3) is never darker than the clear sky behind it above 20°; a base is darker than its sunlit flank; albedo from above within 5 % of two-stream. | WX.2 | L | A |
| WX.4 | **Detail filtered to the pixel** (§7.5). `density(x, y, z, footprint_m=)` band-limits the detail to a footprint; the dome bake passes its texel's, the infrared march its own pixel's, and the march steps at half the finest surviving scale. | **Horizon static:** at elevation < 10°, high-frequency energy against a 4×4 supersampled reference falls below 10 % of today's, in both bands; near clouds keep their 15 m detail (unchanged within noise above 30° at short range); the infrared march's cost on a broken-cumulus frame at most 2× WX.2's. | WX.2 | M | A |
| WX.5 | **Aerial perspective and a dome that resolves the camera's pixel** (in `isaac-weather-fx`): cloud radiance composed through the atmosphere's transmittance to its emission range; the real-time dome's cloud baked to at most two camera pixels per texel over the companion's frustum. | A cloud's contrast against the horizon falls with range as the air's transmittance (within 5 % at 5, 15, 30 km); dome texel ≤ 2 camera pixels at 50° HFOV (today ≈ 9). | WX.3 | M | A |
| WX.6 | **The shape to §7.5's targets** (in `isaac-weather-fx`): measure outline dimension, size exponent and edge width on rendered cloud masks; break the horizontal tiling; remove the stratocumulus stripes. | Area–perimeter dimension 1.3–1.4; size exponent 1.7–2.0 below a 0.5–1 km break; autocorrelation at the tile period < 0.1 (today 1.0 by construction); no along-wind stripe in the 2-D spectrum above the noise floor. | WX.4 | M | A |
| WX.7 | **The volume-asset source to parity** (in `isaac-weather-fx`): hero clouds in the real-time dome as well as the path tracer; several assets with per-instance yaw; large-eddy import (LASSO, htrdr) with σ_vis from liquid water and r_e, and temperature kept; placement by the coverage map; one `CloudSource` contract for both sources. | A hero or LES cloud appears in both tiers and both bands at one position (masks agree to a pixel); an LES field's optical depth equals 3 LWP/(2 ρ_w r_e) column by column; two instances of one asset differ by their yaw. | WX.3 | L | A |
| WX.8 | **The infrared band over either source.** The march reads the contract: volume-asset clouds occlude targets and emit at their own temperature (LES T where present, else the deck's adiabat with the moist rate in cloud); the band ratio from r_e through a Mie κ_B(r_e) table. | A target behind an LES cloud reads as the cloud; an isothermal LES cloud closes to 1e-9; the r_e = 10 µm ratio reproduces 0.50 to 0.02; a target-free sky reproduces AT.14's numbers on the procedural source. | WX.2, WX.7 | M | A |
| WX.9 | **The comparison and the switch.** `clouds.model: procedural · volumes · mixed`; one command renders one scene each way as RGB and IR side by side, encoded to video and placed on the site, for the owner to accept one (ADR on acceptance). | Three clips from one seed and one camera path; each pair passes WX.3's criteria and agrees on cloud pixels between bands. | WX.5, WX.6, WX.8 | M | A |
| WX.10 | **Cloud shadows on surfaces** (§6.7): the direct beam on a facet through the shared field, e^{−τ} along the sun ray, its area mean held to the series' attenuation. | A patch under a 50 % cumulus field shows shadowed and lit cells whose mean equals the uniform model's to 1 %; a drifting shadow leaves a cool footprint that lags it by the patch's time constant. | WX.2 | M | A |
| WX.11 | **Visibility read at 5 %** (`S60`; tell `AT.35`'s owner first: this moves `gamma_aerosol`). `KOSCHMIEDER` 3.912 → MOR 2.996 in `irsim.atmosphere.extinction`; ADR 0049 amended; preset visibilities and goldens refit. | A weather-fx visibility V gives σ_vis = 2.996/V on both sides of the submodule (fails today: 31 % apart); §7.2's haze row refit at its new visibility. | — | S | A |
| WX.12 | **Fog band ratios from droplet size** (`S59`, §7.7): a Mie table of band/visible extinction by r_eff; a fog medium carries r_eff from its type and stage; the `droplet` regime retired. | MWIR, SWIR and NIR fog extinction ≥ the visible for any r_eff ≥ 2 µm (fails today: 0.24–0.60); LWIR/visible 0.35 ± 0.05 at r_eff 3 µm and 1.10 ± 0.05 at 15 µm; Tier 3 "MWIR/SWIR no later than the visible in fog". | WX.11 | M | C |
| WX.13 | **One march for every medium** (§7.6): `march_to` takes a list of media, each a σ_vis field, a class and a temperature, with per-class (r^e, r^a) as band data. | The cloud alone reproduces AT.14 bit for bit; two overlapping media equal one medium of summed extinction to 1e-9; a new class is a data row (the band-scalability guard passes). | WX.8 | M | C |
| WX.14 | **Fog as a layer** (§7.7): weather-fx's fog gains a top, a profile, r_eff and patches; the infrared march carries it with T_fog(z); the visible companion's height fog follows the same field. | From above a 100 m fog, the ground reads through z_top/sin θ of fog, not the range (fails today); the fog top reads colder than the ground air in a thick fog; the two bands agree on the fog's edge. | WX.12, WX.13 | L | C |
| WX.15 | **Rain and snow as media** (§7.8, `S61`, `S62`): Marshall–Palmer extinction; r^e from the pixel's Airy share, r^a the absorbed share; the diffraction halo as a blur; emission at the wet bulb (snow at the ice bulb); visibility split so rain is not folded into fog. | LWIR and visible τ(200 m) at 25 mm h⁻¹ within 5 % of each other (fails today: rain is fog with droplet ratios); path emissivity 0.127 ± 0.01 at 25 mm h⁻¹ over 100 m; a step edge's contrast in rain falls faster than a uniform field's mean. | WX.13 | L | C |
| WX.16 | **Single drops and flakes in the infrared**: the companion's own particle positions, splatted sub-pixel with the bolometer contrast law (cooled detectors: integration time), depth-tested; the particle prims hidden from the G-buffer as the cloud volumes are. | A raindrop keeps 3 ± 1 % of its contrast and a snowflake 39 ± 5 % (§7.8); every flake the companion draws within the near field is at the same pixel in IR; no particle beyond the near field is drawn twice. | WX.15 | M | C |
| WX.17 | **Rain that looks like rain in the visible** (in `isaac-weather-fx`): drops as transparent refracting streaks integrated over the exposure, not opaque white lines. | A streak's brightness follows exposure × drop size × background (Garg & Nayar); the frame's mean over a rain region within the attenuated background. | WX.15 | M | C |
| WX.18 | **The wet surface reads its film** (§4.6, `S63`): ε_wet and the specular fraction from film depth and macrotexture per band; dew and frost onto dry surfaces. | A 0.1 mm film reads water's ε(θ) in LWIR to 0.005 and is specular; a dry patch is bit-identical; a clear night forms dew on a car roof by dawn; Tier 3 "a wet road mirrors the cold sky at grazing angles". | — | M | C |
| WX.19 | **Precipitation carries heat** (§6.7): Q_P = ṁ c_w (T_rain − T_s), snow at its own temperature and L_f while melting. | −58 ± 2 W m⁻² at 10 mm h⁻¹ on a road 5 K above the drops; zero with no rain; an energy budget that closes to 1e-6 with the film. | — | S | C |
| WX.20 | **Water and snow on the window** (§9.6): a blocked fraction of each pixel's beam footprint at the drop's temperature, from the pupil's position. | One drop covering 10 % of the aperture lowers responsivity to 0.90 ± 0.01 with no change of MTF (Jobert et al. 2024); a drop on a window ahead of the pupil shades a soft-edged region. | WX.15 | M | C |

---

## SE — Sea and maritime

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| SE.1 | ✅ **done.** `sea_envelope.py` records the 50° from-nadir limit and reports a frame's fraction beyond it, on the exact `sin θ = (1+h/R) cos δ` (90° at the horizon). Not a gate. ADR 0118. | **Measured.** Shore/mast **1.0000** outside, airborne **0.0000** inside; crossed at **31 m** from 20 m. A **wind axis** too — the 55° drop matches the published 2–3 % only to **7.3 m/s**. On depression the answer **inverts**. Identity held at ε ≡ 0.5, so it is not the test. | — | M | B |
| SE.2 | ✅ **done.** `test_maritime_isaac.py`: one maritime stage, four cameras — the sea declared and undeclared, and a SWIR pair with and without the illumination bundle. 6 cases. | **Measured, in-sim.** The rendered sea **is** `SeaModel`: 293.2 → 290.2 K over 0.5–25° of depression, every band within **0.5 K** of the analytic curve, monotone *down* because the slant path beats the angular emissivity. Horizon at row 72 against 71.36 from the dip, 3.6× nearer curved than flat. Undeclared, the sea reads the **200 K LUT floor** (`IG.17`). SWIR spans 10× more with the bundle. | — | M | B |
| SE.3 | **Sea-surface temperature against ECOSTRESS SST.** The skin model shipped at `c6f98aa` has no external check. ECOSTRESS L2 carries an SST layer valid over all water. | Bias and RMSE against ECOSTRESS SST for a matched place, time and weather record, reported beside ECOSTRESS's own validation accuracy (bias −1.6 K, RMSE 3.1 K against SURFRAD) so the bar is the instrument's, not an aspiration. | XD.9 | M | B |

---

## SC — Sensor chain and published-reference anchoring

The NETD anchor is the strongest part of the repo and the rest of the chain is shipped and tested. What
is missing is **wiring and anchoring**: one path is not reached at all, one term is identically zero in
every config, and three configured values disagree with the project's own measurement or with FLIR's
published acceptance limits. None of this needs a camera — a Boson Engineering Datasheet is free.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| SC.1 | ✅ **done.** `from_sensor` gains `noise_handle` (`auto`/`netd`/`electrons`) and takes the electron budget whenever a photon FPA authors `read_noise_e`. A bolometer keeps the ADR 0025 anchor and is refused `electrons`. | **Measured through the pipeline, not the module.** MWIR InSb σ **533.3 → 350 e⁻** (1.52×, the anchor's own invention); SWIR InGaAs dark **0 → 199.7 e⁻**, larger than its 120 e⁻ read noise. NETD is now a prediction that can disagree — 19.47 mK against a 20 mK claim, where anchored it equalled the claim to 1e-9. A 10 mK claim raises. 11 cases. | — | M | A |
| SC.2 | ✅ **done.** The mutation is alive: `read_noise_e` reaches the rendered σ. `halmstad_boson_320.yaml` takes ME.5's measured ratios; the 640 keeps datasheet limits, settling open question 6. | **Measured.** Under `netd` a 10 % — or a 50 % — perturbation is **bit-identical**, the pre-SC.1 state; under the electron budget σ tracks it exactly, 2.1 % out at the InSb's bright point and the full 10 % where read dominates, including Monte Carlo on 400 frames. Ratios 8.8× / 7.2× / 19.1× out, and `h` only **1.1×** — already right. `vh` 2.64 round-trips. 24 cases. | SC.1 | S | A |
| SC.3 | ✅ **done.** `ffc_interval_s` 180 → **300**, `thermal_time_constant_ms` 10.0 → **8.0**, both `ratios_3d` blocks marked ESTIMATED with Table 13 beside them. [R24] contradicts itself on the FFC defaults; ADR 0091 records which reading wins. | **Measured** at Table 13's conditions (f/1.0 lensless, 20 °C camera, 30 °C scene, averager off): tvh **48.5**, th **2.4**, tv **2.6** mK against < 50 / < 18 / < 18 — compliant and **7× more uniform than guaranteed**. Ratios left to `SC.2`: a ratio of upper bounds is not a typical value. 9 cases. Every golden array bit-identical. | — | M | A |
| SC.4 | ✅ **done.** Both Bosons carry `mtf.aberration_sigma_um` = **1.654 µm**, *solved in code* by `aberration_sigma_for_mtf` from FLIR's 42 % nominal at Nyquist, not pasted; a test re-runs the solve against the YAML. Three generic cameras keep an ideal lens, listed with reasons. ADR 0117. | **Measured.** Lens at Nyquist 0.420 authored vs **0.461** diffraction-only; system 0.267 vs 0.294 — **both inside the 0.27 ± 0.03 band**, so the sharp check is the lens factor, not the system. Goldens: ramp 7.8 mK, hot patch 3.97 K, uniform fields bit-identical. | — | M | A |
| SC.5 | **State the de-trending convention on every noise statistic.** Published measurement moves an uncooled imager's temporal noise by up to **3×** with the filter alone (1.00 unfiltered, 0.68 poly2, 0.34 Gaussian σ=8), and the effect differs between imagers. Larger than the codec floor already guarded. | `decompose_3d` and `temporal_shape` take an explicit `detrend`; every reported σ carries it; the Tier 4 report prints two conventions side by side. Red today: no rendered-versus-real noise bound is reproducible by a third party. | — | M | X |
| SC.6 | **ADR: which NETD irsim means.** NETD is N_im/SiTF and there is no agreed N_im — five incompatible definitions are in current use, and NVESD's own recommendation changed in 1992, 2005 and 2023. | A record plus a docstring change wherever the project writes "NETD". A measured NETD quoted without its definition and filter is not comparable to anything, so this is a prerequisite for SC.3 and SC.9 meaning what they say. | SC.5 | S | X |
| SC.7 | **Adopt the standard bench conditions** so irsim's Tier 2 numbers are comparable rather than project-local: SITF as a differential sweep −10…+20 °C in 5 °C steps with a linear fit over −5…+15; 3-D noise from 128 frames at 25 °C; MTF by ISO 12233 at 0.5 cycles/pixel from 128 averaged frames. Report N_temp and N_spat. | The bench reproduces its own previous numbers under the new conditions within the estimator's stated sampling floor, and the two summary quantities every external source quotes are printed. Red today: the frame counts and sweeps are project-chosen. | SC.6 | M | X |
| SC.8 | **Retire "for when a camera arrives".** All four Tier 2 measured comparisons skip forever against directories never created, keyed to `flir_boson_640_lwir` while the only public measurement belongs to `halmstad_boson_320`. Datasheet limits go on one camera YAML, field-measured ratios on the other, with a schema guard that the two never mix in one `ratios_3d` block. | The four benches run instead of skipping; the framing in `validation/measured.py` and the Tier 2 protocol doc, which contradicts ADR 0003, is deleted. | SC.3 | M | X |
| SC.9 | **NV-IPM Measured System Component export**: four 3-D components in Kelvin, pre-sample MTF arrays, normalised response, pitch, FOV, frame rate. | A third party range-checks irsim's Tier 2 benches independently in a free model with no camera. The export round-trips through NV-IPM's own reader. This is the cheapest external credibility the project can buy under the no-camera constraint. | SC.7 | M | X |
| SC.10 | ✅ **done.** `isp/damping.py` (`TransferDamper`, `damping_over`); every global AGC mode takes `lut_hook` (tiled: its tile stack); `isp.agc_damping` per native frame, scaled by elapsed scene time; Boson 0.85 ([R51] Damping 85). ADR 0170. | **Measured**: background follows a^k·old + (1−a^k)·new to 1e-6; 60 Hz video settles to 1 code in 60 frames; a 6 s time-lapse is bit-identical to undamped. Lag-1 against the real set: EV.14's clip. | — | M | X |
| SC.11 | **Thermal polarity as a config switch and an evaluation axis.** MaCVi 2026 had to normalise inverted thermal scaling across real maritime clips. White-hot stays the default (the owner's preference, and all five configs carry `palette: gray`). | A polarity-flipped condition appears in the detector evaluation and in the ablation switch set. Red today: a detector trained only on white-hot fails on half the real world and nothing measures that. | — | S | X |
| SC.12 | **Record the size-of-source effect** (~0.8–1.0 K for uncooled microbolometers per VDI/VDE 5585, against ~0.1–0.2 K for cooled MCT) as a known omission in `docs/spec-issues.md`. | A spec-issue row with the number. It is larger than several effects the chain does model, so leaving it unrecorded misstates the chain's own error budget. | RP.7 | S | X |
| SC.13 | **Close the ISP temporal-filter question.** `sensor_chain.py:241` says the filter "stays an identity until ME.5's temporal PSD on flat sky shows whether real cores low-pass their output at all". ME.5 landed and measured a one-pole time constant of 2563 frames at a drift fraction of 0.9276, above the 0.5 the report itself calls untrustworthy. | The answer — "not measurable on this set" — is recorded in the ADR and the dangling conditional is removed, so the next session does not re-open a question that was answered. | — | S | X |
| SC.14 | **Carry M9.9's two open criteria rather than dropping them.** The Tier 3 sensor-chain bench landed amber with an aerial edge-asymmetry band (legacy ME.4) and the real-vs-synthetic comparison (legacy ME.6) open; both need statistics measured from public real imagery. This row keeps them tracked and is what `test_tier3_sensor_chain.py` asserts against. | The test reads this row and fails if it stops naming both, or if it is marked done while `EV.6` and `XD` have not supplied the measured bands. An untracked criterion is one nobody will close. | EV.6, XD.1 | S | X |
| SC.15 | **What the camera believes about ε (§11.5).** A `measurement:` block carries `assumed_emissivity`, `reflected_temperature_k` and an assumed atmosphere — the camera's belief, as ADR 0021 already does for the housing — and the pipeline emits `T_meas` beside `T_app`. | Belief equal to truth returns the kinetic temperature inside the LUT's round-trip budget, which fails if any radiance term is misplaced. Belief wrong is checked against the closed form: 0.95-on-0.90 costs −2.04 K at 300 K and −2.98 K at 330 K, and 0.01 of ε is 0.43 K, nine × NETD. | — | M | X |
| SC.16 | **The camera is in the weather too (§9.5).** A lumped camera node on the *same* `WeatherSeries` as the scene: h(v) scales both τ = C/h and ΔT_self = P/h, so wind cools the body and shortens its lag. **Scoped: route it through the FPA node and the NUC residual, not the optics self-emission term.** | Red today: nothing in `detector/`, `isp/` or `noise/` reads wind. **Measured while scoping:** h 8.35 → 27.16 across [R44]'s 0.8–8.5 m s⁻¹ moves the Boson's 4 K still-air rise by **1.66 K**, which through ADR 0016's 87 mK/K is **0.144 K — 34× short** of the published 4.88 K. | — | M | X |
| SC.17 | ✅ **done** (ADR 0145). `apply_optics` computes Φ = A_d Ω_eff [L_h + τ RI (L_scene − L_h)]: RI multiplies scene minus housing, on-axis power unchanged, and `optics.vignetting_map` is loaded, resolved and hashed. A +1 K housing drift at the Boson corner now reads 372 mK on a 300 K scene (was 110) and 938 mK on a 230 K sky; the centre stays 87 mK. | `test_housing_field.py`: sign, closed-form depth, flat at T_housing. Warp device test updated, not run. | — | M | A |
| SC.18 | ✅ **done** (ADR 0148). At power-up and every FFC the chain computes the closed-shutter frame (`shutter_flux`, T_shutter = T_FPA) and `TwoPointNuc.refreshed` moves the display offset so it reads flat; the residual's gain acts on signal − shutter. Boson corner, housing −2 K since the FFC: −563 mK on 300 K, −1446 mK on a 230 K sky. | `test_shutter_reference.py`: closed form to 5 %, monotonic. Warp test not run. | SC.17 | M | A |
| SC.19 | ✅ **done.** `noise.bad_pixel_late_fraction` (schema v12) marks defects that failed after the factory map; `replacement_mask` leaves them in the image, so a late hot pixel reaches the 8-bit output as a white dot. Flags are drawn last, so maps are bit-identical at any fraction. Both Boson configs set 0.01, ESTIMATED from one public clear-sky frame. Warp kernel takes the map; device test updated, not run. | `test_late_defects.py` | — | S | A |
| SC.20 | ✅ **done.** `sky_only.yaml` + `irsim.validation.radial.radial_fit` (plane + paraboloid) + `scripts/validate_sky_flat.py`. Drift bowl sign follows the housing both ways, radial share 0.999; a public clear-sky frame fits a bright centre, share 0.86. The sky itself is dark-centred near the zenith, so that bright centre needs ≳ 1.7 K of cooling since the FFC. | `test_radial_profile.py` | SC.17, SC.18, SC.19 | M | A |
| SC.21 | ✅ **done** (ADR 0147). `agc: information_based` in `irsim.isp.information`: bilateral LP/HP split, the LP histogram plateau-clipped plus an |HP| information histogram, `linear_percent`, detail headroom, HP back at the slope. Opt-in; `linear_percent` also on `plateau_equalization`. | **Measured.** 0.6 % four-part target on a 60 K sky: plateau ≤ 3 codes, `information_based` + λ 0.3 ≥ 25, parts ordered; frame 96 3 → 30. σ_r → 0 is `agc_plateau` bit for bit. Hashes unchanged. 12 cases. | — | M | A |
| SC.22 | ✅ **done** (ADR 0152). The Boson 640 carries [R51] p. 5's published factory values: `information_based`, plateau 0.07, linear percent 0.20, max gain 1.38, headroom 12/255; DDE, ACE and damping recorded as unmapped. Halmstad keeps its recorder's `linear`. | **Measured.** Synthetic sky + drone: 2 → **37** codes, parts ordered; `phantom4_perpart` f96 2 → 20, f48 3 → 26. Goldens: only the two `display8` arrays move; `dn16`/`radiance`/`apparent_t` bit-identical. `test_boson_default_agc.py`. | SC.21 | S | A |
| SC.23 | ✅ **done** (ADR 0151). DN 0 is **zero scene radiance**, not the LUT's 200 K: the layered winter zenith sky is below L_B(200 K) in every preset. `RADIOMETRIC_RANGE_K` keeps the top code and the flat-field blackbodies. | **Measured.** −68…−41 °C sky: 1 code → >900. `phantom4_perpart`: worst frame 66 % → 0 % at DN 0. Coldest preset sky 266 DN on scale. Gain −4.3 %, the closed form; DN16 → T_app ≤ 9.5 mK over 200–470 K. A 200 K floor fails. `test_adc_floor.py`. | — | M | A |
| SC.24 | ✅ **done.** `render_phantom4.py --agc-clip` films the camera's own per-frame `display8` (the float-bin plateau is gone); the `ir` top is the hottest target pixel; readouts sit in a margin. `irsim_eval.video` holds the span, margin, interior and code-count helpers both drivers share. | **Measured** in-engine, same hot-parts mask: camera `display8` and `FIXED_agc` agree within 2 codes on 13 of 14 flight frames of the re-rendered 144-frame clip. Hot parts: 3 codes on the old clip, **19–32** now; whole-airframe interior median 108. `test_demo_display.py`. | SC.21 | S | A |
| SC.25 | ✅ **done** (ADR 0149). The equalising modes gain the controls every vendor ships under some name: `clip_limit_low` (Lepton low clip) and `max_gain` (Boson, Xenics), beside `linear_percent`. Zero is the old operator; the Warp AGC refuses what it has no port of. | **Measured.** Clear sky + 0.6 % target: low clip 1e-3 lifts the target ≤ 3 → ≥ 20 codes, parts ordered. Bland 33 DN sky: `max_gain` 1.25 cuts noise from 256 codes to ≤ 43. Pre-SC.21 Boson ISP hash reproduced. 11 cases. | SC.21 | S | A |
| SC.26 | ✅ **done** (ADR 0149, S54). The NIR silicon config drops the Boson's copied `isp` for a visible-camera display: `linear`, gamma 2.2, no DDE. SWIR keeps equalisation, which its vendors document on-board. | **Measured.** `nir_frame_display8` moves up to 71 codes; `nir_frame_dn16` and `nir_frame_radiance` bit-identical. | — | S | A |
| SC.27 | **Auto-exposure for photon FPAs (§11.3, S54).** NIR/SWIR scene radiance spans five to six decades; every photon config is one fixed integration time. Add an AE loop (integration time and gain mode chosen from the previous frame's histogram, as `PipelineState`) for photon detectors; LWIR is untouched. | Red: a dusk NIR scene renders black at the noon exposure. Green: AE brings the median to its target within N frames from noon to dusk; the chosen time is in the frame sidecar; saturation < 1 %. | — | M | X |
| SC.28 | ✅ **done.** `run_frame` passes `motion_for_integration(planes, sensor)` to `apply_optics`, as `optics_stage` already did. The adapter's synthesised plane (IG.6) now reaches the frame it was made for. | **Measured.** On the moving-edge fixture `run_frame` equals `optics_stage` to 1e-6 of the peak flux; a 2 px/frame edge on the bolometer (duty 1) changes the edge pixels by > 1 % of the step where before it changed nothing; a zero plane and no plane are bit-identical. `test_run_frame_motion_smear.py`, red without the fix. | — | S | A |
| SC.29 | ✅ **done.** `render_phantom4.py` enables the flat field and the M9 chain with the other drivers' `ValueError` retry, gains `--no-flat-field` / `--no-chain`, and its docstring stops claiming the mesh bridge never reached a render. `from_sensor` keeps the flat field off by default: the radiometric branch divides cos⁴ out analytically and the goldens rest on it. | **Measured.** Red: frame 96's dn16 sky rose 3235 → 4465 DN centre to corner, +90 display codes. `test_render_drivers_flat_field.py` walks every `render_*.py` that builds a pipeline and was red on this one alone. | — | S | A |
| SC.30 | ✅ **done** (split). `_scene_radiance_from_signal` subtracts the budget's dark and cold-shield background electrons before inverting -- a calibrated camera's offset, whose mean a dark frame measures and whose shot noise the budget already carries. The cooled camera's out-of-cone housing term and the `g_th` default moved to `SC.34`. | **Measured.** MWIR InSb read a 300 K blackbody as **303.5 K** (pedestal 3.3e5 e⁻ against 3.3e6 signal); now 250, 300 and 350 K round-trip within 10 mK through the photon chain. `test_photon_radiometric_offset.py`, red without the fix. | — | S | A |
| SC.34 | ✅ **done.** `housing_terms(A_d, F, τ, L_h, cold_shielded)` → `(base, per_ri)`; `is_cold_shielded` (photon FPA < 200 K); `apply_optics`, `invert_optics` and the Warp kernel take the two scalars; Boson `g_th_w_per_k` 2e-8 (ESTIMATED, VOx literature). | **Measured.** Warm form bit-identical (every Boson golden unchanged); cooled MWIR corner darker than axis by (1 − τ)RI L_h only (MWIR goldens −0.2 % DN at the corners) and apply→invert to 1e-4 at the corner; Boson floor 33 mK under the 50 mK anchor (was 74.5 over). | — | S | A |
| SC.31 | ✅ **done.** `agc_information(..., detail_threshold_sigma, noise_sigma_dn)` gates |high-pass| at k × σ_TVH for the histogram weighting only; σ_TVH is the detector's, recorded per frame in `PipelineState.sigma_tvh_dn` and passed by `run_frame`, never estimated (an estimate read quantisation as noise: 44 codes on a clean frame). Default 0; Boson 3; ADR 0152 note. | **Measured.** 60 K sky, σ_TVH = 8.5 DN: 3 px target at 0 °C 70 codes ungated, 132 gated = noise-free; 1 px and hotter-than-sky targets move ≤ 6; noise-free frame and k = 0 bit-identical, hash-neutral. | — | S | A |
| SC.32 | ✅ **done.** `irsim.config.catalogue`: `resolve_sensor_path` (path, stem or `sensor_catalogue.yaml` alias; `$IRSIM_SENSOR_DIR` first), `list_sensors`, `migrate_sensor_document` + `MIGRATIONS`; `load_sensor_config` takes a name and migrates; `sensor.extensions:`. ADR 0159. | **Measured.** Five aliases resolve; a user dir shadows `boson640` by file name and a user camera loads by name; frozen v8–v11 Boson fixtures load to today's minus `bad_pixel_late_fraction`; a fake rename migrator walks once; `extensions` round-trips; `f_numbr` is still refused. | — | S | A |
| SC.33 | ✅ **done.** `calibration.geometric` (OpenCV fx, fy, cx, cy, k1–k3, p1, p2) → `Intrinsics.from_sensor`, `effective_distortion`, USD lens schema, ROS info, targets, discs; the CPU builder distorts its own planes. `calibration.radiometric`: hashed gain / offset / bad-pixel `.npy`, `netd_k`, `sitf_dn_per_k`. ADR 0164. | **Measured.** Checkerboard through k1 = −0.3: 1.9 px at the corners, straight under `cv2.undistort` to < 0.1 px; gain map → its flat field to 0.01 DN; bad-pixel map is the replaced set; no block: hash and goldens unchanged. | — | M | A |

---

## OC — Optical focus and defocus

Raised by the owner on 2026-09-23: *"the focusing parameters like focus distance is not being added for
IR camera at all"*. Correct, in all three places it could live. `OpticsSpec` has no focus field; `psf.py`
builds **one** kernel from a single scalar `aberration_sigma_um` and `stage.py:75` convolves the whole
plane with it, independent of `distance_m`; and the camera prim sets `focalLength` and the apertures but
never `focusDistance` or `fStop`, so the RTX camera is a pinhole. The cascade in `mtf.py:3` names
`MTF_defocus`, but ADR 0059 folded it into the aberration Gaussian and `SC.4` then solved that Gaussian
from FLIR's on-axis MTF at Nyquist — an **in-focus** figure. The one number that could have carried
defocus is authored to mean its opposite.

**This is post-process, not a render setting, and that is not a preference.** `gbuffer_isaac.py:6` —
"the renderer is asked for geometry and ids only" — so there is no radiance image in the renderer for
RTX depth of field to blur; enabling `fStop` would blur a distance plane and an instance id. Blurring
temperature would be wrong anyway, because L(T) is nonlinear and the blur is an average over radiance
(the same class of error as non-negotiable #3). The correct plane is the supersampled band radiance, and
that is exactly where `apply_optics` already convolves. The render product is **already** created at
supersample × native, so `distance_m` arrives on the same grid the convolution runs on.

**Hopkins, not the geometric disk.** Geometric optics needs W020 > 2λ, i.e. a blur circle of 16 N λ —
168 µm, or **14 pixels**, for a Boson at F/1.0, reached only inside 1.2 m. Every defocus this project
will render is in the diffraction-dominated transition band, where measured at Nyquist for the Boson at
10 m focused at infinity the three candidates give Hopkins 0.356, geometric 0.173 and a Gaussian 0.203 —
the cheap models lose about half the contrast. Hopkins subsumes diffraction (it reduces to
`mtf_diffraction` at W020 = 0, checked to the 7.7e-9 quadrature floor in `OC.2`), so it replaces the first cascade
factor rather than multiplying onto it, and `SC.4`'s aberration sigma stays valid untouched.

**A receding surface is one surface.** Slicing a frame into depth layers puts two slices of the
*same* surface either side of a bin edge, and `over` treats the nearer one as hiding the farther
one. It does not: the aperture bundle lands on both, so they add. Compositing them with `over`
leaves a quarter of the surface-to-background contrast as a seam along every bin boundary — 8.6 K
on `OC.12`'s cube, growing with `max_layers` — and `OC.11` fixes it by asking whether there is
empty space between two layers before letting one occlude the other (ADR 0134).

**Partial occlusion is bounded, not ignored.** A gather blur cannot let background leak in behind a
defocused foreground silhouette, because a single-layer G-buffer has no behind. `OC.6`–`OC.8` attack it
in the order that costs least: splatting fixes the outward half with no new data, the sky and sea cases
have an **analytic** hidden layer this pipeline can evaluate exactly for zero render cost (`OC.7`), and
only ground clutter falls back to inpainting with a measured error bound (`OC.8`).

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| OC.1 | ✅ **done.** `irsim.optics.defocus`: blur circle, W020 in waves, hyperfocal and the depth-of-field limits. A non-positive distance **raises**, so a G-buffer's `distance_m = 0` sky cannot be read as the nearest thing in the scene. | **Measured.** The oracle is the thin-lens construction, not the module's own algebra: 15 cases agree to 1e-9. The DOF limits are checked by evaluating the blur *at* them. A Boson at infinity is 0.02 waves at 100 m, 0.23 at 10 m, 0.47 at 5 m; geometric optics waits for 168 µm, 14 px. 26 cases. | — | S | X |
| OC.2 | ✅ **done.** Hopkins' defocus OTF by quadrature, `geometric` (jinc) and `gaussian` selectable beside it, `band_average_otf`, and a SciPy-free `bessel_j1`. ADR 0129. | **Measured.** Hopkins equals `mtf_diffraction` at W020 = 0 to the 7.7e-9 quadrature floor — what lets it replace the cascade's first factor. At Nyquist, Boson at 10 m: **0.356** vs geometric 0.173, Gaussian 0.203, in focus 0.461. **Defocus is achromatic** (λ cancels), so averaging moves the contrast-carrying OTF < 0.3 %. 12 cases. | OC.1 | M | X |
| OC.3 | ✅ **done.** `scripts/focus_demo.py` and a two-depth scene: focus on either cube, stills plus a focus-sweep video. Layers blurred in **radiance** and composited back to front on their own blurred coverage. | **Measured.** The 10-90 % edge width tracks the blur circle **in quadrature** — the line-spread constant holds inside 0.05 from 3 m to 80 m of focus. A flat field survives any focus to 1e-9. A half-covered 250/315 K edge reads the radiance-blended temperature, 6.7 K off the Kelvin mean. 5 cases. | OC.2 | S | X |
| OC.4 | ✅ **done.** Schema **v10**: `optics.focus` (`infinity`/`hyperfocal`/`fixed`), `mtf.defocus_model`, `mtf.defocus_apply`, and a `fidelity.defocus` ablation that switches off but never on. ADR 0129. | **Measured.** Every golden array **bit-identical** and every config hash unchanged, because the v10 defaults are dropped from the hash exactly as ME.8's `fidelity` block is — while a model, a focus distance and the ablation give five distinct hashes. Hyperfocal resolves to 16.347 m on a 12 µm pitch. 12 cases. | OC.2 | S | X |
| OC.5 | ✅ **done.** One kernel per frame from a `DefocusKernelBank`, chosen by the **median** range of the geometry in frame — sky excluded via `sky_mask`, since it carries `distance_m = 0`. `PipelineState.defocus_w020_um` reports what was used. | **Measured.** Focusing on a 4 m scene sharpens its edge 1.5x against the same camera at infinity; the reported W020 is the analytic 6.125 µm to 1e-9. A frame three quarters sky takes the geometry's defocus, an all-sky frame takes none. Goldens bit-identical. 6 cases. | OC.4 | M | X |
| OC.6 | ✅ **done.** `irsim.optics.layered`: bins equal-width in **W020**, each layer blurred with its own kernel and coverage, composited back to front and **normalised by the accumulated alpha**. Sky is its own layer at infinity. | **Measured.** A far background's edge stays 1.8x sharper than one global kernel leaves it, while the near slab gains 1.7 px of blur in quadrature — both asserted, because asserting only the first passed while the binning had collapsed to one layer. Flat field exact to 1e-9. 9 cases. | OC.5 | L | B |
| OC.7 | ✅ **done.** An optional `background_t_k` G-buffer plane seeds the layered composite as an opaque backmost layer, so the gap behind a defocused silhouette is filled with the truth. No second render: sky and sea radiance are functions of ray direction the adapter already evaluates. | **Measured** against a two-layer reference: **1e-12** with the plane against **1.98** radiance units without, the error confined to the silhouette — under 1 % of peak sixteen pixels away. 6 cases. | OC.6 | M | B |
| OC.8 | ✅ **done.** `push_pull_fill`/`estimate_background` for scenes with no analytic backdrop, and the bound itself. ADR 0131. | **Measured** against a two-layer reference at three depth ratios, in apparent temperature: normalisation **2.87–5.06 K**, push-pull **2.57–5.03 K**, analytic **0.0000 K**; error under 5 % of peak outside the silhouette band. Push-pull buys 1–10 %, not an order of magnitude — and found two 1.5 K defects in `OC.7`. 7 cases. | OC.6 | M | C |
| OC.9 | ✅ **done.** `autofocus` (passive contrast detection: normalised Tenengrad, multiplicative probe, step that widens when stalled) and `track` (a `semantic_id`'s median range, holding when it leaves frame). | **Measured.** The measure peaks at the true range; it converges from 40 m to 8 m and then does **not** hunt (<5 % over ten frames); it lags a closing target and arrives inside the 5.4–16 m depth of field once it stops; a flat scene never moves the lens. 15 cases. | OC.5 | M | X |
| OC.10 | ✅ **done.** Thermal defocus from the lens and housing materials, folded into an **effective focus distance** so every downstream stage gets it free. Schema v11; `athermal: true` is the default. | **Measured.** β(Ge) = 126.2e-6/K; −1.44 µm per kelvin for a 14 mm lens in aluminium; a **20 K rise moves focus from infinity to 6.8 m**, inside the 16.3 m hyperfocal, leaving 28.7 µm of blur at 1 km. Aluminium beats invar. Hashes and goldens unchanged. 10 cases. | OC.4 | M | X |
| OC.11 | ✅ **done.** The layered composite tells a **surface from a stack**: `over` only across a gap, added where layers abut. ADR 0134. | **Measured** against the `over` chain kept as the test's reference. Slicing one receding surface left `alpha(1-alpha)(L_surface - L_behind)` at every bin edge — **8.6 K** on `OC.12`'s cube, RMS **1.31 → 2.32 K** as the cap went 3 → 8, so the error grew with the only quality knob. Now **0.16 K**, RMS falling with the cap; `OC.7`'s 1e-12 untouched. 7 cases. | OC.6 | M | X |
| OC.12 | ✅ **done.** `scripts/focus_sky_demo.py`: one cube against a **cloudy** sky, the focus pulled from sky to cube and back on the shipped `OC.6`/`OC.7` path with the sky as the analytic background. | **Measured** with `OC.9`'s focus measure per region, because a whole-frame one cannot tell the two settings apart: the sky loses **3.1x** of its contrast when the lens leaves it, the cube **1.7x**. A **clear** sky moves **1.002x** across the same pull and holds three orders of magnitude less structure — defocus is a low-pass filter — so the scene carries cloud. 6 cases. | OC.11 | M | X |
| OC.13 | ✅ **done.** Membership is **fractional**: a pixel is shared between the two W020 bins its blur falls between, so a bin boundary is a ramp and not a step. ADR 0134 addendum. | **Measured.** Adjacent bins carry different kernels, so across a hard boundary the blurred coverage left a shortfall the composite spent on background. Ripple **±0.52 % → ±0.06 %**; on `OC.12`'s cube **0.16 K → 0.041 K** peak, RMS 0.0055 K — a seventh of NETD, and 8.76 K → 0.070 K across the lane. Flat slabs degenerate to the old partition exactly. 11 cases. | OC.11 | M | X |

---

## EV — Evaluation methodology and sim-to-real

The Tier 4 acceptance run at `docs/validation/tier4-2026-09-15` fails and says so, which is right. But
**two of its three failing statistics and its entire attribution table rest on an artefact**, so its
conclusion — that the dominant term is the signal path — is currently not established by the experiment
that reports it. `EV.1`–`EV.4` redo the run before anything is built on top of it.

`EV.9` is the lane's terminus and it is named here, not gestured at: it measures target-crop Sobel
gradient variance and GLCM entropy on irsim renders with the point-wise field **on** versus a per-object
constant, against real target crops. That is the experiment that converts the owner's headline
requirement into an external, cited number, and it can return a negative.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| EV.1 | ✅ **done — pooling could invert a verdict, not just blur one.** Loaders keep clip boundaries; `compare_clips` runs each whole-frame check on every (real, synthetic) mosaic pair, aggregating by **median** — stated in the note, carried as data in the JSON. | At the row's own 55-code spread the pooled composite reads **2.39 codes, a PASS**, while **0 of 36** pairs meet the 8-code target (median 17.4). Each set is also compared with **itself**: synthetic clips differ by 30 codes, so the target was unreachable. One clip a side reduces to `compare_frames` exactly. 6 cases. | — | M | X |
| EV.2 | ✅ **done.** `irsim_eval.admission` runs the three gates the project already had — static, noise scale, flat window — and the loader scans until `limit` clips **pass**, not the first `limit` alphabetically. | Refusals are returned, not dropped: the report prints `N of M admitted (…)`, since "six clips" and "six of forty-one, the rest codec-flattened" are different claims. The **first** failing gate is the reason: a pan invalidates the median the window finder judges against. The synthetic side is audited by the same gate but **not** gated (EV.3 owns that). 8 cases. | — | M | X |
| EV.3 | **Make the synthetic side admissible.** `find_flat_regions` returns `[]` for every rendered clip, so the synthetic side cannot be measured by the route the real side was measured by, in either direction. | Measured on `matched_000`: the best window has structure_ratio 1.50 against a 0.25 limit; a raw 64×64 sky window reads σ_TVH **0.329 codes** against the real set's 7.272 — about **20× too quiet**, the opposite sign to the report's headline. The gap is then stated as a number rather than as a refusal. | EV.2 | M | X |
| EV.4 | **Permutation null for the discriminator.** The analytic `null_sigma` assumes independent patches; patches are drawn four per frame from 60 frames of 6 clips. | The report says the sets are separated "38 null standard errors from chance". At the clip level the null σ is ~0.17 and the run sits ~2.4 σ out — a **~16× overstatement**, with the caveat living in the code and not in the report. A permutation null (repeated refits on shuffled labels) absorbs the dependence and needs no new dependency. | — | M | X |
| EV.5 | **Content-matched patch sampling.** Patches are drawn uniformly at random anywhere in the frame; the real clips are tripod shots containing ground, trees, buildings and horizon, the synthetic ones are sky-only renders. | `gradient_median` and `skew` are the 2nd and 3rd heaviest discriminator features — exactly the statistics content mismatch moves. After: patches are drawn from matched content classes and the report separates the physics question from the scene-composition question. | EV.1 | M | X |
| EV.6 | **The statistics a published LWIR study found separate the sets**: histogram total variation (AGC comb, d = 1.242), target-crop Sobel variance (d = 1.224), target-crop GLCM entropy (d = 0.974), dynamic-range-used (d = 0.899), and rank-order correlation of brightness. | Pure NumPy on synthetic oracles: a comb spectrum returns HTV within 1 % of its closed form; a linear ramp crop the analytic Sobel variance within 1 %; a two-level checkerboard GLCM entropy of exactly 1 bit; rank-order correlation 1.0 under a monotone remap and < 0.5 when shuffled. None exists today. | — | M | X |
| EV.7 | **Score target crops separately, and put a real-vs-real control beside every gap number.** Published controls: FID 41.40 synthetic against 38.74 between two *real* sets; KDD 332 against 262; histogram EMD 29.9 against 13.6–18.8. | Full-image and crop-level gaps told opposite stories in the one study that measured both, and the crop level is where the per-point requirement lives. Without the control, ADR 0068/0085's thresholds stay guessed; with it they are measured. `datasets.yaml` already indexes enough sets to compute it. | EV.5 | M | X |
| EV.8 | **Make the Tier 4 artefact reproducible from itself.** `tier4-2026-09-15.json` records only label, checks, passed and attribution — no clip list, archive SHA-256, `config_hash`, CRF, seed, patch size or count, all of which exist elsewhere and all of which change the numbers. | Re-running from the artefact alone reproduces every number bit-for-bit. ME.5's own report already meets this standard (it records the archive hash and states every number was measured on those bytes); the headline Tier 4 artefact does not. | EV.1 | S | X |
| EV.9 | **The per-point ablation.** Target-crop Sobel gradient variance and GLCM entropy on irsim renders with the point-wise field on versus a per-object constant, against real target crops. Pre-committed: a negative redirects the effort to the ISP. | The published gap driver is d = 1.224, with the note that simulated targets "act as uniform silhouettes rather than noisy, physical heat sources". It must run where the feature is switched on, hence the dependency — a negative from a feature not active in the measured targets is not a negative about the feature. | PT.9, EV.7 | M | X |
| EV.10 | **Viewpoint-distribution alignment.** Sample camera elevation, slant range and target aspect from distributions matched to the intended validation set. | The largest single published ablation in this literature: drone mAP@0.5 **0.464** with fixed pitch, **0.981** with random pitch, **0.995** with pitch drawn from the real set's metadata. It is a config change, not physics, and it is currently unmeasured in irsim's aerial and maritime scene configs — the cheapest leverage available on the sim-to-real deliverable. | — | S | X |
| EV.11 | **Sweep the mixed-to-real ratio** instead of testing one mix. Published: synthetic pre-training plus 100 real images beat real-only in every configuration measured, and the KAIST optimum is only 10–20 % synthetic. | mAP@50:95 against synthetic fraction at 0, 10, 20, 50 and 100 %, three seeds each, on the ~100–200 Python-readable boxes of `XD.11`. Passes when a mixed point beats real-only by more than the seed spread and any point beats the published synthetic-only band 0.41–0.54; a curve without seed spread fails review. | XD.11 | M | X |
| EV.12 | **Training-free dataset-quality proxy.** Evaluate SDQM (public code, Pearson r = 0.87 with YOLO11 mAP50) as a stand-in for the torch-blocked half of ME.7. | A defensible sim-to-real number inside the existing no-GPU gate, making the torch install a confirmation rather than a prerequisite. A clear negative result is an acceptable outcome and must be recorded as one. | — | M | X |
| EV.13 | **Publish the paired RAW-16 / AGC-8 artefact** (open question 7 gates the publication, not the format). No public thermal set offers the pairing and the literature names the 16→8 mapping as the dominant sim-to-real factor. | The float planes invert to apparent temperature within the project's existing 10 mK encode/decode budget while the 8-bit stream fails the same bound — the fp16 negative control that already discriminates. Note honestly that this is a self-consistency check on irsim's own encode/decode, not an external radiometric check; XD.5 is the external one. | IG.13 | M | X |
| EV.14 | ✅ **done.** `aerial_reference_clip.yaml` + `render_quad_outbound.py --reference`: 600 frames at 60 Hz, 50→250 m, cumulus behind, damped AGC, sub-pixel companion at 1.25–1.45 km. ADR 0168. | **Measured**: 26.7→5.4 px; companion in frame 600/600, +0.26–0.56 K over cloud; AGC mean step median 0.01 codes (p95 0.03), lag-1 0.9995. Real-set band: EV.3–EV.7. | AT.21, AT.22, SC.28, SC.29 | M | A |
| EV.15 | ✅ **done.** `irsim.io.labels`: `frame_labels(part_id, legend, targets, category_of, points)` → `Box` (xyxy, COCO xywh, YOLO cxcywh, RLE mask, visibility), `PointLabel(φ)`; `write_frame_labels` (COCO json + YOLO txt per frame), `split_manifest` / `write_split_manifest`. | **Measured.** Boxes equal the mask's extent to the pixel and the RLE decodes to the mask; an absent target gets no box; a sub-pixel target is a point with its φ; visibility is the mask's share inside the frame; a seeded split reproduces; loads in pycocotools when present. Drivers not wired. | — | S | A |

---

## XD — External data anchors

Every set in `data/validation/datasets.yaml` today is 8-bit, post-recorder and mostly lossy-coded, which
is why ADR 0068 declares most of Tier 4 untestable. That is an **indexing** problem, not a fact about the
world: radiometric public data exists, and two of the strongest anchors are not imagery at all.

Since `XD.2` the index says this in fields rather than in prose: `signal_path` is one of `display`,
`recorder`, `radiometric` or `unknown`, `with_signal_path("radiometric")` returns `[]`, and the reader
takes 16-bit frames whenever a set that has them is indexed. So the rows below are now the only thing
missing, and each one turns a `[]` into a set.

Nothing here proposes buying a camera, and `XD.12` is the one action with the highest value per unit of
effort in the whole plan.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| XD.1 | ✅ **done, and the row itself was wrong.** `anti_uav_600` indexed — 600 sequences, 723k IR frames, the largest here; `lrddv3`'s licence and camera corrected; `anti_uav_410`'s two fields left null on purpose. | **Verified at source 2026-09-24.** The row asked for **CC BY 4.0** on `lrddv3`: that is the *paper's* arXiv badge, the dataset page names **CDLA-Permissive-2.0**, and this field opens the fetch gate — so the row would have granted a permission the frames lack. Camera confirmed. 640×512/25 Hz is the RGBT *parent's*, and `probe_clip` reads it from the file. 5 cases. | — | S | X |
| XD.2 | ✅ **done** (ADR 0068 addendum). `signal_path` is one of `display`/`recorder`/`radiometric`/`unknown`; the prose is `signal_path_note`; `Sequence` declares `bit_depth`, the **significant** width — FLIR's ADAS is 14-in-16, 4× in the floor. | 18 measurements declare their paths in one table and both locks turn from it: the index won't load if a set claims what its path can't carry, and `measure_clip` requires the path. ISP and sensor paths are **disjoint**, asserted as a property; `unknown` ⊂ `display`. `noise_3d_kelvin` has `[]` today, pinned — XD.3/4/10 each change it. | XD.1 | M | X |
| XD.3 | ✅ **done, and the row's "16-bit" is unverified.** Indexed: CC BY-NC-SA 4.0 (a *data* grant — LICENSE plus the README's own Copyright line), DOI, 2916 images, 7 classes, ADK, NETD < 50 mK. | The ADK *writes* 16-bit TIFF; nobody states what is in `Images.zip`, so it is `unknown`, not `radiometric` — the XD.1 substitution moved to the field that opens the gate. Also **1423 of 2916 (48.8 %) came off a 320×256 camera** and the release presents 640×512, so `spatial_psd`/`edge_spread` are excluded. NC + ShareAlike is a live limit for a vehicle project. 5 cases. | XD.2 | M | B |
| XD.4 | **LTIR v1.0** — the only 16-bit public source found that is made of *sequences* (20, 8-/16-bit variant), so the only one that can carry temporal PSD, FFC and fixed-pattern-growth work without a codec floor. | Indexed with ADR 0068's provenance fields and `licence: unstated`. The temporal analysers run on one 16-bit sequence and the 1/f knee they report lies inside the band `SC.5` declares for a Boson-class core; on the 8-bit variant of the same sequence they disagree by the codec floor. Red today: they skip. | XD.2 | M | X |
| XD.5 | **FLAME 3** — per-pixel Celsius from a calibrated radiometric response, the only absolutely calibrated public imagery found, open access on IEEE DataPort. | `irsim.radiometry.encoding` round-trips every FLAME 3 pixel, 250 K to the 500 °C cap, within 10 mK; apparent-temperature inversion at FLAME 3's mode and tail returns the input within 1 mK; a low-gain radiometric config reproduces the 0–25 °C mode and the rail at 500 °C (`PH.8`). Frame-level labels only, so no detector claim. | XD.2 | M | X |
| XD.6 | **ARM Infrared Cloud Imager** — radiometrically calibrated full-sky downwelling LWIR, 7.3–14 µm, 320×240 uncooled microbolometer, in W/(m²·sr) to better than 0.5, from a 9-month ARM SGP deployment with netCDF in a free archive. | **The only external check that exists** for ADR 0070 (cloud clutter), ADR 0086 (scattered-sunlight sky) and ADR 0071 (layered slant path) — the physics of the owner's first lane, validated against nothing at all today. Calibrated sky imagery, in the units the sky model predicts, with clouds. | AT.1 | L | A |
| XD.7 | **SURFRAD / BSRN downwelling longwave** — 1-minute pyrgeometer flux with air temperature, RH, wind and solar measured at the same station and minute, US public domain. | `longwave_down` driven by a station's own T_air, RH and cloud lands within 25 W/m² RMSE of the pyrgeometer over a clear month (clear-sky parameterisations sit at ~23 W/m² against BSRN); it cannot detect a second `WeatherSeries` with identical values, so the identity assertion stays the enforcer of #6. Red today: no external W/m² check exists. | AT.1 | M | X |
| XD.8 | **ECOSTRESS / ASTER spectral library** — over 3,000 measured spectra covering asphalt, concrete, soils, vegetation, water, snow and metals, converted to emissivity by Kirchhoff. All 19 irsim materials are `source: literature` and none is anchored to a measurement. | A per-material band-integrated check with a **stated tolerance** — not 1e-6, since the existing closure is exact by construction and this is a different quantity with its own uncertainty — a stated integration convention, and a record of which spectrum each YAML was checked against. | AT.7 | L | X |
| XD.9 | **ECOSTRESS LST / SST** as an absolute surface-temperature check in Kelvin over a known place and time, matchable to the weather that drives the solver. | Sets a defensible bar rather than an aspiration: a 70 m spaceborne product with a full atmospheric correction validates at bias −1.6 K and RMSE 3.1 K against SURFRAD. irsim's solver is assessed on that scale, not on a 0.1 mK convergence tolerance that measures arithmetic. | XD.7 | M | X |
| XD.10 | **FLIR ADAS v2 pre-AGC frames** — 16-bit-in-14 TIFFs in `analyticsData`, Tau 2 640×512, 13 mm f/1.0, T-linear at 0.04 K per count, so the quantisation floor is **11.5 mK**, under a 50 mK NETD. Fires ADR 0068's own "revisit when". | 3-D noise on a flat region in Kelvin, comparable to FLIR's published limits with no SITF inference and no codec floor. Recorded constraints: it is the **ground** lane; it has no in-scene truth so it cannot support a per-material bias claim; Terms of Use are form-gated. | XD.2 | L | C |
| XD.11 | **Python-readable label sources** for the detector half: BIRDSAI (real and AirSim-synthetic aerial TIR with a published baseline), HIT-UAV and MONET (CC BY 4.0, readable boxes), RGBT-Tiny (115 sequences over sky and sea, > 81 % of targets under 16×16 px). | Each set indexed with ADR 0068's provenance fields before download; a loader yields ≥ 100 boxes as NumPy with no MATLAB dependency, and a hash-pinned box count per set fails if the archive changes. Halmstad's MATLAB MCOS boxes stop being the recorded blocker for ME.7. | XD.1 | M | X |
| XD.12 | **Ask the Halmstad authors for the Y16 originals.** Their Data-in-Brief paper states the Boson was run in raw Y16 16-bit mode and that "the raw format is used in the database", then that "all videos are in mp4 format" — the 16 bits existed at capture and the encode destroyed them. | If they survive, the **primary** reference set becomes radiometric, on the **aerial** lane, with the **exact** Boson core the sensor configs model, under a CC licence. Time-boxed (open question 1); the plan does not depend on the answer. | — | S | X |
| XD.13 | 🟡 **A four-band emissivity source.** Done: 7 polished metals (published n, k); 19 surfaces paired from USGS and UCSB or polymer n, k; 30 urban surfaces from SLUM (same sample, both wavebands). Left: paper, fabrics, PE, ABS, rubber (ECOSTRESS unreachable; FLIR's table is the fallback). | After: each covered material's band average reproduces §16.2 within the library's own sample spread, or the gap is named. | — | M | X |
| XD.14 | 🟡 **Five done; the rest needs a licence call.** Curves from CC0 RefractiveIndex.INFO constants, shape from the table, level the authored value: glass and dry soil (silica), snow (ice), both polycarbonates (PC's own). Open: rubber, nylon, ABS, fabrics, skin, paints, concrete. ECOSTRESS has them, reproducible with JPL's credit line but no licence (ADR 0041 wants CC0); USGS splib07a is public domain but refuses this machine. | **Measured.** Nominal bands unchanged. 8–12 / 10–13 µm: glass 0.847 / 0.945, soil 0.898 / 0.963, snow 0.994 / 0.988. | AT.34 | M | A |

---

## IG — Isaac glue integrity

The layer that produces the owner's images is the least verified layer: `src/irsim_isaac` is at 43.5 %
coverage with six files at 0 % totalling 1,101 statements, and `tests/integration` runs in no automated
job. Several of these rows are not new features but *documented invariants that are not true*.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| IG.1 | ✅ **done.** Three flags — `strict_materials`, `strict_thermal_nodes`, `strict_patch_coverage` — one per failure. The six drivers and five integration tests now name the two they meant; coverage keeps its `True` default. | **Measured, engine-free.** `test_camera_strictness.py` drives the real `planes()` on a synthetic frame: each guard fires only on its own failure, and the drivers' flag pair still raises on a patch gap. 5 cases; re-merging turns 3 red. In-sim check not run (workstation in use) — that is `IG.2`. | — | S | 0 |
| IG.2 | ✅ **done.** The oracles are the camera's three named world axes and an authored wall, and `position_frame_residuals` decodes with `world_positions`, not a copy. **Run on the A6000:** `tests/integration` executed for the first time — 15 modules, one Kit. | **Measured, in-sim.** A wall decodes onto its authored plane and a bound patch gathers a ramp across one prim; **225 passed, 1 xfailed** in 2:19. Three defects fell out, two stale: α was still `SC.3`'s old 0.811, the leak oracle eroded its own mask, and the point chain delivers 0.7785 of its model (`PT.23`). | IG.1 | M | A |
| IG.3 | **Enumerate the build's annotator registry** instead of hand-written candidate lists. `SURVEY_CANDIDATES` names 3 motion and 3 occlusion strings; the build registers `Motion2dXYZ`, `MotionVectors`, `OcclusionSD`, `SemanticOcclusionSD` and more. | A committed registry dump (name, dtype, shape) from the 6.1 build, and a test that every ADR 0014 negative names an annotator absent from the dump or one that returned no data when requested — a negative that merely was not requested fails. In-engine; a CPU-only session ships the dump reader and the test on a fixture. | — | S | X |
| IG.4 | **Probe for UV, texcoord and primitive-id channels.** README and ADR 0087 state their absence as *measured*; no probe has ever requested one. Two untested leads: the RTX-sensor `objId` upper index (per-primitive for procedural geometry) and GeomSubsets' distinct `StableIdMap` entries. | The probe requests each channel and records dtype, shape and a checksum; a positive is a plane whose values differ across one mesh, a negative a recorded refusal or a constant plane. Either way ADR 0087's sentence goes from asserted to measured. In-engine; `WM` makes it non-blocking. | — | S | X |
| IG.5 | ✅ **done.** `UNVERIFIED_CHANNELS` keeps `motion` unattached unless a caller names it (the probe does), and `geometry_planes` lost its `"pixels"` default — a motion plane with no convention now raises. Requiring an unverified channel is refused outright. | **Measured.** The three conventions differ by the resolution and the sign of y: raw 0.01 → 0.01 / 1.28 / 2.56 px/frame at 256 px. An all-zero guard would never have caught this — the floor is 6e-5, not 0. 10 cases; reverting the default turns 5 red. | — | S | 0 |
| IG.6 | ✅ **done.** `IrCamera.planes()` sets `motion_px` from `MotionTracker`, which gains an injectable transform reader and a per-frame path set; the three flight drivers declare their moving roots. Refused, not faked, on the first frame and outside the camera frame. | **Measured**, 14 engine-free cases: f·dx/R in pixels, the plane through the real `planes()`, and §16's smear — at 11 px/frame the bolometer's 10–90 edge goes **0 → 8.8 px** and the cooled InSb **0 → 1.1 px**, the 8.33× duty ratio. A wrong premise corrected: a rigid child's offset cancels exactly. | IG.5 | S | A |
| IG.7 | **Re-probe the three ADR 0014 negatives that look like capture-protocol artefacts.** (a) `motion_vectors` with `rt_subframes=0` and the timeline running — IsaacSim #722 says pausing zeroes the motion g-buffer, exactly the 6e-5 signature. (b) `/rtx/rendermode` with NVIDIA's casing `RayTracedLighting`. (c) The Replicator `occlusion` annotator. | One line each in an existing probe. If (b) flips, several negative AOV results were taken under whichever mode that token selected. If (c) delivers, it replaces the unoccluded geometric `V_s`. | IG.3 | S | X |
| IG.8 | ✅ **done.** `precision_critical` is gone; `_as_f64_plane` refuses float16 on every plane. The claim that justified the carve-out was false: ADR 0014's addendum records `normals` as **float32 ×4, full resolution** and the registry agrees. | **Measured.** The fp16 plane in that table is `PtWorldNormal`, which `AovReader` already rejects for being half-res and all-zero — nothing was rescued. Test asserting the opposite inverted; occlusion and motion gained refusals. Reinstating the carve-out turns 3 red. | — | S | 0 |
| IG.9 | ✅ **done.** `_native_rgb` checks the colour buffer against the render product and raises on a mismatch (half, double, one row short) instead of cropping and averaging it; an absent or black buffer is still reported through `rgb_problem`. | **Measured** (engine-free, 3x camera): each native pixel is exactly its own block's mean; every mis-sized buffer raises (`test_companion_rgb_registration.py`). | — | S | X |
| IG.10 | **Mark UNMAPPED pixels in the float outputs** and write `id_coverage` to a sidecar. They are ORed into `sky_mask`, so they skip the atmosphere *as well as* taking ε = 1 against their own temperature; the magenta overlay is display-only; the per-frame coverage number computed in `material_ids.py` is never recorded. | With `strict_materials=False` everywhere this is the live path, not a debug one, and a detector would train on pixels nobody computed. After: `radiance` and `apparent_t` carry a mask plane and every frame's coverage is in the sidecar. | IG.1 | S | X |
| IG.11 | **`heading_deg` and `refresh_pose()`.** `IrCamera.heading_deg` is stored and never read — the live copy is on `SceneIllumination` — so a caller passing it to the camera alone gets the sun in the wrong compass direction silently. `refresh_pose()` is manual and one production caller remembers it. | Two silent-pose footguns in the object that already produced one silently wrong frame. After: the dead parameter goes, and `planes()` compares the cached pose against the prim and raises rather than rendering geometry from the new aim and sky from the old. | — | S | X |
| IG.12 | **A cheap per-frame invariant on every annotator plane** — finite, right shape, not constant — generalising `_reject_reason` beyond required channels. Replicator annotators are documented to return empty arrays on random frames under multi-GPU (IsaacSim #507), and the known-issues page lists AOV texture-dimension mismatches that cause missing exports. | The rule `AovReader` already has at probe time applies on every frame. A silently empty frame in a 10 k-frame dataset export is unrecoverable after the fact, and the microseconds are free next to the radiance kernels. | IG.9 | S | X |
| IG.13 | ✅ **done**, in two commits. `FrameWriter` binds a run's hashes, quantity and container once; `quad_flight`, `aircraft_pass` and `vessel_departure` now write float32 planes and a sidecar, not 8-bit PNGs alone. The sweep went 3 scenes → 7, the 8th declared unswept with its reason (no camera). | **Measured.** Guards ask the bar of each driver in turn; against the parent commit they fail for exactly the three offenders. Two real gaps surfaced: `--rt-subframes` and `--integration-ms` were passed to every child and accepted by three. 33 cases. | — | M | A |
| IG.14 | **One lane driver behind the six render scripts.** Measured: 281 identical lines between `render_quad_flight` and `render_aircraft_pass`, 73 % line similarity. They have already drifted — only three call `write_frame`, only one exposes `flat_field_enabled`. | Adding a lane stops meaning copying 500 lines, and a fix like IG.13 stops meaning fixing it three times. Each existing driver's output is bit-identical before and after the refactor, which is the test. | IG.13 | M | X |
| IG.15 | **Live IR in the Isaac viewport.** The owner's standing requirement. `display_render_var` is RGBA-unorm-only, so this means publishing an 8-bit grayscale AOV (or deliberately overwriting `LdrColor`, which is the documented way to reach existing consumers). | The white-hot display stream appears in the viewport during a render, matching the written PNG to within the AGC's own quantisation. Note the hazard the SPG docs state: AOV name collisions are **silent** and the built-in shadows yours, so the `Ir*` prefix is load-bearing. | DC.1 | M | X |
| IG.16 | **Take the Warp ISP's host readbacks off the per-frame path.** `replace_bad_pixels_warp` calls `counters.numpy()` **inside its pass loop** — a device sync per iteration to read two ints. `agc_lut_warp` pulls the whole 2^bit_depth histogram, plus a second array in plateau mode, then uploads the table. | The loop condition becomes a device flag and the LUT is built in a kernel; M10.7b's bit-exact replacement and M10.8's ±1 display code still hold. Raised by external review; both readbacks verified. | — | M | B |
| IG.17 | ✅ **done.** `mark_unmapped_radiometry` writes NaN over unmapped pixels on `radiance` and `apparent_t`, beside the magenta the display already got; `fold_mask_to_native` marks a detector pixel when **any** of its k×k samples was unmapped. NaN rather than a sentinel kelvin: every sentinel is a number something averages. | **Measured.** `SE.2`'s undeclared sea read **200.1 K** over 72 % of the frame, in range and in kelvin; it is now wholly NaN, and the declared run beside it stays finite and warm. 5 engine-free tests, 6 in Kit. | — | S | X |
| IG.19 | ✅ **done.** `optics/exposure.py` (`exposure_window_s`, `subframe_offsets_s`, `accumulate_subframes`); `IrCamera.get_outputs(rgb_subframes=, pose_at=)` poses along the driver's track and averages the companion; `render_phantom4.py --rgb-subframes`. ADR 0167. | **Measured** (engine-free, `test_companion_exposure.py`): step edge at 3/6/12 px per frame, bolometer and photon windows: companion 10-90 % width = kernel's to < 0.5 px, centred to 0.25 px; static scene bit-identical. Rendered pair: render lane. | SC.28 | S | A |
| IG.20 | ✅ **done.** `irsim_isaac/patch_stage.py`: ENU config → stage frame by one rotation, a quad per patched surface with its own material (shared prims split), fields bound strictly, water film companion-only. `scripts/render_patch_scene.py` (presets, time-lapse, pair). ADR 0172. | **Measured**: rotated road = ENU to 1e-6 K, same shade cells. LWIR: wall shadow step 6.9 K (solved 7.6), road wet 47.9 vs dry 55.5 °C at T+20 min. | IG.2, PT.20, PH.2 | M | P |
| IG.21 | ✅ **done.** `IrCamera.plumes()` → `run_frame(plumes=)` via `world_to_opencv_rotation`; `inject_plumes(sky_mask=)` so a sky pixel's `distance_m = 0` no longer clips the chord; `render_patch_scene.py` scales the eye by the field of view. | **Measured**: rotation = projection path to 1e-9; sky-backed plume = open-air plume. Rendered at cruise: +80.5 K MWIR / +10.8 K LWIR near the pipe, +25.0 / +1.8 K mid-plume. | IG.20, PH.6 | S | P |
| IG.22 | ✅ **done.** Bonnet box drawn 5 mm inside its patch (`bonnet_draw_inset_m`), so strict coverage passes; the overlay's key state is the engine's `load_at`. | **Rendered** (overcast, LWIR, 30 × 60 s): bonnet hot cell 306.8 K at key-off → 314.8 K at T+27 min while the bay cools; peak apparent 303.5 → 310.7 K. The automatic span still clips (use `--span-c`). | TC.6, IG.20 | S | P |
| IG.23 | ✅ **done.** `example_boson_640_wide_calibrated.yaml` (ILLUSTRATIVE k1 −0.25), `calibration_checkerboard.yaml` + preset (13 × 11 tiles, 12 K), `measure_calibration_render.py` (corners before/after `cv2.undistort` with the file's numbers). | **Rendered**: rows bow 9.2 px LWIR / 9.3 px RGB → 0.13 / 0.11 px; IR–RGB corners 0.09 px mean. Engine-free: a half-strength wrong lens stays > 2 px bowed. | SC.33, IG.20 | S | A |
| IG.24 | ✅ **done.** `detector_period_s`, `capture_motion_scale` in `ir_camera.py`: the companion window and the `motion_px` smear use the detector's frame, not a time-lapse's capture interval, as `rotor_veils` did. | **Measured**: `_exposed_companion` poses identical offsets at 17.35 s and 1/60 s capture, 7/8 of the detector window, bolometer and photon; 17 s step → ×1/1020; real-time unchanged. | IG.19, IG.6 | S | A |
| IG.18 | ✅ **done.** `GBuffer.background_t_k` field; `to_dict` emits every `OPTIONAL_KEYS` plane; `to_gbuffer(..., elevation_rad=)` carries the adapter's elevation through validation and `ir_camera.py`'s hand re-add is deleted. | **Measured.** The dataclass's fields equal `REQUIRED_KEYS ∪ OPTIONAL_KEYS`; `from_dict(to_dict(g))` round-trips all twelve planes bit-exactly, dtypes included; extras ride along; an unknown key is still refused. | — | S | A |

---

## AI — Asset ingestion

`AI.1` shipped the half that is engine-free: a third-party model can now be converted, mapped and
audited on the CPU (ADR 0128). `AI.2` finished the other half — the Phantom 4 is solvable geometry
(ADR 0132), it flies on a stage in both bands (ADR 0133), and since the `mesh_fields=` wiring the
solved cells reach the pixels on every prim a scene binds. What is left is a scene question rather
than a code one: `phantom4_parts.yaml` binds 9 of 19 part prims, so the shells that make up most
of the silhouette still render at their node's one temperature.

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| AI.1 | ✅ **done.** Per-asset material map (`configs/assets/`), a new precedence rung above the semantic class, and `scripts/prep_asset.py` — import, rescale, USD, prim dump, audit, all on the CPU via Blender's bundled `pxr`. | **Measured.** The committed Phantom 4 dump goes 20/41 → 41/41, and two *confident* global hits are corrected: the shell's areal heat capacity halves (4399 → 2205 J m⁻² K⁻¹) and the motor housings go ε 0.09 → 0.90. 16 cases; reverting the rung turns 5 red. | — | M | A |
| AI.2 | ✅ **done.** `IrCamera` takes `mesh_fields=`, so `MeshPointBridge` (`WM.3`) has a caller that renders. `MeshBinding.frame` names the Xform the asset is *mounted* by, the patch keeping the archive's own coordinates — the core refuses a patch in a moving frame. `StraightOutTrack` flies it near to far. | **Measured, in-sim.** **4,756** px took a cell of `AI.5`'s part-split asset (9 prims, 23,477 cells, 4 nodes), each propeller holding a gradient across itself: **1.33–2.28 K** on a 50 mK NETD. 4→80 m, 183→9 px, under cumulus worth 40 K of LWIR sky. | AI.1 | L | A |
| AI.3 | ✅ **done.** `irsim.io.asset_budget` measures a prepared archive engine-free and `prep_asset.py` refuses over it: 1.3 M faces total (GT.7's reference), 200 k per prim, plus a *resolution floor* derived from conduction — `sqrt(alpha x 60 s)` = 1.11 mm at the library's slowest material. ADR 0137. | **Measured.** The Phantom 4 is 1,532,656 faces against that budget, and **77 %** of them are finer than the floor; one prim holds 100,926 faces over 2.7 cm2 — a 73 um cell. 19 tests. | AI.1, GT.7 | M | B |
| AI.4 | ✅ **done.** `materials_usd.walk_stage` reads `materialBind` subsets: one record per face group for an audit, one per prim for a render driver (an instance id is per prim), with `subset_meshes` and `shadowed` naming what each reading cannot say. `audit_materials.py` prints both. ADR 0128 addendum. | **Measured.** A committed fixture puts glass, concrete and cladding on **one** mesh, Blender's slot 0 beside them: the walk gives 6 surfaces, not 3, and reading the mesh binding costs **1.59 K** (32 NETD) on the concrete. 10 tests, 5 in Kit. | AI.1 | S | B |
| AI.5 | ✅ **done.** An imported asset is decomposed into **functional parts** by connected component, authored as data in the asset config, and regrouped so that one prim is one part. ADR 0138. | **Measured.** 41 material prims → 31,068 components → 19 parts at 100 % of 0.294 m2; the part-split USD keeps all 2,486,459 faces. Four nodes separate at T+600 s: airframe 26.4 C, battery 32.9 C, ESC 39.5 C, motor 46.0 C. 22 tests. `AI.2` rendered it: the parts scene is what the outbound clip flies. | AI.1 | M | A |
| AI.6 | ✅ **done.** `irsim.io.asset_material_split` plans one prim per material and `prep_asset.py --emit-material-split` regroups the geometry in Blender, on the CPU. Blender's importers put a USD subset, an FBX group and an OBJ `usemtl` on one footing — a per-face slot index — so it is one mechanism for every format. | **Measured, end to end.** The subset fixture: 3 meshes → 6 prims, **35.7 %** of its 14 m² rendered as the wrong material before, and the walk in Kit now finds **no** subset mesh left. The Phantom 4 is 41 → 41 at 0.0 %. 21 tests. | AI.4, AI.5 | M | B |
| AI.7 | ✅ **done.** Every part its own node: `phantom4_perpart.yaml` puts four motors and four ESCs on four throttle histories, six airframe-family parts on their own offsets, each propeller on its own motor's speed; `target_for_part` lets the *scene* set the granularity and `--agc-clip` shows cloud and motors in one frame. ADR 0143. | **Measured.** Engine-free: rear pair 6.9 K over the front in cruise, outer pair 5.9–6.1 K over the inner in each orbit, all four within 0.1 K landed; 5 tests. In-sim: the figure-of-eight clip under fair cumulus, three stretches. | AI.2, AI.5 | S | A |
| AI.8 | ✅ **done.** A link becomes a gated local model: `fetch_sketchfab.py` + `irsim.io.sketchfab` parse the UID, gate the licence **before** the download, fetch the glTF (the API never serves source FBX) and write `ATTRIBUTION.md` + committed sha256-pinned provenance. CC0/BY/BY-SA share; all else quarantines in `3d_models/quarantine/`. ADR 0150. | **Measured, engine-free.** 33 tests pin what no image reveals: the shareable set is exactly {cc0, by, by-sa}; every other known slug quarantines with a reason; an unknown licence never shares; provenance round-trips through YAML. | — | S | A |
| AI.9 | ✅ **done.** The `ingest-asset` skill: a link becomes a library entry. Drives AI.8 and ADR 0128's toolchain: scale against a published dimension, parts by evidence order (metadata → material names → geometry/symmetry → renders; names last), materials into the closure-gated library with `ESTIMATED` flags, then `.blend` + `.usdc` + FBX export. | **Measured.** `--save-blend --emit-fbx` ran headless on the subset fixture: one Blender pass wrote `.usdc` + `.blend` + `.fbx` from one scene; the driver→worker argv is pinned by 3 tests. The skill names every gate it defers to. | AI.8 | M | A |
| AI.10 | ✅ **done.** The `low2high` skill: geometry fit for the camera, up or down. `irsim.io.mesh_facets` gives each edge its silhouette error `(w/2)tan(θ/4)` against half a pixel; `tessellate_step.py` meshes STEP/IGES via OpenCascade (optional `cad` extra) with part names, in metres. ADR 0155. | **Measured.** Phantom 4 as authored: 0/41 meshes faceted at 2 m, 1.36 mrad; decimated to ~100 k faces: 40/41. Sagitta identity exact to 1e-9; 22 tests + 4 with the extra. | AI.9 | M | A |
| AI.11 | ✅ **done.** `PartsConfig.contacts` (`ContactSpec`: a, b, joint, area_m2) and `.hidden_parts` (`HiddenPartSpec`: box, component, mass, c_p, dissipation, status) in `irsim.io.asset_parts`; `load_asset_mapping` checks joints against the table, `assign_parts` areas against measured parts. Phantom 4 authored (ESTIMATED). | **Measured.** Refused by name: missing part, self-contact, name shadowing, unknown joint, area > either part's (box area for hidden), hidden part without mass or component. Round-trips through YAML; the older `phantom4` config loads unchanged; 8 tests. | — | M | A |
| AI.12 | ✅ **done.** `configs/components/*.yaml` (7 entries) + `irsim.config.components`; `HiddenPartSpec.resolved_capacity_j_k` / `resolved_dissipation_w` inherit from the component; `load_asset_mapping` checks the names; `build_full_solve` takes `components`. | **Measured.** Every rated dissipation = `P_in (1 − η)` to 5 % (the loader enforces it); no mass, no source, idle > rated, unknown material, misnamed file: refused; every material named is in the library; the Phantom 4's ESC and controller resolve. DJI pack figures MEASURED, the rest ESTIMATED. | AI.11 | M | A |
| AI.13 | ✅ **done.** `configs/components/{brake_disc,tyre,tail_light,rear_window_heater,differential,ev_drive_motor,ev_pack}.yaml`; `Kind` += brake, tyre, light, heater, gearbox. | **Measured.** Each entry's dissipation is rated × (1 − η) to 5 % and its source names the figure; the pack's loss is (P/V)²R at 360 V and 0.1 Ω; the loader still refuses an entry without mass or source (`tests/unit/test_components.py`). | AI.12 | S | C |
| AI.14 | ✅ **done.** A part may assert its material (`PartSpec.material` → `thermal:material` on the part-split prims) and an asset may exclude geometry that is not it (`PartsConfig.exclude`). Five latent glTF faults in `prep_asset.py` fixed. ADR 0173. | **Measured.** The Mini 3 Pro's component area 40,701 → 0.113 m²; its Blender split 1 → 15 part prims; part-split audits show `override=4` and `=34`. 12 tests, one a scene resolving the asserted material. | AI.5, AI.11 | M | A |
| AI.15 | ✅ **done.** DJI Mini 3 Pro (15 parts), Inspire 3 (26) and Matrice 300 RTK (13, its case excluded) enter the library from CC-BY models; three polymers drafted. | **Measured.** Rotor diagonals 246 / 662–678 / 895 mm vs DJI's 247 / 685–695 / 895. Four propellers agree to ≤ 0.8 %, motor housings ≤ 0.6 %; render and archive splits within 0.5 %. Audits and coverage 100 %. | AI.14 | M | A |
| AI.16 | ✅ **done.** `PartsConfig.granularity: object` judges each source object whole (area-weighted centre, union bounds), so an asset modelled one object per piece is never split across parts; the Inspire 3 and Mini 3 Pro use it. ADR 0174. | **Measured.** Split objects 41/345 → 0 (Inspire) and 21/672 → 0 (Mini); each Inspire propeller is exactly its two 111.39 cm² blades, motors agree to 0.03 %; the Mini's '40 % larger rear motors' were arm leakage. 8 tests. | AI.15 | S | A |
| AI.17 | ✅ **done.** Inspire 3 arms → root + front + rear tube per side (26 parts), Matrice 300 RTK `arms`/`landing_gear` → 4 arms + 2 legs (17); 18 and 10 contacts along motor → tube → body; `scripts/check_asset_contacts.py`. | **Measured.** Every contact touches on 2 cm cells within 3 cm (nearest 0.0–0.2 cm); a far contact is refused. Mirror parts exact (Matrice arms 361.5/341.4, legs 483.7 cm²); 0 objects split; audits PASS. | AI.16, TC.17 | S | A |

---

## GT — Gates, tests and tooling

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| GT.1 | ✅ **done, except the number.** `slow` is applied — 23 validation/end-to-end modules plus 10 tests over a second — and `make test` / `make test-slow` split the suite while `make check` runs both, so nothing escapes the gate by being slow. | **Measured.** 245 of 2,979 marked; fast tier ≈ **65 s of test time** against ~180 s. **30 s is unreachable:** 154 s of the 170 s is in test *bodies* (setup is 15 s), so it would mean marking everything over 0.2 s. Budget is now **open question 11**. | — | M | A |
| GT.2 | ✅ **done.** 15 new arrays over the eight subjects named: MWIR, SWIR and NIR frames through the real configs and LUTs; the layered tau(band, distance, elevation) table; L_sky(theta); the sea vs depression; a half-in-sun field and the point-wise frame from it. 8 -> 23. | **Measured.** Every frame spans the converter without clipping (asserted), and the reflective bands carry real solar irradiance -- emission-only would golden the dark current. A test shows the tau table **would have caught `AT.10`**: SWIR 0.4148 -> 0.3877 at 5 km, which every old golden passed. | SC.1 | M | A |
| GT.3 | **Run `tests/integration` in an automated job.** 15 files, 4,480 lines, all auto-marked `isaac`, running nowhere; `warp_stages.py` at 38 % and six glue files at 0 % in the only gate that executes. | Split into a Warp-only subset needing a CUDA device but not Kit (ADR 0014's 2026-09-12 addendum measured `env.ensure_warp_on_path` from a bare `python.sh`) plus an Isaac subset run manually with its result recorded. Coverage rises above a stated floor and a regression fails a job. | GT.1 | L | X |
| GT.4 | **`scripts/` into `make typecheck`, and the aperture guard's scope extended to it.** 6,495 lines, linted but never type-checked, containing every lane entry point; 21 of 29 scripts have no test, including all six render drivers, `fidelity_ablation.py`, `eval_detector.py` and `train_detector.py`. | mypy runs clean over `scripts/`. The AST aperture guard walks `scripts/` too — today it covers `src/irsim` and `src/irsim_isaac` only, which is the one real hole in non-negotiable #5's coverage. | — | M | X |
| GT.5 | **Test `irsim_eval.decode`.** It is the entry point to the whole Tier 4 public-data lane and has no test at all; its own docstring names two decode facts (luma-plane-only, an unflagged colour range worth a 255/219 gain plus a 16-code offset) that bound every downstream number. Coverage 43 %, all incidental. | A synthetic clip encoded at a known range round-trips, and a range-flag regression fails. Needs ffmpeg and the `validation` extra, neither of which CI installs, so it follows the existing ffmpeg-gated pattern in `test_codec_floor.py`. | — | S | X |
| GT.6 | **Record Tier 3 manual passes with their commit hash** in `docs/validation/tier3-checklist.md`, and commit a small contact sheet per pass. `outputs/` is gitignored, so the owner's way of reading renders is invisible to everyone but the author. | A test parses the checklist: every pass row carries a date, a commit hash that resolves (`git cat-file -e`) and a contact-sheet path that exists; the five rows pointing at open steps (M10.11, M10.19, MM.8) are marked open. Red today: no pass is recorded and the parser finds no rows. | RP.6 | S | X |
| GT.7 | ✅ **done.** `tests/unit/test_cost_budget.py` (slow tier), and the duplicated pass removed: `PlanarPatch.contains_local` lets `sample` reuse the coordinates it already computed instead of `contains` recomputing them. | **Measured.** 102,400 cells over a 48 h spin-up in **11.5 s** (39 ns/cell/tick) against a 90 s budget — 10⁵ cells is affordable, so nobody should coarsen a patch. `apply` is **32.7 ms**/frame at 640×512. The duplicated `local_coords` was **5.3 ms of 23.5 ms**; that one is asserted by *counting calls*, not seconds, so it cannot flake. 5 cases. | GT.1, PT.9 | M | C |
| GT.8 | **`--lane` answers with a startable step.** `next_step.py --lane PT` prints `PT.9` although it waits on `WM.3`; the single-head and `--queue` outputs gain the `waiting on` column the published block already has. | Red today: `--lane PT` names a blocked head. After: the head printed for a lane is its first step whose deps are all ticked, or the line says what it waits on; `test_roadmap_queue.py` gains a `--lane` case. | — | S | X |
| GT.9 | ✅ **done.** `test_colour_does_not_set_emissivity.py`: the three sprayed topcoats share one ε in MWIR and LWIR to 1e-9, with a **control** that they must still differ ≥ 3× in NIR, so the test cannot pass on a library of identical materials. §4.5a. | **Measured.** The folk rule's modest 0.95/0.85 split would put two 320 K panels **5.26 K** apart under a 250 K sky — 105 × NETD, on paint alone — where the library gives 0. Colour's real channel is α_sol 0.94 vs 0.28, worth **24.3 K** of surface temperature at 800 W/m² and h = 15. 4 cases. | — | S | X |
| GT.10 | **Tests that would catch the wiring defects.** Goldens use ε = 1 with no atmosphere or sky; `test_run_frame` never passes an atmosphere, elevation, motion or point target; `test_lut` uses the LUT's own Simpson; `test_convection` compares a formula with itself; `test_material_table_angular.py:155` pins the limb clip. Add a golden with ε < 1, sky, layered atmosphere and slant elevation; `run_frame`-vs-stage equivalence; converged and literature oracles. | Each of AT.21, AT.24, AT.25, SC.28 and PT.25 has a test red at its parent commit and green after, in the fast tier. | — | M | X |
| GT.11 | ✅ **done.** `lut_files.builder_sha256()` (source of `band_integration`, `planck`, `constants` + LUT schema) written into every sidecar; `_read_sidecar` refuses a differing or unrecorded builder naming `make luts`; the multiband goldens hash the builder in. | **Measured.** A fresh bundle records today's hash and loads; the same sidecar with another hash, or none, raises `StaleLUTError` naming `make luts`; the hash is stable and covers `FINE_GRID_FRACTION`. | — | S | X |

---

## DC — Decisions, deferrals and probes

| id | what | verification (red today → green after) | deps | size | phase |
|---|---|---|---|---|---|
| DC.1 | **The SPG probe, split honestly into two.** (a) Write the host-upload experiment into `spg_probe.ALL_EXPERIMENTS` — it lists eleven experiments and the deciding one is not among them, so it does not exist as code. (b) Run it, time-boxed. | Also run the two cheapest temporal-state tests on **this** build: the `:-N` previous-frame `sourceName` suffix, a pure RenderVar rename that fails *silently* by reading the live AOV, so the test must be moving-versus-still; and a `cuda.stateful` output, since the 0.4.0 core binary already contains `allocatePersistentRpResource`. | — | M | X |
| DC.2 | **One ADR for the deferrals that currently have none.** §8.2 narcissus has a deferral row with a reason and a revisit trigger but no ADR, which CLAUDE.md requires for a deliberately skipped fidelity level; same for §13.8 performance, the §6.6 exhaust plume, the composed all-device Warp frame, the ground breadth, and the grey `Atmosphere` as L1-only. | The "Deferred to L3" table is headed "each with its ADR" and has no ADR column. After: it has one, and the plume's "record as ADR 0073" instruction — which would overwrite a live Accepted record — is gone. | RP.4 | M | X |
| DC.3 | **ADR: background photorealism and scene-library breadth are not sim-to-real goals**, labelled as *external* evidence: a published study measured a synthetic set's full-image Vendi diversity at 134.6 against 39.6 for the real set, and that 3.4× inflation drove the gap. | A record. It must say the figure is one external study on one non-irsim generator, name the irsim measurement that supersedes it (`EV.7`'s diversity on irsim frames against the real set), and read "demo scenes must look real" as being about the human viewing the render. | EV.7 | S | X |
| DC.4 | **ADR: allocate ADR numbers at write time, never forward.** The corpus has four numbers cited but never written, and revision 3 instructed a future author to write over ADR 0073, which is Accepted. | The parser from RP.4 enforces it: a roadmap or source citation of `ADR NNNN` must resolve to a file. The roadmap names subjects until a record exists. | RP.4 | S | X |
| DC.5 | **Verify the undocumented `add-thermal-emission` material modifier** in the local Isaac 6.1 build, and record the result either way. A forum post surfaced it in the docs with no reported adoption; NVIDIA's own position (Jan 2025) is that Omniverse has no thermal-IR solution and it is a tracked feature request. | A grep of the local MDL and schema tree settles it in under an hour, and a negative is worth recording so nobody chases it again. This is CLAUDE.md's "flag uncertainty rather than guessing" rule applied to the exact case it was written for. | — | S | X |
| DC.6 | **Post the ADR 0014 fp16 measurement back to IsaacSim discussion #298.** NVIDIA's answer to this project's own feature request recommends encoding temperature as OmniPBR emission and reading `PtSelfIllumination` — the route ADR 0014 measured at ~100 mK against a 10 mK bound. | A concrete, reproducible correction to public guidance, on the project's own thread. **No step is spent on the route itself**, because `WM` needs nothing from the renderer. | — | S | X |

---

## Non-negotiable enforcement map

CLAUDE.md's six rules, with what enforces each **today**, what this plan adds, and — where one exists —
the live exception that is not enforced. A step whose only contribution is *not violating* an existing
guard is not listed as an enforcer.

| # | rule | enforced today | this plan adds | live exception |
|---|---|---|---|---|
| 1 | Engine-free core | `test_layering.py`: 359 AST cases over `src/irsim`, `src/irsim_isaac`, `tests/unit`, `tests/golden`, both directions plus the ML/imaging ban, with a scanner self-test. The best-enforced rule in the repo. | `WM.2`/`WM.3` keep the Warp path split so the field stays in `irsim.thermal` and only the query lives in `irsim_isaac` — the cheap way to keep engine portability as a property (see priority 4) | none found |
| 2 | float32 or better wherever T or L flows | `test_temperature_encoding.py` round trip under 10 mK with an fp16 negative control; `GBuffer` dtype guard and `PRECISION_CRITICAL_KEYS`; LUT float32 in memory and on disk with fp16 files rejected; goldens refuse fp16 and float64; in-Isaac AOV dtype; EXR writers refuse fp16 | ✅ `IG.8` removed the fp16 carve-out: `_as_f64_plane` now refuses float16 on **every** plane. `IG.13` puts float32 planes on the three drivers that emit PNG only; `AT.1`'s elevation plane is float32 **and its value asserted**; `IG.2` is the in-engine end-to-end enforcer that replaces M10.13c | The Isaac adapter's carve-out is closed. Remaining: the three PNG-only drivers (`IG.13`) and the in-engine enforcer (`IG.2`) |
| 3 | Noise in radiance or electron space, never Kelvin | NETD(373)/NETD(300) = 0.576 within 5 % at the detector boundary; `test_residual_not_kelvin_flat`; the NUC residual's mK-to-DN conversion happens once at construction (ADR 0056) | `SC.1` wires the electron budget so the rule reaches photon FPAs at all; `SC.2`'s mutation test then discriminates noise-space correctness rather than wiring; `EV.3` stops the rendered-versus-real comparison being stated in DN8 alone | **True for bolometers only.** `core.py:176` calls `anchor_noise` unconditionally; MWIR renders at σ 533.3 e⁻ against 350 e⁻ datasheet, dark = 0 everywhere. **`SC.1`** |
| 4 | Kirchhoff closure to 1e-6, every material, every band | Structurally unbreakable: the schema refuses more than one authored member of {ε, ρ, τ}, ρ is always derived, and `directional_properties_for` re-derives it at **every angle**. Library walk over 19 materials × 4 bands | The closure is not the exposure; the **inputs** are. `RP.7` reopens S13 (glass τ authored before its k table existed, 0.02 against 0.134 recomputed); `XD.8` tests the values against measured spectra with a stated tolerance; `SE.1` records the sea's angular envelope | All 19 materials are `source: literature`; every NIR and SWIR value is ESTIMATED; the one spectral file is a 7-row sketch its own header calls ESTIMATED |
| 5 | Aperture factor π/(4F²+1) defined once | AST guard, 181 parametrised cases with obfuscated-variant self-tests, forbidding any expression squaring an f-number outside `irsim/optics/aperture.py` | `GT.4` extends the guard's walk to `scripts/` — 6,495 lines containing every lane entry point, currently unscanned. That is the one real hole in this rule's coverage | `scripts/` is not walked |
| 6 | One `WeatherSeries` for thermal and atmosphere | Not code review: an `is` identity assertion in three test files across five consumers (`test_scene.py:42-47`, `test_reflected_environment.py:223`, `test_thermal_scene.py:74`), plus constructors with no path parameter | `XD.7` adds a *physical* consistency check, with its limit stated: SURFRAD measures air T, RH, wind and downwelling longwave at one station and minute, but a pyrgeometer constrains the hemispheric broadband total, not radiance versus elevation, and clear-sky parameterisations sit at ~23 W/m² RMSE — so it **cannot** detect a second object holding identical values. The identity assertion stays the enforcer | none found |

Two enforcers named by revision 3 are removed from this map. **M10.13c** (the 300.000 versus 300.050 K
ΔDN test) was formally deferred by ADR 0061 because SPG 0.4.0 has no persistent device buffer, so the
enforcer cited for rule #2 can never ship; `IG.2` replaces it. **M10.7** is not a row — it was split into
M10.7a and M10.7b and the id dangled; rule #3 lost nothing. **M10.9b** is open and is therefore not
listed as enforcing anything.

---

## Risk register

Carried forward with the rows that need treatment. R1, R4–R10, R12, R14–R25 are unchanged from revision 3
and are not restated here; the ones below either changed state or were being reported wrongly.

| # | risk | state | retired by |
|---|---|---|---|
| R2 | `omni.rtx.spg` cannot hold cross-frame state or read a LUT file from Lua | **Not retired.** Revision 3 said "retired by M2.3, M10.12" and M10.12 is open — and is removed from this plan. Two untried tests exist on the shipped 0.4.0 (the `:-N` suffix; a `cuda.stateful` output) | `DC.1`, or the `WM` lane making the capability unnecessary |
| R3 | The AO AOV is not sky visibility | Bounded further. There is still no AO AOV on this build, but nothing depends on one for a *solved* surface: `PT.21` traces the sky view for every patch cell and `WM.4` for every mesh cell, both through the scene's own occluders. What is left unoccluded is the G-buffer's `V_s` for prims with no field behind them | `IG.7`(c) if the Replicator `occlusion` annotator delivers, for the prims that carry no field |
| R11 | The unit suite drifts past 30 s | **Materialised.** Measured 130 s, a 4.3× overrun. The `slow` marker the mitigation names was never implemented in the Makefile or pyproject | `GT.1` |
| R13 | Estimated data is mistaken for measurement | **Violated where it matters most.** The three example configs mark `ratios_3d` ESTIMATED; the two Boson configs that actually get rendered and compared against reality do not — and those are the values `SC.2` shows are 7–19× out | `SC.3` |
| R26 | The per-point ablation returns a negative | Live. The pre-committed answer is to redirect to the ISP, which the same evidence ranks first — but the two effect sizes are a near-tie (HTV d = 1.242 against target Sobel variance d = 1.224) measured on a different generator, so "ranks first" is not a ranking that survives. See open question 10 | `EV.9` |
| R27 | **New.** Point-wise temperature reaches one lane, and it is the lane ranked third | Live. Two prims in two of eight scene configs, authored only from Python, night only; `PT.1`, `PT.2` and `PT.5` shipped the machinery and no config reaches it | `PT.17`, `PT.18`, `PT.20`, `PT.9`, `PT.10` |
| R28 | **New.** A whole-file write silently reverts another session's work | **Materialised three times**: `68acd1c`, `303b56b`, and the 135-line CHANGELOG deletion live in the working tree now. The tested mitigation `scripts/stage_own_hunk.sh` exists and nothing reaches it | `RP.3` — not the row-length cap, which only makes the diffs reviewable |
| R29 | **New.** A "measured" figure in a project document cannot be reproduced | Live. `tier4-2026-09-15.json` records no clip list, archive hash, `config_hash`, CRF, seed or patch count; no noise statistic anywhere states its de-trending convention, which alone moves the answer by up to 3× | `EV.8`, `SC.5` |
| R30 | **New (revision 6).** Heat never moves between parts: a hot engine renders a warm bonnet and a cold bracket bolted to the block | Live. No inter-cell term, no node-to-node conductance, the engine a scripted ΔT (2026-09-18 audit). The visible symptom is a car whose bonnet glows and whose wings, arches and subframe stay at ambient, which a detector learns as a signature | `TC.1`–`TC.6` |
| R31 | **New (revision 6).** Fire, hot gas and wet surfaces are absent, so the scenes that contain them cannot be generated at all | Live. The gas-slab tables need HITEMP or RadCal offline (open question 14); the latent term needs no external data. A scene author who paints a flame as a hot prim gets a grey emitter that is wrong in every band | `PH.1`, `PH.4`, `PH.5`, `PH.8` |

---

## Deferred deliberately

Each of these is a fidelity level skipped on purpose, and CLAUDE.md requires an ADR for that. `DC.2`
writes the one that covers them; until it lands, this table is the record and it carries the reason and
the revisit trigger, which is more than the old "Deferred to L3" table did — it was headed "each with its
ADR" and had no ADR column, two rows describing work that had since shipped, and one instruction to write
over ADR 0073.

| item | why deferred | revisit when |
|---|---|---|
| **MCT (HgCdTe) detectors** via the Hansen–Schmit E_g(x, T) relation | Raised by an external review as "completely ignored". Narrower than that: `dark_current_a` takes `band_gap_ev` as an authored sensor field (InSb 0.23, InGaAs 0.75, Si 1.12), so an MCT camera is configurable today by authoring its gap; what is missing is only the gap *following* the Cd fraction and temperature. No MCT camera is on hand (ADR 0003), the reference cameras are Boson-class, and the order is aerial → maritime → ground | A scene needs an MCT camera whose Cd fraction or FPA temperature varies, so one authored gap stops describing it |
| **An unconditionally stable two-node solver for thin panels** | ADR 0036 sends thin panels to a single node with a resistive back rather than fixing the 0.235 s explicit bound. An external review proposed Backward Euler. Not free: ADR 0036 chose RK2 over Euler on *bias*, because the T⁴ term makes explicit Euler inflate the diurnal swing — the quantity §6.3's acceptance test measures — and Backward Euler damps the same quantity. The stiffness is all in conduction (1/R₁₂ = 37 500 W m⁻² K⁻¹ against h + 4εσT³ ≈ 43), so **IMEX** — implicit conduction, explicit radiation — is the shape that pays, and `PT.11` already needs that machinery | `PT.11` lands, or a scene needs a thin panel resolved in depth |
| §8.2 narcissus | §8.2 gives only a phenomenological form and no amplitude data | A Tier 4 flat-field PSD shows a radial low-frequency term |
| §13.8 performance | No implementation and no citation outside one open step's spec column | Dataset throughput binds — see the composed Warp frame below |
| §6.6 exhaust plume | ~~Deferred without a number~~ **No longer deferred.** Its trigger — an MWIR Tier 3 bench — fired with `test_tier3_multiband.py`, and the owner asked for fire and hot gas on 2026-09-18. `PH.4`–`PH.6` are the rows; what stays deferred (buoyancy, scattering, flicker, a volume on the Isaac side) is listed in `PH.13` | `PH.6` |
| The composed all-device Warp frame | ~2,200 lines of op-for-op stage twins exist and are held to the CPU oracle, `EQUIVALENCE_STAGES` registers four of them, and nothing composes them into a frame. `ir_camera.py:44-51` still blames M10.7b, which landed 2026-09-13, so the real blockers are recorded nowhere: composing the stages, the AGC/FFC schedule question, and where the host/device seam sits — M10.7a already showed two correct stages can disagree tenfold when the seam moves | Dataset throughput actually binds. State the measurement rather than the belief: `PointwiseTemperature.apply` is 79 ms per frame at 640×512 for one bound prim on the CPU path and the plan binds more prims, so `GT.7` is what decides this |
| Ground and automotive **material** breadth | Building-envelope and roofing materials and the street-canyon reflected environment are real gaps — low-e glazing alone is a ~40 K apparent-temperature error on every modern window — but they serve the lane ranked third. The occlusion input this row used to defer with them is **not** breadth: it is the owner's first requirement and is `PT.18`, `PT.21` and `PT.22` in phase P | Phase C, after the aerial and maritime lanes meet the exit bar |
| The grey `Atmosphere` as a live path | Two models coexist and `Scene.from_config` builds both; every render script uses `scene.layered` while `scene.atmosphere` is the primary attribute | `AT.5` marks it L1-only or guards it; a divergence, not a feature |
| §13.7 options 2, 3 and 4; `THM-16`, `THM-17` | Unchanged from revision 3, and each already carries a reason and a trigger there | Unchanged |
| Turbulence, polarisation, spectral fine structure, scattered-sunlight *path* radiance | App. A #2, #5, #6; §7.4. ADR 0086 shipped the scattered-sunlight **sky**; only the path radiance remains | Ranges beyond 500 m or airborne use |
| `E_star`, `E_artificial`, headlights | No illumination data; airglow and moon cover the night SWIR case | A night urban SWIR scene is needed |

---

## Not adopted, and why

Things a reader might reasonably expect to find here. None of these is in the repository today; this
section exists so nobody adds them.

**MRTD and MDTD (ASTM E1213, ASTM E1311, STANAG 4349).** Never present in this repository — a grep over
every `.md`, `.py` and `.yaml` outside `outputs/` returns zero hits — and they should stay absent. Both
are **observer-in-the-loop by definition**: a human looking through the actual imager at a four-bar
target through a calibrated collimator. The simulator can *predict* them with a TTP-style model; it can
never validate them, with or without a camera. Adding them to a validation ladder would imply a reachable
state that does not exist.

**§15's flat "apparent temperature within 2 K per class", as a pending target.** It should be restated in
`docs/physics-model.md`, not silently retired here — a roadmap can raise a spec issue and propose a
restatement, it cannot amend the document CLAUDE.md names as the source of truth. The reason to restate
it is stronger than "no radiometric capture exists": **no uncooled core is specified well enough to
adjudicate 2 K.** Boson is ±3 °C at 25 °C ambient and ±5 °C at 50 °C, and only under laboratory
conditions (steady state, FPA within 0.2 °C of the last FFC, blackbody emissivity > 98 %); Lepton is the
greater of ±5 °C or 5 %. For scale, DIRSIG — validated against its own instrumented 24-hour collection
with in-scene thermistors — reports roughly 1.8 °C RMS in actual temperature and 5–6 °C in apparent
temperature. So acquiring a camera would not rescue this target, and reporting it as pending implies
otherwise. Raise it as a spec issue against §15 and replace it with a tiered target: solver accuracy
against contact thermometry, and apparent-temperature accuracy through the full chain, each against a
cited precedent. *(Owner-visible: this touches the spec, so it is proposed, not done.)*

**"For when a camera arrives" as a framing anywhere in the repo.** It contradicts ADR 0003 and the
standing public-data-only constraint, and it keeps four measured comparisons skipping forever against
directories that were never created and a camera key (`flir_boson_640_lwir`) that does not match the only
camera anything was measured on (`halmstad_boson_320`). `SC.8` replaces it. **No step in this document
proposes a hardware purchase.**

**Multi Matte as a per-pixel attribute route.** Evaluated and rejected: the id is per mesh or per
material, so it is no finer than `instance_segmentation`, which the project already uses at exact uint32
full resolution. Recorded so nobody rediscovers it — along with the **"Primvar AOV"**, which is a
Houdini/Karma feature and not an Omniverse one. Omniverse's primvar path is MDL-side only and terminates
in fp16 colour or uint8 albedo, so it cannot carry a precise per-pixel parameter out of the renderer.
That is a plausible-looking dead end that costs a day.

**fp16 colour encoding of temperature, UVs or patch coordinates.** ADR 0014 measured ~100 mK at 300 K
against a 10 mK bound, and NVIDIA's own answer to this project's feature request (IsaacSim discussion
#298) recommends exactly that route. Keep the measurement, keep the refusal, post the correction back
(`DC.6`) — but spend **no step** on it. Note the honest form of the argument: fp16's 11 significant bits
are fatal for `(T-200)/800` and would be ample for a 16×8 cell index, so the obstacle for a *UV* is not
precision but the undocumented per-camera exposure scale (~2.8e-4) that puts the ramp in the subnormal
range. The drop stands for a different reason than the one revision 4 gave: `WM` needs nothing from the
renderer, so there is no reason to try.

**GAN image translation and Cosmos Transfer as a fidelity route.** Cosmos Transfer is an RGB appearance
model with no thermal prior; claiming it makes an IR frame look real would violate ADR 0068's rules on
what evaluation data may claim. What transfers is only its posture — structure-preserving augmentation
judged by downstream detector performance — and irsim's equivalent is ADR 0083's hashed
physical-parameter ablation switches. Be precise about the limit of that equivalence: **those switches
have only ever been run against image statistics, never against detector AP**, so the equivalence is
untested, and running it (`EV.9`, and the wider ablation) is itself the publishable result the literature
names as its own unfilled future work.

**The raw test count as a coverage figure.** 540 of 2,756 collected tests come from two per-file
structural scanners that scale with file count. Quote the coverage figures instead: `src/irsim` 93.9 %,
`src/irsim_eval` 85.6 %, `src/irsim_isaac` 43.5 %, with six glue files at 0 % totalling 1,101 statements.

**Revision 3's "Current state (2026-09-10)" section.** Deleted, with the reasons given above. It was the
second section of the document, so a session reading top-down formed a wrong model before reaching a
table.

---

## Recommended, but for the owner to settle — not edited unilaterally

**`IU-29` "engine-interface contract + Unreal port doc", the §14 Unreal mapping, and
`docs/maps/isaac-and-unreal.json`.** The standing guidance is that engine portability stays a *property* —
already CLAUDE.md non-negotiable #1, enforced by `test_layering.py`, and cheap to keep by holding the Warp
kernels free of `omni`/`pxr`/`isaacsim` imports — while no roadmap step, ADR or document is spent on
Unreal-specific deliverables. **CLAUDE.md line 13 ("A port to Unreal Engine follows later") is part of the
same question.** A parallel session may hold the other half of that guidance, so this needs the owner's
word. `IU-29` is marked open in the ledger and left in place until then, and `RP.8` deliberately does not
touch line 13.

---

## Open questions

Decisions only a person can make. Each names the step it blocks, a default answer, and what happens if no
answer arrives — so the plan cannot stall on silence.

| # | question | blocks | default if unanswered |
|---|---|---|---|
| 1 | **Email the Halmstad authors for the Y16 originals.** Action zero, and not engineering work. A positive answer would supersede `XD.10`'s role entirely and switch on every analyser ADR 0068 gates off, on the *aerial* lane, with the *exact* Boson core the configs model | `XD.12`; scopes `XD.10` | Send the email now; time-box the wait at two weeks; on expiry proceed with `XD.3`/`XD.4` and treat any later reply as a bonus. The plan must not depend on the answer |
| 2 | **Unreal.** Does CLAUDE.md line 13 stand, and does `IU-29` leave the plan? | Nothing technical; blocks knowing whether the §14 mapping is dead weight | Leave `IU-29` and line 13 untouched. Spend no step either way |
| 3 | **Detector framework, licence and GPU.** torch plus which small detector — a YOLO-family model (AGPL) or an Apache-licensed alternative? Where does it live, which machine trains it? | `EV.11` | Run `EV.12`'s training-free proxy first, which needs neither, and treat the detector as a confirmation |
| 4 | **FLIR ADAS v2's Terms of Use.** A form-gated click-through with no open terms stated | `XD.10` | Derive statistics on one machine; redistribute nothing; settle before the ingest, not after |
| 5 | **ECOSTRESS / ASTER redistribution terms.** Cite-and-extract is clearly fine; checking CSVs into the repo needs the terms read | `XD.8` | Cite and extract; keep the CSVs out of git, as `docs/maps/materials-surface.json` already flags |
| 6 | ~~**Which camera is "the" reference** — the Boson 640 the spec assumes, or the Halmstad set's Boson 320 the public data was recorded with?~~ **Answered by `SC.3` and `SC.2`: both, with different provenance.** The 640 carries [R24]'s Table 13 limits and says its ratios are ESTIMATED; the 320 carries ME.5's field-measured ratios with their caveats | — | Done. The guard is a test on the *numbers* rather than a schema validator, because the rule is about provenance and a validator cannot see where a number came from |
| 7 | **Publish the paired RAW-16 / AGC-8 dataset?** No public thermal set offers the pairing and the literature names the 16→8 mapping as the dominant sim-to-real factor, so it is a genuine contribution rather than another synthetic drone set. Under what licence? | The *publication* half of `EV.13` only | Build the format and the assertion regardless; hold publication. `EV.13` is split so the unblocked work does not wait on the blocked question |
| 8 | ~~Which tier does the cost-budget test belong to?~~ **Withdrawn as posed**, and now answered: `GT.1` built the tier (`make test` / `make test-slow`, `make check` runs both), so `GT.7` lands in the slow one | — | `GT.7` is marked `slow`; no decision needed |
| 9 | **Commit scopes.** `pipeline`, `validation`, `eval` and `io` are still proposed additions to CLAUDE.md's list | Nothing; a convention | Use `build` for infrastructure and the physics scope a step tests |
| 10 | **What happens if `EV.9` returns a negative** (R26)? | `EV.9`'s interpretation | Redirect to the ISP — but record the caveat *before* the measurement: the two published effect sizes are a near-tie (1.242 against 1.224) measured on a different generator, so the redirect is a decision, not a reading of the evidence. Confirm the intent so the result is not relitigated afterwards |
| 11 | **CLAUDE.md's 30-second unit-test budget is not reachable, and the number should move or be funded.** Measured over 2,979 tests: **154 s of the 170 s is in test bodies**, not in fixtures — setup is only 15 s, so the obvious optimisation (17 modules each building their own `BandLUT`) is worth ~15 s at most. `GT.1` marked the validation benches and everything over a second, which is the defensible line and leaves the fast tier at roughly **65 s of test time**. Reaching 30 s would mean marking every test over 0.2 s — 173 of them — which redefines `slow` to mean five times what it says | `GT.1`'s stated acceptance; nothing else | **Revise the budget to the measured fast-tier number** and record it as a budget the gate enforces, rather than marking 40 % of the suite to hit a figure written when the suite was a tenth of this size. The alternative — funding a real optimisation of the 154 s — is a lane, not a step |
| 13 | **CLAUDE.md has drifted from the tree and is not edited by this plan.** Its commands block calls `make test` the default gate (the gate is `make check`, which also runs the slow tier and the queue check), lists mypy on two packages (three), keeps the 30 s unit budget (open question 11), says Isaac Sim 6.0 (6.1.0-rc.26 measured, ADR 0014), and its layout block omits `irsim/io`, `irsim/pipeline`, `irsim/validation`, `scene.py` and `irsim_eval` | Nothing technical | Leave it; the owner edits CLAUDE.md. README.md is corrected in revision 6 where it repeated the same claims |
| 14 | **HITEMP and RADIS for the hot-gas tables.** HITEMP needs registration and a citation; RADIS is LGPL and stays under `scripts/`; the download is large. RadCal's band tables in the public-domain FDS tree are the fallback | `PH.5` | Try RADIS first, time-boxed to a day; on any block, generate from RadCal and record the swap in `PH.13`. Either way the committed tables are float32 with a hash sidecar and `src/irsim` imports neither |
| 12 | **The Boson's FFC fires on temperature, not only on time, and irsim models only time.** [R24] S5: an FFC is triggered by a 1.0 °C change in FPA temperature, and for the first 90 s after power-up by **one-third** of that. `NucSpec` has `ffc_interval_s` and nothing else, so a warming camera — the first two minutes of every render — shutters far less often in sim than in life, and the NUC residual it leaves is correspondingly larger. `SC.3` corrected the time trigger to 300 s and left this open rather than approximating it | A new `SC` step | Add `ffc_temp_delta_k` and `ffc_startup_period_s` to `NucSpec` and drive them off the existing `housing_tau_s` FPA track, which already exists and is not connected to the controller |

---

## ADR number allocation

The highest ADR is 0126 (0090–0126 were written after this section was first measured; 0123–0126 by `PT.9` (three) and `AT.11`; 0093–0122 by `PT.8`, `TC.1`, `PT.18`, `TC.2`, `TC.4`, `PH.13`, `TC.3`, `TC.5`, `PH.1`, `PT.11`, `PT.12`, `PT.21`, `TC.7`, `PT.15`, `PT.22`, `PH.3`, `PT.9`, `WM.5`, `PT.14`, `WM.6`, `AT.10`, `PH.6`, `PH.7`, `PH.8`, `SC.4`, `SE.1`, `AT.7`, `PH.10`, `PH.11` and `PH.12`). Four numbers below 0089 are cited and were never written: **0042** (the Level B
angular model, cited by `directional.py:21`, `angular.py:24` and four test files), **0079** (sea-water
optical constants and the Cox–Munk slope model, cited by `sea.py:36`, `nk.py:23` and ADR 0078's own
Consequences), and **0062** and **0069**, which are cited only by the roadmap itself as forward
allocations. `RP.4` writes the two that shipped code cites and replaces the two roadmap-only citations
with subjects.

**No number in this document is allocated forward.** In a three-session shared tree a forward-assigned
number is a scheduled collision, and revision 3 demonstrated both failure modes at once: it left four
numbers dangling and it instructed a future author to write over ADR 0073, which is the visible-companion
environment dome, Accepted 2026-09-14. `DC.4` records the rule and the RP.4 parser enforces it.

---

## Review notes

**Revision 6, 2026-09-20.** Amends revision 5 after the owner's 2026-09-18 message and the audit it prompted
(three repository audits and two web-research sweeps, run in parallel; the evidence is
[`docs/research/2026-09-18-thermal-coupling-survey.md`](research/2026-09-18-thermal-coupling-survey.md)).

*What the audit found.* The point-wise machinery revision 5 scheduled and shipped (`PT.1`, `PT.2`, `PT.5`)
is correct and reachable from no scene config: `_build_thermal_field` never reads `SurfaceSpec.patch`, the
schema has no occluder, `cell_shadow` has no caller, no YAML declares a patch, and the only bound fields are
hand-built in `car_demo.py` with no solar term. The roadmap had no row for heat between parts, none for
water beyond the sea, none for fire, and parked lateral conduction in phase C. It also disagreed with itself
about its revision number and step count, its phase table drifted from its rows in seven cells, its WM
gating was circular (phase A's exit waited on a phase-B step), the plume deferral's trigger had fired, and
ten verification cells stated motivation instead of a failure condition.

*What this revision does.* (1) Phase **P** — point-wise and coupled physics, CPU only — between repair and
the aerial lane, with its exit bar in the owner's words; `PHASE_RANK` in `scripts/next_step.py` and the
tiebreak prose follow. (2) Two lanes: `TC` (eight rows: the integrator, the network, contactors, the joint
table, the engine as a solved node with hot soak, the R2 reference scene, the exhaust line, wheel arches)
and `PH` (thirteen rows: the latent term and wet film, the wet/dry road, still water, the gas slab, offline
hot-gas tables, the exhaust plume, fire and what it heats, fire on the camera, steam, snow, vegetation,
people, and the ADR). (3) Six `PT` rows that make what shipped reachable from a config and add sky view,
geometry shadows with neighbours and the penumbra, and the R1 reference scene. (4) `PT.6`–`PT.8`, `PT.11`,
`PT.12`, `PT.14`, `PT.15` and `WM.1`–`WM.6` move to phase P; `PT.11` now depends on `TC.1`, `PT.15` on
`TC.2`, `WM.4` on `PT.22`. (5) The phase table is generated from the rows and guarded by
`tests/unit/test_roadmap_phase_table.py`, so it cannot drift again. (6) Nine spec issues, `S41`–`S49`,
record what §6, §2 and §16.2 lack for the owner's requirements. (7) `RP.10` and `GT.8` carry the two
roadmap defects not fixed here; the rest are fixed in place. Every tolerance in a new row is quoted from a
primary source in the survey and is external evidence until an irsim test reproduces it (R13).

*What this revision deliberately does not do.* It does not re-order the scene lanes: aerial, then maritime,
then ground remains the application order, and the reference scenes in phase P are the smallest scenes that
can show a requirement, not lane deliverables. It does not touch CLAUDE.md (open question 13). It does not
allocate an ADR number. And it does not claim any of the new rows' numbers as irsim measurements.

**Revision 5, 2026-09-15.** Replaces revision 3 (2026-09-10) in full, and supersedes an unpublished
revision 4 that was rejected in review for three reasons this revision fixes.

*What revision 4 got wrong, recorded so it is not reintroduced.* It asserted repository states that do
not hold: that MRTD and MDTD were being dropped from a validation ladder they were never in (zero hits
repo-wide); that §8.2 narcissus was "neither implemented nor properly deferred" when it has a
Deferred-to-L3 row with a reason and a revisit trigger, lacking only an ADR; that §13.8 has "zero
citations anywhere" when it is cited in M10.9b's spec column; and that §13.7 options 3–4 are
"unreferenced" when they hold their own deferral row. It cited ADR 0069 as an existing record and
forward-allocated 0095 and 0102 in a tree whose maximum is 0089. It claimed every number was measured "on
this tree at commit `ca5a663`" when the tree was dirty in seven paths, and two of its own headline figures
(2,748 tests, 298,272 characters) no longer reproduced. And it shipped **no step table at all**, so no
coverage claim about any step could be checked.

*What this revision does differently.* (1) The plan is published: 109 steps at publication (112 by 2026-09-18) across eleven lanes, each with
a verification cell that states what would fail and why the tolerance is that number, plus deps, size and
phase. (2) It is organised by the owner's application order, with a per-lane exit bar taken from the
owner's own words and amended only to require float32 output. (3) The three defects the audits ranked
critical are steps, not omissions: the per-pixel slant path (`AT.1`, 25 % τ and 48 % L_path error at
5 km/45°, on the first lane, where the point-target path already does it correctly so a target disagrees
with its own sky), the photon-FPA electron budget no render path reaches (`SC.1`, MWIR σ 1.52× wrong,
dark current zero everywhere), and the three Tier 4 methodology defects that make the acceptance run's own
attribution table unsound (`EV.1`–`EV.4`). (4) The headline requirement gets six steps that carry it out
of the lane ranked third: per-cell solar, a patch in the scene schema, a local-frame bridge, and bindings
on the aerial and maritime lanes. (5) `WM` is a lane with an error budget, an oracle and a superseding
ADR, because ADR 0087's "a real limit, not a temporary one" is false on this build and leaving it standing
will cost another session a week. (6) The repair lane lands first and fixes the mechanism, not the
symptom: `RP.3` makes `stage_own_hunk.sh` the default, which is what stops a whole-file write; the row cap
only makes the diffs reviewable.

*Sources.* Five subsystem audits run against the code and `git log` on 2026-09-15; a survey of production
IR simulators (DIRSIG, MuSES/RadTherm-IR, OKTAL SE-WORKBENCH, CAMEO-SIM, Fraunhofer IOSB, VIRSuite,
ThRend, AirSim); published camera datasheets (FLIR Boson Rev 340, Lepton Rev 400, Tau 2 Rev 141);
measurement standards (NVESD 3-D noise, NV-IPM, ISO 12233, VDI/VDE 5585, ASTM E1543); public dataset
primary sources; NVIDIA's own ovrtx and Isaac Sim 6.1 documentation and the installed Warp 1.16.0 and
`omni.rtx.spg` 0.4.0 in this build; and the 2026 sim-to-real literature on LWIR drone detection.

*A standing caution about this document's own numbers.* Everything marked *measured* carries the commit
and the tree state above. Anything quoted from outside the repository is labelled as external evidence,
not as an irsim measurement — that distinction is what R13 exists to protect, and revision 4 broke it by
attributing another group's correlation to an irsim ADR that did not exist.
