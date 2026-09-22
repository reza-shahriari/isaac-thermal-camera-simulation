# ADR 0119 — The extrapolated emissivity fraction is reported, and decomposed

**Status:** Accepted. Completes [ADR 0043](0043-emissivity-from-optical-data.md), which decided that
ε comes from the material's optical data and that `total_hemispherical_emissivity` should return
how much of it rests on an extension of the nearest band — the returning was implemented and the
reading never was.
**Date:** 2026-09-22
Roadmap: AT.7 (§6.1)

## Context

`total_hemispherical_emissivity` integrates Planck's weight from 1 to 100 µm. The four nominal
band ranges — NIR 0.75–1.0, SWIR 0.9–1.7, MWIR 3.0–5.0, LWIR 7.5–13.5 µm — cover a minority of
that. Everything outside takes the nearest band's value, and the function returns the Planck
weight that came from doing so as `extrapolated_fraction`.

Every scene build called it and took `.value`. The fraction was computed and discarded at
`scene.py:831` and `scene.py:1217`. So **61 % of the weight that sets every surface temperature
in this project was an assumption, correctly measured and then dropped on the floor**, with no
scene author able to see it and no report carrying it.

## Decision

Report it, and **decompose it**, because the single number is not actionable.

`TotalHemispherical` gains `red_tail_fraction`, `interior_gap_fraction` and `blue_end_fraction`,
integrated from the same mask on the same grid so they sum to the total by construction.
`extrapolation_breakdown(temperature_k)` returns the same without a material. `Scene` gains
`material_names()` and `emissivity_extrapolation()`, and `EPSILON_EVAL_K = 300.0` replaces the two
inline copies of the evaluation temperature so the report and the build cannot disagree.
`scripts/validate_thermal_diurnal.py` prints it directly under the absolute temperatures it quotes.

### Two corrections to the roadmap row

**1. The fraction does not vary across materials.** AT.7 asks a scene to report "the worst
fraction across its materials". There is no worst: it is **bit-identical for all twenty shipped
materials** — 0.606695073 at 300 K for each — because it is a property of the band set and the
Planck weight and of nothing the material does. Their emissivities span 0.113 (bare aluminium) to
0.943 (cotton) and the fraction does not move. A report that ranked materials would invent an
ordering that does not exist and point a reader at the wrong fix.

What does move it is **temperature**, so that is what the scene report is indexed by.

**2. The total hides which assumption is being made, and it flips.** The three shares:

| T | total | red tail (>13.5 µm) | interior gaps | blue end |
|---|---|---|---|---|
| 250 K | 0.687 | 0.641 | 0.046 | 0.000 |
| 300 K | 0.607 | **0.508** | 0.099 | 0.000 |
| 400 K | 0.529 | 0.318 | 0.209 | 0.002 |
| 800 K | 0.465 | 0.071 | **0.394** | 0.000 |
| 1200 K | 0.509 | 0.025 | **0.483** | 0.000 |

At ambient it is four fifths **red tail**, which is the standard thermal-solver assumption and a
defensible one: real dielectrics do stay high and flat beyond 13.5 µm. By 800 K it is five sixths
**interior gaps** — the 1.7–3.0 µm and 5.0–7.5 µm holes — where "extend the nearest band" is a far
weaker claim, because a real spectrum is bounded on both sides and can do anything in between, and
reststrahlen features live exactly there.

This is not academic any more. `PH.6` shipped 594 K exhaust plumes and `PH.7` shipped 1200 K
flames, so the project now renders in the regime where the weaker assumption dominates.

**The fraction is also not monotone in temperature.** It falls to a minimum near 800 K and rises
again — 0.509 at 1200 K — because a 1200 K Planck peak lands at 2.4 µm, squarely in the SWIR–MWIR
gap. A reader who assumed "hotter is better covered" would stop checking exactly where it starts
getting worse. A flame is *less* well covered than a 500 K manifold.

### Sub-decisions

**The evaluation temperature stays a single fixed point.** ε_hemi is evaluated at 300 K for every
surface regardless of what it reaches. Over a 30 K swing ε moves by 0.1 % (0.8531 → 0.8523 for
painted aluminium), far below what the material data is worth, so re-evaluating per surface per
tick would buy nothing real. It is now `EPSILON_EVAL_K` rather than two inline literals, and the
report quotes it, so the number describes the ε actually in the solver rather than a hypothetical.

**Nothing is corrected.** This step makes the assumption visible; it does not narrow it. Closing
the 5.0–7.5 µm gap needs material spectra out to 7.5 µm, which the library does not have — every
NIR and SWIR value in it is already ESTIMATED (rule #4's own exposure note).

**A scene that names no library material says so.** `car_exhaust_plume.yaml` is one: its targets
are exhaust-line solvers with their own thermal model. The summary reports "binds nothing here"
rather than quoting a fraction "for each" of zero materials, which would read as though something
had been checked.

## Consequences

**Which reports carry it is now a registry, not a habit.** `CARRIES_THE_FRACTION` and
`EXEMPT_REPORTS` in `tests/unit/test_emissivity_extrapolation.py` classify every validation
script, each exemption in writing, and a new one fails the test until it decides. The same
discipline as `SC.4`'s `IDEAL_LENS`.

**The Tier 4 acceptance report is exempt on a premise that is now pinned.** AT.7 says "any Tier 4
report quoting an absolute apparent temperature carries it". `scripts/validation_report.py` quotes
none: it works entirely in DN8 because ADR 0068 found no honest conversion from "2 K of apparent
temperature" to "8 codes of histogram shift". A test asserts that `Tier4Targets` has no
kelvin-valued field and that the script mentions no temperature conversion, so if a check ever
starts reporting kelvin the exemption fails rather than silently going stale.

**`validate_sky_r13.py` is exempt for a different reason worth keeping straight.** It does quote
an absolute apparent temperature, but a sky one, produced by the atmosphere's column emission.
There is no surface and no ε_hemi, so there is nothing to extrapolate.

**What this does not fix.** The 61 % is unchanged; it is merely visible. Anyone comparing a
rendered absolute surface temperature against a measurement — `SE.3`'s ECOSTRESS check, any Tier 4
radiometric comparison — now has the number they need to judge whether a kelvin of disagreement is
a solver error or the coverage of the material library.
