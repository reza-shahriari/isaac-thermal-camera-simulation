# ADR 0128 — Per-asset material mapping, and Blender as the CPU-side asset toolchain

**Status:** Accepted
**Date:** 2026-09-23
Extends [ADR 0047](0047-material-mapping-defaults-and-coverage.md) (its "Revisit when" clause
named this file). Evidence: `docs/research/2026-09-23-asset-ingestion-survey.md`.

## Context

Until now every piece of geometry in this project was generated in Python — `phantom3_parts()`
authors eighteen boxes and cylinders and writes `thermal:material` on each as it goes. There was
no import path for a third-party model, and the one written acknowledgement of that
(`src/irsim/config/scene.py:410-415`, "the follow-on to this row") named no owner.

A real asset was then measured: a 62 MB DJI Phantom 4 Pro FBX, 41 meshes, 21 materials,
2,486,459 triangles. Three facts about it forced this decision.

**1. The global glob table maps it to 48.8 %, and two of the hits are wrong.** Measured with
`scripts/audit_materials.py`: 20 of 41 prims mapped, below ADR 0047's 95 % gate. Worse than the
misses are two confident hits:

- `*white*` → `car_paint_white` is paint on steel, 4399 J m⁻² K⁻¹. The shell is moulded ABS,
  2205. Both are white dielectrics at ε ≈ 0.9, so no emissivity check catches the swap — what
  changes is the half that sets the temperature, and the global rule makes the skin **twice as
  sluggish** as it is.
- `*metal*` → `bare_aluminium` puts **ε = 0.09** on the motor housings. At that emissivity the
  part is a mirror: 91 % of what the camera sees is reflected sky rather than the motor's own
  temperature. `irsim_isaac/phantom3.py:249` already records this exact trap for the same
  component on the previous airframe.

A global pattern is a statement about *every* asset. Fixing either of these globally would change
every car and building scene to suit one drone.

**2. The asset is grouped by material, not by function.** 31,068 disconnected shells across 41
prims; one prim is "all the white plastic", scattered over 987 islands. There is no "top shell"
prim and no "leg" prim. Splitting by loose parts — the obvious automatic fix — yields 31,068
objects and is not a route.

**3. Nothing in the file is in metres.** 60.21 source units across against DJI's published 589 mm
tip-to-tip: centimetres, declared nowhere. Left unapplied the aircraft enters a stage 60 m wide
and still renders a perfectly plausible thermal image.

Separately, the survey established that Blender ships a complete `pxr` — OpenUSD 26.03 on the
5.2 LTS build, verified here — so USD can be read and written on the CPU without Kit. That matters
under the standing constraint that Isaac Sim and CUDA workloads are not started on this
workstation, and because the inspect → map → audit loop is run dozens of times per asset and a
15–35 s Kit boot per iteration (ADR 0014) made it a loop nobody would run.

## Options considered

1. **Widen the global globs until the asset passes.** Cheapest, and it makes every other scene
   worse. `*copper*` → something-plausible is a statement about copper everywhere.
2. **Author `thermal:material` on all 41 prims in the DCC.** Works, and it is the per-object
   manual cost ADR 0047 exists to avoid — plus it lives in a `.blend` nobody can diff, and a
   newer version of the asset silently loses it.
3. **A per-asset mapping file layered above the global rules** (chosen). ADR 0047's own
   "Revisit when" clause predicted it: *"then a per-asset mapping file"*.
4. **Split the asset by material into one prim per material.** Still available and orthogonal; it
   fixes the multi-material-mesh case, not the naming case, and this asset does not need it.

## Decision

**Precedence gains one rung, directly below the prim override:**

```
thermal:material override  →  asset material map  →  semantic class  →  global glob  →  loud miss
```

- The asset map sits **above** the semantic rung because it is authored after looking at this
  specific asset, which is a stronger claim than a generic class label.
- It matches the source material name **exactly and case-insensitively, never as a glob**. An
  asset map is authored knowledge; a partial hit there is a mistake, so `Copper_2` must miss
  rather than resolve as `Copper`.
- A miss stays loud. Adding this rung does not add a default.
- `scale_to_metres` lives in the same file. It is metadata the prep tool applies, not something
  the resolver touches — it is recorded here because it is a fact about the asset that is
  otherwise known only to whoever ran the importer once.
- Files live in `configs/assets/<name>.yaml`, schema version 1, and every target is validated
  against the material library at load, exactly as the global rules are.

**Blender is the CPU-side asset toolchain.** `scripts/prep_asset.py` imports the source, applies
the scale, exports USD with `UsdPreviewSurface`, walks the stage into engine-free prim records and
audits them. It runs twice — as a driver under the project interpreter (which has pydantic and
PyYAML) and as a worker inside Blender (which has `bpy` and `pxr`, and neither of the other two).
Nothing in it boots Kit or touches CUDA.

**The prep tool reads `materialBind` subsets, never the mesh-level binding when subsets exist.**
Blender binds the first material slot to the mesh *as well as* writing subsets — a documented
workaround for Hydra not supporting materials on subsets. A reader that took the mesh binding
would map a whole multi-material building to slot 0, silently, and report 100 % coverage. That is
precisely the failure ADR 0047 exists to prevent, arriving through a door ADR 0047 did not cover.
The tool reports any mesh where both are present.

## Consequences

- The Phantom 4 goes **48.8 % → 100 %** coverage, and the two wrong hits become right ones: the
  shell's areal heat capacity halves to the moulding's real value, and the motor housings go from
  ε 0.09 to ε 0.90.
- Adding an asset is: run `prep_asset.py`, read the misses, write a YAML, re-run. No code.
- The per-asset file is the natural home for future asset truth (thickness overrides, interior
  masks, heat-source anchors). None of that is in it yet.
- **Four mappings in `configs/assets/phantom4.yaml` are `ESTIMATED` and flagged in place.** The
  consequential one is `Copper` → `aircraft_aluminium_painted`: those prims sit at the motor
  stations and are the hottest surfaces on the aircraft. Bare copper is ε ≈ 0.03; enamelled
  winding under varnish is ≈ 0.85–0.90, and the coated value is used. If a motor ever needs to be
  right rather than plausible, that is the line to replace with a measurement.
- Generated artefacts (`data/assets/`) are gitignored — the exported `.usdc` is ~400 MB. The
  mapping and a 41-record prim dump fixture are in git; the source FBX is not.
- **This does not make the asset renderable.** It supplies material identity and geometry. Per-cell
  temperature still comes from the mesh field (ADR 0110), which is what makes the material-grouped
  prim structure acceptable: the field solves per cell from each cell's own normal, sky view factor
  and solar incidence, so a prim spanning 987 scattered shells is not one temperature. Heat
  sources, thickness overrides and interior/exterior classification remain unauthored.

## Revisit when

An asset arrives whose meshes carry real `materialBind` subsets (the prep tool handles them; no
committed asset exercises that path yet), or when per-asset files start carrying thermal rather
than optical truth — at which point the schema needs a section, not more top-level keys.
