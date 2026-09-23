# ADR 0137 — An asset's mesh is budgeted twice: by what we can afford, and by what conduction can resolve

**Status:** Accepted
**Date:** 2026-09-24
Extends [ADR 0128](0128-per-asset-material-mapping-and-the-blender-prep-path.md) (the asset
pipeline) and [ADR 0132](0132-the-thermal-mesh-is-not-the-render-mesh.md) (the thermal mesh is not
the render mesh). Uses `GT.7`'s cost measurement. Roadmap: `AI.3`.

## Context

An imported asset arrives with the triangle count a *renderer* wanted. The Phantom 4 of ADR 0128
is 2,486,459 triangles out of a 62 MB FBX, and 1,532,656 after the planar dissolve that
`prep_asset.py --emit-mesh` already applies.

Nothing counted them. A mesh field's cost is linear in cells, so an asset that will take an hour to
solve looks exactly like one that will take a minute for the first thirty seconds — the first
import too heavy to solve is discovered by waiting for it, which is both the slowest way to learn
it and the one that wastes the most of somebody's afternoon.

Counting alone would not have been enough, and this is the part that is easy to miss. A face
budget answers *can we afford it*. It does not answer *is any of it doing anything*, and on real
imported geometry the second answer is usually no. One Phantom 4 prim carries 100,926 faces over
2.7 cm²: a mean cell edge of 73 µm. A test that only counted triangles would pass a million faces
on a postage stamp.

## The physics that sets the second budget

A cell holds one temperature. Two cells closer together than the distance heat diffuses laterally
in one solver tick cannot hold *different* temperatures — conduction erases the difference inside
the step. That distance is `sqrt(α Δt)`.

Walking `configs/materials/` gives the range this project actually spans, at a 60 s tick:

| material | α (m²/s) | `sqrt(α·60s)` |
|---|---|---|
| `etics_render` (the slowest) | 2.06 × 10⁻⁸ | **1.11 mm** |
| `human_skin` | 1.01 × 10⁻⁷ | 2.5 mm |
| `rusted_steel` | 1.23 × 10⁻⁵ | 27 mm |
| `bare_aluminium` (the fastest) | 8.44 × 10⁻⁵ | **71 mm** |

So 1.11 mm is a floor for *any* material in the library, and on the painted aluminium the Phantom
4 is actually made of, the panel is one temperature across 71 mm — its 0.377 mm cells are
over-resolved by a factor of nearly two hundred.

## Options considered

**A. A plain triangle cap.** Cheapest, and it was what the roadmap row asked for. Rejected as the
only measure: it passes the postage stamp, and it gives a person who hits the cap no information
about which triangles to remove.

**B. Derive a per-material cell size and decimate to it in the prep tool.** The physically ideal
answer, and too large a step: it needs the material bound before the mesh is emitted, a decimator
that respects a target edge length rather than a dissolve angle, and a way to keep the areas
within ADR 0128's own area-error gate. Left on the table; the report is what makes it worth doing.

**C. Two budgets, one gated and one reported.** Chosen.

## Decision

1. **`irsim.io.asset_budget` is engine-free and reads the `.npz`.** The archive already carries
   each prim's faces and its measured area, so the budget is about the mesh that will actually be
   solved rather than the one Blender imported, and it costs a file read.

2. **Affordability is gated.** `total_faces = 1,300,000` and `faces_per_prim = 200,000`.
   The first is Fraunhofer's reference scene, the figure `GT.7` sized this lane against
   (1,313,410 triangles, ten layers, five day-night cycles, 252 s). `prep_asset.py` returns
   non-zero when an asset exceeds either, and `--allow-over-budget` overrides it.

3. **Usefulness is reported, not gated.** A prim finer than `MIN_CELL_EDGE_M` is a waste, not an
   error, and the lever that fixes it — `--dissolve-deg` — belongs to the caller, who may have a
   reason (a render-side use of the same archive, a material not yet in the library). Refusing it
   would make the tool argue with someone who knows more than it does.

4. **The floor is derived, not chosen.** `MIN_CELL_EDGE_M = sqrt(SLOWEST_DIFFUSIVITY · 60 s)`,
   and a test walks the whole material library and fails if anything in it is slower. Adding an
   aerogel therefore breaks the test rather than silently lowering the resolution every asset is
   measured against.

5. **`too_fine` is defined as "carries wasted faces"**, not by comparing the edge to the floor a
   second time. `affordable_faces` rounds up, so the two definitions differ by one triangle at the
   boundary, and a report that says "nothing is wasted" and "this prim is too fine" in the same
   breath is a report nobody trusts the rest of.

## Consequences

* The Phantom 4 is **refused** by default: 1,532,656 faces against 1,300,000, with **77 %** of its
  faces finer than the floor. That is the correct answer and it is actionable — a larger
  `--dissolve-deg` is the lever — but it means `prep_asset.py --asset phantom4` now exits 1 unless
  `--allow-over-budget` is passed. The already-generated archive is untouched.
* The budget is quoted at a 60 s tick, the coarsest the drivers in `scripts/` use. A driver
  stepping at 1 s has a floor of 0.14 mm and could justify finer geometry; nothing reads the
  actual tick yet, so the report is generous rather than wrong.
* **The whole face budget buys about 0.8 m² at the floor.** So "finer than the floor" is not an
  invitation to refine anything: a vehicle or a building has to be solved considerably coarser
  than a millimetre, and the report is about geometry finer than even that.
* The areas come from the archive's manifest, which `load_asset_meshes` already cross-checks
  against the arrays, so a budget cannot be passed by an archive whose manifest lies about it.
* A per-material target edge length, and decimation to it, remains unbuilt (option B).
