# 0165 — Coarse thermal cells cluster an over-tessellated mesh's faces; nothing is decimated

Date: 2026-09-29

**Status:** Accepted (2026-09-29, TC.13).

## Context

`TriangleMeshPatch` (ADR 0110) puts at least one thermal cell on every face, and `cell_m` can
only refine a face, never merge two. That is right for a mesh sized to the physics -- a 16-facet
tube, a solver-made box -- and wrong for a product-visualisation asset: the Phantom 4's shells
carry 180-220 k faces each and its motor bells 160 k, 1.5 million faces for an airframe a few
thousand cells describe. ADR 0143 lumped those parts at `T_air + offset` for exactly this
reason and named the mesh solve of the shell as "a cell budget away". It was not: with one cell
per face there is no budget to set, and TC.11's full solve of the object (`solve: full`) could
not be applied to the aircraft it was written for (TC.13).

Two ways were available to make the geometry solvable. Decimate the mesh for the thermal
archive, or coarsen the *cells* and keep the mesh.

## Options

1. **Decimate the thermal archive** (collapse decimation in `prep_asset.py`). Rejected, and
   already rejected once: surface area is a physical quantity here (it sets convection and
   radiation), and the prep tool measured a 0.05-ratio collapse removing 37 % of the area of a
   shell's small features. A decimation that quietly shrinks a surface quietly cools it. The
   closest-point bridge would also then locate pixels on a mesh that is not the one rendered.
2. **Lump the heavy parts** as single nodes with an area. Rejected: it is ADR 0143 again --
   a shell with one temperature cannot show the pack warming it from inside, which is the
   acceptance of TC.13.
3. **Cluster faces into cells** (chosen). A face joins the cell of the voxel its centroid falls
   in, split by the octant of its normal so the inner and outer skins of a thin shell -- which
   share voxels -- never share a cell. A cell's area is the exact sum of its faces' areas, its
   centroid and normal the area-weighted means; a pixel's face, found by the same closest-point
   query, resolves straight to its cluster. `CoarseMeshPatch` presents the surface the rest of
   the code reads (`n_cells`, `cell_area_m2`, `cell_normal`, `cell_centres`, `cell_of`,
   `sample`, `face_mean`, ...), so the field, the forcing, the sky view, the contactors, the
   footprint links and the bridge take it unchanged. Conduction between clusters is a
   finite-volume link across every mesh edge two clusters share, `k δ · L / d` with `d` the
   centroid distance (floored at half a cell). One temperature per cluster is the approximation:
   a gradient finer than `cell_m` is not represented, and the remedy is a smaller `cell_m`.

## Consequences

* The Phantom 4's nineteen parts coarsen to 5,100 cells at 2 cm (3,000 at 3 cm) in under five
  seconds, every square millimetre of area kept, and the object can be solved whole.
* Measured on a 7,680-face tube at 2 cm (80 cells): the heat crossing a plane under a unit
  gradient is `k δ · 2πr` to 10 %; the coarse field tracks the fine field's cluster means to
  2 K through a 20 K step's transient and conserves energy to float32; a scene solving one arm
  coarse beside the same arm fine agrees on the mean to 1.5 K and on the crown-over-underside
  sign at a high sun (`tests/unit/test_mesh_coarse.py`).
* `mesh: {coarse: true, cell_m: ...}` on a surface and `coarse: true` on a `thermal.objects:`
  entry opt in; every existing scene is bit-identical.
* Not supported on a coarse patch: the per-face object exchange of ADR 0090, which needs one
  face per cell; it says so rather than guessing.
