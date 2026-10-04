# 0176 — A coarse cell is never longer than its cell size: long faces are bisected first

Date: 2026-10-01

**Status:** Accepted (2026-10-01, TC.17). Amends [ADR 0165](0165-coarse-thermal-cells-cluster-an-over-tessellated-mesh-s-faces.md).

## Context

ADR 0165 coarsens an over-tessellated mesh by **clustering** its faces: each face joins the
cell of the voxel its centroid falls in. That assumes every face is smaller than a cell. The
Phantom 4 mostly obeyed it, and the thermal archives `prep_asset.py` writes do not have to. Its
planar dissolve merges every coplanar run of faces. On a tube that leaves each side facet as two
triangles running the tube's whole length.

The DJI Inspire 3 (AI.15) showed what that costs, when its arms were split into the tubes they
are made of (AI.17). Each front arm tube is 200 triangles about 16 cm long, so at 2 cm cells:

* **no gradient along the tube.** Every side cell sits at its middle (a strip's two triangles
  have centroids at ±L/6), so a tube heated at its motor end cannot show it. The edge-based
  conduction between those cells is also wrong: a full-length diagonal edge between two cells
  5 cm apart made the sheet operator ~460× too conductive on a test tube;
* **no contact at its ends.** `proximity_contactor` couples cells within a gap of each other.
  The nearest cells of a tube and the junction it plugs into were 4 cm apart, although the
  surfaces touch to 3 mm, so 4 of the Inspire's 18 contacts were refused.

Measured over the archives, faces with an edge over two cells carry 10 % of the Phantom 4's
area, 34 % of the Inspire 3's, 15 % of the Mini 3 Pro's and most of the Matrice 300 RTK's.

## Options

1. **Bisect long edges inside `coarsen_mesh`.** Refine until no edge exceeds `cell_m`, then
   cluster as before.
2. **Stop dissolving tubes in the archive.** Rejected: the dissolve is what keeps the archive
   small, and any other source of large faces (a CAD tessellation, a low-poly game model) would
   bring the same failure back.
3. **Measure contacts surface to surface instead of cell to cell.** Rejected as the fix: it
   repairs the contact and leaves the missing gradient and the wrong conduction.

## Decision

Option 1. `split_long_edges` bisects every edge longer than `max_edge_m` at its midpoint, one
new vertex per edge shared by both faces on it. It then re-triangulates each face by how many of
its edges were split (red-green: one split makes 2 triangles, two make 3 across the shorter
diagonal, three make 4), and repeats until no edge is long. `coarsen_mesh` calls it with
`max_edge_m = cell_m`.

The refinement is exact. Every new vertex lies on the face it split, so area, winding and the
surface the closest-point bridge queries are unchanged. It is also conforming: a closed mesh
stays closed, and every edge is still shared by two faces. That matters because coarse
conduction is built from shared edges.

## Consequences

* A dissolved tube now has cells along its length. The test tube's mid-plane flux under a unit
  gradient matches the sheet value `k δ · 2πr` to 10 %, where the whole-strip clustering was
  ~460× out.
* All 18 Inspire 3 contacts find cells within 3 cm, and the closest pairs are 0.0–0.2 cm apart.
* A coarse patch can hold more faces than its source mesh. The patch's `faces` are the refined
  ones. Nothing indexes a coarse patch by source face: the render bridge locates pixels by
  position, so a pixel finds the same surface.
* The Phantom 4's solved scene and its tests are unchanged in outcome (`test_phantom4_solved`
  passes as before). Its long faces are few and mostly flat panels.
