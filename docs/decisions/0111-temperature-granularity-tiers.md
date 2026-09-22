# ADR 0111 — Temperature granularity tiers, and the rule for picking one

**Status:** Accepted. Replaces ADR 0087's prose about where the boundary between tiers lies;
ADR 0110 already superseded its "a real limit, not a temporary one".
**Date:** 2026-09-22
Roadmap: PT.14 (§6.1, §12.3, §13.1)

## Context

This repository has grown four ways to say how hot a surface is, none of them written down beside
the others. A scene author picking between them has ADR 0087's paragraph about near-planar
surfaces and nothing else, and CLAUDE.md requires an ADR for a deliberately chosen fidelity level.

The production comparison is [DIRSIG](https://dirsig.cis.rit.edu/docs/new/temp_solvers.html),
which offers eight temperature solvers at **four** granularities: per material (THERM, Balfour,
Null, Data-Driven — "all surfaces assigned this material would get assigned the corresponding
temperature"), per object or solid (ODB/GLIST), per facet (a GDB file, or geometry and
temperatures imported from [MuSES](https://dirsig.cis.rit.edu/docs/new/muses_plugin.html), which
solves internal and external temperature in full 3-D), and per pixel through a raster map driving
a surface's temperature. irsim has no per-material tier and no imported 3-D solution; what it has
is the middle of that range, solved rather than supplied.

## Decision

Four tiers. A surface picks exactly one; the render path samples whichever it picked.

| tier | what carries a temperature | authored by | sampled per pixel by |
|---|---|---|---|
| **T0 — per prim** | one facet per §12.3 surface | a `surfaces:` entry with no `patch:` or `mesh:` | the instance-id table (`aerial_bridge`) |
| **T1 — cells on a plane** | `PlanarPatch`, `n_u × n_v` on a rectangle | `patch:` (ADR 0087) | `point_bridge`, **bilinear** |
| **T2 — cells on a mesh** | `TriangleMeshPatch`, `k²` per face | `mesh:` (ADR 0110) | `mesh_bridge`, **piecewise constant** |
| **T3 — a lumped node** | `irsim.thermal.network` nodes in J/K | `nodes:` / `links:` (ADR 0096) | nothing; it drives a surface |

T3 is listed because it is easy to mistake for a tier and is not one: a node is a *mass*, not a
surface, and it reaches a picture only by being a boundary on some T0/T1/T2 surface. There is no
per-material tier and there will not be one — a material is shared by surfaces at different
orientations under different shade, and giving them one temperature is the defect this project
exists to remove.

### What each tier costs

Measured on the shipped scenes, Isaac's CPython, one core:

* **Per tick, cells are nearly free.** `wall_half_in_sun.yaml` steps 288 cells in 0.95 ms and
  2304 cells in 1.66 ms — 8× the cells for 1.75× the time. The fixed cost is the forcing (NOAA sun
  position, the weather sample, the longwave term), about 0.8 ms per field per tick; the marginal
  cost is ~0.4 µs per cell. A mesh is no dearer: `quad_flight_mesh.yaml`'s 336-cell arm costs
  1.82 ms against its 144-cell deck's 1.78 ms.
* **Cells cost at build.** Spin-up evaluates the forcing `spin_up_hours × 60` times, and a mesh is
  **always** spun up per cell (no two of its cells share a normal, so none of them shares a forcing
  history). `WM.4`'s traced sky view is 1740 rays per cell, once: 0.7 s for a 336-cell arm.
* **Cells cost memory only two ticks deep.** ADR 0093's `keep_ticks = 2` holds the two bracketing
  ticks and nothing else, which is what took a day of the road patch from ~240 MB to 166 KB.

So the cell count is not the thing to economise. What it buys, and what limits it, is physical.

### The rule: cells finer than the smoothing length resolve nothing

A thin sheet of conductivity `k` and thickness `δ` losing heat to its surroundings at an effective
`h = h_conv + 4 ε σ T³` cannot hold a temperature feature smaller than

    L = √(k δ / h)

and a sinusoidal gradient of wavelength `λ` survives at

    A / A₀ = 1 / (1 + (2πL/λ)²)

which is the steady fin equation, not a rule of thumb. `L` from the committed material library at
300 K with `h_conv` = 10 W m⁻² K⁻¹:

| material | k (W m⁻¹ K⁻¹) | δ (m) | **L** |
|---|---|---|---|
| vegetation leaf | 0.30 | 0.0003 | **2 mm** |
| painted composite | 0.25 | 0.0020 | **6 mm** |
| carbon fibre | 0.80 | 0.0015 | **9 mm** |
| glass windshield | 1.00 | 0.0050 | **18 mm** |
| dry asphalt | 0.75 | 0.0500 | **49 mm** |
| concrete | 1.40 | 0.1000 | **95 mm** |
| rusted steel | 45.0 | 0.0040 | **109 mm** |
| painted aircraft aluminium | 205 | 0.0016 | **145 mm** |

Read the last row: an aircraft skin panel **cannot** carry a fine thermal pattern, whatever the
forcing does, and a scene that renders one is wrong however many cells it used. Read the first:
a leaf can carry almost anything. The rule that follows:

1. **Choose T0** when the forcing does not vary across the surface — no partial shade, one
   orientation, no hot part behind it — or when the surface subtends few enough pixels that a
   field would be sampled once anyway. A prim is not a *small* thing; it is a thing that is all
   one temperature.
2. **Choose T1** when the forcing varies and the surface is near-planar: a road, a roof, a bonnet,
   a deck, a wall. ADR 0087's projection is exact for these and the sample is bilinear, so it
   staircases only at the patch edge.
3. **Choose T2** when the surface is curved or folds over itself, so no one normal describes it:
   a tube, a wheel, a mast, an exhaust. `WM.4` traces its shadow and its sky view; the sample is
   piecewise constant, so the cell size is the only control on staircasing.
4. **Set the cell size from `L` and the forcing**, never from the pixel count. A cell much finer
   than `L` resolves a feature conduction erases; a cell much coarser than the forcing's own
   feature (a shadow edge, a hot part under the panel) averages away the thing the field exists
   for. Where the two disagree — a shadow edge is sharp and `L` is coarse — `L` wins, because the
   surface wins.

The shipped scenes obey it: `wall_half_in_sun.yaml` puts 250 mm cells on concrete (`L` = 95 mm)
and 500 mm on the ground, both coarser than `L`, and the terminator it reports is 10.3 K.

## Consequences

**The arms in `quad_flight_mesh.yaml` are the exception, on purpose, and it is quantified.** The
tube is 30 mm across with 24 segments, so its circumferential cells are 3.9 mm against carbon
fibre's `L` = 8.8 mm — finer than the rule allows. That is not an authoring mistake: `PT.11`'s
lateral conduction is built on a rectangular grid's four-neighbour edges and does **not** run on a
mesh (`WM.6`), so the cells there conduct nothing and the resolution costs only time. But it means
the 22.1 K the scene reports between crown and underside is an **upper bound**. Solving the same
ring with conduction — 720 cells, `kδT'' − hT + q = 0`, `q` the clipped cosine of a sunlit crown —
gives a span **26 % smaller**, against the analytic `1/(1 + (2πL/λ)²)` = 0.744 for `λ` equal to the
circumference. So `WM.6` is worth roughly a quarter of the headline number on this geometry, and
the scene should be re-measured when it lands.

**A tier is a property of the surface, not of the camera.** Nothing selects a tier by range or by
pixel footprint, and nothing degrades T1 to T0 when a target gets small. That is deliberate: the
field is solved once per tick for the whole scene and sampled per pixel, so a distant target costs
what a near one does, and a target that crosses from 400 px to 17 px during a film (the vessel
departure) does not change its physics halfway through. The cost measurements above are what make
that affordable.

**No tier is chosen automatically.** A surface with no `patch:` or `mesh:` gets T0 silently, which
is the one place this ADR leaves a foot-gun: the difference between "this surface is uniform" and
"nobody got round to it" is not in the config. `PT.19` closes half of it from the other side — a
patch bound to a prim the stage does not know raises before the first frame, and every binding's
pixel count is recorded — but a prim with no patch at all is still just a prim.

**What would add a tier.** A per-facet tier fed by an imported 3-D solution (DIRSIG's MuSES route)
is the obvious next one and is not planned: it would mean carrying a foreign solver's transient
output and trusting its boundary conditions, where every term in this repository is authored
against `docs/physics-model.md`. A per-pixel *solved* tier is not a tier at all — the pixels of one
cell share a forcing history, so solving them separately is the same answer at N times the cost,
which is the cost argument ADR 0087 already made.

## Revisit when

`WM.6` gives a mesh lateral conduction, at which point rule 4 applies to T2 as it does to T1 and
the arm scene's cell count should come down; or a scene needs a surface whose `L` is genuinely
smaller than the geometry can express, which would be the first real argument for a finer tier.
