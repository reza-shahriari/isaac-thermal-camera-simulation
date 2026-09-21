# ADR 0110 — The surface temperature field lives on the mesh

**Status:** Accepted. Supersedes ADR 0087's curved-geometry limitation; ADR 0087's planar patch
remains the shipped path for near-planar surfaces and is not superseded as a whole.
**Date:** 2026-09-21
Roadmap: WM.1–WM.3, WM.5 (§6.1, §6.4, §13.1)

## Context

[ADR 0087](0087-point-wise-surface-temperature.md) gave a surface a temperature *field* by
projecting the per-pixel world position onto a rectangular patch. It was the right call and it is
still what ships for a road, a bonnet, a roof or a deck. It also wrote off everything else:

> **Hard:** curved and small geometry. A wheel, a tyre, an exhaust pipe or a mast is not
> near-planar and still takes one temperature per prim. This is a real limit, not a temporary one:
> raising it means option 1 or 2, and therefore means a renderer capability this build does not
> have.

Options 1 and 2 were a **UV atlas with temperature as a texture** and a **per-triangle id in a
custom AOV**. Both ask the *renderer* to transport a surface parameterisation, and ADR 0014
measured that this build transports exactly two things usefully: an exact integer instance id, and
a float32 world position good to 3.4 mm. No UV AOV, no per-triangle id, every colour AOV fp16.

The premise that made the limit permanent is that the parameterisation must be *transported*. It
does not. It can be **derived**: the instance id says which prim a pixel hit, the position says
where in the world it hit, and a closest-point query against that prim's own mesh returns the
triangle and the barycentric coordinates. `WM.1` measured whether that works on this build, and it
does — see the addendum on ADR 0087 for the numbers.

This ADR records the parameterisation that resulted, its error budget, and the routes rejected.

## Options considered

1. **A UV atlas as the solver domain.** The standard offline-renderer answer. Rejected as the
   *solver* domain — not merely as unavailable. An atlas carries **metric distortion** (equal texel
   areas are unequal surface areas, so a uniform grid is a non-uniform set of facets and §6.1's
   areal heat capacity stops meaning what it says), **seam severing** (a lateral conduction
   operator has to reconnect what the unwrap cut, and a seam is where a thermal gradient looks
   like a crack), and a **conservative-rasterisation tax** (texels straddling a chart boundary need
   gutter padding or they interpolate against background). It also needs an unwrap authored and
   kept in sync for every asset. Still the right answer for a *texture*; the wrong one for a field
   that has to conserve energy.
2. **A per-triangle id in a custom AOV.** Exact, and blocked: it needs an SPG shader (`M10.12`,
   itself blocked) or an integer AOV nobody has found on this build.
3. **Closest-point-method narrow bands.** Solve the surface PDE on a narrow band of a background
   3-D grid and extend off-surface by closest-point extension. Elegant and well-conditioned, and
   wrong here on resolution: the band must be finer than the *thinnest* feature, and a car panel is
   ~1 mm thick, so a vehicle-sized domain would need a grid nobody wants to carry for a quantity
   that lives on a 2-D surface.
4. **Transient surfels.** Scatter oriented disc samples and solve on them. Attractive for a
   renderer, unusable here: `PT.7` spins a field up over 48 hours and the answer depends on that
   memory, so the sample set has to be *stable* across frames. Surfels regenerated per frame cannot
   hold a thermal history, and stabilising them is the same problem as parameterising the mesh with
   extra steps.
5. **Cells per face on the mesh itself, Ptex style, located by a closest-point query.** No atlas,
   no unwrap, no seam, no renderer capability, and the parameterisation is the mesh's own.

## Decision

Option 5.

`irsim.thermal.mesh_field.TriangleMeshPatch` cuts every face into `k²` congruent sub-triangles on
the barycentric grid, at a **per-face** level so resolution follows the thermal gradient rather
than the tessellation. `TriangleMeshField` composes `ThermalField` exactly as `PlanarThermalField`
does, so the fixed tick, the between-tick interpolation and the never-mutate-on-query rule are
reused rather than restated. The cells are ordinary §6.1 facets; the balance is unchanged.

`irsim_isaac.pipeline.mesh_bridge.MeshPointBridge` is the render-path half, and is **additive** in
the same way the planar bridge is: it overwrites only the pixels of prims with a mesh field bound
to them, so an unbound prim is bit-identical and attaching it to an existing scene is a no-op.

**Warp is an accelerator, not the authority.** `irsim.thermal.closest_point_on_mesh` is a
brute-force NumPy test over every triangle. It is the oracle the Warp path is held to *and* the
fallback when Warp is absent, so the bridge produces the same frame on a machine with no GPU, and
the device defaults to Warp's CPU. This is the shape the project's CPU-before-GPU ordering asks for
where a library is needed for a capability rather than for speed.

## Consequences

**Easy.** Any prim with a mesh gets a real temperature field for the cost of authoring a level, with
no atlas, no asset change and no renderer change. A wheel, a tyre, an exhaust pipe or a mast stops
being one number: measured, a pipe goes from 0.000 K across the prim to 34.1 K around its
circumference. `PlanarPatch` is untouched and remains the cheaper, smoother answer for the surfaces
it already covers.

**The error budget.**

* The parameterisation itself costs **0.13 µm** — `mesh_eval_position(face, u, v)` against a point
  already on the surface, four orders inside ADR 0014's 3.4 mm position budget. It is float32
  round-off and **grows with distance from the world origin** (0.73 µm at 4.3 m), so a scene tens of
  kilometres across needs its own check; at vehicle and building scale it is nowhere near the
  budget.
* Under a full 3.4 mm of position error the recovered **face** is unstable — it flips between
  neighbours on 1.9 % of queries on a 12-triangle box and 56 % on a 16 k-triangle sphere — while the
  sampled **position** moves only 2.68 mm on average and never past 3.41 mm. A temperature lookup
  needs the position, so the instability is harmless; anything keyed on the face id is not safe.
* A pixel whose closest point is further than **7 mm** away is refused rather than snapped. That is
  twice the position budget, because a mesh is itself a chord approximation of whatever it
  represents.
* The sample is **piecewise constant within a cell**, where the planar patch is bilinear, so cell
  size is the only control on staircasing.

**Hard, and deliberately deferred.**

* **Smoothing across faces** needs edge adjacency — which cell of the neighbouring face sits across
  this edge, in which orientation — and is `WM.4`'s, with the ray-traced per-cell sky view.
* **Lateral conduction between cells.** `PT.11`'s operator is built on a rectangular grid's
  four-neighbour edges. The mesh equivalent is a cotan Laplacian, and `WM.6` exists because a naive
  one gives negative edge weights wherever two opposite angles sum past π, breaking the discrete
  maximum principle: a cell escapes the range spanned by its neighbours and renders as a bright
  speck indistinguishable from a bad pixel.
* **Normals are per face, not per vertex.** A smooth normal is a better *shading* normal and a worse
  *facet* normal, and §6.1 balances a facet.
* **No scene config can declare a mesh field.** Every shipped scene still renders its curved prims
  at one temperature. Until a scene authors one, this ADR describes a capability the scenes do not
  yet use, and `README.md`'s limitations say so.

**What this does not change.** `PlanarPatch` and `PointwiseTemperature` are unmodified and every
existing scene renders bit-identically. ADR 0087's slab rule, its cost argument (cells are set by
the thermal gradient, not the pixel count) and its projection-error analysis all still stand for
the planar path.

## Revisit when

A scene config can author a mesh field and one does, which is what turns this from a capability into
a picture; or `WM.4` lands and the per-face normal and piecewise-constant sample are replaced by the
ray-traced, adjacency-aware forms, at which point the error budget above needs re-measuring.
