# ADR 0107 — Occlusion behind one adapter, and the sun as a disc

**Status:** Accepted
**Date:** 2026-09-21
Roadmap: PT.22 (§5.2, §6.1; spec issue S44)

## Context

PT.18 (ADR 0095) shadowed the direct beam with authored rectangles and PT.21 (ADR 0104) the sky
with the same ones. Two limits followed. A scene's occluders had to be *rectangles a human typed
into YAML*, which no real asset is; and a shadow edge was a **step**, because the ray test asks
whether one ray reaches one point. The sun is not a point: it subtends 0.5332° (Astronomical
Almanac 2024, §C; 31.46′–32.53′ over a year), so an edge standing `d` above a surface throws a
penumbra `d tan(0.53°)` — 9.3 mm per metre of standoff. At 25 cm cells and a 2 m standoff that
is sub-cell; on a panel gap, a louvre or a wall a hand's breadth from its neighbour it *is* the
edge, and a binary test puts a hard line where a real camera sees a gradient.

## Options considered

1. **More rectangle kinds** (discs, triangles, capsules), still binary. Each new primitive is a
   new intersection routine and none of them is the asset a scene actually has. Rejected.
2. **Convolve the binary shadow with the solar disc's angular kernel.** Correct only for a
   planar occluder parallel to the surface at one distance; a scene's occluders are at many
   distances and orientations at once, and the kernel width is per-cell.
3. **Put the occlusion test behind an `(origins, directions) → hit` adapter and sample the
   sun's disc with several rays.** Geometry becomes a triangle soup or a rectangle set or both,
   the query walks every opaque prim rather than the surface's own, and the visibility a cell
   reports becomes the *fraction* of the disc it can see. Chosen.

## Decision

`irsim.thermal.raycast`. The `Occluders` protocol is one method, `blocked(origins, directions,
max_distance_m)`. Three implementations: `RectangleOccluders` (the oracle — `cell_shadow`'s exact
ray-rectangle test generalised to per-ray origins *and* directions), `TriangleSoup` /
`MeshOccluders` (Möller–Trumbore in NumPy with an axis-aligned box reject per soup, chunked over
rays), and `AnyOccluders` (rectangles beside meshes behind one call). `box_mesh` and
`rectangle_mesh` are the mesh twins of `shadow.box_faces` and a `ShadowRectangle`.

`solar_disc_rays(direction, n_rays)` samples the disc on concentric rings at `r_k = R k / K`
with the **outermost ring exactly on the limb** — that is what makes the sampled penumbra as
wide as the geometric one instead of a fraction of it — giving 1, 7, 19 or 37 rays for
K = 0…3 and weights that are each ray's share of a uniform disc. `sunlit_fraction` turns those
into a per-cell number in [0, 1], with the fully lit and fully shaded ends **snapped** to exactly
1 and 0 so a cell nothing shades stays bit-identical to the per-prim solve (PT.17) at any ray
count. Scene schema v11 adds `thermal.penumbra_rays:`, default 1 — which is PT.18's hard edge to
the bit, so no existing scene moves by a millikelvin.

**No acceleration structure and no dependency.** The tens to low thousands of triangles a
scene's occluders amount to do not need a BVH, and `src/irsim` may not take a dependency
(CLAUDE.md #1). A BVH backend — trimesh with embreex, or Warp on the GPU — drops in behind the
same protocol when a scene needs one, and lives outside the core. `trimesh` is used in the tests
where it happens to be installed, as a third opinion, and the suite skips it where it is not.

## Consequences

**Measured:** the rectangle occluder reproduces `cell_shadow` exactly, and `sunlit_fraction` at
one ray reproduces it to the bit. A box as six rectangles and as twelve triangles gives the same
shadow cell for cell, at sun elevations from 10° to 85°, and the NumPy soup agrees with
trimesh's own intersector on random directions. A wall's own box never shades the wall, and a
lower block 1.5 m west of it shades its lower cells and not its upper ones. The penumbra ramp
comes out within 10 % of `d tan(0.53°)` at standoffs of 0.5, 2 and 6 m for 7, 19 and 37 rays
(18.0 mm against 18.6 mm expected at 2 m, the difference being the 1 mm cell it is sampled on),
is monotone across the terminator, and a binary test gives a ramp of exactly zero width and
fails. On the R1 wall scene `penumbra_rays: 19` puts a thin band of partly lit cells along the
terminator and leaves under a fifth of the wall's cells between 0 and 1.

**What it costs:** `n_rays` occluder passes per forcing evaluation instead of one, on every tick
of the spin-up as well as the run. That is why the default is 1 and the field is opt-in per
scene rather than a global change.

**Not modelled:** limb darkening (the disc is uniform, which changes the ramp's shoulders and not
its width), the 1.7 % annual swing in the sun's apparent diameter (well inside the 10 % the test
allows), and light bouncing off the occluder into the shade — ADR 0088's single-bounce note still
stands, and the shaded side remains a lower bound. **Meshes are not yet authorable from a scene
config:** the adapter takes them, and where the triangles come from is the geometry source's
question (`WM.4`, `IG.2`), so a YAML scene still shades with rectangles.

## Revisit when

A scene needs meshes from an asset rather than from Python, a moving occluder (`PT.9`: the
query is stateless, the caller supplies the origins), or enough triangles that the brute-force
soup stops being fast enough — the first BVH belongs behind `Occluders`, outside `src/irsim`.
