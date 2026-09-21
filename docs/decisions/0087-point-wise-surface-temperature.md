# 0087 — Point-wise surface temperature, parameterised by the position AOV

Date: 2026-09-15
**Status:** Accepted
Roadmap: MP.1 (§6.1, §6.4, §13.1)

## Context

Every rendered frame this project has produced carries **one temperature per prim**. The bridge
that feeds the G-buffer is a `dict[str, float]` keyed by prim path
(`aerial_bridge.AerialThermalBridge.facet_temperatures`), gathered into the `temperature_k` plane
by instance id. `ThermalField` has always been an N-facet solver, but every consumer has mapped one
facet to one prim, so the N has been the number of *objects*, not the number of *points*.

In LWIR the temperature field is the image, and this is the dominant modelling error in the
pipeline for anything larger than a few pixels:

| situation | reality | what irsim rendered |
|---|---|---|
| building wall, half in sun | 10–20 K across one wall | one value |
| bonnet over a running engine | ~15 K from over the block to the wing | one value |
| asphalt beside a warm car | 1–3 K patch, falling with distance | one value for the whole road |

The user who commissioned this work named it as the defect that made a previous simulator unusable:
"whole object is colored as same … if i have a big building, every pixel of that has a single
color, no matter if some of it is under the sun and some other is under shadow."

The constraint that shapes the fix is what the renderer can actually transport per pixel. ADR 0014
measured this build: an exact integer **instance id**, and `Camera3dPositionSD` / `PtWorldPos` as a
**float32 world position good to 3.4 mm**. There is no UV AOV, no per-triangle id, and every colour
AOV is fp16 — which is why temperature does not travel as a colour in the first place.

## Options considered

1. **A UV atlas per prim, temperature as a texture.** The standard offline-renderer answer, and the
   most general. Needs a UV AOV this build does not expose, an unwrap for every asset, and a
   texel-to-cell convention kept in sync with the mesh. Blocked on a renderer capability, so it
   would be a step that cannot be finished.
2. **A per-triangle facet id in a custom AOV.** Exact for arbitrary geometry. Needs either an SPG
   shader (M10.12, itself blocked) or an integer AOV nobody has found on this build.
3. **A world-space voxel grid.** Needs nothing new, but is wrong for surfaces: a voxel containing
   both faces of a panel, or both the road and the car sitting on it, has one temperature, which
   reintroduces the defect at a different scale.
4. **Project the per-pixel world position onto planar patches.** Needs nothing new, is exact for
   near-planar surfaces, and degrades to today's per-prim behaviour for everything else.

## Decision

Option 4. `irsim.thermal.surface_field.PlanarPatch` is a rectangular grid of cells on a plane; its
cells are ordinary §6.1 facets solved by the existing `FacetSolver`, and `PlanarThermalField`
composes `ThermalField` so the fixed tick, the between-tick interpolation and the
never-mutate-on-query rule are reused rather than restated. The spatial part is **only the lookup**:
a point in space becomes a bilinear blend of cells. The balance is unchanged.

Per-cell spatial variation enters through forcing that was already per-facet —
`FacetForcing.q_internal_w_m2`, `q_solar_w_m2`, `h_w_m2_k`. No solver change was needed; the knobs
existed and nothing had ever varied them across one surface.

A patch claims a **slab**, not a rectangle: `thickness_m` is its half-width along the normal. Without
this a road patch would claim a bonnet 0.9 m above it, since the bonnet projects into the same
(u, v) rectangle — and the resulting frame would look entirely reasonable.

## Consequences

**Easy:** any near-planar surface — road, bonnet, roof, wall, deck, container top — gets a real
temperature field for the cost of authoring a patch, with no renderer change, no asset change and no
new dependency. The per-prim path is untouched, so every existing scene renders bit-identically.

**Hard:** curved and small geometry. A wheel, a tyre, an exhaust pipe or a mast is not near-planar
and still takes one temperature per prim. This is a real limit, not a temporary one: raising it
means option 1 or 2, and therefore means a renderer capability this build does not have.

**The error introduced** is the projection error: a surface that deviates from its patch plane by
`d` is sampled at the cell its projection lands in, displaced along the surface by `d·tan(θ)` for
a surface tilted θ from the plane. For a bonnet with 60 mm of crown over a 1.2 m patch and 0.15 m
cells, the worst displacement is under a third of a cell, which the bilinear sample smooths further.
For a surface tilted more than ~30° from its patch plane the displacement exceeds a cell and the
surface should get its own patch instead. This is not checked automatically — `thickness_m` bounds
how far out of plane a claimed point may be, which bounds the error, but the caller chooses it.

**Cost.** Cells are solved, not rendered, so the count is set by the thermal gradient and not by the
pixel count: a 2.4 m bonnet at 0.15 m cells is 16×8 = 128 cells, and a 20 m × 20 m road at 0.25 m is
6 400. Both are far below the per-frame cost of the radiance kernels.

## Revisit when

An SPG shader lands (M10.12/M10.13a–e) and can carry an integer per-triangle attribute, or a build
exposes a UV AOV — either makes option 1 or 2 reachable and this becomes the fallback for surfaces
that do not merit an atlas.

## Addendum, 2026-09-21 (WM.1): the premise of "a real limit" does not hold

This ADR's Consequences call curved geometry "a real limit, not a temporary one: raising it means
option 1 or 2, and therefore means a renderer capability this build does not have". `WM.1` measured
a **third** route that needs nothing from the renderer, and it works.

Options 1 and 2 both ask the *renderer* to transport the parameterisation — a UV AOV, or a
per-triangle id. The third route derives it instead: the instance id says which prim a pixel hit,
the position AOV says where in the world it hit, and a closest-point query against that prim's own
mesh returns the triangle and its barycentric coordinates. `scripts/probe_warp_mesh.py` and
`scripts/probe_warp_prim.py` measure it on this build (Warp 1.16.0, no Kit boot needed for the
first):

* `mesh_eval_position(face, u, v)` reproduces a point already on the surface to **0.13 µm** on a
  0.4 m box and a 16 k-triangle sphere — against this ADR's own 3.4 mm position budget, a margin of
  more than four orders of magnitude. The residual is float32 round-off, and it scales with distance
  from the world origin (0.73 µm at 4.3 m), so a scene tens of kilometres across would need
  checking; at vehicle and building scale it is nowhere near the budget.
* Warp's `(u, v)` are the weights of **v0 and v1**, with `1 − u − v` on v2. The reading a person
  writes down first, `(1−u−v)·v0 + u·v1 + v·v2`, is wrong by **0.56 m on a 0.4 m box** — a silent
  error that samples an unrelated part of the surface.
* Under a full 3.4 mm of position error the recovered *face* changes on 1.9 % of queries on a
  12-triangle box and 56 % on a 16 k-triangle sphere, but the sampled surface point moves only
  2.68 mm on average and never more than 3.41 mm. The face is unstable between neighbours; the
  **position** is stable to the size of the input error, which is what a temperature lookup needs.
* A USD prim needs triangulating (`faceVertexCounts` of 4 for an asset authored as quads) and its
  local-to-world transform applying, and an analytic gprim such as `UsdGeom.Sphere` exposes no
  points to hand Warp at all.

This ADR is **not** superseded here: the planar patch remains what ships, and the decision that
replaces it is `WM.5`'s to write once `WM.3` has built the bridge. What changes now is the claim
that the limit could not be raised on this build. It can.
