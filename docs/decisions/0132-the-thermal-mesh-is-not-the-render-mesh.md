# ADR 0132 — The thermal mesh is not the render mesh, area is what decimation must preserve, and self-occlusion is a decision

**Status:** Accepted
**Date:** 2026-09-23
Extends [ADR 0128](0128-per-asset-material-mapping-and-the-blender-prep-path.md) (the asset
pipeline) and [ADR 0110](0110-the-temperature-field-on-the-mesh.md) (the field lives on the mesh).
Roadmap: `AI.2`.

## Context

ADR 0128 gave an imported asset material identity. It did not make it solvable: `MeshSpec` still
generated a cylinder or a sphere, and `config/scene.py:410-415` recorded reading a real asset's
triangles as an unowned follow-on. This ADR records what that turned out to cost.

The subject is a DJI Phantom 4 Pro FBX: 41 prims, 2,486,459 triangles, **0.2938 m²** of surface.

**The asset is tessellated about a hundred times finer than the physics.** Cells on that aircraft
want to be millimetres; the asset's triangles are a fraction of one. The solver puts at least one
cell on every face, so the tessellation sets the cost and it was chosen for silhouettes.

## Options considered and what they measured

1. **Solve the asset at full resolution.** 1.5–2.5 M cells. `GT.7` cites Fraunhofer running
   1,313,410 triangles through five day–night cycles in 252 s, so this looked affordable; it is
   not, for the reason in *Decision 3* below.
2. **Collapse (quadric) decimation**, the usual answer. **Rejected on measurement:** at ratio 0.05
   it removed **37 %** of the asset's surface area, and at 0.01, **42 %**. Area is not cosmetic
   here — it sets both the radiated power and the convective load — so a 37 % loss is a 37 % loss
   of emitted signal, silently, on geometry that still looks right. The cause is structural: this
   asset has **31,068 disconnected shells**, and a collapse budget spends itself destroying the
   small ones outright.
3. **Planar dissolve** — merge coplanar faces, never move a vertex (chosen). At 3° it removed
   **38 %** of the triangles for **+0.056 %** area. At 5° it removes half for +0.2 %, and at 15 %
   two thirds for +1.7 %.

## Decision

**1. The archive is the *thermal* mesh, and it need not be the render mesh.** ADR 0110's
closest-point query locates a pixel on whatever mesh the field carries, so the renderer keeps the
asset's full tessellation while the solver carries one sized to the physics. What bounds the
divergence is the bridge's distance tolerance, not an equality. `scripts/prep_asset.py --emit-mesh`
writes one `.npz` of world-space triangles per asset and `irsim.io.assets` reads it, so
`src/irsim/` still imports no engine and a scene stays solvable with no renderer present.

**2. Decimation is planar-only, and area is the gate.** The prep tool refuses to write an archive
whose area moved by more than a stated tolerance. The gate is **area-weighted**, not a bare
relative error per prim: a relative gate is right for a prim that carries the signature and wrong
for a 1.1e-5 m² decorative sliver, where 3 % is 4e-7 m² and cannot move any temperature. So the
whole asset is gated on relative area, and a prim is gated individually only once it carries at
least 0.1 % of the total. Everything below is reported, never silent. Zero-area triangles are
dropped and counted — this asset has 126 — because a facet with no area has no normal and §6.1
balances a facet on its normal.

**3. Self-occlusion is a decision a scene makes, not a default it inherits.** Tracing a mesh
against itself costs cells × faces, once for the sky view and **again every tick** for the solar
disc. Measured: one imported prim of 3,320 cells against 3,215 faces spent **248 s** in
`disc_visibility` over a single 6 h spin-up; the largest bound prim would be 41,127 × 41,103 =
**1.69 × 10⁹** ray-triangle tests. The failure mode is not a slow scene, it is a scene that never
finishes, which is why `GT.7`'s cost figure did not predict it — the cost is in *tracing*, not in
stepping.

So `MeshSpec.self_occluding` is a tri-state. `None` keeps "trace unless convex", exact and what
every generated mesh uses. Above `MESH_SELF_OCCLUSION_BUDGET` (2 × 10⁶, calibrated on the 248 s
measurement to about half a minute of tracing) a scene that left it `None` is **refused**, with a
message naming both ways out. The switch reaches the beam as well as the sky view: routing it to
only one leaves the scene just as unable to finish, which is how this was found.

**4. What `self_occluding: false` costs, stated.** Each cell takes the analytic `(1 + cos β)/2`
sky view — the same unoccluded form the in-sim path already uses (ADR 0045). For an airframe in
free air, whose only occluder is itself, that is defensible. It is not free: the gimbal under the
nose genuinely sits in the body's shadow, and with the flag off it does not.

## Consequences

- `configs/scenes/phantom4_pointwise.yaml` is the **first scene in this project whose geometry was
  not authored in Python**. Six prims, 234,923 cells, 0.1628 m² — 55 % of the aircraft's area for
  11 % of its triangles — built in **13.6 s**.
- The result is the physics the lane exists for: black mouldings reach **65.9 °C** where the white
  shell tops out at **33.8 °C** (α 0.94 against 0.25), and every surface carries a *span* —
  7.8 K on the propellers to 39.7 K on the mouldings — rather than one value per object.
- A scene still picks its prims by hand, because nothing yet budgets them. That is `AI.3`.
- `material:` on a surface duplicates what `configs/assets/<name>.yaml` says for the same prim.
  Nothing joins them and they must agree; a follow-on should derive one from the other.
- Schema v16. The pin in `test_mesh_scene.py` that tied a v14 scene to *the newest* version was
  relaxed to the readable range, since an unrelated addition should not fail an old scene.

## Revisit when

`AI.3` lands a triangle budget — at which point tracing may become affordable and
`self_occluding: false` stops being the default choice for an imported asset — or when a scene
needs an imported prim that genuinely shades another, which no aerial scene does and every ground
scene will.
