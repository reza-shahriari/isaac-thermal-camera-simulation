# ADR 0112 — Lateral conduction on a mesh: monotone over consistent

**Status:** Accepted. Extends ADR 0102 (`PT.11`) from a rectangular grid to a triangle mesh;
supplies the term ADR 0111 measured the absence of.
**Date:** 2026-09-22
Roadmap: WM.6 (§6.1, §6.4)

## Context

`PT.11` gave a planar patch's cells a conduction link to their four neighbours, on the grounds
that a metal panel without one renders sharper than aluminium can be. A mesh had none, and
[ADR 0111](0111-temperature-granularity-tiers.md) measured what that costs: on the quadrotor's
carbon arms the crown-to-underside span came out a quarter too strong, because a sheet of `k δ`
losing heat at `h` cannot hold a feature finer than `L = √(kδ/h)` and nothing in the solve knew it.

A rectangular grid makes this easy — the line joining two cell centres is perpendicular to the
edge they share, which is exactly the condition a two-point flux needs. A triangulation does not.

## Options considered

1. **The circumcentric (dual-cotan) weight.** Put the node at each triangle's circumcentre and the
   centre-to-centre line *is* perpendicular to the shared edge, so the scheme is consistent. Its
   weight is `G = 2 k δ / (cot θ_a + cot θ_b)` over the two angles opposite the shared edge — the
   dual of the cotan Laplacian. **Negative whenever those angles sum past π**, and inside one face
   the two sub-triangles across a sub-edge are congruent, so both angles are the parent's own and
   the weight is `k δ tan Θ`: negative for every *obtuse* face and singular for a right-angled one,
   where the circumcentre lies on the shared edge itself. A negative conductance breaks the
   discrete maximum principle — a cell leaves the range its neighbours and its forcing span, and
   renders as a bright speck indistinguishable from a bad pixel.
2. **Intrinsic Delaunay flipping**, which is what the roadmap row named: flip edges intrinsically
   until the triangulation is Delaunay, then option 1's weights are non-negative. Rejected here
   because the cells *are* the mesh's own sub-triangles — the Ptex grid the field lives on and the
   closest-point query resolves to (ADR 0110). Flipping changes which cells exist, so the solve
   would run on one triangulation and the render sample another.
3. **Multi-point flux (MPFA) or a diamond scheme.** Consistent on arbitrary meshes, and gives up
   the M-matrix property in a different way: the stencil is wider and the off-diagonals are not
   signed. It trades one monotonicity failure for a subtler one.
4. **The centroid two-point flux**: `G = k δ w / (d_a + d_b)` over the shared edge length `w` and
   the perpendicular distances from each centroid to it. Non-negative by construction, and the
   same formula ADR 0102 already uses.

## Decision

Option 4. **Monotone over consistent**: a gradient that is 10 % too strong reads as weather, and a
cell outside its neighbours' range reads as a broken sensor.

Inside a face the weight has a closed form that does not depend on the level at all — a sub-edge
parallel to the parent's side `E` carries

    G = (3/4) k δ |E|² / A_face

because the barycentric cut is affine, so every sub-triangle is similar to its parent and `w/d` is
scale-free. Refining a face therefore adds cells without changing how fast heat crosses it, which
is what a discretisation of a continuum has to do. Across a shared mesh edge the boundary cells are
matched by the length they actually overlap, as `TC.3`'s contactors match two patches whose grids
do not line up, so faces of different levels may meet. The result is an ordinary
`ConductionOperator`, so ADR 0094's backward Euler, its prefactorisation per tick size and its
symmetry check are reused rather than restated.

`circumcentric_conductance` ships beside it, unused by the operator, so the tests can show what the
shipped weight is chosen *over* rather than assert it: on an obtuse face it is negative, and
`ConductionOperator` refuses to hold a negative conductance at all, so the cotan assembly fails at
build rather than as a speck in a frame.

## Consequences

**The two schemes agree exactly on an equilateral face.** There the centroid *is* the circumcentre
and both give `k δ tan 60° = √3 k δ`, at every level. That is the anchor: the shipped weight is an
approximation of a specific thing, not of nothing in particular.

**A three-neighbour centroid stencil cannot be both exact and isotropic unless the face is
equilateral**, and this is a theorem rather than an observation. Exactness for linear fields
requires `Σ_s G_s Δx_s = 0` over the three neighbours; the three displacements sum to zero and span
the plane, so that forces `G_0 = G_1 = G_2` — and equal weights on a skewed face give an
anisotropic operator. The scheme keeps the isotropy. Measured residual under a linear field, as a
fraction of scale: **7 × 10⁻¹⁶ on an equilateral face** (machine precision) and **1.5–4.5 % on
right, obtuse and skewed ones**, and it does not fall with refinement, because it is the scheme's
error and not the discretisation's.

**What the tessellation costs, and the rule that follows.** A tube's faces are the two halves of a
`w × a` quad. Solving the steady fin problem on the mesh and backing out the effective `kδ` from
the attenuation gives

| ring / arc | `kδ_eff / kδ` |
|---|---|
| 0.8 | 1.21 |
| 1.0 | 1.12 |
| **1.4** | **1.00** |
| 2.0 | 0.90 |
| 3.0 | 0.82 |
| 12.7 | 0.75 |

So: **cut a tube so its rings are about 1.4 times its circumferential arc.** It is exact there and
saturates at three quarters of the conductivity it should have for long thin quads — which would
render a tube's gradient a quarter too strong, the very error this row exists to remove.
`configs/scenes/quad_flight_mesh.yaml` shipped at 24 × 6, a **12.7 : 1** quad, and is re-cut to
16 × 36 (5.89 mm arc, 8.33 mm ring, **1.41**). A test reads the ratio off the config, because this
is a thing a person editing YAML gets wrong.

**Over-resolution stops being an error and becomes a cost.** ADR 0111's rule — cells finer than
`L` resolve structure conduction erases — was stated when nothing on a mesh conducted, so a mesh
cut finer than `L` was *wrong*, not merely wasteful. With this operator the smoothing happens, and
the same scene cut finer returns the same answer for more time. The arm's 5.89 mm arc against
carbon fibre's `L` = 8.8 mm is now a deliberate over-resolution for the picture's sake.

**Measured on the scene.** `arm_n` on the pad at a 61° sun: crown 42.7 °C over an underside on
27.3 °C, **15.4 K** around the tube, against 26.6 K before `WM.4` traced the pod's shadow and
22.1 K before this operator. The fin equation predicts the last step: for carbon fibre round a
30 mm tube, `L` = 8.8 mm against a 94.2 mm circumference gives `1/(1 + (2πL/λ)²)` = **0.744**, and
the operator reproduces that to **1 %** on a near-square cut.

**Still assumed.** One `k` per material, isotropic and temperature-independent — a CFRP tube
conducts several times better along its fibres than across them, and the library carries one
number. Conduction is in-plane only: `PT.12`'s through-thickness layers do not run on a mesh.
Non-manifold edges (more than two faces at a seam) conduct nothing, because heat sharing between
three sheets is not a two-point flux and guessing at it would be worse than leaving it out.

## Revisit when

An imported asset brings triangles nobody chose — at which point the aspect-ratio rule is no longer
something the scene author controls, and either intrinsic Delaunay flipping on a *separate* solve
triangulation (option 2, with a transfer to the render cells) or an MPFA scheme (option 3) becomes
worth its complexity; or a material needs an anisotropic `k`, which neither the library nor the
operator can express today.
