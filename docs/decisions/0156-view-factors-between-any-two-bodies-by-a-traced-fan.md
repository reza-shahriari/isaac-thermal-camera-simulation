# 0156 — View factors between any two bodies by a traced fan, one-sided faces, sky and back kept apart

Date: 2026-09-28
**Status:** Accepted
Roadmap: TC.9 (§6.1, §6.6); supersedes the "parallel rectangles only" limit of ADR 0088

## Context

ADR 0088 chose a closed form for the configuration factor between a plane cell and a parallel
rectangle, and made every other geometry **raise** rather than be evaluated with a formula that
does not apply. That was right for the two cases it served (an engine bay under a bonnet, an
underbody over asphalt) and it is what stops a downloaded car radiating onto the road beside it, a
battery warming the shell around it, or a deckhouse warming the deck in its lee: none of those is
two parallel rectangles. TC.10 (heat exchange between objects as one scene switch) and TC.11 (a
main object fully solved) both need `F_ij` between arbitrary meshes, occlusion included.

## Options considered

1. **Monte Carlo ray casting.** The textbook general method: random points, random cosine-weighted
   directions, count landings. General and simple, but every result carries sampling noise that a
   seed pins without removing, and a test tolerance then has to cover the noise rather than the
   physics.
2. **A deterministic hemispherical quadrature, traced.** The sky-view code already owns a fan of
   1740 rays of known solid angle (`skyview.sub_rays`, Tregenza patches cut finer). Rotate that fan
   into each face's own frame, cast it from evenly spread points on the face, and credit
   `ω cos θ / π` to whichever face is the **nearest** hit. Occlusion is not a separate pass: a body
   in between simply is the nearest hit. Reproducible to the bit, no seed, and its error is a
   resolution one can measure.
3. **Hemicube or a GPU rasteriser.** Fast, and exactly the wrong first step: CLAUDE.md asks for the
   scalar reference first and the fast path tested against it, and `cpu-reference-before-gpu`
   says the same. The reference is what this ADR is.

## Decision

Option 2, as `irsim.thermal.view_factors.view_factors`. Three further choices inside it:

* **Faces are one-sided and back hits are booked separately.** A face radiates from the side its
  winding normal points to, like every `TriangleSoup` in `raycast`. A ray that lands on the
  *back* of a face is stopped (the body is opaque) but credited to neither that face's `F` nor the
  sky: it is returned as `back`. Without this, a plate facing away from a body would "see" it while
  the body never saw the plate, and reciprocity `A_i F_ij = A_j F_ji` would fail by construction.
  A real two-sided panel is authored as two faces with opposite windings, and then every hit is a
  front hit.
* **The sky share comes with the factors.** Whatever misses every body reaches the sky, and that
  share is returned per face and per body, because ADR 0088's second rule -- a body that radiates
  onto a surface also hides the sky it was seeing -- is what TC.10 will apply with these numbers,
  for any shape.
* **Body-level aggregation is the accurate quantity.** `F_A→B = Σ_{i∈A} A_i Σ_{j∈B} F_ij / A_A`.
  A single face's factor comes from `subdivide²` fans (four by default) and is a coarse estimate;
  the body sum is what the Howell checks and the reciprocity test are held to.

**Verification** (`tests/unit/test_view_factors.py`): unit squares half a unit apart reproduce
Howell C-11 to 0.3 %; a 1 × 1 floor and a 0.5 × 1 wall on its edge reproduce C-14 to 0.3 % and each
other by reciprocity to 0.1 %; 16-gon coaxial discs reproduce C-41 at the equal-area radius to 1 %;
every row of a closed box sums to one to 1e-12 with a sky share of exactly zero; a wide plate
between two squares takes the whole view (`F = 0` exactly) and the sky it hides, and the plate
facing its back exchanges nothing with it in either direction.

## Consequences

* **Cost.** Every face casts `subdivide² × 1740` rays against every triangle: a 30-face pair takes
  a second, a 300-face pair a minute. That is a build-time cost, once per scene, and it is the
  reference the fast path (when there is one) will be tested against. Nothing here is on the
  per-frame path.
* **Resolution, not noise.** The error of a factor is the fan's angular resolution (about 0.0036 sr
  per ray) times the cosine weight at the boundary of the target, averaged over the sample points.
  It is the same for every run; a test that fails has found a change, not a bad seed.
* **ADR 0088 stays.** The closed form remains the exact answer for the parallel-rectangle sources
  the car scenes use and is still what `patch_view_factors` returns; TC.10 decides per pair which
  to use, and the traced factors are what the closed form cannot give.
* **What is still missing.** The factors are geometric only: no emissivity, no temperature, no
  exchange -- that is TC.10's balance. Faces are opaque; glass and semi-transparent bodies are not
  yet distinguished from steel. And a face's own body can shadow it (a concave mesh), which is
  correct and also the reason a coarse mesh of a curved part needs `low2high` first.
