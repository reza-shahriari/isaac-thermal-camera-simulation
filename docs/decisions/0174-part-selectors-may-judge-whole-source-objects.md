# ADR 0174 — Part selectors may judge whole source objects, not only connected shells

**Status:** Accepted
**Date:** 2026-10-01
Amends [ADR 0138](0138-an-asset-is-decomposed-into-parts-by-connected-component.md) (parts by
connected component). Builds on [ADR 0173](0173-a-part-may-assert-its-material-and-an-asset-may-exclude-what-is-not-it.md).
Roadmap: `AI.16`.

## Context

ADR 0138 chose the **connected shell** as the unit a part selector judges. That was right for
the asset it was written against. The Phantom 4's objects are material groups ("all the white
plastic", 987 shells), and only its shells are hardware.

The DJI models of `AI.15` are built the other way. Their artists made **one object per
physical piece** (a blade, a landing foot, a motor can, a lens ring), and many pieces have
several shells. Judged shell by shell, a geometric cut falls through a piece. The owner looked
at the Inspire 3 in Blender and found pieces of different propellers, of the camera and of the
body in the wrong parts. Measured:

* **Inspire 3**: 41 of 345 objects split. 3 % of every blade sat in its rotor head and 8 % of
  every landing foot in the motor; the motor housing split three ways; two camera objects went
  55/45 and 60/40 to the fuselage.
* **Mini 3 Pro**: 21 of 672 objects split. 9 % of every blade sat in its motor and 46 % of each
  rear arm. That arm leak made the rear "motors" 40 % larger than the front, which the AI.15
  config wrongly explained as modelled geometry.

## Decision

`PartsConfig.granularity: component | object`, default `component`. Under `object`, every
shell carries its source object (`Component.source`: the Blender object in the render pass,
the archive prim in the thermal pass, and the two carry the same names in all four library
assets). The selectors then judge each object as one unit (`object_unit`): areas and faces
add, bounds union, the material is the one with the most area, and the centroid is
**area-weighted**. The thermal archive's planar dissolve moves a vertex-mean centroid but
preserves area, so the two passes agree. Exclusion works the same way: an object is excluded
whole or not at all. `PartsConfig.claim_all` implements both granularities, once, for the
report, the archive split and the Blender split.

## Consequences

* Under `object`, no source object is ever split. The Inspire 3 and Mini 3 Pro now have 0
  split objects, and their station parts are whole numbers of physical pieces. The Inspire's
  propeller is exactly its two 111.39 cm² blades, and front and rear motors agree to 0.03 %,
  where they differed by 1.2 % before.
* Choosing the granularity is a judgement about how the artist built the file, recorded in the
  asset YAML. A material-grouped asset (the Phantom 4, the Matrice 300 RTK) must stay at
  `component`. Judged whole, its one "all eight blades" object would fall in a single part.
* A `components.json` measured before this change has no source objects. The driver re-measures
  it when an asset asks for `object`, and the core refuses to guess.
* Object-level selectors are simpler and sturdier. A piece's area is a fixed property (a blade
  is 111.39 cm², a foot 91.95), where a shell's area depended on how the artist happened to
  stitch it.
