# ADR 0192 — A human is a fixed taxonomy labelled by its skeleton, not parts discovered by component

**Status:** Accepted
**Date:** 2026-10-06
Roadmap: HU.2 (§6.1, §16.2). Extends [ADR 0122](0122-a-person-is-two-surfaces-and-iso-7730-is-used-outside-its-range.md)
(a person is two surfaces) and bounds [ADR 0138](0138-an-imported-asset-is-decomposed-into-functional-parts-by-connected-component-and-the-parts-are-data.md)
(parts by connected component), which does not apply to people.
Evidence: `docs/research/2026-10-06-humans-in-the-simulation.md`.

## Context

The owner asked for pedestrians on 2026-10-06: one realistic human that ships, and a fast path for
any downloaded human — a scan, a game character, a parametric export — to arrive with the same
thermal materials, so that one scene can hold a girl in a blue shirt and another in a red one.
"The clothes may change but the human is human."

ADR 0138 recovers an aircraft's parts by splitting the mesh into connected shells and clustering
them: a propeller is a separate shell from its motor even when both are white ABS. A human defeats
that twice. The body is **one** connected shell, so component discovery returns one part. And the
parts it should return are not a property of the mesh at all: every human has a head, a neck, two
upper arms, two forearms, two hands, a chest, a back, a pelvis, two thighs, two lower legs and two
feet — the seventeen segments JOS-3 solves (Takahashi et al. 2021) — each bare or under a garment.

Getting the regions wrong is invisible in a render and large in the physics. Measured people are
not one temperature: forehead 34.7 °C against a nose at 33.5 °C indoors, palms near 28 °C and
fingers near 25 °C in a cool room, so face minus fingers is about 10 K — two hundred NETDs. A hand
labelled as forearm renders a plausible warm arm and is wrong by that much.

## Options considered

1. **Component discovery, as for a drone (ADR 0138).** Returns one part. Even with garments as
   separate shells it names "the shirt" and never "the forearm". Rejected: wrong tool.
2. **Register a parametric body (SMPL-X) to the mesh and transfer its per-vertex part labels.**
   Technically the strongest labeller for scans. Every tool in the family inherits a
   non-commercial licence with a surveillance exclusion, and its commercial channel closed in
   April 2026 (Meshcapade's platforms shut 2026-04-18 after Epic's acquisition). Usable case by
   case under the owner's "any tool" ruling; not the default.
3. **A fixed taxonomy as data, labelled by the skeleton.** Every rigged humanoid carries
   per-vertex bone weights, and the bones have names — Mixamo (65), Rigify (159), SMPL-X (55),
   MPFB GameEngine (53) — that map onto the seventeen segments by a table. An unrigged mesh gets a
   rig with Mixamo names in under a second from an MIT auto-rigger (Make-It-Animatable).
   The argmax of the weights per vertex is the label. Chosen.
4. **Image segmentation projected onto the mesh (Sapiens, 28 classes).** The best garment and
   hair vocabulary; CC BY-NC. Kept as the skin / hair / garment classifier in HU.9, behind the
   skeleton for the body regions, because a skeleton is exact where a projection votes.

## Decision

**The body is data, in `configs/humans/body_schema.yaml`, and `irsim.config.humans` loads and
checks it.** The schema carries:

- the seventeen segments in **JOS-3's names and order**, each with its side, its mirror and JOS-3's
  standard local surface area (1.87 m² in all). The loader refuses any other set, by name, because
  HU.4 binds the solver's per-segment skin temperatures to these strings;
- one **region**, `torso`, that a bone may name when it drives two segments — one spine drives
  Chest and Back — resolved per face by the sign of the body's forward coordinate;
- **garment slots** (head, torso, arms, hands, legs, feet) with what each covers by default; the
  Neck is bare unless a garment's own `covers:` names it;
- one **bone map per rig**, listing every deforming bone and the segment or region it drives, the
  prefixes to strip (`mixamorig:`, `DEF-`), the prefixes that mark non-deforming control bones
  (`ORG-`, `MCH-`), the leaf bones that deform nothing (`HeadTop_End`), and the side tokens.

The loader refuses: a rig that leaves a deforming bone unmapped or a segment undriven (naming
them); a bone whose mirror bone drives anything but the mirror segment; a target that is neither
segment nor region; a slot covering a non-segment. The lookup is one function, `segment_for_bone`:
prefix stripped, exact, pattern, then without a trailing `.NNN` — a Rigify DEF- limb is two bones.

**An asset says `kind: human`** and carries a `human:` block: the phenotype JOS-3 sizes a body
with (sex, age, height, mass) and the garments, each on a slot with a library material, an
insulation in clo and a colour. `kind: human` and the block are required together.

**Three rules are judgements, flagged ESTIMATED in the file.** The lumbar spine bone maps to
Pelvis, because the 65MN lineage's pelvis segment is the pelvis *and* the lower abdomen (its
0.221 m² is the largest torso segment). A clavicle bone maps to the torso region. JOS-3's
"Shoulder" is the upper arm, "Arm" the forearm, "Leg" the lower leg — kept, so that the names
match the solver's, with the meaning written beside each.

## Consequences

- **A new human is a new mesh under the same labels; a new shirt is a material and a colour in one
  slot.** HU.3–HU.8 (man, woman, child, police officer, soldier) add YAML, not code.
- **Colour reaches the reflective bands and the sun, never the long-wave emissivity** (Zhang, Hu
  & Zhang 2009: colour has no effect on a fabric's emissivity). The schema carries the colour;
  HU.5 derives solar absorptivity and NIR reflectance from it. A blue and a red shirt differ in
  LWIR only through temperature.
- **Hard seams.** Splitting at the argmax gives a step between forearm and hand where the body has
  a gradient. The error is bounded by the step between adjacent segments' temperatures — a few K
  at the wrist in the cold, under 1 K across the torso — and the upgrade path is the weights
  themselves as a per-vertex blend. Not needed to get the ordering and the magnitudes right.
- **Chest and Back are a geometric split, not a bone.** A face is Chest if it faces the body's
  forward axis. For a twisted pose the split follows the rest pose's axis unless the prep tool
  evaluates it per bone; HU.3 does it on the rest pose.
- **The Neck is a segment with no default garment.** A collar or a scarf must say so. Chosen over
  a default collar because a bare neck is the common street case and a collar is a wrong
  insulation on the segment the camera sees beside the face.
- Nothing engine-side changes: the schema is pure data and the loader pure Python, tested in 49
  cases with no Isaac Sim.

## Revisit when

- HU.12's validation against radiometric faces shows the single head segment is the limiting
  error — then face sub-regions become segments of a finer taxonomy, and this file's "seventeen"
  becomes "JOS-3's seventeen or the 65MN's sixty-five".
- A rig arrives whose bones do not partition the body (a cloth-only rig, a facial rig alone); then
  the label needs a second oracle (option 4) rather than a longer table.
- The seam error is measured above 1 K on a rendered person at the camera's range — then the
  per-vertex blend is built.
