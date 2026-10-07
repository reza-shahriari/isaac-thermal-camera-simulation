# ADR 0201 — A downloaded human is rigged by Make-It-Animatable and sorted by its own skin colour

**Status:** Accepted
**Date:** 2026-10-07
Roadmap: HU.9 (§6.1, §16.2). Builds on [ADR 0192](0192-a-human-is-a-fixed-taxonomy-labelled-by-its-skeleton.md)
(a human is labelled by its skeleton). The second deliverable the owner asked for on 2026-10-06:
"a fast and forward way to add a new human".

## Context

A human the project generates arrives rigged and in parts. A downloaded scan arrives as one mesh,
with skin, clothes and hair fused, no skeleton and no part names. To become an asset it needs two
answers per triangle: which of JOS-3's seventeen segments it belongs to, and whether it is skin,
garment or hair. The HU.1 survey named the tools: Make-It-Animatable (MIT) to rig any humanoid,
then a priority of cues for the surface (objects and slots, colour, Sapiens, Find3D, a person).

A claim like "the tool labelled 98 % of faces" says nothing about whether the labels are right.
So each step needs a score against a known answer, and a real scan has none.

## Options considered

1. **Wait for a real scan and judge it by eye.** Rejected: there would be no number to judge
   against (`measure-agreement-never-judge-by-eye`).
2. **Score against the project's own people.** `fuse_human.py` turns a person the skeleton
   already labelled into one bare, unrigged mesh, and keeps each triangle's true label. Chosen.
   The real scan becomes the held-out check.
3. **A fixed skin-tone colour locus for skin versus garment.** Rejected after measurement. Brown
   leather sits inside every published locus, and a fixed locus is biased by skin tone.

## Decision

- **Rigging.** `scripts/autorig_mia.py` runs Make-It-Animatable in its own conda environment
  (torch 2.1.2, PyTorch3D, numpy 1.26), never the Isaac interpreter. It drives the tool's
  pipeline functions in-process and writes Mixamo weights, joints and hierarchy.
  - The tool reads a skeleton template from a gated Hugging Face dataset. Inference uses only
    the template's bone names and parents, so `scripts/mixamo_template.py` writes the standard
    52-bone hierarchy instead. No Mixamo asset is copied.
  - It runs on the CPU, pinned by PCI bus order when on a GPU (`IRSIM_GPU`).
- **Segments.** `irsim.io.human_scan.label_faces_from_weights` uses `prep_human.py`'s own
  arithmetic, with the body schema's existing `mixamo` map. The facing comes from the predicted
  joints (left hip to right hip, crossed with hips to head), because a scan declares none.
- **Skin, hair or garment** (`classify_surfaces`):
  - The skin reference is the person's own: the median CIELAB colour of the front, lower head.
  - A triangle's CIE76 difference from it is averaged over three rings of vertex-sharing
    neighbours. A garment shell never averages with the skin under it.
  - A triangle is skin below ΔE\*ab 30. Smoothed skin's 99th percentile was 18–25 on all four
    test people, so 30 is that bound plus about two just-noticeable differences.
  - Non-skin on the head above the eyes is hair. Below the eyes it is the face's own (eyes,
    lips), because a head garment sits above them. The first run called the eyes a head garment.
    Any other non-skin triangle is a garment on the slot that covers its segment.
  - Triangles within ±6 of the threshold are decided anyway and listed in `<name>.review.json`
    for a person: the Blender add-on's job, B14.
  - Otsu's threshold was tried and dropped. It split the bare man's skin at 17, because a body
    with no clothes has no second mode. CMC(2:1) was tried too and separated nothing better.
- **The asset.** `scripts/prep_scan.py` splits the scan into `skin_<Segment>`, `garment_<slot>`
  and `hair` objects, as every human asset names them. Each material is copied per label so the
  asset map binds it. With `--write-config` it writes the asset YAML:
  - the phenotype **stated** (sex, age), stature measured, mass from a BMI;
  - each garment's colour **measured** on the scan, which then drives its α and NIR (ADR 0195);
  - the tools used, with their licences, in the header.

**Measured** on four people fused from the project's own (`fuse_human.py`), rigged on the CPU:

| person | rig time | segments agree (skin area) | surface class correct | decided without a person | correct where decided |
|---|---|---|---|---|---|
| bare man | 3.7 s | 90.4 % | 99.8 % | 98.1 % | 99.8 % |
| dressed man | 6.7 s | 91.0 % | 98.1 % | 94.2 % | 99.3 % |
| woman | 6.5 s | 92.3 % | 84.2 % | 80.5 % | 99.7 % |
| girl, 8 | 6.4 s | 90.7 % | 98.4 % | 92.7 % | 99.6 % |

The segment disagreement is mostly two rig conventions meeting at different heights. Mixamo's
waist and neck bones are not MakeHuman's. The arbiter is JOS-3's own standard area share per
segment. On the bare man, both labellings sit about equally far from it: a summed absolute share
difference of 0.155 for the skeleton and 0.173 for the auto-rig. The auto-rig's pelvis is closer
than the skeleton's; its neck gets a fifth of JOS-3's standard area. The scan-labelled dressed man
passes `prep_asset.py` at 100 % material and part coverage. Rendered in LWIR at the summer
noon, its 17 visible parts match the solver to 0.0005 K. Its trousers read warmer than the
skeleton-labelled man's, because the scan path gives the legs slot its default denim and
0.26 clo where the hand-made asset names wool. The defaults are there to be edited.

## Consequences

- **A downloaded human is minutes of compute plus one review.** HU.9's criterion asks that
  ≥ 95 % of faces be labelled without a person. That held only on the bare man (98.1 %). The
  dressed man fell just short at 94.2 %, the girl at 92.7 % and the woman at 80.5 %. Where the
  colour tier does decide, it is right 99.3–99.8 % of the time.
- **Colour cannot separate what looks like skin.** The woman's near-white top on pale skin is
  15.5 % of her area called skin, and most of it falls outside the review band. Tan leather sits
  at the band's edge.
- **A bias by skin tone, measured and pinned.** Black hair on dark skin is about 25 ΔE from it,
  under the threshold, so it is called skin (`test_human_scan.py`). In LWIR, hair takes the
  head's temperature either way, and the RGB keeps the scan's texture. Still, the error lands on
  one group of people. The fix is a segmentation tier, not a threshold.
- **The four test people are the tuning set.** The threshold and band were set on them. The
  owner's CC-BY Dennis scan is the held-out acceptance test. A real scan also has no skin hidden
  under its clothes, which the fused test people do.
- **The weights are discarded.** The labelled asset is static, like every human (ADR 0199 for
  motion). Keeping the predicted rig for a walking scan is a later step.

## Revisit when

- The Dennis scan or another real scan arrives. Then its score is this ADR's held-out number.
- The review share or the skin-tone bias matters. Then add the next tier: Sapiens (CC BY-NC 4.0,
  allowed by the owner and recorded in provenance) or Find3D (MIT), voting per triangle from
  multi-view segmentation.
