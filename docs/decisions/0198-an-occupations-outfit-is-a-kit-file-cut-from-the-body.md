# ADR 0198 — An occupation's outfit is a kit file, cut from the body by one routine

**Status:** Accepted
**Date:** 2026-10-07
Roadmap: HU.11 (§6.1, §16.2). Builds on [ADR 0196](0196-an-occupation-is-garments-plus-equipment-and-retroreflective-tape-is-not-a-mirror.md)
(garments plus equipment) and [ADR 0197](0197-camouflage-is-an-engineered-nir-signature-not-a-dyed-colour.md).

## Context

The owner wants occupations: police officers, soldiers, firefighters, "etc." MakeHuman's CC0
clothes packs dress a civilian. They have no hi-vis vest, no plate carrier, no turnout coat, no
breathing cylinder. HU.7 and HU.8 cut the missing pieces from the body's own skin in
`make_human.py`, with one Python function per occupation. Its constants were heights as fractions
of stature, offsets and colours. HU.11's acceptance criterion is that each occupation after them
is "a YAML of garments and parts with no new code". Two functions in, the pattern was clear, and
the third would have been a third function.

The firefighter also exposed a gap in the cut. NFPA 1970 asks for a trim band round each sleeve
between wrist and elbow. In MakeHuman's rest pose the forearm is about 30° off horizontal, so a
band cut between horizontal planes is a strip along the forearm, not a ring round it.

## Options considered

1. **A function per occupation.** Rejected: the criterion says otherwise, and each function
   repeats the same cut with different numbers.
2. **A modelled garment per occupation**, bought or drawn. This is better geometry. But it is a
   modelling job per occupation, and a garment must still be fitted and rigged to MakeHuman's
   body. Deferred: a modelled garment drops into a kit's `clothes:` like any MakeHuman asset.
3. **A kit file per occupation, read by one generic routine.** Chosen.

## Decision

- **`configs/humans/kits/<name>.yaml`**, schema `irsim.config.kits.HumanKit`. `clothes:` maps a
  slot to a MakeHuman clothes asset. `pieces:` is a list of generated pieces. Each piece is named
  `garment_<slot>` or `equipment_<name>`, has an RGB base colour, and has one shape:
  - `shell`: the body's skin faces whose dominant bone matches the patterns, pushed `offset_m`
    out. Faces are kept between horizontal `bands` (fractions of stature) or between `rings`
    square to each bone (fractions of its length).
  - `box` or `cylinder`: a rigid shape standing `behind` an earlier piece, `gap_m` clear of it.
- **The kit owns geometry, the asset owns physics.** Materials and clo stay in the asset's
  `human:` block (ADR 0195, ADR 0196). `HumanKit.check_against` refuses a bone pattern that
  matches no bone, and a slot the body lacks. `check_asset` refuses an asset that does not wear
  exactly the kit's slots and carry exactly its equipment. A test refuses equipment that sits
  inside the garment it is on, which would make trim invisible while it is still solved.
- **The police officer and the soldier are kits.** `make_human.py --kit` replaces `--vest` and
  `--plate-carrier`. Rebuilt from their kits, both glTF files are byte-identical to the
  functions' output.
- **Blender reads the YAML.** Its bundled Python has no PyYAML, so PyYAML is installed into
  Blender's user modules directory, as MPFB is a Blender prerequisite. `make_human.py` reads the
  kit as plain data, and the pydantic schema validates the same file in the engine-free tests.
- **The firefighter.** `firefighter`: a turnout coat, sleeves and trousers in
  `turnout_aramid_shell`. Its MWIR emissivity, 0.87, is NISTIR 6299's measured absorptance of a
  Nomex IIIA shell; its LWIR emissivity, 0.90, is ESTIMATED for polymer cloth. Insulation comes
  from Kuklane et al. 2022's manikin, ensemble C6, zone by zone: torso 3.38 clo, arms 2.54,
  legs 2.73, gloves 1.02, boots 1.53. Each is the zone's total insulation less the nude
  manikin's air layer, ESTIMATED because the clothing area factor is ignored. There is a painted
  helmet (0.12 clo, as the soldier's). There is 76 mm retroreflective trim on the coat (hem and
  chest), round each forearm (a ring) and above each trouser cuff. A 6.8 L carbon-composite
  breathing cylinder (157 × 532 mm, EN 12245) stands on the back as equipment in `carbon_fibre`.

## Consequences

- **A new occupation is a kit plus an asset seed**, with no code. A piece the routine cannot cut
  (a brim, a visor, a pouch) is a modelled garment in `clothes:`, or a new shape kind. A new
  shape kind is code, and it should be rare.
- **Shells follow the anatomy.** A coat cut from the skin sits like a vest, closer than a real
  turnout coat, which hangs off the shoulders. A Laplacian relaxation was tried and dropped. On
  these open shells it threw spikes at the seams and pushed the trim inside the coat. The
  camera's temperature is unaffected, since a garment is one temperature (ADR 0195). The
  silhouette is a few centimetres slim.
- **The cylinder is the coat's temperature plus its own sun.** As equipment it is solved with the
  torso garment's insulation (ADR 0196). A real cylinder is a mass of compressed air, cooled
  further as it discharges. Neither the mass nor the discharge is modelled.
- **No breathing mask, no hood.** The face is bare, as at a scene before entry. A masked
  firefighter is a different thermal picture: the face hidden behind a visor.

## Amendment, HU.11 part 2 (2026-10-07)

The criterion held. `construction_worker` and `cyclist` were added as two kits, two asset seeds
and four scenes, with no change to any Python file. The worker's vest was first cut 12 mm out,
with no armour under it. MakeHuman's tucked T-shirt stands up to 2 cm off the belly and showed
through, so the vest is 25 mm out, as on the police officer. A kit's offsets must clear whatever
MakeHuman garment lies beneath; only the render shows when they do not.

## Revisit when

- Modelled turnout gear or a helmet with a brim becomes available under a licence that allows
  redistribution. It then enters through `clothes:`.
- A breathing set in use is wanted. The cylinder then needs a heat capacity and a discharge, and
  the face needs a mask.
