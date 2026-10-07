# ADR 0196 — An occupation is garments plus equipment, and retroreflective tape is not a long-wave mirror

**Status:** Accepted
**Date:** 2026-10-07
Roadmap: HU.7 (§4.2, §6.1, §16.2). Builds on [ADR 0195](0195-a-garments-colour-dyes-its-sun-and-its-nir-and-its-surface-is-solved-on-the-skin-beneath.md)
(garments) and [ADR 0194](0194-a-persons-skin-is-solved-by-jos-3-on-the-scenes-weather.md) (the skin beneath).
Corrects the premise of HU.7's roadmap row.

## Context

The owner wants occupations — police officers, soldiers, firefighters — not only people in street
clothes. An occupation differs from a civilian in two ways a camera sees. Its garments are
heavier or layered: a police officer's torso is a uniform pullover, soft body armour and a hi-vis
vest. And it carries things that are neither skin nor garment: retroreflective tape, a duty belt,
later a helmet or a breathing cylinder. Each has its own material, so its own emissivity and its
own share of the sun.

HU.7's row assumed the vest's tape is "a mirror in LWIR — the AT.18 hazard" and asked that a tape
ε ≤ 0.3 be reachable only through the asset map. Before building on that, the claim was checked.
No measurement of vest tape in 8–14 µm was found. What exists says the opposite. Glass-bead
retroreflective sheeting puts glass beads and a polymer binder at the surface; both are opaque and
strongly emissive in the long-wave, and the aluminium behind the beads is a visible-band mirror
the long-wave never reaches. The building-envelope literature describes glass-bead
retroreflective materials as "high solar reflectance and high thermal emittance", and a
glass-sphere-in-polymer film reaches 0.96 over 8–13 µm. The low-emissivity tapes that read cold in
a thermal camera are a different product: metallised marking tapes made to be low-emissivity.

## Options considered

1. **Build the tape as the row says, a low-ε mirror.** Rejected: the evidence says it is wrong,
   and a cold tape on a warm vest is exactly the plausible-looking error the project exists to avoid.
2. **Equipment as more garment slots.** A belt as a second `legs` garment. Rejected: a slot is one
   garment with one insulation, and the belt adds no insulation to the leg; it sits on the
   trousers.
3. **Equipment as its own kind of part**, sitting on a worn garment slot: that garment's
   insulation over that garment's skin, with the equipment's own material and its own absorbed
   sun. Chosen.

## Decision

**Equipment.** `HumanSpec.equipment` maps a name to `{material, on}`. The object and its material
are `equipment_<name>`; `on` must be a slot the person wears, and the loader refuses equipment on
a bare slot. `HumanBodySolver` hands out `<person>.equipment_<name>` targets whose temperature is
the slot's ISO 7730 balance on the JOS-3 skin beneath, with the equipment material's α_sol. Two
consequences follow and are tested. With no sun, the tape and the vest are one temperature. At
noon they differ only by the sun each absorbs.

**The tape.** `retroreflective_tape`: LWIR ε 0.88, MWIR 0.85, ESTIMATED at the glass value since
no measurement of vest tape exists; NIR 0.40 and SWIR 0.45 (silver); α_sol 0.35. It is emissive in
the thermal bands and bright in the sun's. The retroreflective lobe, which returns a coaxial NIR
illuminator's light to the camera, is directional and is not modelled.

**The vest, tape and belt are generated.** MakeHuman's CC0 packs have no hi-vis vest, so
`make_human.py --vest` cuts one from the body itself. It takes the torso faces between the waist
and the shoulders by their dominant bone, cut by exact horizontal planes, and pushes them 25 mm out
for armour and vest. Two 50 mm bands, the EN ISO 20471 minimum, become the tape, and a band at the
waist 35 mm out becomes the belt. Each piece is made from the morphed body, after MakeHuman's
shape keys are baked, and keeps the body's weights, so it deforms with the body.

**The officer.** `police_officer`. Torso: a fluorescent-yellow polyester vest over soft armour
over a wool pullover, 1.33 clo on chest and back: 0.28 pullover, plus 1.0 armour, plus 0.05 vest.
The armour figure is from Potter et al. 2015 (PLoS ONE 10:e0132698). Whole-body insulation rose
0.20–0.26 clo with armour; over the roughly 18 % of skin that chest and back armour covers, that is
about 1 clo locally, consistent with the thermal resistance of an 8 mm aramid pack. Arms: the
navy pullover, 0.28. Legs and pelvis: navy wool trousers, 0.28. A navy cap and leather ankle
boots. The tape sits on the torso and the belt on the legs.

## Consequences

- **HU.7's criterion is restated.** "The tape's ε ≤ 0.3 reachable only through the asset map"
  becomes "the tape is emissive in the long-wave, ESTIMATED and sourced as such, and reads as its
  vest does when there is no sun". AT.18's guard against a glob reaching a mirror is unchanged and
  still tested (`test_material_mapping.py`).
- **In winter the armoured torso sits nearer the air than the sleeves.** At −5 °C under a clear
  sky, it reads ISO 7730's figure for 1.33 clo on the chest's skin against 0.28 on the arm's; the
  test computes both by hand.
- **The generated vest is a single surface** with no thickness and jagged armhole and neck edges,
  where the bone boundary runs through whole faces. It reads as a vest to both cameras; a modelled
  vest is the upgrade. The pullover's torso, hidden under the vest, renders at the sleeves'
  temperature. A garment is one temperature (ADR 0195).
- **Equipment is a template for the occupations after this.** The soldier's helmet and plate
  carrier, the firefighter's breathing cylinder: each is a material on a slot, with no new code.

## Revisit when

- A measurement of hi-vis tape in 8–14 µm turns up. Then the ESTIMATED 0.88 becomes it, and if it
  is low, this ADR's correction is itself corrected.
- An NIR camera with an illuminator is simulated. Then the tape's retroreflective lobe matters and
  needs a directional term.
- Equipment that makes heat, such as a radio, a light or a heated vest, is wanted. Then equipment
  gains a dissipation, as hidden parts have.
