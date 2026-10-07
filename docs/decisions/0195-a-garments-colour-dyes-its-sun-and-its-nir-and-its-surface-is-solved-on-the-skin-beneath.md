# ADR 0195 — A garment's colour dyes its sun and its NIR, never its long-wave; its surface is solved on the skin beneath it

**Status:** Accepted
**Date:** 2026-10-06
Roadmap: HU.5 (§4.2, §6.1, §16.2). Extends [ADR 0122](0122-a-person-is-two-surfaces-and-iso-7730-is-used-outside-its-range.md)
(the clothing balance) and [ADR 0194](0194-a-persons-skin-is-solved-by-jos-3-on-the-scenes-weather.md)
(the skin beneath). Logs spec issue S66 as still open.

## Context

The owner's requirement for people is as much about RGB as about IR: one scene holds a girl in a
blue shirt and another in a red one, and the same asset must serve both cameras. The question is
what a colour *is* to the infrared, and how a garment gets a temperature of its own.

Two measurements decide the first. "Color has no effect on surface emissivity for the same fabric"
(Zhang, Hu & Zhang 2009, J. Textile Inst. 100:90): in the long-wave bands a blue shirt and a red
shirt are one material. Dye dominates reflectance in the visible and the near infrared (an olive
dye 23–32 % visible, 50–58 % NIR) and fades in the SWIR, where "all dyed samples gave similar
signatures" (Kaur et al. 2024, Appl. Spectrosc.). So colour reaches the infrared in exactly two
ways: through the sun the cloth absorbs — the dark shirt runs warmer, and LWIR sees the
temperature — and through the NIR camera's own band.

PH.12 (ADR 0122) solves a clothing surface from ISO 7730's balance over an authored skin set
point, with no sun. HU.4 (ADR 0194) made the skin beneath a solved field: JOS-3's segment, which
at 0 °C is 28 °C on the head and 9 °C on the hands. A coat solved over 34 °C when the skin under
it is 15 °C is wrong by the difference.

## Options considered

1. **A colour per garment as a separate material file** (`cotton_blue`, `cotton_red`, …). Every
   colour a library entry; the walk, the table and the audit know nothing of why two files share
   every number but one. Rejected: it hides the physics that colour changes one thing.
2. **A colour that edits the long-wave emissivity by its luminance.** What a renderer's
   "darker is hotter" intuition does. Rejected: it is the measured wrong answer (Zhang 2009).
3. **A colour that derives a variant of the base material**, moving `solar_absorptivity` and the
   NIR emissivity and nothing else, made at scene load, never written to the library. Chosen.
4. **For the garment's temperature: JOS-3's own clothing node.** JOS-3 reports skin, not the
   clothing's outer surface. Rejected for the surface; it supplies the skin beneath.
5. **PH.12's ISO 7730 balance per garment, on the JOS-3 skin beneath, with absorbed sun.**
   Chosen: the equation ADR 0122 chose for the reason it gave (published, comparable), with two
   additions that leave it coefficient for coefficient the standard's when they are zero.

## Decision

**Colour** (`irsim.materials.colour`). A garment's `colour_rgb` is linear reflectance. Its
luminance (Rec. 709) is the visible reflectance ρ_vis; the NIR reflectance is ρ_nir = 0.15 +
0.60 ρ_vis (white cloth 0.75, black 0.15 — the range `cotton_clothing.yaml` already quoted);
`solar_absorptivity` = 1 − (ρ_vis + ρ_nir)/2 (the solar spectrum is about half visible, half
NIR); `emissivity_per_band['nir']` = 1 − ρ_nir. SWIR, MWIR and LWIR are the base material's,
untouched. The variant is `<base>__rgb<hex>`, a full `Material`, added to the library the
renderer packs and mapped to the garment's source material `garment_<slot>` at render time
(`render_phantom4.py`), never committed. Measured: black against white cotton, Δε_LWIR = 0,
Δα_sol = 0.48.

**Surface** (`HumanBodySolver.garment_temperature_k`, `GarmentView`). For the garment on slot
*s*: the skin beneath is JOS-3's, area-weighted over the segments the garment covers (JOS-3's
standard areas); the insulation is the garment's clo; the air, the long-wave-only radiant
temperature and the wind are the body's forcing at that tick; the garment absorbs its own
α_sol of the short-wave the body is offered (ADR 0194's `radiant_terms`). ISO 7730's balance gains
`t_skin_c` and `absorbed_solar_w_m2` as keyword arguments and is otherwise unchanged: with the
defaults it still gives 13.96 °C for 1 clo at 0 °C (ADR 0122's number). The garment is a target
`<name>.garment_<slot>`, so its prim `garment_<slot>` renders at that temperature.

**Six materials** join the library for the garments a person wears: `polyester_clothing`,
`denim`, `wool_clothing`, `leather`, `shoe_rubber`, `hair`. Cotton, nylon and polyester at
Belliveau et al. 2020's measured 0.88 over 8–12 µm; wool below them (Zhang 2009's order); leather,
rubber and hair from the spec's rubber row and vendor tables, every such number marked ESTIMATED
in its `reference`. **`cotton_clothing` stays at §16.2's 0.95** — the spec's row — while the new
fabrics carry the measurement; S66 records the 0.07 disagreement for the spec owner.

**Garments are objects.** `make_human.py --garment slot=asset` dresses a MakeHuman slot with a CC0
clothes asset, fitted and rigged by MPFB, named `garment_<slot>` with a material of the same name;
`prep_human.py` keeps it whole as a part, writes the `garments:` block (an existing config's
entries survive a re-run; new slots take ISO 9920 defaults: T-shirt 0.09 clo, trousers 0.26,
shoes 0.03) and **dyes** the garment's diffuse texture: a copy of the image, multiplied by the
colour pixel by pixel, replaces it in the material. A multiply node was tried first and is what the
glTF exporter writes as `baseColorFactor`, but Blender's USD exporter drops it, so the first render
wore an undyed grey shirt; a texture survives both exports. One YAML edit and a re-run change the
shirt in both cameras.

## Consequences

- **The NIR rule is a rule, not a measurement.** ρ_nir = 0.15 + 0.60 ρ_vis is the range the
  library already quoted for cotton; a measured dye curve (Kaur 2024's samples) would replace it
  per material. Flagged ESTIMATED in the variant's description.
- **Luminance is the whole of colour here.** Two colours of equal luminance (a saturated red and
  a saturated green) absorb the same sun and reflect the same NIR. That is right to first order for
  α_sol and wrong in detail for the NIR of specific dyes; the upgrade is per-channel curves.
- **A garment is one temperature.** One ISO 7730 node per garment, the skin beneath it the
  area-weighted mean of what it covers: a long-sleeved shirt's sleeve over a 10 °C arm and its
  front over a 28 °C chest are one number. Per-segment garments (a garment part per covered
  segment) are the upgrade, bounded by the skin spread under the garment.
- **Measured at the summer noon** (26.2 °C air, 276 W/m² offered, 2.5 m/s, chest 33.4 °C): a
  0.09 clo T-shirt reads 31.8 °C white and 33.9 °C black, **2.0 K apart**; at 0.5 clo, 29.2
  against 34.7 °C, **5.4 K**. Blue (α 0.753) and red (α 0.743) are **0.03 K** apart: the two
  shirts the owner asked for differ in RGB and NIR and not in LWIR, exactly as Zhang 2009 says.
- **The winter criterion as written was wrong, and the measurement says so.** HU.5's row asked
  for a 1 clo coat "within 3 K of background" with "face and hands brightest". On the clear
  January night (−5.4 °C air, sky 12 K colder, 1.5 m/s, two hours out): a 1 clo wool coat over
  torso and arms reads **0.4 °C, 5.8 K above the air**, the head **24.3 °C** — the brightest thing
  on him by 20 K — and the bare hands **2.6 °C**: JOS-3 lets a bare hand fall to within 8 K of the
  air in two hours, so the hands are *not* bright; gloves at 0.3 clo bring them to 7.9 °C. The
  face carries the winter scene alone, which is also what winter thermography shows. In the
  T-shirt the committed asset wears, the shirt reads 12.9 °C over a 13.1 °C chest.

## Revisit when

- A measured dye reflectance curve is wanted per garment: then `colour_rgb` becomes a spectrum
  and the NIR rule is retired.
- HU.8's soldier: NIR-compliant camouflage is a material whose NIR *does not* follow its colour,
  so it is authored as its own material and never dyed by this rule.
- A garment spans segments 10 K apart and the seam shows: then a garment part per segment.
