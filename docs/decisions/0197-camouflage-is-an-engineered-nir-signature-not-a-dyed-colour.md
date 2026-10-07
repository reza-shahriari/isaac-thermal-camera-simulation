# ADR 0197 — Camouflage is an engineered NIR signature, not a dyed colour

**Status:** Accepted
**Date:** 2026-10-07
Roadmap: HU.8 (§4.2, §12.1, §16.2). Amends [ADR 0195](0195-a-garments-colour-dyes-its-sun-and-its-nir-and-its-surface-is-solved-on-the-skin-beneath.md)
(a garment's colour dyes its sun and its NIR) with the case its "Revisit when" named. Uses
[ADR 0196](0196-an-occupation-is-garments-plus-equipment-and-retroreflective-tape-is-not-a-mirror.md)'s
equipment.

## Context

ADR 0195 lets a garment's colour derive its solar absorptance and its near-infrared reflectance:
ρ_nir = 0.15 + 0.60 ρ_vis. That is right for ordinary dyed cloth, and it is exactly what military
camouflage is engineered not to be. A dark green dye reflects little in the near infrared
(ρ_nir ≈ 0.30 for this soldier's olive by the rule), while foliage reflects up to about 0.60 over
700–1000 nm. An image-intensifier or an NIR camera then sees an ordinary green uniform as a black
figure on a bright field. Camouflage specifications therefore set NIR reflectance targets per
pattern colour, matched to the vegetation, soil and bark the pattern hides against: green
45–55 %, brown 25–40 %, beige 60–70 % and blue-black 5–20 % over 700–1100 nm (Yahaya et al. 2026,
J. Industrial Textiles, and the literature it cites). In the SWIR, dyed fabrics converge (Kaur et
al. 2024), and in the long-wave, colour does not reach a fabric's emissivity at all (Zhang, Hu &
Zhang 2009).

## Options considered

1. **Dye the camouflage like any garment.** Rejected: the rule gives ρ_nir 0.30, below the pattern's
   band. The soldier would be the NIR outlier his uniform exists not to be.
2. **A camouflage material per pattern colour, as separate parts.** Right in principle; the pattern
   is a texture, not geometry, and splitting a garment by its texture's colours is a mesh problem
   with no physics in it. Deferred.
3. **A camouflage material carrying the pattern's area-weighted mean, and a garment switch
   `dye_optics: false`** under which the colour tints the RGB companion only. Chosen.

## Decision

- `GarmentSpec.dye_optics` (default true). When false, `irsim.materials.colour` and the scene's
  absorptance leave the library material as authored. The colour still dyes the RGB texture in
  `prep_human.py`.
- `camouflage_nir_compliant`: a woodland NYCO cloth. NIR 0.45 reflected is the pattern's mean for
  about 40 % green, 30 % brown, 20 % beige and 10 % black against the targets. SWIR is 0.40
  reflected, LWIR 0.88 (Belliveau's nylon and cotton) and MWIR 0.86. All are ESTIMATED where the
  file says so.
- `helmet_aramid_painted`: a flat NIR-matched paint over an 8 mm aramid shell; ε 0.91 in LWIR. The
  helmet's insulation is the garment's clo: 0.12 on the head segment. That is the headform
  measurement of 0.029–0.055 m²K/W (0.19–0.35 clo) over the half of JOS-3's head segment the helmet
  covers; the face is the other half and is bare.
- `soldier`: combat shirt and trousers in that cloth, olive in the visible. A coyote plate carrier
  cut from the body (`make_human.py --plate-carrier`, 45 mm out), 1.38 clo on chest and back: 0.28
  shirt, plus 1.0 armour (ADR 0196), plus 0.1 carrier. An aramid helmet, leather boots, and two
  pieces of equipment: a rigger belt on the trousers and a 30 × 14 × 40 cm assault pack on the
  carrier.

## Consequences

- **The pattern is a mean.** One NIR value for a pattern whose colours span 0.05–0.70; an NIR
  camera close enough to resolve the blotches would see their contrast, and this asset shows none.
  The upgrade is option 2.
- **The RGB companion shows a uniform olive, not a pattern.** It is the same mean in the visible.
- **The generated carrier and pack are simple surfaces.** The carrier is the police vest's cut,
  further out; the pack is a box. Both read correctly to the cameras, and neither has pouches or
  straps.
- **ADR 0195's rule stands for everything else.** Only a garment that says `dye_optics: false` is
  exempt, and the test pins both sides.

## Revisit when

- Per-colour NIR curves for a real pattern are wanted. Then option 2.
- A SWIR-compliance specification is adopted. Modern specifications reach into the SWIR, and this
  entry's SWIR value is a convergence estimate, not a target.
