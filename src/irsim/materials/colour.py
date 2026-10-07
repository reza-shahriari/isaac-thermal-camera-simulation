"""A garment's colour, and what it does and does not change about its material (HU.5).

docs/physics-model.md §4.2 (α_sol is a property of the pigment), §16.2; roadmap HU.5; ADR 0195.

A blue shirt and a red shirt are the same cloth. In the long-wave bands they are the same
material: "color has no effect on surface emissivity for the same fabric" (Zhang, Hu & Zhang
2009, J. Textile Inst. 100:90, doi:10.1080/00405000701692486). What the dye changes is how much
sun the garment absorbs -- the dark shirt runs warmer, and that *is* visible in LWIR, through
temperature -- and how it reflects in the near infrared, where dye still dominates (an olive dye
23-32 % in the visible, 50-58 % in the NIR), fading in the SWIR where "all dyed samples gave
similar signatures" (Kaur et al. 2024, Appl. Spectrosc., doi:10.1177/00037028241258111).

So a colour derives a **variant of a library material** that differs in exactly two places:

- ``thermal.solar_absorptivity`` = 1 − ρ_vis, where ρ_vis is the colour's luminance in linear
  reflectance (Rec. 709 weights). The solar spectrum is about half visible and half NIR; the
  NIR half is folded in by the same rule below, so α_sol = 1 − (ρ_vis + ρ_nir)/2.
- the emissivity of every band that lies mostly below :data:`DYE_CUTOFF_UM` = 1 − ρ_nir, with
  ρ_nir = 0.15 + 0.60 ρ_vis: white cloth about 0.75, black about 0.15 -- the range
  `cotton_clothing.yaml` already quotes for the near infrared.

Every other band is left exactly as the base material has it. Which bands are "below the dye's
cut-off" is read from the band registry's nominal ranges (`irsim.config.bands`), never named here:
bands are data, and a new reflective band below 1.1 µm is dyed without a change to this file.

The variant is named ``<base>__rgb<hex>`` and is a full
:class:`~irsim.materials.library.Material`, so the packed table, the Kirchhoff walk and the
resolver treat it like any other entry.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from irsim.config.humans import HumanSpec
from irsim.materials.library import Material, MaterialLibrary

__all__ = [
    "DYE_CUTOFF_UM",
    "GARMENT_PREFIX",
    "coloured_material",
    "dress_asset_materials",
    "dyed_bands",
    "luminance",
    "dyed_reflectance",
    "variant_name",
]

#: A garment object, prim and source material are all ``garment_<slot>`` (make_human.py,
#: prep_human.py): the slot is the one key that joins the asset's `garments:` block to the mesh.
GARMENT_PREFIX = "garment_"
#: Where dye stops mattering: "all dyed samples (irrespective of colour, concentration, and dyeing
#: temperature) gave similar signatures in the SWIR", while below about 1.1 µm the dye dominates
#: (Kaur et al. 2024, Appl. Spectrosc., doi:10.1177/00037028241258111).
DYE_CUTOFF_UM = 1.1


def luminance(rgb: tuple[float, float, float]) -> float:
    """Rec. 709 luminance of a linear reflectance colour: the visible reflectance of the cloth."""
    r, g, b = (float(c) for c in rgb)
    if not all(0.0 <= c <= 1.0 for c in (r, g, b)):
        raise ValueError(f"colour {rgb} is not linear reflectance in 0..1")
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def dyed_reflectance(rho_vis: float) -> float:
    """Dyed cloth in the near infrared: 0.15 + 0.60 ρ_vis (white 0.75, black 0.15)."""
    return min(1.0, max(0.0, 0.15 + 0.60 * float(rho_vis)))


def dyed_bands(bands: Iterable[str]) -> frozenset[str]:
    """The bands whose nominal range lies mostly (by more than half) below the dye's cut-off."""
    from irsim.config.bands import nominal_range_for

    out = set()
    for band in bands:
        lo, hi = nominal_range_for(band)
        below = max(0.0, min(hi, DYE_CUTOFF_UM) - lo)
        if hi > lo and below / (hi - lo) > 0.5:
            out.add(band)
    return frozenset(out)


def variant_name(base: str, rgb: tuple[float, float, float]) -> str:
    code = "".join(f"{int(round(c * 255)):02x}" for c in rgb)
    return f"{base}__rgb{code}"


def coloured_material(base: Material, rgb: tuple[float, float, float]) -> Material:
    """``base`` dyed ``rgb``: α_sol and the dyed bands follow the colour, nothing else moves."""
    rho_vis = luminance(rgb)
    rho_dyed = dyed_reflectance(rho_vis)
    spec = base.spec
    optical = spec.optical
    if optical.emissivity_per_band is None:
        raise ValueError(
            f"{spec.name!r} authors its emissivity as a curve or a single value; a colour variant "
            "needs `emissivity_per_band` to move the dyed bands alone"
        )
    eps = dict(optical.emissivity_per_band)
    for band in dyed_bands(eps):
        eps[band] = round(1.0 - rho_dyed, 4)
    new_spec = spec.model_copy(
        update={
            "name": variant_name(spec.name, rgb),
            "source": "estimated",
            "description": (
                f"{spec.description} -- dyed rgb={tuple(round(c, 3) for c in rgb)}: "
                f"alpha_sol and the NIR reflectance follow the colour (irsim.materials.colour)"
            ),
            "thermal": spec.thermal.model_copy(
                update={"solar_absorptivity": round(1.0 - 0.5 * (rho_vis + rho_dyed), 4)}
            ),
            "optical": optical.model_copy(update={"emissivity_per_band": eps}),
        }
    )
    return Material(spec=new_spec, path=base.path, curve=base.curve, n_k_path=base.n_k_path)


def dress_asset_materials(
    materials: Mapping[str, str], human: HumanSpec, library: MaterialLibrary
) -> tuple[dict[str, str], MaterialLibrary]:
    """Point every coloured garment's source material at its dyed variant, in a library that has it.

    ``materials`` is the asset's source → library map. For each garment on slot *s* with a
    ``colour_rgb``, the source material ``garment_<s>`` is remapped to the variant of the
    garment's library material; garments with no colour keep the base material. Returns the new
    map and a library extended with the variants (the committed library is never written to).
    """
    out = dict(materials)
    extra: dict[str, Material] = {}
    for slot, garment in human.garments.items():
        if garment.colour_rgb is None or not garment.dye_optics:
            continue
        base = library[garment.material]
        variant = coloured_material(base, garment.colour_rgb)
        extra[variant.name] = variant
        out[f"{GARMENT_PREFIX}{slot}"] = variant.name
    if not extra:
        return out, library
    return out, MaterialLibrary({**{n: library[n] for n in library}, **extra})
