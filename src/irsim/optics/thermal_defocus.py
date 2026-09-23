"""Thermal defocus: the focus an unathermalised IR lens loses when it warms (`OC.10`, ADR 0129).

Germanium's dn/dT is about 396e-6 K^-1 -- roughly 250 times a visible glass -- so an LWIR lens's
focal length is a strong function of its own temperature, while the housing that holds the detector
at the back focus expands on its own, much smaller, coefficient. The two do not cancel by accident,
and the residue moves the image plane off the detector.

This is the distinctly *infrared* focus effect. It is also the one this project was already halfway
to modelling and was not using: `irsim.optics.HousingTemperature` (M9.3) solves the housing node
over a diurnal run for the self-emission term, and until now nothing else read it.

The defocus coefficient of a single-element lens in a housing (docs/physics-model.md §8.3 does not
give this; the form is the standard one from IR optical design):

    dz/dT = f * [ alpha_housing - ( (dn/dT)/(n - 1) - alpha_lens ) ]

The bracketed term is the **thermo-optic coefficient** of the lens material, often written beta. A
lens is *passively athermalised* when the housing's expansion cancels it -- which is what an
aluminium barrel is chosen to do over a stated range, and what pairing germanium with a negative-
going chalcogenide does inside the lens. `athermal=True` models that as an exact cancellation,
which is the right level for a camera whose data sheet quotes "thermal gradient compensation" and
gives no residual.

Longitudinal defocus becomes wavefront error by ``W020 = dz / (8 F^2)``, the same relation
`irsim.optics.defocus` uses in the form ``W020 = c / (8F)`` with ``c = dz / F``.
"""

from __future__ import annotations

from typing import Literal

from irsim.radiometry.constants import (
    CTE_ALUMINIUM,
    CTE_AMTIR1,
    CTE_GERMANIUM,
    CTE_INVAR,
    CTE_SILICON,
    CTE_STEEL,
    CTE_TITANIUM,
    CTE_ZNSE,
    DN_DT_AMTIR1,
    DN_DT_GERMANIUM,
    DN_DT_SILICON,
    DN_DT_ZNSE,
    N_AMTIR1_10UM,
    N_GERMANIUM_10UM,
    N_SILICON_10UM,
    N_ZNSE_10UM,
)

__all__ = [
    "HOUSING_CTE",
    "effective_focus_distance_m",
    "LENS_MATERIALS",
    "LensMaterial",
    "HousingMaterial",
    "thermal_defocus_um",
    "thermal_defocus_w020_um",
    "thermo_optic_coefficient",
]

LensMaterial = Literal["germanium", "silicon", "zinc_selenide", "amtir1"]
HousingMaterial = Literal["aluminium", "steel", "titanium", "invar"]

#: (dn/dT, n, CTE) per lens material.
LENS_MATERIALS: dict[str, tuple[float, float, float]] = {
    "germanium": (DN_DT_GERMANIUM, N_GERMANIUM_10UM, CTE_GERMANIUM),
    "silicon": (DN_DT_SILICON, N_SILICON_10UM, CTE_SILICON),
    "zinc_selenide": (DN_DT_ZNSE, N_ZNSE_10UM, CTE_ZNSE),
    "amtir1": (DN_DT_AMTIR1, N_AMTIR1_10UM, CTE_AMTIR1),
}
HOUSING_CTE: dict[str, float] = {
    "aluminium": CTE_ALUMINIUM,
    "steel": CTE_STEEL,
    "titanium": CTE_TITANIUM,
    "invar": CTE_INVAR,
}


def thermo_optic_coefficient(lens_material: str = "germanium") -> float:
    """beta = (dn/dT)/(n - 1) - alpha_lens, K^-1 -- how fast the lens loses focal length.

    Germanium comes out at about 126e-6 K^-1, which is what makes it the material that has to be
    athermalised; ZnSe is a third of that and AMTIR-1 lower still, which is why they are the usual
    partners in a passively athermal pair.
    """
    try:
        dn_dt, n, cte = LENS_MATERIALS[lens_material]
    except KeyError:
        raise ValueError(
            f"unknown lens material {lens_material!r}; expected one of {sorted(LENS_MATERIALS)}"
        ) from None
    return dn_dt / (n - 1.0) - cte


def thermal_defocus_um(
    focal_length_mm: float,
    delta_t_k: float,
    lens_material: str = "germanium",
    housing_material: str = "aluminium",
) -> float:
    """Longitudinal image-plane shift (µm) for a temperature change of ``delta_t_k``.

    Signed: positive means the image forms **behind** the detector. A germanium lens in an
    aluminium barrel comes out at −103e-6 per kelvin of its focal length, so a 14 mm Boson lens
    moves −1.44 µm per kelvin — about −29 µm over a 20 K rise, which at F/1.0 is a third of a wave
    and well past Rayleigh's quarter. That is why nobody sells one unathermalised.
    """
    if focal_length_mm <= 0.0:
        raise ValueError("focal length must be positive")
    if housing_material not in HOUSING_CTE:
        raise ValueError(
            f"unknown housing material {housing_material!r}; expected one of {sorted(HOUSING_CTE)}"
        )
    beta = thermo_optic_coefficient(lens_material)
    net = HOUSING_CTE[housing_material] - beta
    return float(focal_length_mm * 1e3 * delta_t_k * net)


def thermal_defocus_w020_um(  # noqa: PLR0913
    focal_length_mm: float,
    f_number: float,
    delta_t_k: float,
    lens_material: str = "germanium",
    housing_material: str = "aluminium",
    athermal: bool = True,
) -> float:
    """Signed wavefront defocus (µm of W020) from a temperature change; 0 when athermalised.

    ``W020 = dz / (8 F^2)``. The sign is kept because thermal defocus and scene defocus are the
    same Zernike term and add algebraically -- a warm lens looking at something nearer than its
    focus setting is *less* defocused than a cold one, not more, and a model that took magnitudes
    would get that backwards.
    """
    if f_number <= 0.0:
        raise ValueError("f-number must be positive")
    if athermal:
        return 0.0
    dz_um = thermal_defocus_um(focal_length_mm, delta_t_k, lens_material, housing_material)
    return float(dz_um / (8.0 * f_number * f_number))


def effective_focus_distance_m(
    focus_distance_m: float | None, focal_length_mm: float, defocus_um: float
) -> float | None:
    """Where the lens is *actually* focused once a longitudinal defocus has moved the image plane.

    This is the whole wiring of `OC.10`. A thermal image-plane shift is not a new kind of blur: it
    is the same defocus as looking at the wrong distance, so folding it into an effective focus
    distance means every stage downstream -- the global kernel, the layered composite, the
    autofocus servo -- gets it for free and none of them needs a thermal term of its own.

    The sensor sits at the back focus ``v = f s/(s - f)``; a shift ``dz`` puts it at ``v - dz``,
    and the distance that now images there is ``f v' / (v' - f)``. A shift that pushes focus past
    infinity returns ``None``, which is what every other part of this package means by infinity.

    Measured for a 14 mm F/1.0 germanium lens in an aluminium barrel: a **20 K rise moves a lens
    focused at infinity to being focused at 6.8 m** -- inside its own hyperfocal distance, so
    distant objects go soft. This is the effect `HousingTemperature` has been solving the input for
    since M9.3 with nothing reading it.
    """
    if focal_length_mm <= 0.0:
        raise ValueError("focal length must be positive")
    f_mm = float(focal_length_mm)
    if defocus_um == 0.0:
        return focus_distance_m
    v = (
        f_mm
        if focus_distance_m is None
        else f_mm * focus_distance_m * 1e3 / (focus_distance_m * 1e3 - f_mm)
    )
    v_prime = v - defocus_um * 1e-3
    if v_prime <= f_mm:
        return None  # focused at or beyond infinity
    return float(f_mm * v_prime / (v_prime - f_mm) * 1e-3)
