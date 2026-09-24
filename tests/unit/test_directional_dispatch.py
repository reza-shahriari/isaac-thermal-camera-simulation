"""M7.7 — one entry point for ε(θ), dispatching on the level the *material* declares (§4.2, §13.3).

Level A is Fresnel from a checked-in n/k table, B is the baked empirical falloff of M7.6, C is a
constant. The choice belongs to the material, not to the caller, and the CPU answer here is the
oracle the Warp/SPG kernel's ε(θ) is compared against.

§4.2 permits Level C "only for rough, high-emissivity surfaces (ε > 0.93)" and names vehicle
bodies, glass and water as violating it. That bound is **enforced, not warned about**, because the
error it hides is large and one-sided — a constant ε renders every limb too warm, everywhere, in a
way that looks like a perfectly plausible scene. Enforcing it immediately found one violation in
the committed library, which this module pins.

docs/physics-model.md §4.2, §13.3, §13.5; ADR 0042
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.materials.angular import (
    FIT_MAX_ANGLE_DEG,
    emissivity_empirical,
    fit_band_level_b,
    fit_from_nk,
)
from irsim.materials.directional import (
    LEVEL_C_MIN_EPSILON,
    angular_level,
    directional_emissivity,
)
from irsim.materials.library import MaterialLibrary, nominal_response
from irsim.materials.nk import band_directional_emissivity, load_nk_table
from irsim.radiometry.spectral_response import load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
COS_70 = math.cos(math.radians(FIT_MAX_ANGLE_DEG))


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load(REPO / "configs" / "materials")


@pytest.fixture(scope="module")
def boson():  # type: ignore[no-untyped-def]
    return load_spectral_response(DATA / "spectra" / "responses" / "boson_vox.csv")


def _dispatch(material, band, cos_theta, boson):  # type: ignore[no-untyped-def]
    return directional_emissivity(material, band, cos_theta, boson, data_dir=DATA)


# ---------------------------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------------------------


def test_every_committed_material_declares_a_level(library) -> None:  # type: ignore[no-untyped-def]
    levels = {name: angular_level(library[name]) for name in library.names}
    assert set(levels.values()) <= {"A", "B", "C"}
    assert "B" in levels.values(), "nothing uses the empirical model M7.6 exists to bake"


def test_the_dispatch_preserves_float32(library, boson) -> None:  # type: ignore[no-untyped-def]
    material = library["car_paint_black"]
    cos_theta = np.linspace(1.0, 0.0, 17, dtype=np.float32)
    out = _dispatch(material, "lwir", cos_theta, boson)
    assert out.dtype == np.float32
    assert out.shape == cos_theta.shape
    with pytest.raises(TypeError, match="float16"):
        _dispatch(material, "lwir", cos_theta.astype(np.float16), boson)


def test_a_back_facing_normal_uses_the_absolute_cosine(library, boson) -> None:  # type: ignore[no-untyped-def]
    material = library["car_paint_black"]
    front = _dispatch(material, "lwir", np.float32(0.4), boson)
    back = _dispatch(material, "lwir", np.float32(-0.4), boson)
    assert float(front) == float(back)


def test_the_answer_is_finite_at_grazing(library, boson) -> None:  # type: ignore[no-untyped-def]
    """cos θ = 0 must be a number. A formulation that divided by cos θ would not be."""
    for name in library.names:
        material = library[name]
        if (
            angular_level(material) == "C"
            and material.band_properties("lwir").emissivity < LEVEL_C_MIN_EPSILON
        ):
            continue
        value = _dispatch(material, "lwir", np.float32(0.0), boson)
        assert np.isfinite(value).all(), name
        assert 0.0 <= float(value) <= 1.0, name


def test_normal_incidence_returns_the_band_value(library, boson) -> None:  # type: ignore[no-untyped-def]
    """ε(0) must be the ε_B every other consumer of the material already uses."""
    for name in library.names:
        material = library[name]
        if (
            angular_level(material) == "C"
            and material.band_properties("lwir").emissivity < LEVEL_C_MIN_EPSILON
        ):
            continue
        expected = float(material.band_properties("lwir", boson).emissivity)
        assert float(_dispatch(material, "lwir", np.float32(1.0), boson)) == pytest.approx(
            expected, rel=1e-6
        ), name


def test_the_limb_is_darker_than_the_centre_for_every_dielectric(library, boson) -> None:  # type: ignore[no-untyped-def]
    for name in library.names:
        material = library[name]
        if angular_level(material) != "B":
            continue
        centre = float(_dispatch(material, "lwir", np.float32(1.0), boson))
        limb = float(_dispatch(material, "lwir", np.float32(COS_70), boson))
        assert limb < centre, name


# ---------------------------------------------------------------------------------------------
# Level C's bound, enforced
# ---------------------------------------------------------------------------------------------


def _with_constant_angular(material):  # type: ignore[no-untyped-def]
    """The same material, forced to Level C, so the guard can be exercised on a real ε."""
    from irsim.config.materials import ConstantAngular

    optical = material.spec.optical.model_copy(
        update={"angular_model": ConstantAngular(type="constant")}
    )
    return type(material)(
        spec=material.spec.model_copy(update={"optical": optical}),
        path=material.path,
        spectrum=material.spectrum,
        n_k_path=material.n_k_path,
    )


def test_level_c_is_refused_below_the_section_4_2_bound(library, boson) -> None:  # type: ignore[no-untyped-def]
    """⚠️ Enforcing this bound changed two committed materials, which is the point of enforcing it.

    `bare_aluminium` declared `constant` at ε_LWIR = 0.09 and `asphalt_dry` at ε_NIR = 0.92, both
    under §4.2's 0.93. Aluminium is now Level A — a metal's ε *rises* with angle, so Level B
    cannot represent it at any (a, p) either — and asphalt is Level B with a ≈ 0, which is what
    §4.2 itself prescribes for rough dielectrics. The guard is exercised here on a material forced
    back to Level C, so it keeps its ability to fail now that nothing in the library trips it.
    """
    forced = _with_constant_angular(library["glass_windshield"])
    assert float(forced.band_properties("lwir", boson).emissivity) < LEVEL_C_MIN_EPSILON
    with pytest.raises(ValueError, match="§4.2"):
        _dispatch(forced, "lwir", np.float32(1.0), boson)


def test_level_c_is_allowed_for_a_rough_high_emissivity_surface(library, boson) -> None:  # type: ignore[no-untyped-def]
    """The other side of the bound: human skin at ε = 0.98 is exactly what a constant is for."""
    forced = _with_constant_angular(library["human_skin"])
    assert float(forced.band_properties("lwir", boson).emissivity) >= LEVEL_C_MIN_EPSILON
    values = _dispatch(forced, "lwir", np.linspace(1.0, 0.0, 9, dtype=np.float32), boson)
    assert len(set(values.tolist())) == 1, "a constant model must return one value, bit for bit"


def test_no_committed_material_relies_on_an_illegal_constant(library, boson) -> None:
    """A survey, so a new material that gets this wrong fails here and not in a render."""
    for name in library.names:
        material = library[name]
        if angular_level(material) != "C":
            continue
        for band in ("nir", "swir", "mwir", "lwir"):
            epsilon = float(material.band_properties(band).emissivity)
            assert epsilon >= LEVEL_C_MIN_EPSILON, f"{name} band {band}: ε = {epsilon}"


def test_the_bare_metals_are_the_only_materials_whose_emissivity_rises_with_angle(
    library, boson
) -> None:  # type: ignore[no-untyped-def]
    """§4.2's dielectric/metal split, as a property of the whole committed library.

    Bare aluminium goes 0.090 at normal to 0.146 at 70°; everything else falls. This is why it
    needs Level A: the sign of its angular slope is one Level B cannot produce.

    `AT.17` made the set two rather than one, and the *third* aluminium is the interesting case.
    `aluminium_polished` rises, because it is the same exposed metal with less oxide on it.
    `aluminium_anodised` does **not**: its emitting surface is micrometres of oxide, a rough
    dielectric, so it falls toward grazing like a paint does even though the substance beneath it
    is the same metal. A library keyed on substance could not hold that; one keyed on surface
    state (§4.5) does, and this assertion is what says so.
    """
    rising = []
    for name in library.names:
        material = library[name]
        centre = float(_dispatch(material, "lwir", np.float32(1.0), boson))
        limb = float(_dispatch(material, "lwir", np.float32(COS_70), boson))
        if limb > centre:
            rising.append(name)
    assert sorted(rising) == ["aluminium_polished", "bare_aluminium"], rising
    assert "aluminium_anodised" not in rising
    aluminium = library["bare_aluminium"]
    assert float(_dispatch(aluminium, "lwir", np.float32(1.0), boson)) == pytest.approx(
        0.09, abs=1e-6
    )
    assert float(_dispatch(aluminium, "lwir", np.float32(COS_70), boson)) == pytest.approx(
        0.146, abs=0.01
    )


def test_level_a_takes_its_magnitude_from_the_authored_band_value(library, boson) -> None:  # type: ignore[no-untyped-def]
    """⚠️ The table supplies the **shape**; the material supplies the **magnitude**.

    Ideal Drude aluminium is ε = 0.012 at 10 µm and §16.2 gives bare aluminium 0.09 — an oxide
    layer and a little roughness are worth almost an order of magnitude. Taking the table's
    absolute value would fix a ~30 % error in angular shape by introducing an 8× error in the
    emissivity itself. Scaling also keeps ε(0) meaning the same thing at all three levels.
    """
    from irsim.materials.nk import band_directional_emissivity, load_nk_table

    aluminium = library["bare_aluminium"]
    authored = float(aluminium.band_properties("lwir", boson).emissivity)
    table = load_nk_table(str(aluminium.n_k_path), DATA)
    raw = float(band_directional_emissivity(table, boson, 1.0))
    assert raw < 0.02, "the ideal-metal model should be far below the authored value"
    assert authored / raw > 5.0
    assert float(_dispatch(aluminium, "lwir", np.float32(1.0), boson)) == pytest.approx(
        authored, abs=1e-6
    )


# ---------------------------------------------------------------------------------------------
# Level A against baked Level B
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow  # GT.1: over a second on its own
def test_baked_level_b_reproduces_level_a_on_the_sphere_fixture(
    gbuffer_sphere: dict[str, np.ndarray], boson
) -> None:  # type: ignore[no-untyped-def]
    """The acceptance criterion, on real geometry: agreement to 0.02 wherever cos θ > cos 70°.

    Water, because it is the one material with a checked-in n/k table (M7.5) and because a sea
    surface is the case the whole angular model exists for — its ε collapses from 0.99 at nadir
    to 0.70 at 80°, which no constant can describe.
    """
    table = load_nk_table("water", DATA)
    cos_theta = np.asarray(gbuffer_sphere["normal_dot_view"], dtype=np.float64)
    inside = cos_theta > COS_70
    assert inside.sum() > 100, "the fixture does not reach enough of the fitted range"

    level_a = band_directional_emissivity(table, boson, cos_theta[inside])
    fit = fit_band_level_b(table, boson, "water")
    level_b = emissivity_empirical(fit.epsilon_0, fit.a, fit.p, cos_theta[inside])
    assert float(np.max(np.abs(level_a - level_b))) < 0.02

    # ⚠️ and the fit must see the *band*, not a representative wavelength. Fitting at 10 µm and
    # comparing against the band average is a real error: 0.0275 at 70°, past the criterion.
    single = fit_from_nk(
        float(np.interp(10.0, table.wavelength_um, table.n)),
        float(np.interp(10.0, table.wavelength_um, table.k)),
        "water at 10 um",
    )
    single_b = emissivity_empirical(single.epsilon_0, single.a, single.p, cos_theta[inside])
    assert float(np.max(np.abs(level_a - single_b))) > 0.02


def test_the_sphere_fixture_reaches_grazing_where_the_two_levels_must_diverge(
    gbuffer_sphere: dict[str, np.ndarray], boson
) -> None:
    """Outside 70° the forms part company, which is why the fit stops there and why the
    agreement above is claimed only inside it."""
    table = load_nk_table("water", DATA)
    cos_theta = np.asarray(gbuffer_sphere["normal_dot_view"], dtype=np.float64)
    grazing = cos_theta[(cos_theta > 0.0) & (cos_theta < 0.1)]
    assert grazing.size > 0, "the fixture never reaches grazing incidence"
    level_a = band_directional_emissivity(table, boson, grazing)
    assert float(level_a.min()) < 0.75, "water's angular collapse is missing"


# --- glass takes Level A (M7.5) -----------------------------------------------------------------


def test_glass_declares_level_a(library) -> None:  # type: ignore[no-untyped-def]
    assert angular_level(library["glass_windshield"]) == "A"


def test_level_a_keeps_the_authored_band_emissivity_at_normal(library, boson) -> None:  # type: ignore[no-untyped-def]
    """ε(0) is the value in the YAML, not the value in the table. The table is a *shape*.

    This is the convention M7.5 established on aluminium, and glass is where it earns its keep in
    the other direction: the fused-silica table gives ε ≈ 0.81 at 10 µm where the library authors
    0.88 for soda-lime. Letting the table set the magnitude would silently replace a soda-lime
    windshield with a quartz one, and ε(0) would mean one thing for a Fresnel material and another
    for every other material in the library.
    """
    glass = library["glass_windshield"]
    for band in ("nir", "swir", "mwir", "lwir"):
        authored = float(glass.band_properties(band, boson).emissivity)
        got = float(_dispatch(glass, band, np.float32(1.0), boson))
        assert got == pytest.approx(authored, abs=1e-6), (band, got, authored)


def test_glass_limb_darkens_and_monotonically(library, boson) -> None:  # type: ignore[no-untyped-def]
    """§4.2's headline effect: a curved dielectric shows a cooler rim at uniform temperature."""
    glass = library["glass_windshield"]
    cos_theta = np.cos(np.radians(np.linspace(0.0, 89.0, 24))).astype(np.float32)
    for band in ("nir", "swir", "mwir", "lwir"):
        eps = np.asarray(_dispatch(glass, band, cos_theta, boson), dtype=np.float64)
        assert np.all(np.diff(eps) <= 1e-7), band  # never rises: glass is not a metal
        assert eps[-1] < 0.75 * eps[0], (band, eps[-1] / eps[0])
        assert np.all(eps >= 0.0) and np.all(eps <= 1.0), band


def test_glass_needs_a_per_band_shape_that_one_level_b_cannot_give(library, boson) -> None:  # type: ignore[no-untyped-def]
    """The reason glass is Level A rather than a fitted Level B, stated as a measurement.

    `angular_model` is one setting for the whole material, so a Level B glass gets a single
    (a, p) for all four bands. The Si-O reststrahlen band sits inside the LWIR window and nowhere
    near the other three, so the LWIR shape is genuinely different: fitting each band separately
    gives a ≈ 1.4 in LWIR against 0.6-0.7 elsewhere. Level A gets this for free because it
    integrates the table over each band's own response.

    A fitted a > 1 is also not merely inaccurate. ε₀(1 − a(1−cos θ)^p) crosses zero at 85.1° and
    is clipped there, so beyond that angle the model reports a windshield edge as having *no*
    emissivity at all -- a perfect mirror -- while Level A still has 0.37 of normal at 85° and
    0.18 at 88°. Those are the angles a windshield is seen at from across a street.
    """
    glass = library["glass_windshield"]
    table = load_nk_table("glass", DATA)
    fits = {
        band: fit_band_level_b(table, nominal_response(band), name=f"glass:{band}")
        for band in ("nir", "swir", "mwir", "lwir")
    }
    assert fits["lwir"].a > 1.0, fits["lwir"]
    assert max(fits[b].a for b in ("nir", "swir", "mwir")) < 0.8

    # The consequence, measured where it bites. The fit is good to 70° by construction (that is
    # the range §4.2 fits over); what it cannot do is carry past it.
    e0 = float(glass.band_properties("lwir", boson).emissivity)
    crossing_deg = math.degrees(math.acos(1.0 - (1.0 / fits["lwir"].a) ** (1.0 / fits["lwir"].p)))
    assert 84.0 < crossing_deg < 86.0, crossing_deg
    for angle, max_ratio in ((80.0, 0.65), (85.0, 0.10)):
        cos_theta = np.float32(math.cos(math.radians(angle)))
        level_a = float(_dispatch(glass, "lwir", cos_theta, boson))
        level_b = float(emissivity_empirical(e0, fits["lwir"].a, fits["lwir"].p, float(cos_theta)))
        assert level_a > 0.3 * e0, (angle, level_a)  # Level A stays a real emitter
        assert level_b < max_ratio * level_a, (angle, level_a, level_b)


def test_level_a_leaves_kirchhoff_closable_on_a_semi_transparent_material(library, boson) -> None:  # type: ignore[no-untyped-def]
    """ε(θ) + τ must stay under 1, or the derived ρ goes negative (CLAUDE.md #4).

    Glass is the library's only semi-transparent material, and it is now the only one whose ε
    varies with angle from a table rather than from a formula bounded by construction. ρ is
    derived as 1 − ε(θ) − τ and clamped at zero downstream, so a violation would not raise -- it
    would quietly stop conserving energy in the NIR and SWIR, where τ is 0.77 and 0.70.
    """
    glass = library["glass_windshield"]
    cos_theta = np.cos(np.radians(np.linspace(0.0, 90.0, 46))).astype(np.float32)
    for band in ("nir", "swir", "mwir", "lwir"):
        props = glass.band_properties(band, boson)
        eps = np.asarray(_dispatch(glass, band, cos_theta, boson), dtype=np.float64)
        rho = 1.0 - eps - float(props.transmittance)
        assert np.all(rho >= -1e-9), (band, float(rho.min()))
        assert np.all(rho <= 1.0 + 1e-9), band
        # And ε(θ) only ever frees up more reflectance, never less, as the angle opens.
        assert np.all(np.diff(rho) >= -1e-9), band


# --- the painted class is fitted, not estimated (M7.5 paint proxy) --------------------------------

PAINTED = ("car_paint_black", "car_paint_white", "aircraft_aluminium_painted", "painted_composite")


def test_the_painted_class_carries_the_fitted_parameters(library) -> None:  # type: ignore[no-untyped-def]
    """Every painted material takes the same (a, p), and it is the one the proxy produces.

    §4.2 says to fit (a, p) once per *class* and bake it, so the four sharing one value is the
    point, not a coincidence -- and a fifth painted material added later should join them rather
    than acquire an estimate of its own.
    """
    for name in PAINTED:
        model = library[name].spec.optical.angular_model
        assert model.type == "empirical", name
        assert (model.a, model.p) == (0.75, 4.0), (name, model.a, model.p)


def test_the_baked_parameters_are_what_the_proxy_actually_fits(library) -> None:  # type: ignore[no-untyped-def]
    """Re-derive the bake from the checked-in table: the YAML must not drift from its source.

    This is the test that makes the numbers in four YAML files traceable. Fitting the paint proxy
    band by band gives a = 0.732-0.770 with p pinned at 4; the baked 0.75 sits inside that spread,
    so an edit that moved it back toward §4.2's quoted 0.15-0.35 -- or a re-fetch that changed the
    table under it -- fails here rather than quietly flattening every painted limb in the library.
    """
    table = load_nk_table("paint_proxy", DATA)
    fits = {
        band: fit_band_level_b(table, nominal_response(band), name=f"paint:{band}")
        for band in ("nir", "swir", "mwir", "lwir")
    }
    assert {f.p for f in fits.values()} == {4.0}, fits
    values = [f.a for f in fits.values()]
    assert min(values) > 0.72 and max(values) < 0.78, fits
    baked = library["car_paint_black"].spec.optical.angular_model
    assert min(values) - 0.02 <= baked.a <= max(values) + 0.02, (baked.a, values)
    assert all(f.rms_residual < 0.02 for f in fits.values()), fits


def test_the_fit_is_far_from_the_estimate_it_replaced(library, boson) -> None:  # type: ignore[no-untyped-def]
    """The size of the correction, because "we re-fitted it" understates what changed.

    The estimated (0.25, 5.0) held ε at 0.97 of its normal value at 70°, where Fresnel on an
    acrylic gives 0.86. Against a 250 K sky that is 3.8 K of apparent temperature on a car door
    seen obliquely -- comfortably past the 2 K Tier 4 target, so it is not a refinement.
    """
    eps0 = float(library["car_paint_black"].band_properties("lwir", boson).emissivity)
    cos_70 = math.cos(math.radians(70.0))
    estimate = float(emissivity_empirical(eps0, 0.25, 5.0, cos_70))
    fitted = float(emissivity_empirical(eps0, 0.75, 4.0, cos_70))
    assert estimate / eps0 > 0.96, estimate / eps0
    assert 0.85 < fitted / eps0 < 0.87, fitted / eps0
    assert estimate - fitted > 0.09, (estimate, fitted)


def test_the_unpainted_materials_did_not_borrow_the_paint_fit(library) -> None:  # type: ignore[no-untyped-def]
    """Carbon fibre and rubber keep their estimate: the proxy is PMMA, and they are not painted.

    They sat on the same estimated (0.25, 5.0) as the paints, so the tempting move was to sweep
    them along. That would have borrowed a measurement of a different material without saying so,
    which is exactly what ADR 0041's PROXY label exists to prevent.
    """
    for name in ("carbon_fibre", "propeller_rubber"):
        model = library[name].spec.optical.angular_model
        assert (model.a, model.p) == (0.25, 5.0), (name, model.a, model.p)
