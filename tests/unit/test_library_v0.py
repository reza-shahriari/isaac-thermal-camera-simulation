"""M7.9 — the fifteen §16.2 materials, with provenance, and what separates the bands (§16.2, §12.3).

The §16.2 table gives ε_LWIR, ε_MWIR, α_sol and the three thermal properties, and nothing about
NIR or SWIR. Every NIR/SWIR value in the library is therefore ESTIMATED and labelled so — but two
of them are not guesses at all, and they are the reason a multi-band library is worth having:

* a **leaf** in the near-infrared reflects ~0.45 and *transmits* ~0.45, absorbing a tenth. That is
  the red edge, and it makes a canopy brilliant and semi-transparent at 0.9 µm while being an
  ordinary opaque high-ε dielectric at 10 µm.
* **snow** reflects ~0.85 at 0.9 µm and ~0.10 at 1.6 µm, because ice absorbs strongly in the SWIR.
  A swing of 0.75 in ε between adjacent bands — larger than any other material here — and the
  reason snow and cloud are indistinguishable in the visible and obvious at 1.6 µm.

A single-band library cannot be wrong about either, because it cannot say anything about them.

docs/physics-model.md §16.2, §12.3, §4.3, §4.4; ADR 0040, ADR 0043
"""

from __future__ import annotations

import pathlib

import pytest

from irsim.config.bands import BAND_IDS
from irsim.materials.directional import angular_level
from irsim.materials.library import MaterialLibrary

REPO = pathlib.Path(__file__).resolve().parents[2]
CLOSURE_TOL = 1e-6

#: §16.2's table, verbatim: (ε_LWIR, ε_MWIR, α_sol, ρ, c_p, k) keyed by this library's name.
SECTION_16_2: dict[str, tuple[float, float, float, float, float, float]] = {
    "asphalt_dry": (0.94, 0.92, 0.90, 2200, 920, 0.75),
    "concrete": (0.92, 0.90, 0.65, 2300, 880, 1.4),
    "car_paint_black": (0.90, 0.88, 0.94, 7800, 470, 45),
    "car_paint_white": (0.90, 0.88, 0.28, 7800, 470, 45),
    "bare_aluminium": (0.09, 0.06, 0.15, 2700, 900, 205),
    "rusted_steel": (0.85, 0.82, 0.80, 7800, 470, 45),
    "glass_windshield": (0.88, 0.85, 0.10, 2500, 840, 1.0),
    "rubber_tyre": (0.95, 0.94, 0.94, 1100, 2000, 0.16),
    "human_skin": (0.98, 0.97, 0.65, 1050, 3500, 0.37),
    "cotton_clothing": (0.95, 0.93, 0.70, 300, 1300, 0.06),
    "vegetation_leaf": (0.97, 0.96, 0.50, 700, 3000, 0.30),
    "soil_dry": (0.92, 0.90, 0.75, 1500, 800, 0.30),
    "soil_wet": (0.96, 0.95, 0.85, 2000, 1500, 1.5),
    "water": (0.96, 0.98, 0.93, 1000, 4180, 0.60),
    "snow": (0.99, 0.98, 0.15, 300, 2100, 0.20),
}


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load(REPO / "configs" / "materials")


def test_all_fifteen_section_16_2_materials_are_present(library) -> None:  # type: ignore[no-untyped-def]
    missing = set(SECTION_16_2) - set(library.names)
    assert not missing, f"§16.2 materials still unauthored: {sorted(missing)}"
    assert len(library.names) >= 15


@pytest.mark.parametrize("name", sorted(SECTION_16_2))
def test_each_material_reproduces_its_section_16_2_row(name: str, library) -> None:  # type: ignore[no-untyped-def]
    """The thermal-band ε and every thermal property, exactly. ADR 0043 reads the ε columns as
    normal incidence, which is what `band_properties` returns."""
    eps_lwir, eps_mwir, alpha, rho, cp, k = SECTION_16_2[name]
    material = library[name]
    assert material.band_properties("lwir").emissivity == pytest.approx(eps_lwir, abs=1e-6)
    assert material.band_properties("mwir").emissivity == pytest.approx(eps_mwir, abs=1e-6)
    thermal = material.spec.thermal
    assert thermal.solar_absorptivity == pytest.approx(alpha, abs=1e-9)
    assert thermal.density_kg_m3 == pytest.approx(rho, rel=1e-9)
    assert thermal.specific_heat_j_kgk == pytest.approx(cp, rel=1e-9)
    assert thermal.conductivity_w_mk == pytest.approx(k, rel=1e-9)


@pytest.mark.parametrize("name", sorted(SECTION_16_2))
def test_each_material_declares_its_provenance(name: str, library) -> None:  # type: ignore[no-untyped-def]
    """`source` is required by the schema; this checks it is *used*, and that anything estimated
    says so. A number nobody can trace is a number nobody can correct."""
    material = library[name]
    assert material.spec.source in ("measured", "literature", "estimated")
    reference = (material.spec.reference or "").lower()
    assert "16.2" in reference or "section" in reference, name
    text = material.path.read_text(encoding="utf-8").lower()
    assert "estimated" in text or material.spec.source == "measured", name


def test_every_material_closes_in_every_band(library) -> None:  # type: ignore[no-untyped-def]
    """CLAUDE.md #4, over the whole library and all four bands."""
    for name in library:
        for band in BAND_IDS:
            props = library[name].band_properties(band)
            closure = props.emissivity + props.reflectance + props.transmittance
            assert abs(closure - 1.0) < CLOSURE_TOL, (name, band, closure)
            assert 0.0 <= props.reflectance <= 1.0
            assert 0.0 <= props.transmittance <= 1.0


def test_every_material_has_a_valid_angular_model(library) -> None:  # type: ignore[no-untyped-def]
    """After M7.7's §4.2 guard, a Level C material must clear ε = 0.93 in **every** band."""
    for name in library.names:
        material = library[name]
        if angular_level(material) != "C":
            continue
        for band in BAND_IDS:
            assert material.band_properties(band).emissivity >= 0.93, (name, band)


# ---------------------------------------------------------------------------------------------
# what a multi-band library can say that a single-band one cannot
# ---------------------------------------------------------------------------------------------


def test_a_leaf_is_semi_transparent_in_the_near_infrared_and_opaque_in_lwir(library) -> None:  # type: ignore[no-untyped-def]
    """The red edge: a canopy is not opaque at 0.9 µm, and is at 10 µm."""
    leaf = library["vegetation_leaf"]
    nir = leaf.band_properties("nir")
    lwir = leaf.band_properties("lwir")
    assert nir.transmittance > 0.4, "a leaf must transmit in the NIR plateau"
    assert nir.emissivity < 0.15, "a leaf absorbs little in the NIR"
    assert lwir.transmittance == 0.0
    assert lwir.emissivity > 0.95
    # and the SWIR sits between them, because leaf water has begun to absorb
    swir = leaf.band_properties("swir")
    assert nir.transmittance > swir.transmittance > lwir.transmittance


def test_snow_swings_further_between_adjacent_bands_than_anything_else(library) -> None:  # type: ignore[no-untyped-def]
    """ε 0.15 in NIR and 0.90 in SWIR — the snow/cloud discriminator, as a material property."""
    snow = library["snow"]
    nir = snow.band_properties("nir").emissivity
    swir = snow.band_properties("swir").emissivity
    assert swir - nir > 0.7
    swings = {
        name: abs(
            library[name].band_properties("swir").emissivity
            - library[name].band_properties("nir").emissivity
        )
        for name in library.names
    }
    assert max(swings, key=lambda n: swings[n]) == "snow", swings


def test_glass_shows_itself_in_lwir_and_what_is_behind_it_in_swir(library) -> None:  # type: ignore[no-untyped-def]
    """§4.4's headline, and the row §16.2 says is most likely to break a naïve simulator."""
    glass = library["glass_windshield"]
    assert glass.band_properties("swir").transmittance > 0.5
    assert glass.band_properties("lwir").transmittance == 0.0
    assert glass.band_properties("lwir").emissivity > 0.85


def test_bare_metal_is_the_other_row_that_breaks_naive_simulators(library) -> None:  # type: ignore[no-untyped-def]
    """§16.2: "aluminium because ε = 0.09 means it is a mirror, not a surface".

    `AT.17` made this a rule rather than an enumeration. Every material in the library that is
    nearly a mirror must be a **bare metal** — a surface state of `polished` or `oxidised`, and
    Level A, because a metal's ε rises toward grazing and no Level B (a, p) can produce that sign.
    Stating it as "everything except this one name" would let the next low-emissivity material in
    without a word; stating it as a property of the surface state is what §4.5 is for.
    """
    aluminium = library["bare_aluminium"]
    assert aluminium.band_properties("lwir").reflectance == pytest.approx(0.91, abs=1e-9)
    assert angular_level(aluminium) == "A", "a metal needs Fresnel; Level B cannot rise with angle"
    mirrors = [n for n in library.names if library[n].band_properties("lwir").emissivity <= 0.2]
    assert sorted(mirrors) == ["aluminium_polished", "bare_aluminium"], mirrors
    for name in mirrors:
        assert library[name].spec.surface_treatment in ("polished", "oxidised"), name
        assert angular_level(library[name]) == "A", name
    # And the third aluminium is not one, though it is the same metal: the anodic oxide is the
    # surface, so it reads as a near-blackbody and takes a dielectric's angular model.
    anodised = library["aluminium_anodised"]
    assert anodised.band_properties("lwir").emissivity > 0.8
    assert angular_level(anodised) != "A"


def test_the_wet_and_dry_soil_pair_differ_where_it_matters(library) -> None:  # type: ignore[no-untyped-def]
    """The pair exists so a scene can show what rain does to a surface's diurnal behaviour."""
    dry = library["soil_dry"].spec.thermal
    wet = library["soil_wet"].spec.thermal
    assert wet.heat_capacity_j_m2_k > 2.0 * dry.heat_capacity_j_m2_k
    assert wet.conductivity_w_mk > 4.0 * dry.conductivity_w_mk
    assert (
        library["soil_wet"].band_properties("swir").emissivity
        > library["soil_dry"].band_properties("swir").emissivity
    ), "wet soil is darker at 1.6 µm — the basis of soil-moisture retrieval"


def test_the_paint_pair_differ_only_where_they_should(library) -> None:  # type: ignore[no-untyped-def]
    """Black and white paint are the same in the thermal bands and opposite in the solar ones —
    which is why a white car and a black car look identical in LWIR and nothing alike at noon."""
    black, white = library["car_paint_black"], library["car_paint_white"]
    # abs=1e-9, not ==: black paint's ε comes from a *spectral curve* band-averaged by quadrature
    # and white's from an authored scalar, so they agree in the last float64 place, not before.
    assert black.band_properties("lwir").emissivity == pytest.approx(
        white.band_properties("lwir").emissivity, abs=1e-9
    )
    assert black.band_properties("mwir").emissivity == pytest.approx(
        white.band_properties("mwir").emissivity, abs=1e-9
    )
    assert black.spec.thermal.solar_absorptivity > 3.0 * white.spec.thermal.solar_absorptivity
    assert black.band_properties("nir").emissivity > 3.0 * white.band_properties("nir").emissivity


def test_water_carries_the_only_measured_optical_constants_in_the_library(library) -> None:  # type: ignore[no-untyped-def]
    """And it is the one material a maritime scene stands on, which is why it gets Level A."""
    water = library["water"]
    assert angular_level(water) == "A"
    assert water.n_k_path is not None and water.n_k_path.name == "water.csv"
    assert "segelstein" in water.n_k_path.read_text(encoding="utf-8").lower()
