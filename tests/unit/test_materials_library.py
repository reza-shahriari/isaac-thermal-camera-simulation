"""Material schema and library (M7.2): author one property, derive the rest; closure to 1e-6 after
load for the committed library, the §12.3 example and a tau = 0.7 glass; refusals that keep
non-negotiable #4 honest; the round trip; §16.2 sanity."""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from irsim.config.bands import BAND_IDS
from irsim.config.materials import MaterialConfig
from irsim.materials.library import (
    CLOSURE_TOL,
    MaterialLibrary,
    load_material,
    nominal_response,
)
from irsim.materials.spectra import load_property_spectrum
from irsim.radiometry.spectral_response import SpectralResponse

REPO = pathlib.Path(__file__).resolve().parents[2]

# docs/physics-model.md §12.3, in the one-file-per-material layout (ADR 0040)
EXAMPLE_12_3 = {
    "schema_version": 2,
    "material": {
        "name": "car_paint_black",
        "source": "literature",
        # `AT.17`: §4.5 requires the surface state the optics were measured on, and the field has
        # no default, so every fixture names one too.
        "surface_treatment": "painted",
        "thermal": {
            "density_kg_m3": 7800,
            "specific_heat_j_kgk": 470,
            "conductivity_w_mk": 45,
            "thickness_m": 0.0012,
            "solar_absorptivity": 0.94,
        },
        "optical": {
            "spectral_emissivity": "spectra/car_paint_black.csv",
            "roughness_per_band": {"nir": 0.35, "swir": 0.30, "mwir": 0.18, "lwir": 0.12},
            "angular_model": {"type": "fresnel", "n_k_file": "nk/acrylic_paint.csv"},
            "transmittance_per_band": {"nir": 0.0, "swir": 0.0, "mwir": 0.0, "lwir": 0.0},
        },
    },
}


def _data_root(
    tmp_path: pathlib.Path, spectrum: str = "0.3,0.94\n2.5,0.94\n3,0.88\n5,0.88\n7.5,0.9\n15,0.9\n"
) -> pathlib.Path:
    root = tmp_path / "data"
    (root / "spectra").mkdir(parents=True, exist_ok=True)
    (root / "nk").mkdir(exist_ok=True)
    (root / "spectra" / "car_paint_black.csv").write_text("# test\n" + spectrum)
    (root / "nk" / "acrylic_paint.csv").write_text(
        "# placeholder n,k table\n1.0,1.5,0.01\n10.0,1.5,0.05\n"
    )
    return root


def _write(tmp_path: pathlib.Path, raw: dict, name: str = "m.yaml") -> pathlib.Path:  # type: ignore[type-arg]
    p = tmp_path / name
    p.write_text(yaml.safe_dump(raw, sort_keys=False))
    return p


def test_committed_library_closes_in_every_band() -> None:
    """The CLAUDE.md #4 library walk: every material, every standard band, ε + ρ + τ = 1 to 1e-6."""
    lib = MaterialLibrary.load()
    assert set(lib.names) >= {
        "car_paint_black",
        "car_paint_white",
        "bare_aluminium",
        "glass_windshield",
        "asphalt_dry",
        "human_skin",
    }
    for name in lib:
        for band in BAND_IDS:
            props = lib[name].band_properties(band)
            closure = props.emissivity + props.reflectance + props.transmittance
            assert abs(closure - 1.0) < CLOSURE_TOL, (name, band, closure)
            assert 0.0 <= props.reflectance <= 1.0
    assert (
        lib["car_paint_white"].spec.thermal.solar_absorptivity
        < lib["car_paint_black"].spec.thermal.solar_absorptivity
    )
    assert lib["car_paint_black"].band_properties("lwir").emissivity == pytest.approx(
        0.90, abs=1e-6
    )
    assert lib["car_paint_black"].band_properties("mwir").emissivity == pytest.approx(
        0.88, abs=1e-6
    )
    glass = lib["glass_windshield"]
    assert (
        glass.band_properties("swir").transmittance == 0.70
        and glass.band_properties("lwir").transmittance == 0.0
    )
    assert glass.band_properties("swir").reflectance == pytest.approx(0.08, abs=1e-12)
    assert lib["bare_aluminium"].band_properties("lwir").reflectance == pytest.approx(
        0.91, abs=1e-12
    )
    assert len(lib.content_hash()) == 64


def test_spec_12_3_example_loads_round_trips_and_closes(tmp_path: pathlib.Path) -> None:
    root = _data_root(tmp_path)
    cfg = MaterialConfig.model_validate(EXAMPLE_12_3)
    again = MaterialConfig.model_validate(
        yaml.safe_load(yaml.safe_dump(cfg.model_dump(mode="json")))
    )
    assert again == cfg
    m = load_material(_write(tmp_path, EXAMPLE_12_3), data_dir=root)
    assert m.n_k_path is not None and m.spectrum is not None
    for band in BAND_IDS:
        p = m.band_properties(band)
        assert abs(p.emissivity + p.reflectance + p.transmittance - 1.0) < CLOSURE_TOL
    assert m.band_properties("lwir").emissivity == pytest.approx(0.90, abs=1e-9)
    assert m.band_properties("nir").emissivity == pytest.approx(0.94, abs=1e-9)
    # the same file with the photon weighting and a real response still closes
    resp = SpectralResponse(np.array([7.5, 9.0, 13.5]), np.array([0.5, 1.0, 0.6]), "r", "")
    p = m.band_properties("lwir", response=resp, form="photon")
    assert abs(p.emissivity + p.reflectance + p.transmittance - 1.0) < CLOSURE_TOL
    h1 = m.content_hash()
    (root / "spectra" / "car_paint_black.csv").write_text(
        "# test\n0.3,0.94\n2.5,0.94\n3,0.88\n5,0.88\n7.5,0.91\n15,0.91\n"
    )
    assert load_material(_write(tmp_path, EXAMPLE_12_3), data_dir=root).content_hash() != h1


def test_authoring_both_raises_and_reflectance_plus_tau_derives_emissivity(
    tmp_path: pathlib.Path,
) -> None:
    raw = copy.deepcopy(EXAMPLE_12_3)
    raw["material"]["optical"]["spectral_reflectance"] = "spectra/x.csv"
    with pytest.raises(ValueError, match="exactly one"):
        MaterialConfig.model_validate(raw)
    raw = copy.deepcopy(EXAMPLE_12_3)
    raw["material"]["optical"]["emissivity_per_band"] = {"lwir": 0.9}
    with pytest.raises(ValueError, match="exactly one"):
        MaterialConfig.model_validate(raw)
    raw = copy.deepcopy(EXAMPLE_12_3)
    del raw["material"]["optical"]["spectral_emissivity"]
    with pytest.raises(ValueError, match="exactly one"):
        MaterialConfig.model_validate(raw)
    glass = copy.deepcopy(EXAMPLE_12_3)
    glass["material"]["name"] = "glass"
    glass["material"]["optical"] = {
        "reflectance_per_band": {"swir": 0.08, "lwir": 0.12},
        "transmittance_per_band": {"swir": 0.70, "lwir": 0.0},
    }
    m = load_material(_write(tmp_path, glass), data_dir=_data_root(tmp_path))
    swir = m.band_properties("swir")
    assert swir.authored == "reflectance" and swir.emissivity == pytest.approx(0.22, abs=1e-12)
    assert (
        swir.transmittance == 0.70
        and abs(swir.emissivity + swir.reflectance + swir.transmittance - 1) < 1e-12
    )
    assert m.band_properties("lwir").emissivity == pytest.approx(0.88, abs=1e-12)


def test_epsilon_plus_tau_over_one_refused_scalar_and_spectral(tmp_path: pathlib.Path) -> None:
    raw = copy.deepcopy(EXAMPLE_12_3)
    raw["material"]["optical"] = {
        "emissivity_per_band": {"lwir": 0.9},
        "transmittance_per_band": {"lwir": 0.2},
    }
    with pytest.raises(ValueError, match="Kirchhoff"):
        MaterialConfig.model_validate(raw)
    raw = copy.deepcopy(EXAMPLE_12_3)
    raw["material"]["optical"]["transmittance_per_band"] = {"lwir": 0.2}
    m = load_material(_write(tmp_path, raw), data_dir=_data_root(tmp_path))
    with pytest.raises(ValueError, match="Kirchhoff"):
        m.band_properties("lwir")  # 0.90 + 0.2 > 1, caught after the band average
    assert m.band_properties("mwir").emissivity == pytest.approx(0.88, abs=1e-9)


def test_extra_band_loads_and_needs_a_response_for_spectra(tmp_path: pathlib.Path) -> None:
    raw = copy.deepcopy(EXAMPLE_12_3)
    raw["material"]["optical"] = {"emissivity_per_band": {"lwir": 0.9, "lwir_wide": 0.91}}
    m = load_material(_write(tmp_path, raw), data_dir=_data_root(tmp_path))
    assert m.declared_bands == {"lwir", "lwir_wide"}
    assert m.band_properties("lwir_wide").emissivity == 0.91
    with pytest.raises(KeyError, match="no emissivity authored"):
        m.band_properties("mwir")
    spectral = load_material(_write(tmp_path, EXAMPLE_12_3), data_dir=_data_root(tmp_path))
    with pytest.raises(KeyError, match="nominal range"):
        spectral.band_properties("lwir_wide")
    wide = SpectralResponse(np.array([7.5, 14.0]), np.array([1.0, 1.0]), "w", "")
    assert spectral.band_properties("lwir_wide", response=wide).emissivity == pytest.approx(
        0.90, abs=1e-9
    )
    with pytest.raises(ValueError, match="covers"):
        spectral.band_properties(
            "lwir",
            response=SpectralResponse(np.array([7.0, 16.0]), np.array([1.0, 1.0]), "too_wide", ""),
        )


def test_file_layout_and_data_guards(tmp_path: pathlib.Path) -> None:
    root = _data_root(tmp_path)
    with pytest.raises(ValueError, match="materials:' wrapper"):
        load_material(
            _write(tmp_path, {"materials": {"x": EXAMPLE_12_3["material"]}}), data_dir=root
        )
    raw = copy.deepcopy(EXAMPLE_12_3)
    raw["material"]["optical"]["spectral_emissivity"] = "spectra/missing.csv"
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        load_material(_write(tmp_path, raw), data_dir=root)
    raw = copy.deepcopy(EXAMPLE_12_3)
    del raw["material"]["source"]
    with pytest.raises(ValueError, match="source"):
        MaterialConfig.model_validate(raw)
    raw = copy.deepcopy(EXAMPLE_12_3)
    raw["material"]["thermal"]["thickness_m"] = -1.0
    with pytest.raises(ValueError):
        MaterialConfig.model_validate(raw)
    (root / "spectra" / "bad.csv").write_text("# bad\n1.0,0.5\n0.9,0.6\n")
    with pytest.raises(ValueError, match="increasing"):
        load_property_spectrum(root / "spectra" / "bad.csv")
    (root / "spectra" / "bad.csv").write_text("# bad\n1.0,0.5\n2.0,1.2\n")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        load_property_spectrum(root / "spectra" / "bad.csv")
    assert nominal_response("lwir").support_um == (7.5, 13.5)
    thermal = MaterialLibrary.load()["car_paint_black"].spec.thermal
    assert thermal.heat_capacity_j_m2_k == pytest.approx(7800 * 470 * 0.0012)
    assert thermal.thermal_inertia == pytest.approx((45 * 7800 * 470) ** 0.5)


# --- AT.17: surface state is a named property (§4.5) --------------------------------------------


def test_every_material_names_the_surface_state_its_optics_were_measured_on() -> None:
    """§4.5: emissivity is a property of the surface, not of the substance under it.

    One measurement campaign on aluminium window profiles reports 0.834-0.856 for anodised
    exterior surfaces and 0.055-0.82 across untreated cavities in the same frames [R39] -- one
    metal, one paper, a fifteen-fold spread. A file that does not say which surface it describes is
    not a material specification, and the field has no default for that reason: a default would be
    a state nobody chose, on exactly the materials whose state is least obvious.
    """
    from irsim.config.materials import SURFACE_TREATMENTS

    library = MaterialLibrary.load()
    for name in library.names:
        treatment = library[name].spec.surface_treatment
        assert treatment in SURFACE_TREATMENTS, f"{name}: {treatment!r}"


def test_the_three_aluminiums_are_one_metal_in_three_surface_states() -> None:
    """The library's own demonstration of §4.5, and the defect `AT.17` was raised for.

    `bare_aluminium` shipped described as *polished* and valued as *oxidised*: 0.04 against 0.09 at
    8 µm [R39], a factor of 2.25 in the term that decides whether a surface reports itself or the
    sky. The three files now differ **only** in surface state: they share ρ, c_p and k exactly,
    because those belong to the metal, and they span the whole measured range in emissivity,
    because that belongs to the surface. A file copied from one to another without changing its
    optics fails here rather than rendering a plausible new material.
    """
    library = MaterialLibrary.load()
    names = ("aluminium_polished", "bare_aluminium", "aluminium_anodised")
    specs = [library[n].spec for n in names]

    assert [s.surface_treatment for s in specs] == ["polished", "oxidised", "anodised"]
    for field in ("density_kg_m3", "specific_heat_j_kgk", "conductivity_w_mk"):
        values = {getattr(s.thermal, field) for s in specs}
        assert len(values) == 1, f"{field} differs across one metal: {values}"

    epsilon = [float(library[n].band_properties("lwir").emissivity) for n in names]
    assert epsilon == sorted(epsilon), f"the states are not ordered by emissivity: {epsilon}"
    assert epsilon[-1] - epsilon[0] >= 0.7, f"the three states span only {epsilon[-1] - epsilon[0]}"
    # [R39]'s own figures, which is what makes this a measurement rather than a spread.
    assert epsilon[0] == pytest.approx(0.04, abs=0.005)
    assert 0.834 <= epsilon[-1] <= 0.856


def test_a_material_that_does_not_name_its_state_is_refused() -> None:
    """Silence is the failure mode: neither 0.04 nor 0.09 is wrong, unlabelled."""
    raw = {
        "schema_version": 2,
        "material": {
            "name": "unlabelled_alloy",
            "source": "estimated",
            "thermal": {
                "density_kg_m3": 2700,
                "specific_heat_j_kgk": 900,
                "conductivity_w_mk": 205,
                "thickness_m": 0.002,
                "solar_absorptivity": 0.15,
            },
            "optical": {"emissivity_per_band": {"lwir": 0.09}},
        },
    }
    with pytest.raises(ValidationError, match="surface_treatment"):
        MaterialConfig.model_validate(raw)

    raw["material"]["surface_treatment"] = "shiny"  # type: ignore[index]
    with pytest.raises(ValidationError, match="surface_treatment"):
        MaterialConfig.model_validate(raw)
