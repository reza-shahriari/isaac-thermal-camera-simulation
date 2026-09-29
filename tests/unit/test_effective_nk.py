"""AT.25 -- an angular model for oxidised, anodised and painted metal.

Level A scaled the clean metal's Fresnel shape by the authored magnitude and clipped to 1, so
`bare_aluminium` went ε 0.09 → 1.0 at 89° and a 280 K skin against a 230 K sky brightened
236 → 280 K at the limb: a ring on every unpainted airframe. With `fit: effective_nk` the table's
n and k are scaled together until the band's normal-incidence emissivity is the authored ε_B,
and the angular law is that conductor's own: it peaks ≤ 0.3 at 80–85° and falls to 0 at 90°;
its hemispherical emissivity sits where a conductor's does; dielectrics are bit-identical.
Roadmap AT.25; ADR 0163; §4.2.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.materials.directional import directional_emissivity
from irsim.materials.library import MaterialLibrary
from irsim.materials.nk import band_directional_emissivity, effective_nk_table, load_nk_table
from irsim.materials.table import MaterialTable
from irsim.radiometry.spectral_response import load_spectral_response

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"


@pytest.fixture(scope="module")
def library() -> MaterialLibrary:
    return MaterialLibrary.load()


@pytest.fixture(scope="module")
def boson():  # type: ignore[no-untyped-def]
    return load_spectral_response(DATA / "spectra" / "responses" / "boson_vox.csv")


def _hemispherical(eps_of_cos, n: int = 4000) -> float:  # type: ignore[no-untyped-def]
    """ε_h = 2 ∫₀¹ ε(μ) μ dμ."""
    mu = (np.arange(n) + 0.5) / n
    return float(2.0 * np.sum(np.asarray(eps_of_cos(mu), dtype=np.float64) * mu) / n)


def test_the_effective_table_reproduces_the_authored_normal_emissivity(library, boson) -> None:  # type: ignore[no-untyped-def]
    table = load_nk_table("aluminium", DATA)
    clean = float(band_directional_emissivity(table, boson, 1.0))
    target = float(library["bare_aluminium"].band_properties("lwir", boson).emissivity)
    assert clean < 0.06 < target, (clean, target)
    fitted, magnitude = effective_nk_table(table, boson, target)
    assert magnitude == 1.0
    assert float(band_directional_emissivity(fitted, boson, 1.0)) == pytest.approx(target, abs=1e-9)
    assert np.all(fitted.k < table.k) and np.all(fitted.n < table.n), "an oxide dims a conductor"
    assert effective_nk_table(table, boson, target)[0] is fitted, "cached"
    # a value below the clean metal's is a brighter conductor (n, k up), still exact
    brighter, one = effective_nk_table(table, boson, 0.5 * clean)
    assert one == 1.0 and np.all(brighter.k > table.k)
    assert float(band_directional_emissivity(brighter, boson, 1.0)) == pytest.approx(
        0.5 * clean, abs=1e-9
    )
    # beyond the conductor family's reach (0.43 for aluminium in LWIR): the most emissive
    # conductor, and the rest as a magnitude on its shape -- still 0 at grazing, never clipped
    best, rest = effective_nk_table(table, boson, 0.6)
    assert rest > 1.0 and float(
        band_directional_emissivity(best, boson, 1.0)
    ) * rest == pytest.approx(0.6, rel=1e-9)
    with pytest.raises(ValueError, match="epsilon_normal"):
        effective_nk_table(table, boson, 1.0)


def test_bare_aluminium_peaks_short_of_the_limb_and_goes_dark_at_grazing(library, boson) -> None:  # type: ignore[no-untyped-def]
    cos_theta = np.cos(np.radians(np.linspace(0.0, 90.0, 901))).astype(np.float32)
    eps = np.asarray(
        directional_emissivity(library["bare_aluminium"], "lwir", cos_theta, boson, data_dir=DATA),
        dtype=np.float64,
    )
    theta = np.linspace(0.0, 90.0, 901)
    assert eps[0] == pytest.approx(0.09, abs=0.005)
    peak = int(np.argmax(eps))
    assert eps.max() <= 0.3, eps.max()
    assert 75.0 <= theta[peak] <= 87.0, theta[peak]
    assert eps[-1] < 1e-3, "zero at grazing"
    assert np.all(eps <= 1.0) and eps.max() < 0.99, "nothing clips"
    hemi = _hemispherical(
        lambda mu: directional_emissivity(
            library["bare_aluminium"],
            "lwir",
            np.asarray(mu, dtype=np.float32),
            boson,
            data_dir=DATA,
        )
    )
    assert 1.10 * eps[0] <= hemi <= 1.40 * eps[0], (hemi, eps[0])  # a conductor's ratio


def test_dielectrics_are_bit_identical_and_the_limb_ring_is_gone(library, boson) -> None:  # type: ignore[no-untyped-def]
    """Water keeps the magnitude scaling: identical to the bit. And a 280 K aluminium skin under
    a 230 K sky no longer reads 280 K at the limb: the apparent temperature at 89° stays near
    the sky's, as a mirror's should."""
    cos_theta = np.linspace(1.0, 0.0, 361, dtype=np.float32)
    water = library["water"]
    assert water.spec.optical.angular_model.fit == "magnitude"
    eps_w = directional_emissivity(water, "lwir", cos_theta, boson, data_dir=DATA)
    shape = band_directional_emissivity(load_nk_table("water", DATA), boson, cos_theta)
    at_normal = band_directional_emissivity(load_nk_table("water", DATA), boson, 1.0)
    magnitude = float(water.band_properties("lwir", boson).emissivity)
    assert np.array_equal(eps_w, np.clip(magnitude * shape / at_normal, 0, 1).astype(np.float32))
    from irsim.radiometry.lut import BandLUT

    lut = BandLUT.build(boson, n=1001)
    l_skin, l_sky = float(lut.lookup(280.0)[()]), float(lut.lookup(230.0)[()])
    al = library["bare_aluminium"]
    eps_limb = float(
        directional_emissivity(
            al, "lwir", np.float32(np.cos(np.radians(89.0))), boson, data_dir=DATA
        )
    )
    l_app = eps_limb * l_skin + (1.0 - eps_limb) * l_sky
    t_app = float(lut.apparent_temperature(np.float32(l_app))[()])
    assert t_app < 245.0, t_app  # not 280 K


def test_the_packed_table_carries_the_fit(library, boson) -> None:  # type: ignore[no-untyped-def]
    packed = MaterialTable.from_library(library, "lwir", angle_lut=True, response=boson)
    ids = np.full(5, packed.id_for("bare_aluminium"), dtype=np.int32)
    cos = np.array([1.0, 0.5, 0.17, 0.05, 0.0], dtype=np.float32)
    eps = np.asarray(packed.epsilon_at(ids, cos), dtype=np.float64)
    assert eps.max() <= 0.31 and eps[0] == pytest.approx(0.09, abs=0.01)


@pytest.mark.parametrize(
    "camera,band",
    [
        ("example_mwir_insb_640", "mwir"),
        ("example_swir_ingaas_640", "swir"),
        ("example_nir_si_1280", "nir"),
    ],
)
def test_every_band_of_the_library_fits_and_no_metal_clips(library, camera, band) -> None:  # type: ignore[no-untyped-def]
    """A scene packs every material in every band, so the fit must resolve them all: MWIR and
    SWIR inside the conductor family, NIR (authored 0.08 against a reachable 0.06) by the most
    emissive conductor plus a magnitude -- in each the law peaks short of 1 and is 0 at 90°."""
    from irsim.config.loader import load_sensor_config
    from irsim.radiometry.lut_files import load_band_response_for_config

    response = load_band_response_for_config(load_sensor_config(camera))
    cos_theta = np.cos(np.radians(np.linspace(0.0, 90.0, 901))).astype(np.float32)
    for name in ("bare_aluminium", "aluminium_polished"):
        material = library[name]
        target = float(material.band_properties(band, response).emissivity)
        eps = np.asarray(
            directional_emissivity(material, band, cos_theta, response, data_dir=DATA), float
        )
        assert eps[0] == pytest.approx(target, abs=2e-3), (band, name, eps[0], target)
        assert eps.max() < 0.99 and eps[-1] < 1e-3, (band, name, eps.max(), eps[-1])
