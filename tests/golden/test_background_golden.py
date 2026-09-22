"""Golden arrays for the backgrounds a target is seen against (GT.2).

The sky and the sea are where drift is hardest to see by eye: a sky two kelvin warm still looks
like a sky, and a sea profile that bends the wrong way past the horizon still looks like water.
Neither had a reference array, and both feed every aerial and maritime frame this project makes.

The layered atmosphere is here for the same reason and with a worked example: `AT.10` changed
SWIR's transmittance by **6.5 % at 5 km** while leaving the 200 m anchor exact, and not one golden
moved, because all eight of them came from one LWIR config. The table below would have moved.

docs/physics-model.md §7.1, §7.4, §5.3; ADR 0004, ADR 0071, ADR 0078, ADR 0080, ADR 0113.
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import pytest
from golden_store import GoldenStore

from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.atmosphere.sea import SeaModel
from irsim.atmosphere.sky import SkyModel
from irsim.config.bands import BAND_KEYS
from irsim.config.environment import load_environment_preset
from irsim.config.loader import load_sensor_config
from irsim.materials.nk import load_nk_table
from irsim.radiometry.lut_files import load_band_lut_for_config, load_band_response_for_config
from irsim.thermal import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
PRESET = "us_standard_clear"
ENVIRONMENT = "clear_dry"

SENSORS = {
    "lwir": "flir_boson_640_lwir",
    "mwir": "example_mwir_insb_640",
    "swir": "example_swir_ingaas_640",
    "nir": "example_nir_si_1280",
}

#: One weather sample, stated here rather than loaded, so the golden's inputs are all in this file.
WEATHER = WeatherSample(288.15, 0.46, 1.0, 0.0, 0.0, 3.0, 23000.0, 0.0)

DISTANCES_M = (200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0)
ELEVATIONS_DEG = (0.0, 2.0, 5.0, 15.0, 30.0, 60.0, 90.0)
DEPRESSIONS_DEG = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 45.0)


def _key(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


@pytest.fixture(scope="module")
def weather() -> WeatherSeries:
    return WeatherSeries.constant(WEATHER, 3600.0)


@pytest.fixture(scope="module")
def bands():  # type: ignore[no-untyped-def]
    """``band -> (config, LUT, response)`` for every band the registry names a sensor for."""
    out = {}
    for band, name in SENSORS.items():
        cfg = load_sensor_config(REPO / "configs" / "sensors" / f"{name}.yaml", DATA)
        out[band] = (
            cfg,
            load_band_lut_for_config(cfg, DATA / "lut", DATA),
            load_band_response_for_config(cfg, DATA),
        )
    return out


def test_the_registry_and_this_file_name_the_same_bands() -> None:
    """A fifth band must not quietly go ungoldened, which is how LWIR came to be the only one."""
    assert set(SENSORS) | {"visible"} == set(BAND_KEYS)


def test_golden_layered_transmittance_table(golden: GoldenStore, weather, bands) -> None:  # type: ignore[no-untyped-def]
    """``tau(band, distance, elevation)`` for all four bands, in one array.

    The 200 m column is the anchor the grey preset is fitted at and is pinned by its own test; the
    value here is the rest of the curve, which is where a spectral-class change shows up and where
    nothing was watching.
    """
    atmosphere = LayeredAtmosphere(
        load_atmosphere_preset(PRESET),
        weather,
        {b: lut for b, (_c, lut, _r) in bands.items()},
        {b: resp for b, (_c, _l, resp) in bands.items()},
    )
    table = np.array(
        [
            [
                [
                    float(atmosphere.transmittance(band, 0.0, d, np.deg2rad(e)))
                    for e in ELEVATIONS_DEG
                ]
                for d in DISTANCES_M
            ]
            for band in sorted(SENSORS)
        ],
        dtype=np.float32,
    )
    golden.check(
        "layered_tau_band_distance_elevation",
        table,
        config_hash=_key(PRESET, WEATHER, DISTANCES_M, ELEVATIONS_DEG, sorted(SENSORS)),
        atol=1e-6,
        units="transmittance",
    )
    assert np.all((table > 0.0) & (table <= 1.0))
    # and it is a table worth comparing: every band attenuates, and the slant path beats the
    # horizontal one at every range because the air thins with height.
    for i in range(table.shape[0]):
        assert table[i, -1, 0] < table[i, 0, 0]
        assert table[i, -1, -1] > table[i, -1, 0]


def test_golden_sky_radiance_against_elevation(golden: GoldenStore, weather, bands) -> None:  # type: ignore[no-untyped-def]
    """`MS.2`'s ``L_sky(theta)`` for the two emissive bands: the background every aerial target
    is seen against, and the one that decides whether a drone reads hot or cold against it."""
    rows = []
    for band in ("lwir", "mwir"):
        _cfg, lut, resp = bands[band]
        atmosphere = LayeredAtmosphere(
            load_atmosphere_preset(PRESET), weather, {band: lut}, {band: resp}
        )
        sky = SkyModel(atmosphere, load_environment_preset(ENVIRONMENT), band, lut)
        rows.append(np.asarray(sky.radiance(0.0, np.deg2rad(np.array(ELEVATIONS_DEG)))))
    table = np.asarray(rows, dtype=np.float32)
    golden.check(
        "sky_radiance_elevation_lwir_mwir",
        table,
        config_hash=_key(PRESET, ENVIRONMENT, WEATHER, ELEVATIONS_DEG),
        rtol=1e-5,
        units="W/m2/sr",
    )
    # The sky gets colder as you look up, in both bands, which is the whole shape of the thing.
    assert np.all(np.diff(table, axis=1) < 0.0)


def test_golden_sea_apparent_temperature_against_depression(
    golden: GoldenStore, weather, bands
) -> None:  # type: ignore[no-untyped-def]
    """`MM`'s sea profile: what a camera 20 m up reads looking down at water.

    Near the horizon the surface is grazing, water's emissivity collapses and the sea reads as
    reflected sky; at nadir it reads as the skin. That collapse is the maritime lane's whole
    phenomenology and it had no reference array.
    """
    cfg, lut, resp = bands["lwir"]
    atmosphere = LayeredAtmosphere(
        load_atmosphere_preset(PRESET), weather, {"lwir": lut}, {"lwir": resp}
    )
    sky = SkyModel(atmosphere, load_environment_preset(ENVIRONMENT), "lwir", lut)
    sea = SeaModel(sky, load_nk_table("water"), resp, bulk_sst_k=290.0, camera_height_m=20.0)
    profile = np.asarray(
        sea.apparent_temperature_k(0.0, np.deg2rad(np.array(DEPRESSIONS_DEG))), dtype=np.float32
    )
    golden.check(
        "sea_apparent_temperature_depression",
        profile,
        config_hash=_key(PRESET, ENVIRONMENT, WEATHER, DEPRESSIONS_DEG, 290.0, 20.0),
        atol=1e-3,
        units="K (1 mK)",
    )
    assert float(profile[-1]) > float(profile[0])  # nadir reads the skin, the horizon the sky
    assert float(profile.min()) > 270.0 and float(profile.max()) < 300.0


def test_the_transmittance_golden_would_have_caught_at_10(
    golden: GoldenStore, weather, bands
) -> None:  # type: ignore[no-untyped-def]
    """The worked example, checkable rather than claimed.

    Before `AT.10`, SWIR's hand-written window ran 0.80-1.10 µm at ×0.5 and swallowed the 0.94 µm
    water band; deriving the classes from one ladder handed that stretch to `h2o_0p94` at ×10.
    `tau_SWIR(5 km)` went 0.4148 -> 0.3877 and **every golden array passed**, because all eight of
    them came from one LWIR config. This asserts the table holds the new value and that the old one
    is outside its tolerance -- which is what "a golden would have caught it" means.
    """
    _cfg, lut, resp = bands["swir"]
    atmosphere = LayeredAtmosphere(
        load_atmosphere_preset(PRESET), weather, {"swir": lut}, {"swir": resp}
    )
    now = float(atmosphere.transmittance("swir", 0.0, 5000.0, 0.0))
    before_at_10 = 0.414804696154634
    assert now == pytest.approx(0.387690895559268, rel=1e-9)
    assert abs(now - before_at_10) > 1e-6 * 1000  # a thousand times the table's own tolerance
