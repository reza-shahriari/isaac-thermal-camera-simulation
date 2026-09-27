"""A reflective band sees the sun off the ground and in the clouds (S55, `AT.20`, ADR 0153).

The four-band Phantom 4 render found it: in SWIR the white ABS shell (ρ 0.78) rendered **black**
against the sky while its RGB companion was white, and the SWIR sky had no clouds. Both came from
the environment being thermal only. The ground half of L_env was L_B(T_ground), ~0 in NIR/SWIR,
so a downward-facing surface (V_s → 0) reflected nothing. And an opaque cloud swapped the column's
thermal emission for its own, ~0 there too, so a cloud pixel read exactly the clear sky.

These tests pin the fix to closed forms. With a response that covers the whole solar spectrum
both band fractions are 1, so a V_s = 0 panel must read ρ · albedo · GHI / π, the acceptance
taken literally, from the weather's own DNI and DHI and the site's own sun. They also pin the
cloud base to the visible dome's own expression, the night to exact zero, and LWIR to its old
value bit for bit.
"""

from __future__ import annotations

import math
import pathlib
from datetime import datetime, timezone

import numpy as np
import pytest

from irsim.atmosphere.cloud import CLOUD_BASE_ALBEDO, cloud_reflectance
from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.atmosphere.sky import SkyModel
from irsim.atmosphere.skylight import DiffuseSkylight
from irsim.config.environment import load_environment_preset
from irsim.config.loader import load_sensor_config
from irsim.pipeline.environment import environment_radiance
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.solar import load_solar_spectrum
from irsim.radiometry.spectral_response import SpectralResponse, load_spectral_response
from irsim.scene import ground_reflectance
from irsim.thermal.scene_forcing import solar_terms_at
from irsim.thermal.weather import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
DATA = REPO / "data"
GROUND_SPECTRUM = DATA / "spectra" / "solar" / "direct_normal_am1p5.csv"
#: 45 N on the Greenwich meridian, at midsummer: at t = 12 h of a series whose epoch is midnight
#: UTC the sun is near its local transit, ~68 deg up, and at t = 0 it is below the horizon.
SITE = (45.0, 0.0)
EPOCH = datetime(2026, 6, 21, tzinfo=timezone.utc)
NOON = 12 * 3600.0
MIDNIGHT = 0.0
WHITE_PANEL_RHO = 0.78  # abs_plastic_white in SWIR, the shell that rendered black


def _weather(cloud_fraction: float = 0.0, dni: float = 650.0, dhi: float = 140.0) -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(293.15, 0.45, 3.0, cloud_fraction, dni, dhi, 23000.0, 0.0),
        86400.0,
        epoch_utc=EPOCH,
    )


def _wide(tmp_path: pathlib.Path) -> SpectralResponse:
    """A top-hat from the model's trusted floor (0.7 um, ADR 0064) to the end of the solar file."""
    spectrum = load_solar_spectrum(GROUND_SPECTRUM)
    hi = float(spectrum.wavelength_um.max()) + 0.01
    path = tmp_path / "wide.csv"
    path.write_text(f"# 0.7 um to the end of the solar file\n0.7,1.0\n{hi},1.0\n")
    return load_spectral_response(path)


def _oracle_fractions() -> tuple[float, float]:
    """(f_dir, f_diff) for `_wide`, integrated here from the file with numpy alone.

    f_dir is the beam's share above 0.7 um; f_diff the same share of the λ⁻⁴-weighted (Rayleigh)
    sky. Independent of the module under test, which interpolates onto its own grid.
    """
    spectrum = load_solar_spectrum(GROUND_SPECTRUM)
    lam, e = spectrum.wavelength_um, spectrum.irradiance_w_m2_um
    band = lam >= 0.7
    rayleigh = e * lam**-4.0
    f_dir = np.trapezoid(e[band], lam[band]) / np.trapezoid(e, lam)
    f_diff = np.trapezoid(rayleigh[band], lam[band]) / np.trapezoid(rayleigh, lam)
    return float(f_dir), float(f_diff)


def _oracle_irradiance(weather: WeatherSeries, t_s: float) -> tuple[float, float]:
    """(E_B for `_wide`, μ0): the weather's GHI split through the two oracle fractions."""
    f_dir, f_diff = _oracle_fractions()
    terms = solar_terms_at(weather, *SITE, t_s)
    mu0 = math.sin(math.radians(terms.elevation_deg))
    return f_dir * terms.dni_w_m2 * mu0 + f_diff * terms.dhi_w_m2, mu0


def _sky(
    band: str,
    environment: str,
    weather: WeatherSeries,
    response: SpectralResponse,
    *,
    sunlit: bool = True,
    skylight_response: SpectralResponse | None = None,
) -> SkyModel:
    lut = BandLUT.build(response, n=1001)
    layered = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), weather, {band: lut}, {band: response}
    )
    env = load_environment_preset(environment)
    skylight = DiffuseSkylight.from_spectrum(
        load_solar_spectrum(GROUND_SPECTRUM), skylight_response or response, "lb", band
    )
    if not sunlit:
        # the sky as it was before AT.20: ADR 0086's skylight, and nothing lit by the sun
        return SkyModel(layered, env, band, lut, "lb", skylight=skylight)
    return SkyModel(
        layered,
        env,
        band,
        lut,
        "lb",
        skylight=skylight,
        site=SITE,
        ground_albedo=ground_reflectance(env, band),
    )


def _response(band: str) -> SpectralResponse:
    sensors = {"nir": "example_nir_si_1280", "swir": "example_swir_ingaas_640"}
    cfg = load_sensor_config(REPO / "configs" / "sensors" / f"{sensors[band]}.yaml")
    return load_spectral_response(cfg.sensor.band.spectral_response)


def test_a_downward_facing_white_panel_reads_rho_albedo_ghi_over_pi(tmp_path: pathlib.Path) -> None:
    """The acceptance: V_s = 0, ρ · albedo · GHI / π through the band's solar fractions, to 5 %."""
    weather = _weather()
    sky = _sky("swir", "clear_dry", weather, _response("swir"), skylight_response=_wide(tmp_path))
    e_band, mu0 = _oracle_irradiance(weather, NOON)
    assert mu0 > 0.5, "the test wants a high sun"
    albedo = ground_reflectance(load_environment_preset("clear_dry"), "swir")
    assert albedo == pytest.approx(0.26)  # soil_dry: 1 − ε_SWIR 0.74

    l_env = float(environment_radiance(sky, sky.lut, NOON, np.zeros((1, 1)), "lb")[0, 0])
    panel = WHITE_PANEL_RHO * l_env
    assert panel == pytest.approx(WHITE_PANEL_RHO * albedo * e_band / math.pi, rel=0.05)
    assert sky.band_irradiance(NOON) == pytest.approx(e_band, rel=5e-3)


def test_without_the_ground_material_the_same_panel_is_black() -> None:
    """The red half: a ground that reflects nothing leaves the underside at its thermal floor."""
    weather = _weather()
    lit = _sky("swir", "clear_dry", weather, _response("swir"))
    thermal = _sky("swir", "clear_dry", weather, _response("swir"), sunlit=False)
    below = np.zeros((1, 1))
    bright = float(environment_radiance(lit, lit.lut, NOON, below, "lb")[0, 0])
    dark = float(environment_radiance(thermal, thermal.lut, NOON, below, "lb")[0, 0])
    assert dark < 1e-3 * bright, (dark, bright)


def test_the_band_takes_its_own_share_of_the_beam_and_of_the_sky() -> None:
    """E_B = f_dir DNI sin(el) + f_diff DHI, and NIR's share of the beam beats its share of the
    blue sky (Rayleigh λ⁻⁴): a band is not lit by a fixed fraction of GHI."""
    weather = _weather()
    sky = _sky("nir", "clear_dry", weather, _response("nir"))
    skylight = sky.skylight
    assert skylight is not None and skylight.per_dni is not None
    assert skylight.per_dni > math.pi * skylight.per_dhi
    terms = solar_terms_at(weather, *SITE, NOON)
    expected = (
        skylight.per_dni * terms.dni_w_m2 * math.sin(math.radians(terms.elevation_deg))
        + math.pi * skylight.per_dhi * terms.dhi_w_m2
    )
    assert sky.band_irradiance(NOON) == pytest.approx(expected, rel=1e-12)
    assert sky.ground_shine(NOON) == pytest.approx(0.30 * expected / math.pi, rel=1e-12)


def test_night_is_exactly_dark() -> None:
    """A real night -- sun down, no diffuse light in the weather -- is 0.0, not a small number."""
    sky = _sky("swir", "scattered_cumulus", _weather(0.5, dni=0.0, dhi=0.0), _response("swir"))
    assert solar_terms_at(sky.weather, *SITE, MIDNIGHT).elevation_deg < 0.0
    assert sky.ground_shine(MIDNIGHT) == 0.0
    assert sky.cloud_shine(MIDNIGHT) == 0.0
    np.testing.assert_array_equal(
        sky.cloud_shine(MIDNIGHT, np.array([0.0, 3.0, 30.0])), np.zeros(3)
    )


def test_below_the_horizon_the_beam_is_cut_and_the_weather_s_diffuse_is_kept() -> None:
    """The weather is the authority on DHI (twilight is real); only the beam needs the sun up."""
    sky = _sky("swir", "clear_dry", _weather(dni=650.0, dhi=40.0), _response("swir"))
    skylight = sky.skylight
    assert skylight is not None
    assert sky.band_irradiance(MIDNIGHT) == pytest.approx(math.pi * skylight.per_dhi * 40.0)


def test_a_cloud_base_is_the_visible_dome_s_expression_in_this_band(
    tmp_path: pathlib.Path,
) -> None:
    """R(τ, μ0) · E / π: the dome's `_cloud_base` with the band's irradiance in place of the
    luminous one, so the two bands light a cloud with the same shape."""
    weather = _weather(0.5)
    sky = _sky(
        "swir", "scattered_cumulus", weather, _response("swir"), skylight_response=_wide(tmp_path)
    )
    e_band, mu0 = _oracle_irradiance(weather, NOON)
    column = np.array([1.0, 10.0, 30.0])
    expected = cloud_reflectance(column, mu0) * e_band / math.pi
    shine = sky.cloud_shine(NOON, column)
    np.testing.assert_allclose(shine, expected, rtol=5e-3)
    assert shine[0] < shine[1] < shine[2]  # thin edge dim, deep core bright: the dome's texture


def test_the_uniform_cloud_uses_the_preset_depth_or_the_dome_s_base_albedo() -> None:
    weather = _weather(0.5)
    cumulus = _sky("swir", "scattered_cumulus", weather, _response("swir"))
    sheet = _sky("swir", "overcast", weather, _response("swir"))
    e_band = cumulus.band_irradiance(NOON)
    mu0 = math.sin(math.radians(solar_terms_at(weather, *SITE, NOON).elevation_deg))
    depth = load_environment_preset("scattered_cumulus").clouds.optical_depth
    assert depth is not None
    assert cumulus.cloud_shine(NOON) == pytest.approx(
        float(cloud_reflectance(depth, mu0)[()]) * e_band / math.pi, rel=1e-12
    )
    assert load_environment_preset("overcast").clouds.optical_depth is None
    assert sheet.cloud_shine(NOON) == pytest.approx(CLOUD_BASE_ALBEDO * e_band / math.pi, rel=1e-12)


def test_swir_clouds_are_brighter_than_the_clear_sky_beside_them() -> None:
    """The deck in SWIR at noon: cloudy rays outshine clear ones. Without sunlight they did not
    -- an opaque cloud replaced the column's thermal emission with its own, both ~0 here."""
    weather = _weather(0.5)
    lit = _sky("swir", "scattered_cumulus", weather, _response("swir"))
    thermal = _sky("swir", "scattered_cumulus", weather, _response("swir"), sunlit=False)
    deck = lit.cloud_deck(NOON, seed=3, min_elevation_deg=20.0)
    rng = np.random.default_rng(0)
    elevation = np.radians(rng.uniform(25.0, 85.0, 4000))
    azimuth = rng.uniform(0.0, 2.0 * math.pi, 4000)
    covered = deck.march(elevation, azimuth).emissivity() > 0.9
    clear = deck.march(elevation, azimuth).emissivity() == 0.0
    assert covered.sum() > 100 and clear.sum() > 100

    bright = lit.radiance_field_from_deck(NOON, elevation, azimuth, deck)
    flat = thermal.radiance_field_from_deck(NOON, elevation, azimuth, deck)
    assert np.median(bright[covered]) > 2.0 * np.median(bright[clear])
    # before AT.20 the cloud's thermal excess was nothing beside the skylight: no contrast
    assert np.median(flat[covered]) < 1.02 * np.median(flat[clear])


def test_lwir_is_bit_identical_whatever_the_ground_is_made_of() -> None:
    """An emissive band has no skylight, so neither term exists: the environment and the cloud
    field come out the same bits with the ground material named or removed."""
    cfg = load_sensor_config(REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml")
    response = load_spectral_response(cfg.sensor.band.spectral_response)
    lut = BandLUT.build(response, n=1001)
    weather = _weather(0.5)
    layered = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), weather, {"lwir": lut}, {"lwir": response}
    )
    env = load_environment_preset("scattered_cumulus")
    bare = env.model_copy(update={"ground": env.ground.model_copy(update={"material": None})})
    named, unnamed = SkyModel(layered, env, "lwir", lut), SkyModel(layered, bare, "lwir", lut)
    assert not named.sunlit and named.ground_shine(NOON) == 0.0
    view = np.linspace(0.0, 1.0, 11)[None, :]
    np.testing.assert_array_equal(
        environment_radiance(named, lut, NOON, view), environment_radiance(unnamed, lut, NOON, view)
    )
    deck = named.cloud_deck(NOON, seed=3, min_elevation_deg=20.0)
    el, az = np.radians([30.0, 60.0, 85.0]), np.radians([10.0, 120.0, 250.0])
    np.testing.assert_array_equal(
        named.radiance_field_from_deck(NOON, el, az, deck),
        unnamed.radiance_field_from_deck(NOON, el, az, deck),
    )


def test_an_albedo_without_the_sun_is_refused() -> None:
    weather = _weather()
    response = _response("nir")
    lut = BandLUT.build(response, n=1001)
    layered = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), weather, {"nir": lut}, {"nir": response}
    )
    with pytest.raises(ValueError, match="site"):
        SkyModel(layered, load_environment_preset("clear_dry"), "nir", lut, "lb", ground_albedo=0.3)


def test_every_shipped_environment_names_a_ground_the_library_has() -> None:
    for path in sorted((REPO / "configs" / "environments").glob("*.yaml")):
        env = load_environment_preset(path.stem)
        assert env.ground.material is not None, path.stem
        for band in ("nir", "swir", "mwir", "lwir"):
            assert 0.0 <= ground_reflectance(env, band) <= 1.0
