"""Cloud clutter (MS.3): thick cloud = uniform L_B(T_base), cloud = 0 is MS.2 exactly, the
generator's PSD slope self-test, exact coverage and determinism, the LCL base from the weather,
and cloud edges warmer than the clear zenith sky by > 20 K for a 1 km base."""

from __future__ import annotations

import math

import numpy as np
import pytest

from irsim.atmosphere import LayeredAtmosphere, SkyModel, load_atmosphere_preset
from irsim.atmosphere.cloud import (
    cloud_base_temperature_k,
    cloud_radiance,
    generate_cloud_field,
    lifting_condensation_level_m,
    psd_slope,
)
from irsim.atmosphere.humidity import dew_point_k
from irsim.config.environment import EnvironmentConfig, load_environment_preset
from irsim.radiometry.constants import DRY_ADIABATIC_LAPSE_K_PER_M
from irsim.radiometry.lut import BandLUT
from irsim.thermal import WeatherSample, WeatherSeries

T_AIR = 288.15


def _sky(lut: BandLUT, cloud: float, rh: float, tau: float = 0.0) -> SkyModel:
    """The session-scoped 7.5-13.5 um top-hat LUT is also the layered model's nominal LWIR
    response, so no per-module LUT build is needed here (the unit suite is near its budget)."""
    w = WeatherSeries.constant(WeatherSample(T_AIR, rh, 1.0, cloud, 0.0, 0.0, 23000.0, 0.0), 3600.0)
    atm = LayeredAtmosphere(load_atmosphere_preset("us_standard_clear"), w, {"lwir": lut})
    env = load_environment_preset("clear_dry")
    if tau > 0.0:
        raw = {"schema_version": 2, "environment": env.model_dump(mode="json")}
        raw["environment"]["clouds"]["tau"] = tau
        env = EnvironmentConfig.model_validate(raw).environment
    return SkyModel(atm, env, "lwir", lut)


def test_dew_point_and_lcl_from_the_weather() -> None:
    assert dew_point_k(293.15, 1.0) == pytest.approx(293.15, abs=1e-9)
    assert dew_point_k(293.15, 0.5) == pytest.approx(282.4, abs=0.3)  # ~9.3 C at 20 C / 50 %
    assert lifting_condensation_level_m(T_AIR, 1.0) == 0.0, "saturated air: base at the surface"
    z = lifting_condensation_level_m(T_AIR, 0.5)
    assert z == pytest.approx(125.0 * (T_AIR - dew_point_k(T_AIR, 0.5)), rel=1e-12)
    assert 1000.0 < z < 1500.0
    assert cloud_base_temperature_k(T_AIR, 1000.0, 6.5e-3) == pytest.approx(T_AIR - 6.5)
    with pytest.raises(ValueError):
        lifting_condensation_level_m(T_AIR, 0.0)


def test_thick_cloud_is_uniform_at_the_base_temperature_and_clear_is_ms2(
    tophat_lwir_lut: BandLUT,
) -> None:
    """Saturated air puts the base at the surface: zero range, so an opaque sky is L_B(T_base)
    at every elevation exactly, and T_base is T_air. A base a kilometre up is read *through*
    the air in front of it (ADR 0126's range term, on the plane-parallel blend since ADR 0146):
    never colder than the base, never warmer than the air, and warmer toward the horizon where
    the path is longer. Clear sky is MS.2's column, untouched."""
    sky = _sky(tophat_lwir_lut, cloud=1.0, rh=1.0)
    els = np.radians(np.array([[2.0, 15.0, 45.0], [60.0, 80.0, 90.0]]))
    cov = np.ones(els.shape, dtype=bool)
    t = sky.apparent_temperature_field(0.0, els, cov)
    t_base = sky.cloud_base_temperature_k(0.0)
    assert sky.cloud_base_m(0.0) == 0.0
    assert t_base == pytest.approx(T_AIR, abs=1e-9)
    assert np.max(np.abs(t - t_base)) * 1e3 < 1.0, (
        "thick cloud at zero range: uniform L_B(T_base) at every elevation"
    )
    for el in (2.0, 45.0, 90.0):
        assert float(sky.radiance(0.0, math.radians(el))) == pytest.approx(
            float(tophat_lwir_lut.lookup(np.float64(t_base))[()]), rel=1e-9
        )

    lifted = _sky(tophat_lwir_lut, cloud=1.0, rh=0.5)
    base_m = lifted.cloud_base_m(0.0)
    assert 1000.0 < base_m < 1500.0
    # Dry-adiabatic to the base (ADR 0146): the LCL is where surface air, cooled at g/c_p,
    # saturates. ADR 0070's environmental 6.5 K/km overwarmed this base by 4 K.
    t_lifted = lifted.cloud_base_temperature_k(0.0)
    assert t_lifted == pytest.approx(T_AIR - DRY_ADIABATIC_LAPSE_K_PER_M * base_m, rel=1e-12)
    seen = lifted.apparent_temperature_field(0.0, els, cov)
    assert np.all(seen >= t_lifted - 1e-3) and np.all(seen < T_AIR)
    assert seen[0, 0] >= seen[1, 2], "the grazing ray reads the base through more warm air"
    blend = np.asarray([float(lifted.apparent_temperature_k(0.0, e)) for e in els.ravel()])
    assert np.all(blend >= t_lifted - 1e-3) and np.all(blend < T_AIR)
    assert blend[0] >= blend[-1]

    clear = _sky(tophat_lwir_lut, cloud=0.0, rh=0.5)
    none = np.zeros(els.shape, dtype=bool)
    np.testing.assert_array_equal(
        clear.radiance_field(0.0, els, none), clear.clear_radiance(0.0, els)
    )
    np.testing.assert_array_equal(clear.radiance(0.0, els), clear.clear_radiance(0.0, els))


def test_cloud_edges_warmer_than_clear_zenith_and_thin_cloud_in_between(
    tophat_lwir_lut: BandLUT,
) -> None:
    # a ~1 km base: T - T_d = 8 K -> RH such that the dew point is 280.15 K
    rh = 0.59
    sky = _sky(tophat_lwir_lut, cloud=0.5, rh=rh)
    base = sky.cloud_base_m(0.0)
    assert 800.0 < base < 1200.0, base
    zenith = math.radians(90.0)
    t_clear = float(
        sky.apparent_temperature_k(0.0, zenith)
        if False
        else sky._lut.apparent_temperature(sky.clear_radiance(0.0, zenith), "lb")
    )
    cov = np.array([[True, False]])
    t = sky.apparent_temperature_field(0.0, np.array([[zenith, zenith]]), cov)
    assert t[0, 0] - t[0, 1] > 20.0, (t, t_clear)
    assert t[0, 1] == pytest.approx(t_clear, abs=1e-6)
    thin = _sky(tophat_lwir_lut, cloud=0.5, rh=rh, tau=0.5)
    t_thin = thin.apparent_temperature_field(0.0, np.array([[zenith]]), np.array([[True]]))[0, 0]
    assert t[0, 1] < t_thin < t[0, 0], "thin cloud lies between the clear sky and the thick cloud"
    # the uniform blend is the expectation of the structured field, **at the base's range**
    # (ADR 0126 on the plane-parallel blend, ADR 0146): L_clear + c eps tau (L_base - L_beyond),
    # which is (1 - c eps) L_clear + c eps L_base only when the base is at the surface.
    l_clear = float(sky.clear_radiance(0.0, zenith))
    l_base = float(sky._lut.lookup(np.float64(sky.cloud_base_temperature_k(0.0)))[()])
    tau_grid, beyond_grid = sky._cloud_path(0.0)
    tau_up, beyond_up = float(tau_grid[-1]), float(beyond_grid[-1])
    assert 0.5 < tau_up < 1.0, "a kilometre of air is neither clear nor opaque in the window"
    assert float(sky.radiance(0.0, zenith)) == pytest.approx(
        l_clear + 0.5 * tau_up * (l_base - beyond_up), rel=1e-12
    )
    assert float(thin.radiance(0.0, zenith)) == pytest.approx(
        l_clear + 0.25 * tau_up * (l_base - beyond_up), rel=1e-12
    )
    # and the range-free blend it replaced would have read the base through no air at all:
    assert float(sky.radiance(0.0, zenith)) != pytest.approx(0.5 * l_clear + 0.5 * l_base, rel=1e-3)


@pytest.mark.parametrize("beta", [1.5, 2.0, 2.5])
def test_generator_psd_slope_self_test(beta: float) -> None:
    field = generate_cloud_field((256, 256), beta, 0.4, seed=3)
    assert abs(-psd_slope(field.field) - beta) < 0.1
    assert field.coverage.sum() == round(0.4 * 256 * 256)
    assert abs(field.field.mean()) < 1e-9 and abs(field.field.std() - 1.0) < 1e-9
    again = generate_cloud_field((256, 256), beta, 0.4, seed=3)
    np.testing.assert_array_equal(again.field, field.field)
    other = generate_cloud_field((256, 256), beta, 0.4, seed=4)
    assert not np.array_equal(other.field, field.field)


def test_coverage_limits_and_radiance_guards() -> None:
    full = generate_cloud_field((16, 16), 1.8, 1.0, seed=0)
    assert full.coverage.all()
    empty = generate_cloud_field((16, 16), 1.8, 0.0, seed=0)
    assert not empty.coverage.any()
    white = generate_cloud_field((256, 256), 0.0, 0.5, seed=1)
    assert abs(-psd_slope(white.field)) < 0.15
    clear = np.full((4, 4), 30.0)
    out = cloud_radiance(clear, 40.0, 0.25, full.coverage[:4, :4])
    assert np.allclose(out, 0.75 * 40.0 + 0.25 * 30.0)
    with pytest.raises(ValueError):
        cloud_radiance(clear, 40.0, 1.0, full.coverage[:4, :4])
    with pytest.raises(ValueError):
        generate_cloud_field((1, 8), 1.8, 0.5, 0)


@pytest.mark.skip(
    reason="ME.5 landed and *refused* this band: a cloud PSD slope needs a labelled cloud region "
    "on the published clips, and selecting one automatically would be inventing an annotation and "
    "calling it data (docs/validation/reference-stats-2026-09-15.md). The analyser is implemented "
    "and tested (ME.4 `clutter_slope`); what is missing is a region list somebody has to draw. The "
    "comparison that *can* be made without one runs end to end in M12.2's acceptance report."
)
def test_display_domain_cloud_slope_inside_the_reference_band() -> None:  # pragma: no cover
    raise AssertionError("unreachable")


def test_environment_clouds_block_and_presets() -> None:
    env = load_environment_preset("overcast")
    assert env.clouds.tau == 0.0 and env.clouds.emissivity == 1.0 and env.clouds.beta == 1.8
    raw = {"schema_version": 2, "environment": env.model_dump(mode="json")}
    raw["environment"]["clouds"] = {"tau": 1.0}
    with pytest.raises(ValueError):
        EnvironmentConfig.model_validate(raw)
    raw["environment"]["clouds"] = {"cloud_fraction": 0.5}
    with pytest.raises(ValueError, match="WeatherSeries"):
        EnvironmentConfig.model_validate(raw)
