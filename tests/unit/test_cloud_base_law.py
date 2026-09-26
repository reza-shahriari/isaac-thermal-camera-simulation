"""The temperature at a cumulus base, and whose base it is (ADR 0146).

Two things a cloud in the infrared gets wrong quietly. Its base temperature: the base is the
lifting condensation level of *surface* air, which got there by cooling dry-adiabatically, so
lapsing it at the free atmosphere's mean rate overwarms the cloud by a kelvin or two per kilometre
of base -- always toward a brighter cloud. And whose base: a marched deck knows where its rays
entered; taking the temperature at a second base computed from the weather makes the geometry and
the temperature disagree the moment the two are not drawn from one state, which is how a stale
run rendered a 600 m cloud at the temperature of a 1.8 km one.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.atmosphere.cloud import cloud_base_temperature_k, lifting_condensation_level_m
from irsim.atmosphere.humidity import dew_point_k
from irsim.radiometry.constants import DEW_POINT_LAPSE_K_PER_M, DRY_ADIABATIC_LAPSE_K_PER_M


def test_the_lifted_parcel_meets_its_dew_point_at_the_base() -> None:
    """Espy's 125 m/K is (Γ_d − Γ_dew)⁻¹: at the LCL the parcel, cooled at the dry rate, is
    within a tenth of a kelvin of its dew point, cooled at the dew-point rate. That closure is
    what makes the dry rate the right one to the base, and it fails by 2 K at 6.5 K/km."""
    t_air = 293.15
    for rh in (0.3, 0.5, 0.8):
        z = lifting_condensation_level_m(t_air, rh)
        t_dew = dew_point_k(t_air, rh)
        parcel = cloud_base_temperature_k(t_air, z, DRY_ADIABATIC_LAPSE_K_PER_M)
        dew_at_base = t_dew - DEW_POINT_LAPSE_K_PER_M * z
        assert parcel == pytest.approx(dew_at_base, abs=0.1)
        environmental = cloud_base_temperature_k(t_air, z, 6.5e-3)
        assert environmental - parcel == pytest.approx((9.761e-3 - 6.5e-3) * z, rel=1e-6)


def test_the_sky_model_lapses_dry_adiabatically_to_the_base(aerial_cloudy_sky) -> None:  # type: ignore[no-untyped-def]
    """`SkyModel.cloud_base_temperature_k` is the surface air cooled at g/c_p over the LCL."""
    sky = aerial_cloudy_sky
    base = sky.cloud_base_m(0.0)
    t_air = sky.weather.at(0.0).t_air_k
    assert base > 500.0
    assert sky.cloud_base_temperature_k(0.0) == pytest.approx(
        t_air - DRY_ADIABATIC_LAPSE_K_PER_M * base, abs=1e-9
    )
    # A caller that owns the geometry names the base; the law is the same.
    assert sky.cloud_base_temperature_k(0.0, 300.0) == pytest.approx(
        t_air - DRY_ADIABATIC_LAPSE_K_PER_M * 300.0, abs=1e-9
    )


def test_a_marched_deck_is_read_at_its_own_base(aerial_cloudy_sky) -> None:  # type: ignore[no-untyped-def]
    """Two decks of the same field at different bases give different opaque temperatures, each
    the dry-adiabatic one for *its* base -- not the weather's LCL for both."""
    from irsim.atmosphere.weather_fx import (
        WeatherFxDeck,
        cloud_field_from_spec,
        weather_fx_available,
    )

    if not weather_fx_available():
        pytest.skip("the isaac-weather-fx submodule is not checked out")
    sky = aerial_cloudy_sky
    t_air = sky.weather.at(0.0).t_air_k
    el = np.radians(np.full((6, 48), 40.0))
    az = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)[None, :] * np.ones((6, 1))
    readings: dict[float, float] = {}
    warming: dict[float, float] = {}
    for base_m in (300.0, 1500.0):
        field = cloud_field_from_spec(
            cover=0.95, base_m=base_m, seed=3, cells=64, levels=24, cell_m=140.0
        )
        deck = WeatherFxDeck(field)
        march = deck.march(el, az)
        t_app = sky.apparent_temperature_field_from_deck(0.0, el, az, deck)
        opaque = march.emissivity() > 0.98
        assert opaque.any()
        readings[base_m] = float(np.median(t_app[opaque]))
        # An opaque base reads between its own temperature and the air's: the emitting level is
        # inside the cloud (colder, by the in-cloud lapse), and the kilometres of warmer air in
        # front of a high base add a little back (the range term of ADR 0126). Measured here:
        # a 1.5 km base at 273.5 K reads 276.6 K through 2.3 km of 288 K air at 40 degrees.
        t_base = t_air - DRY_ADIABATIC_LAPSE_K_PER_M * base_m
        assert t_base - 6.0 < readings[base_m] < t_air
        warming[base_m] = readings[base_m] - t_base
    # The two bases are 11.7 K apart dry-adiabatically; the higher one gets more path warming,
    # so the observed split is smaller than that but well clear of the noise.
    assert readings[300.0] - readings[1500.0] > 6.0
    assert warming[1500.0] > warming[300.0]
