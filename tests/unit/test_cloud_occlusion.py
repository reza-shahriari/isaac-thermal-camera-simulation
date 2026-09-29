"""A cloud between the camera and a target occludes it (AT.14, ADR 0162).

Before this the infrared band marched the shared cloud field only where the ray hit nothing, so
a drone behind a cumulus was drawn through clear air while the visible companion hid it. These
tests are about the finite-range march -- the Schwarzschild integral ended at the hit -- and
about stage 2 composing its answer: a target behind an opaque cloud reads as the cloud, in front
of it unchanged, inside it between the two. docs/clouds-in-the-infrared.md has the sources.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.atmosphere.weather_fx import (
    WeatherFxDeck,
    cloud_field_from_spec,
    weather_fx_available,
)
from irsim.pipeline.atmosphere import (
    CloudOcclusion,
    apply_atmosphere_gbuffer,
    apply_layered_gbuffer,
    cloud_from_planes,
)

needs_weather_fx = pytest.mark.skipif(
    not weather_fx_available(), reason="the isaac-weather-fx submodule is not checked out"
)

SMALL = dict(cells=64, levels=24, cell_m=140.0)


def deck(**overrides) -> WeatherFxDeck:  # type: ignore[no-untyped-def]
    kwargs = dict(cover=0.6, genus="cumulus", temperature_c=20.0, dewpoint_c=12.0, seed=5, **SMALL)
    kwargs.update(overrides)
    return WeatherFxDeck(cloud_field_from_spec(**kwargs))


def _rays() -> tuple[np.ndarray, np.ndarray]:
    el = np.radians(np.linspace(15.0, 70.0, 12))[:, None]
    az = np.radians(np.linspace(0.0, 350.0, 36))[None, :]
    return np.broadcast_arrays(el, az)


def _constant(value: float):  # type: ignore[no-untyped-def]
    return lambda h: np.full(np.shape(h), value)


# --- the march to a range ----------------------------------------------------------------------


@needs_weather_fx
def test_a_hit_short_of_the_cloud_meets_no_cloud() -> None:
    """A target 50 m away under a 600 m base is seen through clear air: transmittance 1,
    nothing emitted, no emission range -- whatever the sky behind it holds."""
    d = deck(cover=0.95)
    el, az = _rays()
    m = d.march_to(el, az, 50.0, _constant(5.0))
    assert np.all(m.transmittance == 1.0)
    assert np.all(m.path_radiance == 0.0)
    assert np.all(m.emission_range_m == 0.0)
    assert np.all(m.optical_depth == 0.0)


@needs_weather_fx
def test_an_infinite_range_is_the_sky_march() -> None:
    """Ended at infinity the new march must be the old one: the same samples, so the band's
    transmittance is `1 − emissivity` of `march` to rounding. Otherwise the sky pixels and the
    geometry pixels beside them would carry two clouds."""
    d = deck(cover=0.6)
    el, az = _rays()
    sky = d.march(el, az)
    m = d.march_to(el, az, np.inf, _constant(1.0))
    assert np.allclose(m.transmittance, 1.0 - sky.emissivity(), atol=1e-9)
    assert np.any(m.transmittance < 0.5)  # the field is not empty


@needs_weather_fx
def test_a_uniform_emitter_closes_the_schwarzschild_integral() -> None:
    """For an isothermal cloud the integral is exact: ``L_path = (1 − τ_c) L_B``. This is the
    discretisation being checked -- each sample emits exactly what it absorbs, through the
    cloud already crossed -- not the field."""
    d = deck(cover=0.6)
    el, az = _rays()
    m = d.march_to(el, az, np.inf, _constant(7.25))
    assert np.allclose(m.path_radiance, (1.0 - m.transmittance) * 7.25, atol=1e-9)


@needs_weather_fx
def test_transmittance_falls_with_range_and_the_emission_lies_between() -> None:
    """Ending the ray deeper in the slab can only add cloud: τ_c is non-increasing in range,
    and where cloud was met the emission is centred between the slab's entry and the hit.
    At a fixed dense step count, because the samples move with the span and the quadrature
    noise of a near-opaque ray is larger than the physics being asserted."""
    d = deck(cover=0.8)
    el, az = _rays()
    direction = np.stack([np.cos(el) * np.sin(az), np.sin(el), -np.cos(el) * np.cos(az)], -1)
    near, far = d.field.slab_span(np.zeros_like(direction), direction)
    previous = np.ones(el.shape)
    for fraction in (0.1, 0.3, 0.6, 1.0):
        rng = near + fraction * (far - near)
        m = d.march_to(el, az, rng, _constant(1.0), steps=2048)
        assert np.all(m.transmittance <= previous + 0.02)
        cloudy = m.transmittance < 1.0 - 1e-9
        assert np.all(m.emission_range_m[cloudy] >= near[cloudy] - 1e-6)
        assert np.all(m.emission_range_m[cloudy] <= rng[cloudy] + 1e-6)
        previous = m.transmittance
    assert np.any(previous < 0.01)  # some rays go opaque at the far side


@needs_weather_fx
def test_the_cloud_emits_at_its_own_height(aerial_cloudy_sky) -> None:  # type: ignore[no-untyped-def]
    """Through the sky model each sample emits at the base temperature lapsed to its height, so
    the emission lies between the black base and the black top of the cloud, and a ray that
    met no cloud contributes nothing. A native deck answers None rather than a guess."""
    from irsim.atmosphere.cloud_deck import CloudDeck

    sky = aerial_cloudy_sky
    d = deck(cover=0.7)
    el, az = _rays()
    m = sky.cloud_occlusion(0.0, el, az, d, np.inf)
    t_base = sky.cloud_base_temperature_k(0.0, d.base_m)
    lapse = sky.atmosphere.preset.profile.lapse_rate_k_per_m
    l_base = float(sky.lut.lookup(np.float64(t_base), sky.quantity)[()])
    l_top = float(sky.lut.lookup(np.float64(t_base - lapse * d.thickness_m), sky.quantity)[()])
    eps = 1.0 - m.transmittance
    cloudy = eps > 1e-6
    assert cloudy.any()
    assert np.all(m.path_radiance[cloudy] <= eps[cloudy] * l_base + 1e-9)
    assert np.all(m.path_radiance[cloudy] >= eps[cloudy] * l_top - 1e-9)
    assert np.all(m.path_radiance[~cloudy] == 0.0)
    native = CloudDeck.__new__(CloudDeck)
    assert sky.cloud_occlusion(0.0, el, az, native, np.inf) is None


# --- stage 2 composes it ----------------------------------------------------------------------


def _planes(shape: tuple[int, int], tau: float, radiance: float, range_m: float) -> dict:
    return {
        "cloud_transmittance": np.full(shape, tau, dtype=np.float32),
        "cloud_radiance": np.full(shape, radiance, dtype=np.float32),
        "cloud_range_m": np.full(shape, range_m, dtype=np.float32),
    }


def test_a_target_behind_an_opaque_cloud_reads_as_the_cloud(aerial_cloudy_sky) -> None:  # type: ignore[no-untyped-def]
    """The whole point. A 40 W/m²/sr target at 2 km behind an opaque cloud whose emission is
    centred 300 m away comes back as the cloud's radiance through 300 m of air, not as the
    target through 2 km of it; and the sky pixel beside it is untouched, because its cloud is
    already in its temperature."""
    sky = aerial_cloudy_sky
    atm, band, q = sky.atmosphere, "lwir", sky.quantity
    shape = (4, 6)
    l_in = np.full(shape, 40.0, dtype=np.float32)
    d = np.full(shape, 2000.0, dtype=np.float32)
    el = np.full(shape, np.radians(20.0))
    sky_mask = np.zeros(shape, dtype=bool)
    sky_mask[0, 0] = True
    clear = apply_layered_gbuffer(atm, band, 0.0, l_in, d, q, sky_mask=sky_mask, elevation_rad=el)
    cloud = cloud_from_planes(_planes(shape, tau=0.0, radiance=9.0, range_m=300.0))
    assert cloud is not None
    out = apply_layered_gbuffer(
        atm, band, 0.0, l_in, d, q, sky_mask=sky_mask, elevation_rad=el, cloud=cloud
    )
    tau_300 = float(atm.transmittance(band, 0.0, 300.0, np.radians(20.0))[()])
    assert out[1, 1] == pytest.approx(9.0 * tau_300, rel=1e-5)
    assert out[1, 1] != pytest.approx(clear[1, 1], rel=1e-3)
    assert out[0, 0] == clear[0, 0] == l_in[0, 0]
    assert out.dtype == np.float32


def test_no_cloud_is_no_change_and_a_thin_one_is_in_between(aerial_cloudy_sky) -> None:  # type: ignore[no-untyped-def]
    """τ_c = 1 with nothing emitted is bit-for-bit the clear result; a half-transparent cloud
    lands between the clear target and the opaque cloud."""
    sky = aerial_cloudy_sky
    atm, band, q = sky.atmosphere, "lwir", sky.quantity
    shape = (3, 3)
    l_in = np.full(shape, 40.0, dtype=np.float64)
    d = np.full(shape, 800.0)
    el = np.full(shape, np.radians(35.0))
    clear = apply_layered_gbuffer(atm, band, 0.0, l_in, d, q, elevation_rad=el)
    none = cloud_from_planes(_planes(shape, tau=1.0, radiance=0.0, range_m=0.0))
    same = apply_layered_gbuffer(atm, band, 0.0, l_in, d, q, elevation_rad=el, cloud=none)
    assert np.array_equal(same, clear)
    opaque = apply_layered_gbuffer(
        atm,
        band,
        0.0,
        l_in,
        d,
        q,
        elevation_rad=el,
        cloud=cloud_from_planes(_planes(shape, tau=0.0, radiance=9.0, range_m=400.0)),
    )
    thin = apply_layered_gbuffer(
        atm,
        band,
        0.0,
        l_in,
        d,
        q,
        elevation_rad=el,
        cloud=cloud_from_planes(_planes(shape, tau=0.5, radiance=4.5, range_m=400.0)),
    )
    assert np.all((thin > np.minimum(clear, opaque)) & (thin < np.maximum(clear, opaque)))
    assert cloud_from_planes({"radiance": l_in}) is None


def test_the_grey_atmosphere_composes_the_same_way() -> None:
    """M8.1's Beer-Lambert path gets the cloud too, its emission through exp(−γ R_c)."""
    shape = (2, 2)
    l_in = np.full(shape, 30.0, dtype=np.float32)
    d = np.full(shape, 1000.0, dtype=np.float32)
    gamma, l_air = 2e-4, 12.0
    cloud = CloudOcclusion(
        np.zeros(shape, np.float32),
        np.full(shape, 9.0, np.float32),
        np.full(shape, 250.0, np.float32),
    )
    out = apply_atmosphere_gbuffer(l_in, d, gamma, l_air, cloud=cloud)
    assert np.allclose(out, 9.0 * np.exp(-gamma * 250.0), rtol=1e-6)
    with pytest.raises(TypeError):
        apply_atmosphere_gbuffer(l_in.astype(np.float16), d, gamma, l_air, cloud=cloud)
