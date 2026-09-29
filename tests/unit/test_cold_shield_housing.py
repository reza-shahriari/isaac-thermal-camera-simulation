"""SC.34 -- a cold-shielded FPA sees the shield out of cone; VOx's G_th is a literature number.

`housing_power_field` added A_d Ω (1 − τ RI) L_h to every FPA. Behind a cold shield the
out-of-cone hemisphere is a surface ~200 K colder than any scene, so only the lens's own emission
arrives, A_d Ω RI (1 − τ) L_h: the corner shading carries the lens sign, forward and inverse
round-trip, and the Warp twin takes the same two scalars. The Boson now authors its VOx thermal
conductance, so the reported floor sits below the 50 mK anchor. Roadmap SC.34; §8.2, §9.1.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim.optics.aperture import aperture_factor
from irsim.optics.self_emission import (
    COLD_SHIELD_MAX_FPA_K,
    housing_power_axis,
    housing_power_field,
    housing_terms,
    is_cold_shielded,
)
from irsim.optics.stage import apply_optics, invert_optics, optics_field

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_which_cameras_are_cold_shielded() -> None:
    assert is_cold_shielded(load_sensor_config("mwir640").sensor)
    for name in ("boson640", "swir640", "nir1280"):
        assert not is_cold_shielded(load_sensor_config(name).sensor), name
    assert COLD_SHIELD_MAX_FPA_K == 200.0


def test_the_two_scalars_reproduce_the_warm_form_and_flip_the_corner_sign() -> None:
    a_d, f, tau, lh = 2.9e-10, 1.0, 0.9, 10.0
    axis = housing_power_axis(a_d, f, tau, lh)
    base, per_ri = housing_terms(a_d, f, tau, lh, cold_shielded=False)
    assert base == axis and per_ri == pytest.approx(-axis * tau)
    ri = np.array([1.0, 0.8])
    warm = housing_power_field(a_d, f, tau, lh, ri)
    assert np.allclose(warm, axis * (1.0 - tau * ri)), "the pre-SC.34 form, bit for bit"
    assert warm[1] > warm[0], "a warm camera's corner sees more housing"
    cold = housing_power_field(a_d, f, tau, lh, ri, cold_shielded=True)
    base_c, per_c = housing_terms(a_d, f, tau, lh, cold_shielded=True)
    assert base_c == 0.0 and per_c == pytest.approx(axis * (1.0 - tau))
    assert np.allclose(cold, axis * (1.0 - tau) * ri)
    assert cold[1] < cold[0], "behind a cold shield the corner sees less lens"
    assert cold[0] == pytest.approx(0.1 * axis) and cold[0] == pytest.approx(
        warm[0]
    )  # same on axis
    assert cold[1] < warm[1], "and less than the warm camera off axis"


def test_cooled_corner_shading_carries_the_lens_sign_and_round_trips_to_ten_millikelvin() -> None:
    """A uniform scene colder than the housing: on the MWIR InSb the corners are now *brighter*
    than the axis by only the lens's emission share, not darker by the housing's; and
    apply → invert recovers the scene radiance at the corner to the equivalent of 10 mK."""
    sensor = load_sensor_config("mwir640").sensor
    lb_housing, lb_scene = 3.0, 1.5  # a scene colder than the housing, in W/m2/sr
    h, w = sensor.fpa.height, sensor.fpa.width
    k = sensor.optics.supersample_factor
    radiance = np.full((h * k, w * k), lb_scene, dtype=np.float32)
    phi = apply_optics(radiance, sensor, lb_housing, supersample=k)
    ri = optics_field(sensor)
    a_d = sensor.detector_active_area_m2
    f, tau = sensor.optics.f_number, sensor.optics.transmittance
    expect = a_d * aperture_factor(f) * ri * (tau * lb_scene + (1.0 - tau) * lb_housing)
    assert np.allclose(phi, expect, rtol=1e-6)
    corner, axis = phi[0, 0], phi[h // 2, w // 2]
    assert corner < axis, "cold-shielded: everything the pixel gets scales with RI"
    back = invert_optics(phi, sensor, lb_housing)
    assert np.allclose(back, lb_scene, rtol=1e-6)
    # 10 mK at 300 K in MWIR is ~1e-4 relative in band radiance
    assert abs(float(back[0, 0]) - lb_scene) / lb_scene < 1e-4


def test_the_warm_camera_is_unchanged_and_the_boson_reports_a_floor_below_its_anchor() -> None:
    sensor = load_sensor_config("boson640").sensor
    assert sensor.fpa.g_th_w_per_k == pytest.approx(2.0e-8)
    from irsim.detector.netd import bolometer_floors
    from irsim.radiometry.lut_files import load_band_lut_for_config

    cfg = load_sensor_config("boson640")
    floors = bolometer_floors(cfg.sensor, load_band_lut_for_config(cfg, REPO / "data" / "lut"))
    reported_mk = 1e3 * max(
        v for k, v in floors.items() if k.startswith("netd_") and k.endswith("_k")
    )
    assert reported_mk < sensor.noise.netd_mk_at_300k, floors
    assert reported_mk == pytest.approx(33.3, abs=0.5), "VOx at G_th = 2e-8 W/K: 33 mK, not 74.5"
    h, w = sensor.fpa.height, sensor.fpa.width
    radiance = np.full((h, w), 2.0, dtype=np.float32)
    phi = apply_optics(radiance, sensor, 3.0, supersample=1)
    ri = optics_field(sensor)
    a_d = sensor.detector_active_area_m2
    f, tau = sensor.optics.f_number, sensor.optics.transmittance
    warm = a_d * aperture_factor(f) * (3.0 + tau * ri * (2.0 - 3.0))
    assert np.allclose(phi, warm, rtol=1e-6), "the bolometer keeps ADR 0145's warm form"
