"""Point-target injection (MS.6): 1/R^2 with tau = 1, tau(R)/R^2 with the atmosphere and the exact
per-class identity, zero excess at the sky-beyond radiance, flux conservation over a sub-pixel
sweep, handoff continuity in the phase mean, the rasteriser's measured error, the F-number ratio
3.4, and the whole chain."""

from __future__ import annotations

import copy
import math
import pathlib

import numpy as np
import pytest
import yaml

from irsim.atmosphere import Atmosphere, LayeredAtmosphere, load_atmosphere_preset
from irsim.config.sensor import SensorConfig
from irsim.materials import MaterialTable
from irsim.optics.aperture import aperture_factor
from irsim.optics.sampling import box_downsample
from irsim.pipeline import PipelineConfig, PipelineState, PointTarget, run_frame
from irsim.pipeline.point_target import excess_power, excess_radiance, fill_fraction, splat
from irsim.radiometry.lut import BandLUT
from irsim.thermal import WeatherSample, WeatherSeries

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON = yaml.safe_load((REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml").read_text())


def _sensor(
    width: int = 16, height: int = 8, supersample: int = 1, f_number: float = 1.0
) -> SensorConfig:
    d = copy.deepcopy(BOSON)
    d["sensor"]["fpa"].update(width=width, height=height)
    d["sensor"]["optics"]["supersample_factor"] = supersample
    d["sensor"]["optics"]["f_number"] = f_number
    return SensorConfig.model_validate(d)


def _weather(t_air: float = 288.15, rh: float = 0.3) -> WeatherSeries:
    return WeatherSeries.constant(
        WeatherSample(t_air, rh, 1.0, 0.0, 0.0, 0.0, 23000.0, 0.0), 3600.0
    )


def _lb(lut: BandLUT, t: float) -> float:
    return float(lut.lookup(np.float64(t))[()])


def _area_for(sensor, phi: float, range_m: float) -> float:  # type: ignore[no-untyped-def]
    f = sensor.optics.focal_length_mm * 1e-3
    return phi * range_m**2 * sensor.pixel_area_m2 / (f * f)


def test_fill_fraction_and_inverse_square_without_atmosphere(tophat_lwir_lut: BandLUT) -> None:
    sensor = _sensor().sensor
    f = sensor.optics.focal_length_mm * 1e-3
    lt = _lb(tophat_lwir_lut, 400.0)
    ref = None
    area = _area_for(sensor, 0.5, 200.0)
    for r in (200.0, 500.0, 1000.0, 5000.0):
        t = PointTarget(area, r, lt, (4.5, 3.5))
        phi = fill_fraction(area, r, f, sensor.pixel_area_m2)
        assert phi == pytest.approx(area * f * f / (r * r * sensor.pixel_area_m2), rel=1e-12)
        ex = excess_radiance(t, sensor, None, "lwir", 0.0, tophat_lwir_lut, background_radiance=2.0)
        assert ex == pytest.approx(phi * (lt - 2.0), rel=1e-12)
        scaled = ex * r * r
        ref = scaled if ref is None else ref
        assert scaled == pytest.approx(ref, rel=1e-9)
    with pytest.raises(ValueError, match="rasterise"):
        excess_radiance(
            PointTarget(50.0, 100.0, lt, (4.5, 3.5)), sensor, None, "lwir", 0.0, tophat_lwir_lut
        )


def test_layered_atmosphere_identity_and_tau_over_r2(tophat_lwir_lut: BandLUT) -> None:
    sensor = _sensor().sensor
    f = sensor.optics.focal_length_mm * 1e-3
    atm = LayeredAtmosphere(
        load_atmosphere_preset("us_standard_clear"), _weather(), {"lwir": tophat_lwir_lut}
    )
    lt = _lb(tophat_lwir_lut, 600.0)
    theta = math.radians(15.0)
    es = atm.exponential_sum("lwir", 0.0)
    ratios = []
    area = _area_for(sensor, 0.5, 200.0)
    for r in (200.0, 500.0, 1000.0, 2000.0, 5000.0):
        t = PointTarget(area, r, lt, (4.5, 3.5), theta)
        ex = excess_radiance(t, sensor, atm, "lwir", 0.0, tophat_lwir_lut)
        phi = fill_fraction(area, r, f, sensor.pixel_area_m2)
        tau_k = atm.class_transmittances("lwir", 0.0, r, theta)
        beyond = atm.sky_beyond_per_class("lwir", 0.0, r, theta)
        expect = phi * float(np.sum(es.weights * tau_k * (lt - beyond)))
        assert ex == pytest.approx(expect, rel=1e-12), "the per-class identity"
        tau_band = float(np.dot(es.weights, tau_k))
        ratios.append(ex * r * r / tau_band)
    assert max(ratios) / min(ratios) - 1.0 < 0.02, (
        "tau(R)/R^2 times a constant within 2 % (hot target)"
    )
    # a target at the effective sky-beyond radiance has zero excess at every range
    for r in (100.0, 1000.0, 10000.0):
        l_beyond = atm.sky_beyond("lwir", 0.0, r, theta)
        t = PointTarget(_area_for(sensor, 0.3, r), r, l_beyond, (4.5, 3.5), theta)
        ex = excess_radiance(t, sensor, atm, "lwir", 0.0, tophat_lwir_lut)
        assert abs(ex) < 1e-9 * l_beyond
    # the sky beyond at R -> 0 is the sky itself; at very large R it vanishes
    assert atm.sky_beyond("lwir", 0.0, 1e-3, theta) == pytest.approx(
        atm.sky_radiance("lwir", 0.0, theta), rel=1e-6
    )
    assert atm.sky_beyond("lwir", 0.0, 200e3, theta) < 0.01 * atm.sky_radiance("lwir", 0.0, theta)
    grey = Atmosphere(
        load_atmosphere_preset("us_standard_clear"), _weather(), {"lwir": tophat_lwir_lut}
    )
    t = PointTarget(0.02, 1000.0, lt, (4.5, 3.5))
    tau = float(grey.transmittance("lwir", 0.0, 1000.0))
    expect = (
        fill_fraction(0.02, 1000.0, f, sensor.pixel_area_m2)
        * tau
        * (lt - grey.air_radiance("lwir", 0.0))
    )
    assert excess_radiance(t, sensor, grey, "lwir", 0.0, tophat_lwir_lut) == pytest.approx(
        expect, rel=1e-12
    )


def test_sub_pixel_sweep_conserves_flux_and_handoff_is_continuous() -> None:
    k = 4
    base = np.zeros((32, 32), np.float64)
    total = []
    for px in np.linspace(0.0, 1.0, 16, endpoint=False):
        for py in np.linspace(0.0, 1.0, 16, endpoint=False):
            out = splat(base, 3.0, (4.0 + px, 4.0 + py), k)
            native = box_downsample(out.astype(np.float32), k)
            total.append(float(native.astype(np.float64).sum()))
    assert np.max(np.abs(np.array(total) / 3.0 - 1.0)) < 1e-6
    # handoff: a rasterised square of side 0.99 px, averaged over a fine phase grid spanning one
    # supersample period (the expectation over continuous phase is exactly the area; a coarse
    # phase grid aliases with the sample grid and biases the mean), equals the injected flux to 1 %
    side = 0.99 * k
    fluxes = []
    xs = np.arange(32) + 0.5
    for px in (np.arange(50) + 0.5) / 50.0:
        for py in (np.arange(50) + 0.5) / 50.0:
            cx, cy = 4.0 * k + px, 4.0 * k + py
            inside = (np.abs(xs[None, :] - cx) <= side / 2) & (np.abs(xs[:, None] - cy) <= side / 2)
            native = box_downsample(np.where(inside, 3.0, 0.0).astype(np.float32), k)
            fluxes.append(float(native.astype(np.float64).sum()))
    injected = 0.99**2 * 3.0
    assert abs(np.mean(fluxes) / injected - 1.0) < 0.01
    with pytest.raises(ValueError, match="outside"):
        splat(base, 1.0, (-1.0, 4.0), k)


@pytest.mark.parametrize(
    "size_px, max_mean_err, min_rms_err", [(1.2, 0.05, 0.10), (2.0, 0.07, 0.08), (4.0, 0.04, 0.04)]
)
def test_rasteriser_flux_error_vs_size_is_as_recorded(
    size_px: float, max_mean_err: float, min_rms_err: float
) -> None:
    """ADR 0071: the k = 4 rasteriser is unbiased in the phase mean (within 5 % for a 1.2 px
    target) but its per-phase error is large below a few pixels -- the evidence for the 1 px
    handoff and for supersampling small targets further."""
    k = 4
    side = size_px * k
    errs = []
    xs = np.arange(64) + 0.5
    for px in np.linspace(0.0, 1.0, 16, endpoint=False):
        for py in np.linspace(0.0, 1.0, 16, endpoint=False):
            cx, cy = (8.0 + px) * k, (8.0 + py) * k
            inside = (np.abs(xs[None, :] - cx) <= side / 2) & (np.abs(xs[:, None] - cy) <= side / 2)
            native = box_downsample(inside.astype(np.float32), k)
            errs.append(float(native.astype(np.float64).sum()) / size_px**2 - 1.0)
    e = np.array(errs)
    assert abs(e.mean()) < max_mean_err
    assert math.sqrt(float((e**2).mean())) > min_rms_err, "the per-phase error is not small"


def test_point_target_power_f_number_ratio_is_3p4(tophat_lwir_lut: BandLUT) -> None:
    s1, s2 = _sensor(f_number=1.0).sensor, _sensor(f_number=2.0).sensor
    p1, p2 = excess_power(1.0, s1), excess_power(1.0, s2)
    assert p1 / p2 == pytest.approx(3.4, rel=1e-12)
    assert p1 / p2 == pytest.approx(aperture_factor(1.0) / aperture_factor(2.0), rel=1e-12)


def test_through_the_chain(tophat_lwir_lut: BandLUT) -> None:
    """A sub-pixel 400 K target on a uniform 290 K blackbody scene, no atmosphere: the injected
    pixel's apparent temperature is the exact inversion of L_bg + phi (L_t - L_bg) within 1 mK
    (PSF off), and with the PSF on the total excess over the frame is conserved to 1e-4."""
    shape = (32, 32)  # large enough that the PSF tails stay inside the frame
    planes = {
        "temperature_k": np.full(shape, 290.0, np.float32),
        "normal_dot_view": np.ones(shape, np.float32),
        "distance_m": np.full(shape, 50.0, np.float32),
        "material_id": np.ones(shape, np.int32),
        "sky_view_factor": np.ones(shape, np.float32),
    }
    sensor_cfg = _sensor(width=32, height=32)
    sensor = sensor_cfg.sensor
    lt = _lb(tophat_lwir_lut, 400.0)
    lbg = _lb(tophat_lwir_lut, 290.0)
    r = 2000.0
    area = _area_for(sensor, 0.3, r)
    target = PointTarget(area, r, lt, (16.5, 16.5))
    phi = fill_fraction(area, r, sensor.optics.focal_length_mm * 1e-3, sensor.pixel_area_m2)
    assert phi == pytest.approx(0.3, rel=1e-9)
    for psf_enabled in (False, True):
        cfg = PipelineConfig.from_sensor(
            sensor_cfg,
            MaterialTable.constant(1.0),
            lut=tophat_lwir_lut,
            noise_enabled=False,
            psf_enabled=psf_enabled,
        )
        with_t = run_frame(
            planes, cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k), point_targets=[target]
        )
        without = run_frame(planes, cfg, PipelineState(housing_temp_k=cfg.t_housing_cal_k))
        assert (
            with_t.radiance is not None
            and without.radiance is not None
            and with_t.apparent_t is not None
        )
        excess = with_t.radiance.astype(np.float64) - without.radiance.astype(np.float64)
        assert excess.sum() == pytest.approx(phi * (lt - lbg), rel=1e-4)
        if not psf_enabled:
            expect = float(
                tophat_lwir_lut.apparent_temperature(np.asarray(lbg + phi * (lt - lbg)))[()]
            )
            assert abs(float(with_t.apparent_t[16, 16]) - expect) * 1e3 < 1.0
            assert np.count_nonzero(excess > 1e-6 * excess.max()) == 1
        else:
            assert np.count_nonzero(excess > 1e-3 * excess.max()) > 1, "the PSF spreads it"


@pytest.mark.parametrize("k", [1, 2, 4])
def test_the_safe_interval_for_a_splat_is_half_a_supersample_cell_in_from_each_edge(k: int) -> None:
    """`IrCamera` drops an analytic target outside the frame instead of failing the render, and
    the bounds it uses have to be `splat`'s own or it either raises anyway or throws away a
    target it could have injected.

    `splat` spreads over the four supersample cells around the position, so the last safe place is
    ``0.5/k`` in from either edge -- and one cell further out, the weight on the missing cell
    stops being zero and it raises. The aerial demo found this by rendering: its targets are
    placed by angle, so the InSb's narrower field puts one at x = 700 px on a 640 px frame and the
    whole MWIR render stopped at that `ValueError`.
    """
    width = height = 8
    base = np.zeros((height * k, width * k), dtype=np.float32)
    margin = 0.5 / k
    for x, y in ((margin, margin), (width - margin, height - margin), (margin, height - margin)):
        splat(base, 1.0, (x, y), k)  # must not raise
    for x, y in ((margin * 0.5, 4.0), (4.0, margin * 0.5), (width - margin * 0.5, 4.0)):
        with pytest.raises(ValueError, match="outside"):
            splat(base, 1.0, (x, y), k)
