"""The camera's flat-field correction (M9.12): what it removes, and what it must not touch.

`TwoPointNuc` was built in M5 and nothing applied it, so the 8-bit picture carried every fixed
spatial structure the optics put into the DN plane -- most visibly cos⁴ vignetting, 21 % at the
corner of the Boson's 14 mm lens, which plateau equalisation then stretches into black corners.
No real camera looks like that, because every one of them flat-fields before the AGC.

The subtle half is what the correction must *not* do. The radiometric branch already divides cos⁴
out per pixel analytically in ``invert_optics``, which is why apparent temperature was flat across
a row while the picture was not. Correcting it again would remove the same term twice, so the flat
field belongs to the display branch alone and `dn16` stays the raw ADC plane.

docs/physics-model.md §11.1, §11.2; ADR 0021.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from irsim.config.loader import load_sensor_config
from irsim.isp.nuc import TwoPointNuc
from irsim.materials.table import MaterialTable
from irsim.pipeline.core import PipelineConfig, PipelineState
from irsim.pipeline.flat_field import calibrate_flat_field, uniform_signal_dn
from irsim.pipeline.frame import run_frame

# GT.1: this whole module is the slow tier -- a validation bench or an end-to-end frame rather
# than a unit test. `make test` skips it; `make test-slow` and `make check` run it.
pytestmark = pytest.mark.slow

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"

#: cos⁴ at the corner of the Boson format (ADR 0015's known answer). This is the depth of the
#: artefact the correction exists to remove.
COS4_CORNER = 0.79240


@pytest.fixture(scope="module")
def sensor():  # type: ignore[no-untyped-def]
    return load_sensor_config(BOSON_YAML)


@pytest.fixture(scope="module")
def config(sensor, boson_lut):  # type: ignore[no-untyped-def]
    materials = MaterialTable.constant(0.95, ids=(1,), band_id="lwir")
    return PipelineConfig.from_sensor(
        sensor, materials, boson_lut, noise_enabled=False, flat_field_enabled=False
    )


@pytest.fixture(scope="module")
def corrected_config(sensor, boson_lut):  # type: ignore[no-untyped-def]
    materials = MaterialTable.constant(0.95, ids=(1,), band_id="lwir")
    return PipelineConfig.from_sensor(
        sensor, materials, boson_lut, noise_enabled=False, flat_field_enabled=True
    )


def uniform_planes(config, temperature_k: float):  # type: ignore[no-untyped-def]
    """A G-buffer of one blackbody filling the field, at the supersampled grid."""
    rows, cols = config.sensor.sensor.fpa_shape
    k = config.supersample
    shape = (rows * k, cols * k)
    return {
        "temperature_k": np.full(shape, temperature_k, np.float32),
        "material_id": np.ones(shape, np.int32),
        "distance_m": np.zeros(shape, np.float32),
        "normal_dot_view": np.ones(shape, np.float32),
        "sky_view_factor": np.ones(shape, np.float32),
    }


def corner_over_centre(plane: np.ndarray, half: int = 24) -> float:
    """Mean of the four corners over the centre -- the vignetting signature in one number."""
    h, w = plane.shape
    corners = np.mean(
        [
            plane[:half, :half].mean(),
            plane[:half, -half:].mean(),
            plane[-half:, :half].mean(),
            plane[-half:, -half:].mean(),
        ]
    )
    centre = plane[h // 2 - half : h // 2 + half, w // 2 - half : w // 2 + half].mean()
    return float(corners / centre)


# --- what it removes ----------------------------------------------------------------------------


def test_the_raw_dn_plane_really_does_carry_cos4(config) -> None:  # type: ignore[no-untyped-def]
    """The premise, measured: an un-corrected camera looking at a scene warmer than its housing
    darkens toward the corners.

    Stated first so the fix below is measured against a number rather than an impression. Since
    SC.17 (§8.2, ADR 0145) the relative illumination multiplies the scene *minus* the housing, so
    the depth depends on how far the scene is from the housing: 400 K through a 300 K housing is
    the case this module was written for.
    """
    raw = uniform_signal_dn(config, 400.0)
    ratio = corner_over_centre(raw)
    assert ratio < 0.90, f"corner/centre {ratio:.3f}: the premise of this module is wrong"


def test_the_raw_shading_follows_the_scene_minus_the_housing(config) -> None:  # type: ignore[no-untyped-def]
    """SC.17: the sign of the raw shading is the sign of L_scene − L_housing (§8.2).

    A pixel off axis sees less scene and more housing. Warmer than the housing, the corners are
    darker; colder -- a clear sky -- they are *brighter*; at the housing temperature the camera is
    an isothermal enclosure and there is no shading at all. The old form, cos⁴ on the scene alone
    plus one self-emission number, darkened the corners at every temperature.
    """
    t_housing = config.t_housing_cal_k
    warm = corner_over_centre(uniform_signal_dn(config, t_housing + 60.0))
    cold = corner_over_centre(uniform_signal_dn(config, t_housing - 60.0))
    same = corner_over_centre(uniform_signal_dn(config, t_housing))
    assert warm < 1.0 < cold, f"warm {warm:.4f}, cold {cold:.4f}"
    assert abs(same - 1.0) < 1e-3, f"scene at the housing temperature: corner/centre {same:.5f}"


def test_a_uniform_scene_comes_out_uniform(corrected_config) -> None:  # type: ignore[no-untyped-def]
    """The definition of a flat field: one blackbody filling the field reads the same everywhere.

    To 0.1 %, which is far below the 21 % it started at and below a single DN code at these
    levels, so what is left cannot darken a corner.
    """
    raw = uniform_signal_dn(corrected_config, 300.0)
    corrected = corrected_config.flat_field.apply(raw)
    ratio = corner_over_centre(np.asarray(corrected))
    assert abs(ratio - 1.0) < 1e-3, f"corner/centre {ratio:.5f} after correction"


def test_it_is_flat_at_a_temperature_it_was_not_calibrated_at(corrected_config) -> None:  # type: ignore[no-untyped-def]
    """Two points calibrate a line, so the correction holds between and beyond them.

    Checked at 300 K, nowhere near either blackbody (233 K and 473 K): a correction that only
    worked at its calibration points would be describing those frames, not the camera.
    """
    for temperature in (250.0, 300.0, 400.0):
        corrected = corrected_config.flat_field.apply(
            uniform_signal_dn(corrected_config, temperature)
        )
        ratio = corner_over_centre(np.asarray(corrected))
        assert abs(ratio - 1.0) < 2e-3, f"{temperature} K: corner/centre {ratio:.5f}"


def test_the_correction_is_of_the_right_size(corrected_config) -> None:  # type: ignore[no-untyped-def]
    """The gain has to be about 1/cos⁴ at the corner, or it is correcting something else."""
    gain = np.asarray(corrected_config.flat_field.gain)
    h, w = gain.shape
    corner = float(np.mean([gain[0, 0], gain[0, -1], gain[-1, 0], gain[-1, -1]]))
    centre = float(gain[h // 2, w // 2])
    assert corner / centre == pytest.approx(1.0 / COS4_CORNER, rel=0.08)


# --- what it must not touch ---------------------------------------------------------------------


def test_dn16_and_the_radiometric_outputs_are_bit_identical_either_way(
    config, corrected_config
) -> None:  # type: ignore[no-untyped-def]
    """The flat field is a *display* correction; the linear outputs must not move by one bit.

    This is the half that is easy to get wrong and impossible to see. `invert_optics` already
    divides cos⁴ out per pixel, so a flat field applied before the fork would remove it twice and
    every apparent temperature off-axis would come back too warm -- smoothly, plausibly, and
    wrongly.
    """
    planes = uniform_planes(config, 300.0)
    plain = run_frame(planes, config, PipelineState())
    flat = run_frame(planes, corrected_config, PipelineState())

    assert np.array_equal(plain.dn16, flat.dn16)
    assert np.array_equal(plain.apparent_t, flat.apparent_t)
    assert np.array_equal(plain.radiance, flat.radiance)
    assert not np.array_equal(plain.display8, flat.display8), "the picture should have changed"


def gradient_planes(config, low_k: float = 262.0, high_k: float = 300.0):  # type: ignore[no-untyped-def]
    """A scene whose temperature varies with **row only** -- a sky gradient, in effect.

    A uniform scene is the wrong instrument for a display-level test: with the noise off, plateau
    equalisation sees a constant frame and stretches whatever residual is left across the whole
    8-bit range, so the picture is saturated nonsense however good the correction is. A gradient
    gives the AGC a real range to work with, and because the temperature is constant *along* a
    row, any left-to-right difference in the picture is the camera and nothing else.
    """
    rows, cols = config.sensor.sensor.fpa_shape
    k = config.supersample
    shape = (rows * k, cols * k)
    column = np.linspace(high_k, low_k, shape[0], dtype=np.float32)[:, None]
    planes = uniform_planes(config, 300.0)
    planes["temperature_k"] = np.broadcast_to(column, shape).astype(np.float32).copy()
    return planes


def row_band(image: np.ndarray, centre_row: int, half: int = 12) -> tuple[float, float]:
    """(edge, centre) mean grey over one band of rows -- constant scene temperature across it."""
    band = image[centre_row - half : centre_row + half]
    edge = float(np.concatenate([band[:, :24].ravel(), band[:, -24:].ravel()]).mean())
    middle = float(band[:, band.shape[1] // 2 - 24 : band.shape[1] // 2 + 24].mean())
    return edge, middle


def test_the_picture_stops_having_shaded_edges(config, corrected_config) -> None:  # type: ignore[no-untyped-def]
    """End to end, in the 8-bit image: the artefact the complaint was about.

    Along a row the scene temperature is constant, so the edges and the centre must read the same.
    Uncorrected they do not, which is a shading no real thermal clip has, because every real
    camera flat-fields before its AGC. The sign is not asserted: this scene is colder than the
    housing, so since SC.17 the uncorrected edges come out *brighter* (§8.2); the size is.
    """
    planes = gradient_planes(config)
    plain = np.asarray(run_frame(planes, config, PipelineState()).display8)[..., 0].astype(float)
    flat = np.asarray(run_frame(planes, corrected_config, PipelineState()).display8)[..., 0].astype(
        float
    )

    row = plain.shape[0] // 2
    plain_edge, plain_centre = row_band(plain, row)
    flat_edge, flat_centre = row_band(flat, row)

    shading = abs(plain_centre - plain_edge)
    assert shading > 20.0, (
        f"uncorrected should shade at the edges: {plain_edge:.1f} vs {plain_centre:.1f}"
    )
    assert abs(flat_edge - flat_centre) < 0.2 * shading, (
        f"still shading after correction: {flat_edge:.1f} vs {flat_centre:.1f}"
    )


def test_the_apparent_temperature_was_already_flat(config) -> None:  # type: ignore[no-untyped-def]
    """Why the correction is display-only, stated as a measurement rather than an argument."""
    planes = uniform_planes(config, 300.0)
    t_app = np.asarray(run_frame(planes, config, PipelineState()).apparent_t)
    assert abs(corner_over_centre(t_app) - 1.0) < 1e-3


# --- the operator -------------------------------------------------------------------------------


def test_the_pedestal_defaults_to_the_spec_convention() -> None:
    """Without ``restore_pedestal`` the corrected cold blackbody reads 0, as §11.2 says."""
    lo = np.array([[10.0, 12.0], [11.0, 13.0]], np.float32)
    hi = lo + np.array([[100.0, 110.0], [105.0, 95.0]], np.float32)
    plain = TwoPointNuc.calibrate(lo, hi)
    assert plain.pedestal == 0.0
    assert np.allclose(plain.apply(lo), 0.0, atol=1e-5)


def test_restoring_the_pedestal_keeps_the_dn_range() -> None:
    """With it, the corrected cold frame sits at the cold frame's own mean level.

    That is what lets the AGC downstream keep working in the units it was written for: a
    flat-fielded frame that read zero on a cold scene would move the histogram for reasons that
    have nothing to do with the scene.
    """
    lo = np.array([[10.0, 12.0], [11.0, 13.0]], np.float32)
    hi = lo + np.array([[100.0, 110.0], [105.0, 95.0]], np.float32)
    restored = TwoPointNuc.calibrate(lo, hi, restore_pedestal=True)
    assert restored.pedestal == pytest.approx(float(lo.mean()))
    assert np.allclose(restored.apply(lo), float(lo.mean()), atol=1e-5)
    assert np.allclose(restored.apply(hi), float(hi.mean()), atol=1e-4)


def test_calibration_refuses_a_pair_that_does_not_respond(config) -> None:  # type: ignore[no-untyped-def]
    """A pixel that answers the hot blackbody no more than the cold one has no gain to compute."""
    with pytest.raises(ValueError, match="hotter"):
        calibrate_flat_field(config, 400.0, 300.0)
