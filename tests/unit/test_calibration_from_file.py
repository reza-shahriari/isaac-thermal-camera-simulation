"""SC.33 -- calibration loaded from the camera file.

A camera file may now carry what a bench measured on the unit: ``calibration.geometric`` (OpenCV's
pinhole and Brown-Conrady lens, in pixels) and ``calibration.radiometric`` (gain, offset and
bad-pixel maps as hashed ``.npy`` sidecars, a measured NETD and SITF). Each is used when present
and synthesised when absent, and a file with no ``calibration:`` block is the camera it always was,
bit for bit -- which the goldens keyed on the Boson's config hash prove in the same gate. The
geometric half is checked against OpenCV itself: a checkerboard rendered through k1 = −0.3
straightens under ``cv2.undistort`` to a tenth of a pixel, and the corners in the distorted frame
sit where ``distort_normalised`` puts them. Roadmap SC.33; ADR 0164; §8.4, §10.3, §10.4, §12.2.
"""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest
import yaml

from irsim.config.loader import config_hash, load_sensor_config
from irsim.config.sensor import GeometricCalibrationSpec, SensorConfig
from irsim.materials.table import MaterialTable
from irsim.noise.defects import replacement_mask
from irsim.optics.calibration import distort_image, distort_planes, is_identity, source_positions
from irsim.optics.projection import Intrinsics, distort_normalised
from irsim.pipeline import PipelineConfig, PipelineState, attach_sensor_chain, run_frame

REPO = pathlib.Path(__file__).resolve().parents[2]
BOSON_YAML = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
SHAPE = (64, 80)

#: The Boson's own pinhole in pixels (14 mm on 12 µm) with a strong barrel lens.
GEO = GeometricCalibrationSpec(fx_px=1166.6667, fy_px=1166.6667, cx_px=320.0, cy_px=256.0, k1=-0.3)


def _checkerboard(
    width: int = 640, height: int = 512, square: int = 48, n: tuple[int, int] = (9, 7)
):  # type: ignore[no-untyped-def]
    """A centred board of ``n`` squares, float32 in [0.2, 0.8]; its inner corners in pixel-edge
    coordinates (a corner at x = j means between pixels j-1 and j)."""
    img = np.full((height, width), 0.5, dtype=np.float32)
    x0 = (width - n[0] * square) // 2
    y0 = (height - n[1] * square) // 2
    for r in range(n[1]):
        for c in range(n[0]):
            img[
                y0 + r * square : y0 + (r + 1) * square, x0 + c * square : x0 + (c + 1) * square
            ] = 0.8 if (r + c) % 2 == 0 else 0.2
    corners = np.array(
        [(x0 + c * square, y0 + r * square) for r in range(1, n[1]) for c in range(1, n[0])],
        dtype=np.float64,
    )
    return img, corners, (n[0] - 1, n[1] - 1)


def _find_corners(cv2, img: np.ndarray, pattern: tuple[int, int]) -> np.ndarray:  # type: ignore[no-untyped-def]
    u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    ok, pts = cv2.findChessboardCorners(u8, pattern)
    assert ok, "OpenCV must find the board"
    pts = cv2.cornerSubPix(
        u8, pts, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)
    )
    # OpenCV's pixel-centre convention (pixel j centred at j) vs this repo's pixel-edge one
    return pts.reshape(-1, 2).astype(np.float64) + 0.5


def _sorted(pts: np.ndarray) -> np.ndarray:
    return pts[np.lexsort((np.round(pts[:, 0]), np.round(pts[:, 1] / 8.0)))]


# --- the lens ------------------------------------------------------------------------------------


def test_a_checkerboard_through_the_lens_straightens_under_opencv_to_a_tenth_of_a_pixel() -> None:
    cv2 = pytest.importorskip("cv2")
    ideal, truth, pattern = _checkerboard()
    intr = Intrinsics(GEO.fx_px, GEO.fy_px, GEO.cx_px, GEO.cy_px, 640, 512)
    distorted = distort_image(ideal, intr, GEO.distortion)
    assert distorted.dtype == np.float32 and distorted.shape == ideal.shape
    assert not np.array_equal(distorted, ideal)
    # where the lens puts each corner, by the projection model this repo already trusts
    xn, yn = (truth[:, 0] - GEO.cx_px) / GEO.fx_px, (truth[:, 1] - GEO.cy_px) / GEO.fy_px
    xd, yd = distort_normalised(xn, yn, GEO.distortion)
    predicted = np.stack([xd * GEO.fx_px + GEO.cx_px, yd * GEO.fy_px + GEO.cy_px], axis=1)
    found = _find_corners(cv2, distorted, pattern)
    err = np.linalg.norm(_sorted(found) - _sorted(predicted), axis=1)
    assert err.max() < 0.15, err.max()  # cornerSubPix on a twice-resampled edge
    assert np.linalg.norm(_sorted(predicted) - _sorted(truth), axis=1).max() > 1.5, "it did bend"
    # and OpenCV's own undistort, given the same K and D, puts the corners back
    k_mat = np.array([[GEO.fx_px, 0, GEO.cx_px - 0.5], [0, GEO.fy_px, GEO.cy_px - 0.5], [0, 0, 1]])
    dist = np.array([GEO.k1, GEO.k2, GEO.p1, GEO.p2, GEO.k3])
    straight = cv2.undistort(np.clip(distorted * 255, 0, 255).astype(np.uint8), k_mat, dist)
    back = _find_corners(cv2, straight.astype(np.float32) / 255.0, pattern)
    err_back = np.linalg.norm(_sorted(back) - _sorted(truth), axis=1)
    assert err_back.max() < 0.1, err_back.max()


def test_a_pinhole_is_the_identity_and_planes_resample_by_their_dtype() -> None:
    intr = Intrinsics(GEO.fx_px, GEO.fy_px, GEO.cx_px, GEO.cy_px, 640, 512)
    pinhole = GeometricCalibrationSpec(fx_px=1.0, fy_px=1.0, cx_px=0.0, cy_px=0.0).distortion
    assert is_identity(pinhole) and not is_identity(GEO.distortion)
    img = np.random.default_rng(1).random((512, 640), dtype=np.float32)
    assert distort_image(img, intr, pinhole) is img
    ids = np.arange(512 * 640, dtype=np.int32).reshape(512, 640)
    mask = ids % 7 == 0
    planes = {"temperature_k": img, "part_id": ids, "sky_mask": mask, "observer_height_m": 3.0}
    out = distort_planes(planes, intr, GEO.distortion)
    assert out["observer_height_m"] == 3.0
    assert out["part_id"].dtype == np.int32 and out["sky_mask"].dtype == np.bool_
    assert set(np.unique(out["part_id"])) <= set(np.unique(ids)), "nearest: no invented ids"
    assert out["temperature_k"].dtype == np.float32
    xs, ys = source_positions(intr, GEO.distortion)
    # a barrel lens pulls the field in: the corner pixel shows a ray from outside the ideal frame
    assert xs[0, 0] < 0.5 and ys[0, 0] < 0.5 and abs(xs[256, 320] - 320.5) < 1e-6
    with pytest.raises(TypeError, match="float16"):
        distort_image(img.astype(np.float16), intr, GEO.distortion)


def test_the_measured_pinhole_reaches_the_projection_and_the_ros_camera_info() -> None:
    d = yaml.safe_load(BOSON_YAML.read_text())
    d["sensor"]["calibration"] = {
        "geometric": {"fx_px": 1200.0, "fy_px": 1180.0, "cx_px": 330.0, "cy_px": 250.0, "k1": -0.2}
    }
    spec = SensorConfig.model_validate(d).sensor
    intr = Intrinsics.from_sensor(spec, 2)
    assert (intr.fx_px, intr.fy_px, intr.cx_px, intr.cy_px) == (2400.0, 2360.0, 660.0, 500.0)
    assert spec.effective_distortion.model == "brown_conrady"
    assert spec.effective_distortion.coeffs == [-0.2, 0.0, 0.0, 0.0, 0.0]
    plain = load_sensor_config("boson640").sensor
    assert plain.calibration is None and plain.effective_distortion == plain.optics.distortion
    from irsim_isaac.ros2_bridge import camera_info

    info = camera_info(SensorConfig.model_validate(d), t_s=0.0)
    assert info.k[0] == 1200.0 and info.k[2] == 330.0 and info.d[0] == -0.2


# --- the radiometric maps -------------------------------------------------------------------------


def _boson(**calibration):  # type: ignore[no-untyped-def]
    d = copy.deepcopy(yaml.safe_load(BOSON_YAML.read_text()))
    d["sensor"]["fpa"].update(width=SHAPE[1], height=SHAPE[0])
    d["sensor"]["optics"].update(
        supersample_factor=1, housing_temp_mode="fixed", housing_temp_k=295.0
    )
    if calibration:
        d["sensor"]["calibration"] = {"radiometric": calibration}
    return SensorConfig.model_validate(d)


def _uniform_frame(sensor: SensorConfig, lut, t_k: float = 300.0):  # type: ignore[no-untyped-def]
    config = PipelineConfig.from_sensor(
        sensor, MaterialTable.constant(1.0), lut=lut, sensor_seed=5, noise_enabled=False
    )
    config = attach_sensor_chain(config, defects_enabled=False, residual_enabled=False)
    planes = {
        "temperature_k": np.full(SHAPE, t_k, dtype=np.float32),
        "material_id": np.ones(SHAPE, dtype=np.int32),
        "distance_m": np.zeros(SHAPE, dtype=np.float32),
    }
    return np.asarray(
        run_frame(planes, config, PipelineState()).signal_dn, dtype=np.float64
    ), config


def test_a_loaded_gain_map_reproduces_its_own_flat_field_to_a_dn(
    tmp_path: pathlib.Path, boson_lut
) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(9)
    gain = (1.0 + 0.02 * rng.standard_normal(SHAPE)).astype(np.float32)
    offset = (5.0 * rng.standard_normal(SHAPE)).astype(np.float32)
    np.save(tmp_path / "gain.npy", gain)
    np.save(tmp_path / "offset.npy", offset)
    ideal, _ = _uniform_frame(_boson(), boson_lut)
    calibrated = _boson(
        gain_map=str(tmp_path / "gain.npy"), offset_map=str(tmp_path / "offset.npy")
    )
    measured, config = _uniform_frame(calibrated, boson_lut)
    assert np.abs(measured - (gain * ideal + offset)).max() < 1e-2, "the map is the flat field"
    assert np.abs(measured - ideal).max() > 50.0, "and it is not the ideal one"
    assert config.chain is not None
    assert not np.any(config.chain.pattern.pattern.vh), "the drawn pattern is not added on top"
    assert np.any(_uniform_frame(_boson(), boson_lut)[1].chain.pattern.pattern.vh)
    # the maps are inputs: their content is in the hash, and a different map is a different camera
    h1 = config_hash(calibrated)
    np.save(tmp_path / "gain.npy", (gain * 1.001).astype(np.float32))
    assert config_hash(calibrated) != h1 and h1 != config_hash(_boson())


def test_a_loaded_bad_pixel_map_is_the_camera_s_own_map(tmp_path: pathlib.Path, boson_lut) -> None:  # type: ignore[no-untyped-def]
    mask = np.zeros(SHAPE, dtype=bool)
    mask[3, 4] = mask[40, 70] = mask[41, 70] = True
    np.save(tmp_path / "bad.npy", mask)
    _, config = _uniform_frame(_boson(bad_pixel_map=str(tmp_path / "bad.npy")), boson_lut)
    chain = config.chain
    assert chain is not None and chain.bad_pixels.count == 3
    assert np.array_equal(replacement_mask(chain.bad_pixels, chain.defect_state), mask)
    np.save(tmp_path / "bad16.npy", mask.astype(np.uint16))
    with pytest.raises(TypeError, match="boolean"):
        _uniform_frame(_boson(bad_pixel_map=str(tmp_path / "bad16.npy")), boson_lut)
    np.save(tmp_path / "gain16.npy", np.ones(SHAPE, dtype=np.float16))
    with pytest.raises(TypeError, match="float32"):
        _uniform_frame(_boson(gain_map=str(tmp_path / "gain16.npy")), boson_lut)
    np.save(tmp_path / "wrong.npy", np.ones((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        _uniform_frame(_boson(offset_map=str(tmp_path / "wrong.npy")), boson_lut)


def test_a_measured_netd_and_sitf_are_the_anchors(boson_lut) -> None:  # type: ignore[no-untyped-def]
    """The Boson's datasheet says 50 mK; a bench on this unit says 30 mK and 150 DN/K."""
    _, plain = _uniform_frame(_boson(), boson_lut)
    _, measured = _uniform_frame(_boson(netd_k=0.030, sitf_dn_per_k=150.0), boson_lut)
    ratio = measured.detector.budget.sigma_gaussian / plain.detector.budget.sigma_gaussian
    assert ratio == pytest.approx(0.6, abs=0.01), ratio  # a bolometer's floor is the anchor's
    assert measured.chain is not None and measured.chain.residual.dn_per_k == 150.0
    assert plain.chain is not None and plain.chain.residual.dn_per_k != 150.0


def test_a_file_without_a_calibration_block_is_the_camera_as_before() -> None:
    """The Boson's config hash is the one its goldens were recorded against."""
    cfg = load_sensor_config("boson640")
    assert cfg.sensor.calibration is None
    d = yaml.safe_load(BOSON_YAML.read_text())
    d["sensor"]["calibration"] = None
    spelled = SensorConfig.model_validate(d)
    assert spelled.sensor.calibration is None and config_hash(spelled) == config_hash(cfg)
    d["sensor"]["calibration"] = {"geometric": None, "radiometric": None}
    assert config_hash(SensorConfig.model_validate(d)) == config_hash(cfg), (
        "an empty block: nothing"
    )
    d["sensor"]["calibration"] = {"radiometric": {"netd_k": 0.03}}
    assert config_hash(SensorConfig.model_validate(d)) != config_hash(cfg), (
        "a measurement: a camera"
    )
