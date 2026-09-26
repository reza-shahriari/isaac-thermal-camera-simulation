"""SC.20: the radial bowl of a featureless frame, measured one way for rendered and real frames.

`irsim.validation.radial.radial_fit` fits a plane plus a paraboloid about the principal point. The
plane takes the sky's elevation gradient; the paraboloid is the camera's radial shading (§8.2,
§11.2). These tests pin the fit on synthetic frames with a known answer, then check the sign a
rendered sky-only frame gives against the one §11.2 predicts for its housing history.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

from irsim.validation.radial import radial_fit

REPO = pathlib.Path(__file__).resolve().parents[2]
SHAPE = (120, 160)


def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = SHAPE
    yy, xx = np.mgrid[:h, :w]
    x, y = xx + 0.5 - w / 2, yy + 0.5 - h / 2
    rc = np.hypot(w / 2, h / 2)
    return x / rc, y / rc, (x * x + y * y) / rc**2


def test_a_known_bowl_under_a_sky_gradient_is_recovered() -> None:
    x, y, r2 = _grid()
    rng = np.random.default_rng(0)
    img = 100.0 + 30.0 * y - 8.0 * x - 40.0 * r2 + rng.normal(0.0, 0.5, SHAPE)
    fit = radial_fit(img)
    assert fit.k == pytest.approx(-40.0, rel=0.02)
    assert fit.plane[2] == pytest.approx(30.0, rel=0.02)
    assert fit.sign == -1 and fit.monotonic and fit.radial_share > 0.95


def test_a_plane_alone_has_no_bowl() -> None:
    x, y, _ = _grid()
    rng = np.random.default_rng(1)
    fit = radial_fit(50.0 + 20.0 * y + rng.normal(0.0, 1.0, SHAPE))
    assert abs(fit.k) < 0.5 and fit.radial_share < 0.05 and fit.sign == 0


def test_a_hot_pixel_and_a_masked_region_do_not_move_it() -> None:
    _, _, r2 = _grid()
    img = 10.0 + 12.0 * r2
    img[30, 40] = 1e4
    fit = radial_fit(img)
    assert fit.k == pytest.approx(12.0, rel=1e-3) and fit.outliers >= 1
    masked = img.copy()
    masked[:20] = 999.0  # a horizon, say
    keep = np.ones(SHAPE, bool)
    keep[:20] = False
    assert radial_fit(masked, mask=keep).k == pytest.approx(12.0, rel=1e-3)


def test_an_rgb_frame_and_float16_are_handled() -> None:
    _, _, r2 = _grid()
    img = 10.0 + 5.0 * r2
    rgb = np.repeat(img[..., None], 3, axis=-1)
    assert radial_fit(rgb).k == pytest.approx(5.0, rel=1e-6)
    with pytest.raises(TypeError, match="float16"):
        radial_fit(img.astype(np.float16))


@pytest.mark.slow  # renders two 640x512 sky frames through the full chain
@pytest.mark.parametrize(("drift_k", "sign"), [(-1.5, -1), (1.5, +1)])
def test_a_rendered_sky_bowls_the_way_the_housing_moved(
    tmp_path: pathlib.Path, drift_k: float, sign: int
) -> None:
    """The bench end to end: a housing that cooled since the shutter leaves a bright centre."""
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        import validate_sky_flat as bench
    finally:
        sys.path.pop(0)
    frames = bench.render_pair(bench.DEFAULT_SENSOR, bench.DEFAULT_SCENE, 45.0, drift_k, 120.0)
    bowl = radial_fit(frames["drift_bowl"])
    assert bowl.sign == sign and bowl.monotonic and bowl.radial_share > 0.95
    # right after the shutter the camera adds nothing: what is left is the sky's own shape
    assert abs(radial_fit(frames["at_shutter"]).k) > 0.0  # the sky is not flat...
    assert abs(bowl.k) > 0.3 * abs(radial_fit(frames["at_shutter"]).k)  # ...and the drift shows
