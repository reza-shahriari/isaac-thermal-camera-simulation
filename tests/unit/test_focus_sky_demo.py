"""OC.12 -- the focus pulled from a cloudy sky onto a cube (ADR 0129, ADR 0134).

The demonstration is a **crossover**, so the tests are about the crossover rather than about
anything being blurry: each region must be sharpest when the lens is on it and measurably softer
when the lens is on the other. A test that only asserted "the sky got blurrier" would pass against
a render that blurred everything, which is the failure worth catching.

`test_a_clear_sky_barely_changes_at_all` is the one that says why the demo has clouds in it. It is
not a robustness check; it is the physics claim the scene is built on, asserted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from focus_sky_demo import (  # noqa: E402
    BAND_UM,
    F_NUMBER,
    FOCAL_MM,
    PITCH_UM,
    SIGMA_ABERR_UM,
    apparent_temperature,
    blur_circles_um,
    focus_schedule,
    render,
    sharpness,
    sky_cube_scene,
)

from irsim.optics.defocus import blur_circle_um  # noqa: E402
from irsim.optics.psf import DefocusKernelBank  # noqa: E402

WIDTH, HEIGHT, SS = 96, 80, 2
CUBE_M, SIDE_M, BORESIGHT_DEG = 3.0, 0.6, 20.0
SKY_FOCUS_M = 2000.0
#: The demo's own cloud grid, deliberately. A coarser one halves the measured effect -- 1.7x
#: instead of 3.1x at an eighth of a degree -- because blur attenuates high spatial frequencies
#: hardest, so a test run on coarser cloud would be measuring a different camera than the one that
#: ships. Cells per *pixel* is what matters and that is set by the lens, not by the frame size.


@pytest.fixture(scope="module")
def bank() -> DefocusKernelBank:
    return DefocusKernelBank(
        0.5 * (BAND_UM[0] + BAND_UM[1]), F_NUMBER, SIGMA_ABERR_UM, PITCH_UM, SS
    )


@pytest.fixture(scope="module")
def scene():
    return sky_cube_scene(WIDTH, HEIGHT, SS, CUBE_M, SIDE_M, BORESIGHT_DEG)


def _measure(scene, bank, focus_m: float) -> dict[str, float]:
    return sharpness(scene, apparent_temperature(render(scene, focus_m, bank)))


def test_each_region_is_sharpest_when_the_lens_is_on_it(scene, bank) -> None:
    """The crossover, which is the whole demonstration."""
    on_sky = _measure(scene, bank, SKY_FOCUS_M)
    on_cube = _measure(scene, bank, CUBE_M)
    assert on_sky["sky"] > on_cube["sky"], "the sky must soften when the lens leaves it"
    assert on_cube["cube"] > on_sky["cube"], "the cube must sharpen when the lens arrives"
    # A crossover, not a global blur: if the render simply blurred harder at one setting, both
    # regions would move the same way and this pair of assertions could not both hold.
    assert on_sky["sky"] / on_cube["sky"] > 2.0, "the sky's loss must be worth looking at"


def test_a_clear_sky_barely_changes_at_all(bank) -> None:
    """Why the demo carries cloud, asserted rather than claimed in a docstring.

    Defocus is a low-pass filter: it removes detail, and a clear LWIR sky is a smooth ramp in
    elevation with almost none to remove. Pulling the lens the whole way across a cloudless sky
    must therefore do far less to it than the same pull does to a cloudy one -- so a demo built on
    a clear sky would render two frames a reader could not tell apart, and would be a fair picture
    of the physics while being a useless picture of focus.
    """
    clear = sky_cube_scene(WIDTH, HEIGHT, SS, CUBE_M, SIDE_M, BORESIGHT_DEG, cloud_fraction=0.0)
    cloudy = sky_cube_scene(WIDTH, HEIGHT, SS, CUBE_M, SIDE_M, BORESIGHT_DEG)
    ratio = {}
    for name, s in (("clear", clear), ("cloudy", cloudy)):
        far, near = _measure(s, bank, SKY_FOCUS_M)["sky"], _measure(s, bank, CUBE_M)["sky"]
        ratio[name] = far / near
    assert ratio["clear"] < 1.1, f"a clear sky should hardly notice the pull: {ratio['clear']:.3f}"
    assert ratio["cloudy"] > 2.0 * ratio["clear"], f"cloud is what carries the effect: {ratio}"


def test_the_blur_circles_are_the_thin_lens_ones(scene) -> None:
    """The readout burnt into every frame has to be the real quantity, not an illustration."""
    on_cube = blur_circles_um(scene, CUBE_M)
    assert on_cube["cube"] == pytest.approx(0.0, abs=1e-12), "focused on it is focused on it"
    assert on_cube["sky"] == pytest.approx(
        float(blur_circle_um(1e9, FOCAL_MM, F_NUMBER, CUBE_M)), rel=1e-12
    )
    on_sky = blur_circles_um(scene, SKY_FOCUS_M)
    assert on_sky["cube"] == pytest.approx(
        float(blur_circle_um(CUBE_M, FOCAL_MM, F_NUMBER, SKY_FOCUS_M)), rel=1e-12
    )
    # Focusing 2 km away is all but focusing at infinity for this lens, which is why the sweep can
    # start there and still be called "focused on the sky".
    assert on_sky["sky"] < 0.01 * on_cube["sky"]


def test_the_background_is_defined_behind_the_cube_too(scene) -> None:
    """`OC.7`'s precondition: the analytic sky must be finite on the rays the cube hides.

    This is what makes the demo exact rather than bounded. If the background plane were only
    defined where sky is visible, the gap opened behind the cube's defocused silhouette would be
    filled with a guess, and `OC.8`'s 2.9-5.1 K would apply here too.
    """
    hidden = scene.background_radiance[scene.cube_mask]
    assert hidden.size > 0 and np.all(np.isfinite(hidden)) and np.all(hidden > 0.0)
    assert hidden.min() < hidden.max(), "and it must carry the cloud structure, not one value"


def test_the_sky_warms_toward_the_horizon_and_cloud_is_warmer_still(scene) -> None:
    """The two gradients the sky is made of, each measured where it is the one in charge.

    The elevation ramp is read on a **clear** sky on purpose. Cloud is a random field, so which
    quarter of a cloudy frame holds more of it is a coin toss worth tens of kelvin, against the
    ramp's few across this lens's 7.5 degrees -- measuring the ramp through cloud reads the coin.
    """
    clear = sky_cube_scene(WIDTH, HEIGHT, SS, CUBE_M, SIDE_M, BORESIGHT_DEG, cloud_fraction=0.0)
    t_clear = apparent_temperature(clear.background_radiance)
    rows = t_clear.shape[0]
    top, bottom = t_clear[: rows // 4].mean(), t_clear[-rows // 4 :].mean()
    assert bottom > top + 1.0, f"the slant path is longer low in frame: {top:.1f} -> {bottom:.1f} K"

    # Cloud is the warm population: it is opaque at its base temperature while the clear column
    # only reaches a fraction of air temperature through the 8-14 um window.
    t_k = apparent_temperature(scene.background_radiance)
    assert np.percentile(t_k, 95) - np.percentile(t_k, 5) > 15.0, "cloud must stand out of the sky"


def test_the_pull_is_linear_in_dioptres_and_returns(scene) -> None:
    """A sweep linear in metres would spend nine tenths of itself with nothing moving."""
    schedule = focus_schedule(CUBE_M, SKY_FOCUS_M, hold=3, ramp=9)
    assert len(schedule) == 2 * (3 + 9)
    assert schedule[0] == pytest.approx(SKY_FOCUS_M) and schedule[-1] == pytest.approx(SKY_FOCUS_M)
    assert min(schedule) == pytest.approx(CUBE_M), "the pull must reach the cube"
    ramp = np.array(schedule[3:12])
    steps = np.diff(1.0 / ramp)
    assert np.allclose(steps, steps[0]), "equal steps in dioptres, so the blur moves at one rate"
