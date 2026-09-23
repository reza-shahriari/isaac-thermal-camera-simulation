"""OC.9 -- focus that moves (ADR 0129).

The servo is a model of passive contrast detection, not a shortcut to the right answer: it may only
look at the picture. The tests hold it to the three behaviours that distinguish a real one -- it
converges, it does not hunt, and it lags.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.optics.autofocus import AutofocusServo, focus_measure, track_distance_m
from irsim.optics.defocus import blur_circle_um, defocus_w020_um, depth_of_field_m
from irsim.optics.psf import DefocusKernelBank, apply_psf

FOCAL_MM, F_NUMBER = 14.0, 1.0
TRUTH_M = 8.0
SIZE = 64


@pytest.fixture(scope="module")
def bank() -> DefocusKernelBank:
    return DefocusKernelBank(10.5, F_NUMBER, 1.654, 12.0, 1, "hopkins")


def _target(seed: int = 0) -> np.ndarray:
    """A textured patch -- contrast detection needs edges to detect."""
    rng = np.random.default_rng(seed)
    plane = np.full((SIZE, SIZE), 1.0)
    plane[16:48, 16:48] = 4.0 + rng.uniform(0.0, 2.0, (32, 32))
    return plane


def _evaluator(bank, scene_m: float, plane: np.ndarray):
    def evaluate(focus_m: float) -> float:
        c = float(blur_circle_um(scene_m, FOCAL_MM, F_NUMBER, focus_m))
        kernel = bank.kernel_for(float(defocus_w020_um(c, F_NUMBER)))
        return focus_measure(apply_psf(plane, kernel))

    return evaluate


def test_the_focus_measure_peaks_where_the_lens_is_focused(bank) -> None:
    """Without this the servo is climbing the wrong hill and every other test is meaningless."""
    evaluate = _evaluator(bank, TRUTH_M, _target())
    distances = np.geomspace(2.0, 60.0, 21)
    scores = [evaluate(float(d)) for d in distances]
    assert distances[int(np.argmax(scores))] == pytest.approx(TRUTH_M, rel=0.2)


def test_the_measure_is_invariant_to_scene_radiance(bank) -> None:
    """Unnormalised, a servo would 'focus' by finding the hottest frame of a diurnal run."""
    plane = _target()
    assert focus_measure(plane * 7.3) == pytest.approx(focus_measure(plane), rel=1e-9)


def test_it_converges_on_a_static_scene(bank) -> None:
    servo = AutofocusServo(distance_m=40.0)
    evaluate = _evaluator(bank, TRUTH_M, _target())
    for _ in range(40):
        servo.update(evaluate)
    assert servo.distance_m == pytest.approx(TRUTH_M, rel=0.25), (
        f"converged to {servo.distance_m:.2f} m, target {TRUTH_M} m"
    )


def test_it_does_not_hunt_once_it_has_converged(bank) -> None:
    """A servo with no hysteresis steps forever on quantisation alone; the artefact is a focus
    that breathes on a static scene, which is visible and wrong."""
    servo = AutofocusServo(distance_m=40.0)
    evaluate = _evaluator(bank, TRUTH_M, _target())
    for _ in range(40):
        servo.update(evaluate)
    settled = [servo.update(evaluate) for _ in range(10)]
    assert max(settled) / min(settled) < 1.05, f"hunting: {np.round(settled, 3)}"


def test_it_lags_a_moving_target_and_catches_up_when_it_stops(bank) -> None:
    """The distinctive artefact, and the honest form of it.

    A servo that only ever probes one step either side cannot outrun a target closing faster than
    its step, so the error *grows* while the target moves -- that is the lag, and it is real. What
    the servo must do is keep pointing the right way and settle once the target does.
    """
    servo = AutofocusServo(distance_m=25.0)
    plane = _target()
    closing = list(np.geomspace(25.0, 8.0, 12))
    for scene_m in closing:
        servo.update(_evaluator(bank, float(scene_m), plane))
    during = abs(servo.distance_m - closing[-1]) / closing[-1]
    assert during > 0.05, "a closing target must be lagged, not tracked instantly"
    assert servo.distance_m < 25.0, "but the servo must be following it, not sitting still"

    for _ in range(40):  # the target stops; the servo should arrive
        servo.update(_evaluator(bank, 8.0, plane))

    # "Arrive" means inside the depth of field, not at the metre. A contrast measure cannot resolve
    # a lens position finer than the band over which the blur stays under one pixel, so demanding
    # better would be demanding the servo see something that is not in the picture. At 8 m this
    # Boson's depth of field runs from about 5.4 m to about 16 m.
    near, far = depth_of_field_m(FOCAL_MM, F_NUMBER, 12.0, 8.0)
    assert near < servo.distance_m < far, (
        f"{servo.distance_m:.2f} m is outside the {near:.1f}-{far:.1f} m depth of field of 8 m"
    )
    assert abs(servo.distance_m - 8.0) / 8.0 <= during + 1e-9, "it must not have drifted away"


def test_a_scene_with_no_contrast_leaves_the_lens_alone(bank) -> None:
    servo = AutofocusServo(distance_m=12.0)
    flat = np.full((SIZE, SIZE), 3.0)
    before = servo.distance_m
    for _ in range(5):
        servo.update(_evaluator(bank, TRUTH_M, flat))
    assert servo.distance_m == pytest.approx(before, rel=1e-12)


def test_the_probe_step_is_multiplicative() -> None:
    """Depth of field is multiplicative: a fixed metre step is wild at 5 m, invisible at 50."""
    near, far = AutofocusServo(distance_m=5.0), AutofocusServo(distance_m=50.0)
    for servo in (near, far):
        lo, here, hi = servo.probes()
        assert hi / here == pytest.approx(here / lo, rel=1e-9)


def test_tracking_takes_the_target_median_and_says_when_it_is_absent() -> None:
    distance = np.full((SIZE, SIZE), 100.0)
    distance[10:20, 10:20] = 7.0
    ids = np.ones((SIZE, SIZE), np.int32)
    ids[10:20, 10:20] = 42
    assert track_distance_m(distance, ids, 42) == pytest.approx(7.0)
    assert track_distance_m(distance, ids, 99) is None
    sky = np.zeros((SIZE, SIZE), bool)
    sky[10:15, 10:20] = True  # half the target is really sky, mis-segmented
    assert track_distance_m(distance, ids, 42, sky) == pytest.approx(7.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"step_ratio": 0.0}, "step_ratio"),
        ({"step_ratio": 1.5}, "step_ratio"),
        ({"damping": 0.0}, "damping"),
        ({"hysteresis": -0.1}, "hysteresis"),
    ],
)
def test_impossible_servos_are_refused(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        AutofocusServo(distance_m=10.0, **kwargs)
