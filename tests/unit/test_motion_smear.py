"""Within-frame motion smear (§8.3, §9.2): the spatial twin of the cascade's ``mtf_motion``.

The MTF cascade has carried |sinc(v t_int xi)| since M5 and nothing applied it, so a rendered
frame was sharp however fast the scene crossed it. The test that matters most here is therefore
not that the operator blurs -- anything blurs -- but that it blurs by **exactly the amount the
cascade already claims**. Two descriptions of one effect that disagreed would be worse than one.

That cross-check found a real error. Taps placed at the segment's *endpoints* look natural and
implement a boxcar of length s + s/(N-1): measured MTF 0.7182 where sinc said 0.7842, which is
precisely the Dirichlet kernel of the longer smear. The operator was self-consistent and
describing the wrong thing, which is the failure mode no amount of "does it look blurred" would
have caught.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from irsim.optics.smear import (
    NEGLIGIBLE_SMEAR_PX,
    apply_motion_smear,
    smear_decay_frames,
    smear_duty,
)

SIZE = 256

#: Agreement required with the analytic sinc. The residual is bilinear interpolation at taps that
#: do not land on integers, worst where the smear is short (few taps) and the frequency high;
#: measured worst case 0.015 over the sweep below.
MTF_TOLERANCE = 0.02


def uniform_motion(vx: float, vy: float = 0.0) -> np.ndarray:
    motion = np.zeros((SIZE, SIZE, 2))
    motion[..., 0] = vx
    motion[..., 1] = vy
    return motion


def ripple(frequency_cyc_px: float) -> np.ndarray:
    x = np.arange(SIZE)
    return np.tile(1.0 + 0.5 * np.sin(2.0 * np.pi * frequency_cyc_px * x), (SIZE, 1))


def measured_mtf(smear_px: float, frequency_cyc_px: float) -> float:
    out = apply_motion_smear(ripple(frequency_cyc_px), uniform_motion(smear_px), 1.0)
    row = out[SIZE // 2, 60:200]
    return float((row.max() - row.min()) / 2.0 / 0.5)


# --- the cross-check against the cascade -------------------------------------------------------


@pytest.mark.parametrize("smear_px", [3.0, 6.0, 12.0, 20.0])
@pytest.mark.parametrize("frequency", [1 / 48, 1 / 32, 1 / 16, 1 / 12])
def test_the_smear_matches_the_cascade_it_implements(smear_px: float, frequency: float) -> None:
    """|sinc(s f)| -- the same number `mtf_motion` returns, from a spatial operator.

    This is the whole justification for the operator: the frequency-domain term was already in the
    cascade and describes the same physics, so a spatial implementation that disagreed with it
    would mean one of the two is wrong and there would be no way to tell which from a picture.
    """
    assert measured_mtf(smear_px, frequency) == pytest.approx(
        abs(np.sinc(smear_px * frequency)), abs=MTF_TOLERANCE
    )


def test_smearing_past_the_first_zero_kills_the_pattern() -> None:
    """At s f = 1 the boxcar nulls that frequency outright; a blur that merely softened it would
    be a different operator wearing the same name."""
    assert measured_mtf(12.0, 1.0 / 12.0) < 0.05


def test_more_smear_means_less_contrast_at_every_frequency() -> None:
    for frequency in (1 / 32, 1 / 16):
        values = [measured_mtf(s, frequency) for s in (0.5, 3.0, 6.0, 12.0)]
        assert all(a > b for a, b in zip(values, values[1:], strict=False)), values


# --- the operator's own guarantees --------------------------------------------------------------


def test_a_still_scene_is_returned_untouched() -> None:
    """Bit-identical, not merely close: a static frame must not pay a numerical cost for the
    existence of a smear stage."""
    image = ripple(1 / 16)
    assert apply_motion_smear(image, uniform_motion(0.0), 1.0) is image or np.array_equal(
        apply_motion_smear(image, uniform_motion(0.0), 1.0), image
    )


def test_sub_pixel_motion_is_left_alone() -> None:
    """Below a quarter pixel the detector's own box filter already covers it."""
    image = ripple(1 / 16)
    assert np.array_equal(
        apply_motion_smear(image, uniform_motion(NEGLIGIBLE_SMEAR_PX * 0.5), 1.0), image
    )


def test_zero_duty_smears_nothing() -> None:
    """An infinitely short integration freezes the scene however fast it is moving."""
    image = ripple(1 / 16)
    assert np.array_equal(apply_motion_smear(image, uniform_motion(20.0), 0.0), image)


def test_the_smear_is_centred_not_trailing() -> None:
    """A feature's centroid must not move. A trailing segment would displace every moving thing
    by half its smear -- a shift that looks like a timing error because it is one."""
    image = np.zeros((SIZE, SIZE))
    image[:, SIZE // 2] = 1.0
    out = apply_motion_smear(image, uniform_motion(11.0), 1.0)
    column = np.arange(SIZE)
    centroid = float((out[SIZE // 2] * column).sum() / out[SIZE // 2].sum())
    assert centroid == pytest.approx(SIZE / 2, abs=0.05)


def test_it_smears_along_the_motion_direction() -> None:
    """Vertical motion blurs vertically and leaves a vertical line alone; horizontal does the
    opposite. Swapped axes would still produce a plausible blur."""
    image = np.zeros((SIZE, SIZE))
    image[:, SIZE // 2] = 1.0  # a vertical line
    along = apply_motion_smear(image, uniform_motion(0.0, 11.0), 1.0)
    across = apply_motion_smear(image, uniform_motion(11.0, 0.0), 1.0)
    assert np.allclose(along, image, atol=1e-9), "motion along a line does not blur it"
    assert float(across.max()) < 0.2, "motion across a line spreads it out"


def test_a_still_region_beside_a_moving_one_stays_sharp() -> None:
    """Spatially varying, which is the case the scenes here actually need.

    Under a tracking mount the target is stationary on the focal plane while the sky sweeps past,
    so a single convolution cannot serve both. If the operator used one kernel for the frame, the
    tracked half of this test would blur.
    """
    image = ripple(1 / 16)
    motion = np.zeros((SIZE, SIZE, 2))
    motion[:, SIZE // 2 :, 0] = 12.0
    out = apply_motion_smear(image, motion, 1.0)
    still, moving = out[:, : SIZE // 2 - 20], out[:, SIZE // 2 + 20 :]
    assert np.allclose(still, image[:, : SIZE // 2 - 20], atol=1e-9)
    assert float(moving.max() - moving.min()) < 0.5 * float(
        image[:, SIZE // 2 + 20 :].max() - image[:, SIZE // 2 + 20 :].min()
    )


def test_it_conserves_the_total() -> None:
    """A smear redistributes energy; it does not create or destroy it, away from the borders."""
    rng = np.random.default_rng(4)
    image = rng.random((SIZE, SIZE)) + 1.0
    out = apply_motion_smear(image, uniform_motion(7.0), 1.0)
    interior = slice(20, SIZE - 20)
    assert float(out[interior, interior].mean()) == pytest.approx(
        float(image[interior, interior].mean()), rel=2e-3
    )


def test_it_refuses_inputs_it_cannot_interpret() -> None:
    image = ripple(1 / 16)
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        apply_motion_smear(np.zeros((4, 4, 2)), np.zeros((4, 4, 2)), 1.0)
    with pytest.raises(ValueError, match="motion_px must be"):
        apply_motion_smear(image, np.zeros((SIZE, SIZE, 3)), 1.0)
    with pytest.raises(ValueError, match="duty"):
        apply_motion_smear(image, uniform_motion(1.0), 1.5)


# --- the duty, which is where the two detector families part ------------------------------------


def test_a_bolometer_integrates_the_whole_frame() -> None:
    """`integration_time_ms` is None for a bolometer *because* it has no integration window.

    Reading that as zero would make every uncooled camera in the repository sharper than it is --
    and sharper is the direction that flatters a simulator, which is why it gets a test.
    """
    assert smear_duty(1 / 60, None) == 1.0


def test_a_cooled_detector_integrates_a_fraction_of_it() -> None:
    assert smear_duty(1 / 60, 0.008) == pytest.approx(0.48)
    assert smear_duty(1 / 60, 0.0) == 0.0


def test_the_duty_cannot_exceed_the_frame() -> None:
    """A config that claims a longer integration than its frame period is clamped, not trusted."""
    assert smear_duty(1 / 60, 1.0) == 1.0


def test_the_duty_refuses_nonsense() -> None:
    with pytest.raises(ValueError):
        smear_duty(0.0, None)
    with pytest.raises(ValueError):
        smear_duty(1 / 60, -0.001)


# --- the wiring, which is the part that was missing for a year --------------------------------


def boson_sensor() -> object:
    """The committed Boson config -- the uncooled case, whose `integration_time_ms` is None."""
    from irsim.config.loader import load_sensor_config

    return load_sensor_config(
        pathlib.Path(__file__).resolve().parents[2]
        / "configs"
        / "sensors"
        / "flir_boson_640_lwir.yaml"
    ).sensor


def radiance_ramp(sensor: object) -> np.ndarray:
    """A band-radiance plane on the sensor's own grid, with structure along x to smear."""
    height, width = sensor.fpa.height, sensor.fpa.width  # type: ignore[attr-defined]
    return np.tile(np.linspace(5.0, 40.0, width), (height, 1)) + np.tile(
        2.0 * np.sin(np.arange(width) * 0.6), (height, 1)
    )


def test_the_optics_stage_actually_applies_it() -> None:
    """A `motion_px` plane must change the rendered flux. The term existed and did nothing.

    `mtf_motion` has been in the cascade since M5 and no code path called it, so every frame this
    simulator has produced was sharp regardless of how fast the scene crossed it. This asserts the
    connection rather than the formula -- the formula is checked against sinc above.
    """
    from irsim.optics.stage import apply_optics

    sensor = boson_sensor()
    radiance = radiance_ramp(sensor)
    still = apply_optics(radiance, sensor, 0.0, supersample=1)
    motion = np.zeros((*radiance.shape, 2))
    motion[..., 0] = 9.0
    moving = apply_optics(radiance, sensor, 0.0, supersample=1, motion_px=motion)
    # Compared *relatively*. Pixel power here is of order 1e-9 W, and `np.allclose`'s default
    # atol of 1e-8 is larger than the whole signal -- it called a 15 % change "close". An absolute
    # tolerance against a quantity whose scale you have not checked is not a comparison.
    relative = float(np.abs(moving - still).max() / np.abs(still).max())
    assert relative > 0.01, f"motion_px changed the flux by only {relative:.2%}"


def test_a_bolometer_smears_more_than_a_cooled_detector() -> None:
    """The §16 checklist line, as arithmetic: same motion, same frame rate, different duty.

    A microbolometer integrates continuously and smears over the whole frame period; a cooled
    photon detector integrates briefly and is idle for the rest. So for identical scene motion the
    uncooled camera is the blurrier one -- which is the opposite of the intuition that a
    "better" detector must be sharper in every respect, and is why it is worth pinning.
    """
    from irsim.pipeline.optics import motion_for_integration

    sensor = boson_sensor()
    assert sensor.fpa.integration_time_ms is None, "the Boson is the uncooled case"
    planes = {"motion_px": np.full((8, 8, 2), 10.0)}
    bolometer = motion_for_integration(planes, sensor)
    assert bolometer is not None
    assert float(bolometer.max()) == pytest.approx(10.0), "the whole frame period"

    cooled = sensor.model_copy(
        update={
            "fpa": sensor.fpa.model_copy(
                update={
                    "type": "photon",
                    "quantum_efficiency": 0.7,
                    "well_capacity_e": 5e6,
                    "integration_time_ms": 4.0,
                    "dark_current_model": "fixed",
                    "thermal_time_constant_ms": None,
                    "tcr_per_k": None,
                    "g_th_w_per_k": None,
                }
            )
        }
    )
    photon = motion_for_integration(planes, cooled)
    assert photon is not None
    assert float(photon.max()) == pytest.approx(10.0 * 4.0e-3 * 60.0), "4 ms of a 16.7 ms frame"
    assert float(photon.max()) < float(bolometer.max())


def test_a_scene_without_motion_costs_nothing() -> None:
    """`motion_px` is optional in the G-buffer, so a still scene must take the old path exactly."""
    from irsim.optics.stage import apply_optics
    from irsim.pipeline.optics import motion_for_integration

    sensor = boson_sensor()
    assert motion_for_integration({}, sensor) is None
    radiance = radiance_ramp(sensor)
    assert np.array_equal(
        apply_optics(radiance, sensor, 0.0, supersample=1),
        apply_optics(radiance, sensor, 0.0, supersample=1, motion_px=None),
    )


# --- the bolometer's membrane kernel (ADR 0077 amendment) ---------------------------------------

#: The Boson: 8 ms membrane, 60 Hz frame.
BOSON_DECAY = 8.0e-3 * 60.0


def moving_line(px_per_frame: float, decay: float | None) -> np.ndarray:
    """One row of a bright vertical line at column 60, moving +x, smeared."""
    image = np.zeros((8, 96))
    image[:, 60] = 1.0
    motion = np.zeros((8, 96, 2))
    motion[..., 0] = px_per_frame
    return apply_motion_smear(image, motion, 1.0, decay_frames=decay)[4]


def test_the_membrane_kernel_trails_and_leaves_nothing_ahead() -> None:
    """The head stays where the line is now; the tail lies behind it, along where it came from.
    A centred or leading kernel would put signal ahead of the line, which no membrane can."""
    row = moving_line(11.0, BOSON_DECAY)
    assert float(row[62:].sum()) < 1e-12, "signal ahead of the line"
    assert float(row[:60].sum()) > 0.5, "the tail is behind"
    assert float(row.sum()) == pytest.approx(1.0, rel=1e-9)


def test_the_membrane_kernel_is_the_truncated_exponential() -> None:
    """Energy within d pixels behind the head is the exponential's CDF over the frame,
    F(d) = (1 - e^(-(d/L)/a)) / (1 - e^(-1/a)), a = tau/T -- to the bilinear taps' resolution."""
    length = 11.0
    row = moving_line(length, BOSON_DECAY)
    # Summed backwards from the line's own column. Bilinear taps split each sample over two
    # pixels, so the sum through column 60 - d holds the energy out to d + 1/2 pixels of travel.
    behind = np.cumsum(row[60::-1])
    for d in (2.0, 4.0, 6.0, 9.0):
        expected = (1.0 - math.exp(-((d + 0.5) / length) / BOSON_DECAY)) / (
            1.0 - math.exp(-1.0 / BOSON_DECAY)
        )
        measured = float(behind[int(d)])
        assert measured == pytest.approx(expected, abs=0.05), (d, measured, expected)


def test_the_membrane_concentrates_what_the_box_spreads() -> None:
    """Same travel, same mean delay class: the membrane puts half its energy within ~3 px of the
    head where the box needs half the travel, and its brightest pixel is about twice the box's."""
    exp_row, box_row = moving_line(11.0, BOSON_DECAY), moving_line(11.0, None)

    def half_extent(row: np.ndarray) -> int:
        centre = int(np.argmax(np.cumsum(row) >= 1.0))  # the leading edge
        return int(np.argmax(np.cumsum(row[centre::-1]) >= 0.5))

    assert half_extent(exp_row) <= 3
    assert half_extent(box_row) >= 5
    assert float(exp_row.max()) > 1.8 * float(box_row.max())


def test_the_decay_is_the_bolometer_s_and_only_the_bolometer_s() -> None:
    assert smear_decay_frames(1.0 / 60.0, None, 8.0e-3) == pytest.approx(BOSON_DECAY)
    assert smear_decay_frames(1.0 / 60.0, 2.0e-3, None) is None  # a photon FPA keeps its box
    with pytest.raises(ValueError, match="thermal time constant"):
        smear_decay_frames(1.0 / 60.0, None, None)
    with pytest.raises(ValueError, match="decay_frames"):
        apply_motion_smear(np.zeros((8, 8)), np.ones((8, 8, 2)), 1.0, decay_frames=0.0)
