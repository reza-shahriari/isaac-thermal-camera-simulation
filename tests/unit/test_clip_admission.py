"""Which clips a Tier 4 comparison may use, and the refusals it has to report (EV.2).

The acceptance report took the first six clips of the archive in alphabetical order and measured
whatever they turned out to be. The project had already built three gates and applied none of them
here, and ME.5 had already measured the cost on the reference set: 81 of 365 clips are moving,
306 of 365 have a robust noise scale at or below one code, and only 28 support a noise table at
all. Six-alphabetically is therefore six clips that are, on those proportions, most likely
unusable -- which is exactly the condition under which a discriminator separates the two sets on
`noise_scale` for reasons unrelated to the simulator.

These tests are about the gate saying **no**, and about it saying *why*: a refusal is a result
here, and a report that kept only the survivors would state a sample size it did not have.

docs/physics-model.md §15 T4; ADR 0068, ADR 0023 addendum.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim_eval.admission import ADMITTED, admission_summary, admit_clip, admit_clips
from irsim_eval.reference import COLLAPSED_SCALE_CODES

SIZE = 128
FRAMES = 32
RNG = np.random.default_rng(20260924)


def _smooth(field: np.ndarray, radius: int) -> np.ndarray:
    out = np.zeros_like(field)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out += np.roll(np.roll(field, dy, 0), dx, 1)
    return out / out.std()


def _scene(shape: tuple[int, int] = (SIZE, SIZE)) -> np.ndarray:
    """Textured ground under flat sky, as ME.5's own fixtures build it.

    The two gates want opposite things and the geometry is how both are satisfied: phase
    correlation needs two-dimensional non-periodic structure or a still clip reads as moving,
    while the window finder needs a patch with no scene in it at all. So the structure goes in a
    band at the top and the rest is left flat -- which is also what a camera pointed just above a
    treeline sees.
    """
    field = np.random.default_rng(4).normal(0.0, 1.0, shape)
    scene = np.full(shape, 110.0)
    band = shape[0] // 3
    scene[:band] += 25.0 * _smooth(field, 3)[:band]
    return scene


def _usable(sigma: float = 4.0) -> np.ndarray:
    """A tripod clip with sensor noise still in it: static, measurable, with a flat window."""
    scene = _scene()
    fixed = RNG.normal(0.0, 1.5, scene.shape)
    return np.clip(scene + fixed + RNG.normal(0.0, sigma, (FRAMES, *scene.shape)), 0, 255).astype(
        np.uint8
    )


def _moving() -> np.ndarray:
    """The same camera, panning two pixels a frame: displacement accumulates past the threshold."""
    cube = _usable()
    return np.stack([np.roll(f, 2 * i, axis=1) for i, f in enumerate(cube)])


def _flattened() -> np.ndarray:
    """What a codec leaves on a tripod: the scene, with the noise and the fixed pattern gone.

    Static on purpose. ME.5's own fixture drifts as well, which is realistic and no use here: the
    motion gate runs first, so a clip that both pans and is flattened would be refused as `moving`
    and would never exercise the noise gate at all.
    """
    scene = np.round(_scene())
    return np.stack([scene for _ in range(FRAMES)]).astype(np.uint8)


def test_a_usable_clip_is_admitted_and_says_so() -> None:
    """The gate has to let the good case through, or it is a filter rather than a gate.

    A tripod clip with sensor noise still in it passes all three: it does not move, its noise
    scale is above the quantiser, and it has a window with no scene in it.
    """
    verdict = admit_clip(_usable(), name="good")
    assert verdict.admitted and verdict.reason == ADMITTED
    assert verdict.static
    assert verdict.noise_scale_codes > COLLAPSED_SCALE_CODES
    assert verdict.flat_regions > 0


def test_a_panning_clip_is_refused_because_the_statistic_would_measure_the_pan() -> None:
    """81 of the reference set's 365 clips are this, and the old loader measured them anyway."""
    verdict = admit_clip(_moving(), name="pan")
    assert not verdict.admitted
    assert verdict.reason == "moving"
    assert verdict.max_displacement_px > 1.0


def test_a_codec_flattened_clip_is_refused_rather_than_read_as_quiet() -> None:
    """306 of 365. The ruler is the quantiser, so "low noise" is not a measurement of the sensor."""
    verdict = admit_clip(_flattened(), name="flat")
    assert not verdict.admitted
    assert verdict.reason == "noise_floor_collapsed"
    assert verdict.noise_scale_codes <= COLLAPSED_SCALE_CODES + 1e-9


def test_the_first_failing_gate_is_the_reason_not_the_last() -> None:
    """A moving clip's window count is not interesting: the finder judges against a median a pan
    has already invalidated, so reporting `no_flat_window` there would name the wrong problem."""
    verdict = admit_clip(_moving(), name="pan")
    assert verdict.reason == "moving"


def test_admission_counts_clips_that_pass_not_clips_that_were_opened() -> None:
    """The behaviour change EV.2 is about: stop at six *usable* clips, not at the sixth file.

    The first three here are unusable. Taking the first two alphabetically -- what the report did
    -- would have measured a pan and a frozen clip and reported a Tier 4 number from them.
    """
    clips = [_moving(), _flattened(), _moving(), _usable(), _usable(), _usable()]
    names = [f"c{i}" for i in range(len(clips))]
    kept, verdicts = admit_clips(clips, names, want=2)

    assert len(kept) == 2, "two admitted"
    assert len(verdicts) == 5, "and it had to open five files to find them"
    assert [v.reason for v in verdicts] == [
        "moving",
        "noise_floor_collapsed",
        "moving",
        ADMITTED,
        ADMITTED,
    ]


def test_every_refusal_is_returned_not_dropped() -> None:
    """A report that kept only the survivors would state a sample size it does not have."""
    clips = [_moving(), _usable(), _flattened()]
    kept, verdicts = admit_clips(clips, ["a", "b", "c"])
    assert len(kept) == 1
    assert len(verdicts) == 3
    assert [v.name for v in verdicts] == ["a", "b", "c"]


def test_the_summary_names_the_sample_size_and_what_refused_the_rest() -> None:
    """ "six clips" and "six of forty-one, the rest codec-flattened" are different claims."""
    _, verdicts = admit_clips([_moving(), _usable(), _flattened()], ["a", "b", "c"])
    line = admission_summary(verdicts)
    assert line.startswith("1 of 3 clips admitted")
    assert "1 moving" in line
    assert "1 noise_floor_collapsed" in line
    assert admission_summary([]) == "no clips examined"


def test_a_clip_too_short_to_judge_is_an_error_not_a_quiet_pass() -> None:
    """Four frames is the floor for a temporal statistic; fewer cannot be judged either way.

    Returning "admitted" would put an unmeasurable clip into the sample, and returning "refused"
    would file it beside clips that were actually examined and found wanting.
    """
    with pytest.raises(ValueError, match="at least four frames"):
        admit_clip(np.zeros((2, 16, 16), np.uint8), name="stub")
