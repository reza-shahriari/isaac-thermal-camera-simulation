"""The signal-path vocabulary and what it refuses (XD.2).

This file is mostly about *refusals*, because a permission table is only worth having if it says
no. Three properties are pinned and each one is a mistake that has actually been made in this
project or is one step away from being made:

* the table and the index agree -- a permission written in YAML that the table would refuse cannot
  be loaded at all, so the two cannot drift;
* the paths that carry the ISP and the paths that carry the sensor are disjoint, which is the
  physical content of the whole vocabulary and not a naming convention;
* ``unknown`` is not a synonym for ``display``.

docs/physics-model.md §11.3, §15 Tier 4; ADR 0068.
"""

from __future__ import annotations

import pytest

from irsim.validation.signal_path import (
    ANALYSERS,
    SIGNAL_PATHS,
    allows,
    permitted_analysers,
    require_signal_path,
)

#: What only the camera's own ISP can produce.
ISP_MEASUREMENTS = ("agc_signature", "dde_overshoot")
#: What only an unrescaled frame can carry.
SENSOR_MEASUREMENTS = ("noise_3d", "spatial_psd", "temporal_psd", "fixed_pattern_growth")


def test_the_vocabulary_is_the_four_words_and_nothing_else() -> None:
    """A closed vocabulary is the point: a fifth value would be a claim nobody defined."""
    assert SIGNAL_PATHS == ("display", "recorder", "radiometric", "unknown")


def test_the_isp_and_the_sensor_live_on_disjoint_paths() -> None:
    """The physical content of the vocabulary, stated as a property rather than a table read.

    A frame the camera's AGC produced has had its scene statistics rewritten per frame; a frame it
    did not has no AGC in it to measure. No path can satisfy both, and if one ever did, one of the
    two groups of measurements would be quietly wrong on it rather than refused.
    """
    for isp in ISP_MEASUREMENTS:
        for sensor in SENSOR_MEASUREMENTS:
            shared = ANALYSERS[isp].paths & ANALYSERS[sensor].paths
            assert not shared, f"{isp} and {sensor} both claim {sorted(shared)}"


def test_an_undocumented_path_is_not_a_display_path() -> None:
    """``unknown`` is the most common value in the index and the easiest to treat as ``display``.

    It is weaker in a specific way: an undocumented path cannot be assumed monotone, so it cannot
    even say whether the picture is white-hot or black-hot. Everything it supports, ``display``
    supports too -- the inclusion is strict.
    """
    unknown = set(permitted_analysers("unknown"))
    display = set(permitted_analysers("display"))
    assert unknown < display
    assert "contrast_polarity" in display - unknown


def test_only_a_radiometric_path_reaches_kelvin() -> None:
    """The measurement XD.10 exists to make, and the one no 8-bit display set may ever claim."""
    assert "noise_3d_kelvin" in permitted_analysers("radiometric")
    for path in ("display", "recorder", "unknown"):
        assert not allows("noise_3d_kelvin", path)
        with pytest.raises(ValueError, match="linear in radiance"):
            require_signal_path("noise_3d_kelvin", path)


def test_a_refusal_carries_the_reason_not_the_rule() -> None:
    """The message is the only record of *why* once the traceback is in somebody's terminal."""
    with pytest.raises(ValueError, match="no DDE in it to measure"):
        require_signal_path("dde_overshoot", "recorder")
    with pytest.raises(ValueError, match="rescales every frame on its own content"):
        require_signal_path("noise_3d", "display")


def test_a_name_that_is_not_a_measurement_is_refused_not_ignored() -> None:
    """A typo in a permission list is a permission granted to nothing, or to the wrong thing."""
    with pytest.raises(KeyError, match="unknown analyser"):
        allows("noise3d", "recorder")
    with pytest.raises(KeyError, match="unknown analyser"):
        require_signal_path("agc_signatures", "display")


def test_a_path_that_is_not_in_the_vocabulary_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown signal path"):
        require_signal_path("noise_3d", "y16")
    with pytest.raises(ValueError, match="unknown signal path"):
        permitted_analysers("raw")


def test_every_measurement_is_possible_on_at_least_one_path() -> None:
    """A row permitting nothing would be a measurement that can never run, written as if it can."""
    for name, requirement in ANALYSERS.items():
        assert requirement.paths, name
        assert requirement.paths <= set(SIGNAL_PATHS), name
        assert requirement.because.strip(), f"{name} refuses without saying why"


def test_the_geometry_measurements_survive_every_path() -> None:
    """A box's extent in pixels is not radiometry, and no tone map can change it."""
    for name in ("target_size_prior", "scr_prior", "size_vs_range", "target_size_and_scr"):
        for path in SIGNAL_PATHS:
            assert allows(name, path), f"{name} on {path}"
