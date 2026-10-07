"""A scene with a layered atmosphere will not hand out the grey one (AT.5).

`Scene.from_config` builds **both** models whenever the scene names an environment preset, and
they do not agree. The grey L1 model (§8.5) takes one band-averaged optical depth on a horizontal
path; the layered one (§8.6) integrates the exponential-sum terms along the real slant path. Every
render script passes `scene.layered`, and `pipeline/atmosphere.py` branches between them -- but the
attribute named `atmosphere`, the one a reader would take for the scene's atmosphere, held the
fallback.

Nothing was reading the wrong one when this guard was written. That is the point: both live
readers want the *preset*, which is one object and cannot disagree, and the next caller would have
had no way to tell that the primary-looking name was not what the renders use.

The divergence is not small, and it is not the one `AT.1` measured. AT.1 compared the layered
model against *itself* on a horizontal path (tau 0.5995 against 0.7230). Grey against layered, on
the shipped aerial scene at 5 km and 20 degrees elevation, is **tau 0.057 against 0.590** -- an
order of magnitude -- with path radiance 45.1 against 18.7 W/m^2/sr.

That gap is the k-distribution, not a bug in either model: a single band-averaged optical depth
over-attenuates, because exp(-mean tau) is not the mean of exp(-tau) over a band whose lines vary
by orders of magnitude across it. The exponential sum exists to fix exactly that (§8.6), which is
why the grey model is L1 and why handing it out under the primary name is a trap. Measured here
rather than quoted, because a guard whose justification is a number in a docstring rots.

docs/physics-model.md §8.5, §8.6; ADR 0033; roadmap AT.1, AT.5.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import yaml

from irsim.atmosphere.layered import LayeredAtmosphere
from irsim.atmosphere.model import Atmosphere
from irsim.config.atmosphere import AtmospherePreset
from irsim.config.scene import SceneConfig
from irsim.scene import Scene

REPO = pathlib.Path(__file__).resolve().parents[2]
AERIAL = REPO / "configs" / "scenes" / "sky_target_clear_day.yaml"
SCENES = REPO / "configs" / "scenes"


@pytest.fixture(scope="module")
def layered_scene(tophat_lwir_lut):  # type: ignore[no-untyped-def]
    return Scene.from_file(AERIAL, {"lwir": tophat_lwir_lut})


@pytest.fixture(scope="module")
def grey_only_scene(tophat_lwir_lut):  # type: ignore[no-untyped-def]
    """A scene with no `environment_preset:`, so `from_config` builds no layered model.

    It has to be constructed here because **none of the eight shipped scenes is one** -- which is
    itself the measurement: `scene.atmosphere` was not the model any scene in this repository
    renders with, on every scene in this repository.
    """
    raw = yaml.safe_load((SCENES / "sky_target_clear_day.yaml").read_text(encoding="utf-8"))
    raw["scene"].pop("environment_preset", None)
    return Scene.from_config(SceneConfig.model_validate(raw), {"lwir": tophat_lwir_lut})


def test_every_shipped_scene_carries_a_layered_model(tophat_lwir_lut) -> None:  # type: ignore[no-untyped-def]
    """So the refused attribute was wrong on all eight, not on an unusual one."""
    names = sorted(p.name for p in SCENES.glob("*.yaml"))
    assert len(names) >= 8
    for name in names:
        raw = yaml.safe_load((SCENES / name).read_text(encoding="utf-8"))
        assert raw["scene"].get("environment_preset") is not None, name


def test_the_primary_name_is_refused_when_it_is_not_what_renders(layered_scene) -> None:  # type: ignore[no-untyped-def]
    assert layered_scene.layered is not None
    with pytest.raises(AttributeError, match="not the one it renders with"):
        _ = layered_scene.atmosphere


def test_the_message_names_all_three_ways_out(layered_scene) -> None:  # type: ignore[no-untyped-def]
    """A guard that refuses without saying what to use instead just moves the confusion."""
    with pytest.raises(AttributeError) as excinfo:
        _ = layered_scene.atmosphere
    message = str(excinfo.value)
    for door in ("transfer_atmosphere", "atmosphere_preset", "grey_atmosphere"):
        assert door in message, door


def test_a_scene_with_no_layered_model_still_hands_out_its_atmosphere(grey_only_scene) -> None:  # type: ignore[no-untyped-def]
    """The guard is about ambiguity, not about the grey model, which is still L1 for §8.5."""
    assert grey_only_scene.layered is None
    assert isinstance(grey_only_scene.atmosphere, Atmosphere)
    assert grey_only_scene.atmosphere is grey_only_scene.grey_atmosphere
    assert grey_only_scene.transfer_atmosphere is grey_only_scene.grey_atmosphere


def test_transfer_atmosphere_is_the_one_the_renders_use(layered_scene, grey_only_scene) -> None:  # type: ignore[no-untyped-def]
    assert layered_scene.transfer_atmosphere is layered_scene.layered
    assert isinstance(layered_scene.transfer_atmosphere, LayeredAtmosphere)
    assert isinstance(grey_only_scene.transfer_atmosphere, Atmosphere)


def test_the_preset_is_one_object_so_it_cannot_disagree(layered_scene) -> None:
    """What both live readers actually wanted, and the reason nothing was broken yet."""
    preset = layered_scene.atmosphere_preset
    assert isinstance(preset, AtmospherePreset)
    assert preset is layered_scene.grey_atmosphere.preset
    assert preset is layered_scene.layered.preset


def test_the_two_models_disagree_by_far_more_than_the_netd(layered_scene) -> None:
    """Measured here rather than quoted: the guard's whole justification is this gap.

    At 5 km and 20 degrees the grey model puts the path at tau 0.047 (0.057 before AT.27 gave
    LWIR its humidity square; this scene's air is wetter than the clear-dry anchor) and the
    layered one at 0.587 (0.576 under AT.27's square, 0.590 before it; AT.37 refit the square
    smaller and gave it its own half-depth column) -- the grey path is essentially opaque where
    the real one passes more than half. A target seen
    through the first is almost entirely path radiance; through the second it is mostly itself.
    Against a 50 mK NETD this is not a tolerance question by any reading.
    """
    band, t_s = "lwir", layered_scene.t0_s
    distance_m = np.array([5000.0])
    elevation_rad = np.array([np.deg2rad(20.0)])

    grey = layered_scene.grey_atmosphere
    layered = layered_scene.layered
    assert layered is not None

    tau_grey = float(np.atleast_1d(grey.transmittance(band, t_s, distance_m))[0])
    tau_layered = float(
        np.atleast_1d(layered.exponential_sum(band, t_s).transmittance(distance_m, elevation_rad))[
            0
        ]
    )
    assert tau_grey == pytest.approx(0.047, abs=0.01), tau_grey
    assert tau_layered == pytest.approx(0.587, abs=0.01), tau_layered
    # An order of magnitude. The bound is loose against the measurement on purpose: what is being
    # pinned is "these are different models", not either model's own value, which `AT.1` and the
    # exponential-sum tests own.
    assert tau_layered / tau_grey > 5.0, (tau_grey, tau_layered)

    # The grey model has no `path_radiance`: §8.5's emitted term is (1 - tau) L_B(T_air), which
    # is what `apply` adds to a zero input. Computed that way rather than restated.
    zero = np.zeros_like(distance_m)
    path_grey = float(np.atleast_1d(grey.apply(band, t_s, zero, distance_m, "lb"))[0])
    path_layered = float(
        np.atleast_1d(
            layered.path_radiance_plane(band, t_s, distance_m, elevation_rad, "lb")
        ).ravel()[0]
    )
    assert path_grey == pytest.approx(45.1, rel=0.05), path_grey
    assert path_layered == pytest.approx(18.7, rel=0.05), path_layered
    assert path_grey / path_layered > 2.0, (path_grey, path_layered)


def test_the_grey_model_is_still_a_registered_consumer(layered_scene) -> None:
    """The one-weather guard (CLAUDE.md #6) must keep seeing it: refused is not removed."""
    consumers = layered_scene.consumers
    assert consumers["atmosphere"] is layered_scene.grey_atmosphere
    assert consumers["layered"] is layered_scene.layered
    assert layered_scene.grey_atmosphere.weather is layered_scene.weather
