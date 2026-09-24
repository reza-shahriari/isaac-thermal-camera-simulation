"""What a set's signal path allows to be measured on it, decided before anything is measured.

roadmap XD.2; docs/physics-model.md §11.3, §15 Tier 4; ADR 0068.

Every number this project compares against reality was measured on somebody else's file, and what
happened to the photons between the detector and that file decides what the number means. A
three-dimensional noise decomposition describes a sensor if the frames came off a linear stream and
describes a *recorder* if something converted them; a histogram's flatness is an AGC's fingerprint
if the camera made the 8-bit picture and is nothing at all if a recorder did. Those are not
tolerances to widen -- they are the difference between a measurement and a plausible number.

So the path is a small closed vocabulary rather than prose, and each measurement declares which
paths it can live on:

``display``
    The camera's own output. AGC, DDE, palette and any temporal filtering are baked in, which is
    what makes the ISP measurable here and the sensor not.
``recorder``
    A linear stream -- Y16 off the core -- that something downstream converted to 8 bits. The
    sensor's noise, striping and shutter freezes survive; the ISP never happened.
``radiometric``
    Counts linear in radiance, straight off the core, with no AGC and no tone map. The only path on
    which a measurement can be carried into radiance or Kelvin without inventing a SITF.
``unknown``
    The publisher did not say. Not a synonym for ``display``: an undocumented path cannot even be
    assumed monotone, so it cannot tell white-hot from black-hot.

**Bit depth is a separate axis and is deliberately not encoded here.** A radiometric path at 8 bits
is quantiser-limited and a display path at 16 is still an AGC's output; the two facts constrain
different things and collapsing them into one field would let either hide behind the other. The
depth travels with the frames (``irsim_eval.data.Sequence.bit_depth``) and the floor it puts under a
statistic is measured from the data, not read off the declaration
(:func:`~irsim.validation.codec.quantiser_step`).

**A per-pixel-temperature path is not in this vocabulary yet.** A calibrated set such as FLAME 3
stores Celsius rather than counts, which is a different container and a different set of claims;
it lands with the set that needs it (roadmap XD.5) rather than being guessed at in advance.

The table below states the *necessary* condition -- what the path itself permits. A set may still
exclude a measurement the path allows, for reasons the path knows nothing about (a lossy codec, an
unverified sensor, a frame rate too low for anything temporal). Those live per set in
``data/validation/datasets.yaml``; this module cannot and does not replace them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

__all__ = [
    "SignalPath",
    "SIGNAL_PATHS",
    "AnalyserRequirement",
    "ANALYSERS",
    "allows",
    "permitted_analysers",
    "require_signal_path",
]

#: What happened to the photons between the detector and the file. See the module docstring.
SignalPath = Literal["display", "recorder", "radiometric", "unknown"]

#: The vocabulary, in order of how much they support: the widest claim last but for ``unknown``.
SIGNAL_PATHS: tuple[SignalPath, ...] = get_args(SignalPath)


@dataclass(frozen=True)
class AnalyserRequirement:
    """One measurement, the paths it may run on, and why the others are refused."""

    name: str
    paths: frozenset[str]
    because: str

    def allows(self, path: str) -> bool:
        return path in self.paths


#: Geometry and labels: a box's extent in pixels survives any map at all, monotone or not.
_ANY = frozenset(SIGNAL_PATHS)
#: The camera's own 8-bit picture, and nothing else.
_ISP = frozenset({"display"})
#: Paths on which the *sensor* is still visible: nothing rescaled the frame on its own content.
_SENSOR = frozenset({"recorder", "radiometric"})
#: Paths somebody documented. Excludes ``unknown``, where not even the polarity is known.
_DOCUMENTED = frozenset({"display", "recorder", "radiometric"})
#: Counts linear in radiance: the only path that converts to a physical unit without a guess.
_LINEAR = frozenset({"radiometric"})


def _r(name: str, paths: frozenset[str], because: str) -> tuple[str, AnalyserRequirement]:
    return name, AnalyserRequirement(name=name, paths=paths, because=because)


#: Every measurement name the index may use, and the paths it may use it on. A name that is not
#: here cannot appear in ``datasets.yaml``: a typo in a permission list is a permission granted by
#: accident, and this is the only place that can catch one.
ANALYSERS: dict[str, AnalyserRequirement] = dict(
    (
        _r(
            "agc_signature",
            _ISP,
            "the output histogram's flatness is the camera's AGC only if the camera made the "
            "8-bit frame; on a converted stream it is the recorder's conversion rule",
        ),
        _r(
            "dde_overshoot",
            _ISP,
            "the overshoot at a step is DDE's gain, and a stream that never went through the "
            "camera's ISP has no DDE in it to measure",
        ),
        _r(
            "noise_3d",
            _SENSOR,
            "an AGC rescales every frame on its own content, which moves temporal noise into the "
            "scene's variance and back; an undocumented path cannot be attributed to a sensor",
        ),
        _r(
            "noise_3d_kelvin",
            _LINEAR,
            "a noise figure in Kelvin needs counts linear in radiance -- on any other path the "
            "counts-to-Kelvin step is an inferred SITF, and the answer is that inference",
        ),
        _r(
            "spatial_psd",
            _SENSOR,
            "row and column structure is the sensor's, but a per-frame rescale changes its "
            "amplitude relative to the scene it is measured against",
        ),
        _r(
            "temporal_psd",
            _SENSOR,
            "an AGC's own time constant appears in the temporal spectrum as if it were the "
            "detector's",
        ),
        _r(
            "fixed_pattern_growth",
            _SENSOR,
            "drift of the fixed pattern between shutter events is a sensor quantity; a per-frame "
            "rescale removes exactly the slow component being measured",
        ),
        _r(
            "bad_pixels",
            _SENSOR,
            "a replaced pixel is exactly the mean of its four neighbours, and that identity is "
            "affine: it survives a linear conversion and not a tone map",
        ),
        _r(
            "edge_spread",
            _SENSOR,
            "DDE rings the very edge this measures, so an edge-spread function read off a "
            "display frame is the ISP's and not the optics'",
        ),
        _r(
            "smear",
            _SENSOR,
            "smear is the integration time written across the scene; sharpening rewrites it",
        ),
        _r(
            "ffc_freeze",
            _DOCUMENTED,
            "a run of identical frames is identical whatever mapped them, so any documented path "
            "carries a shutter freeze -- but an undocumented one may have dropped or duplicated "
            "frames of its own",
        ),
        _r(
            "ffc_interval",
            _DOCUMENTED,
            "as ffc_freeze: the interval between freezes is a count of frames, not a radiometry",
        ),
        _r(
            "cloud_clutter_psd",
            _DOCUMENTED,
            "clutter is read as a shape across spatial scales, which survives the camera's own "
            "tone map; its amplitude in codes is not a radiance, and only a radiometric path "
            "makes it one",
        ),
        _r(
            "contrast_polarity",
            _DOCUMENTED,
            "white-hot and black-hot are the same picture inverted, and a path nobody documented "
            "cannot say which one it is",
        ),
        _r(
            "target_size_and_scr",
            _ANY,
            "a box's extent is geometry and survives any map; the contrast half is in whatever "
            "units the path left behind, which is what the set's role, not its path, bounds",
        ),
        _r(
            "target_size_prior",
            _ANY,
            "a distribution of box sizes needs no radiometry at all",
        ),
        _r(
            "scr_prior",
            _ANY,
            "a prior bounds a distribution rather than stating a value, which is the claim an "
            "undocumented path can still support",
        ),
        _r(
            "size_vs_range",
            _ANY,
            "size against range is the optics and the label, not the tone map",
        ),
    )
)


def allows(analyser: str, path: str) -> bool:
    """Whether ``analyser`` may run on frames that came down ``path``."""
    try:
        requirement = ANALYSERS[analyser]
    except KeyError:
        raise KeyError(f"unknown analyser {analyser!r}; known: {sorted(ANALYSERS)}") from None
    return requirement.allows(path)


def permitted_analysers(path: str) -> tuple[str, ...]:
    """Every measurement ``path`` supports, sorted. The upper bound on a set with that path."""
    if path not in SIGNAL_PATHS:
        raise ValueError(f"unknown signal path {path!r}; known: {list(SIGNAL_PATHS)}")
    return tuple(sorted(name for name, r in ANALYSERS.items() if r.allows(path)))


def require_signal_path(analyser: str, path: str) -> None:
    """Raise unless ``path`` supports ``analyser``, with the reason in the message.

    Refusing is the point, and the message carries the physics rather than the rule: the failure
    this guards against is a number that reaches a table and cannot afterwards be attributed to
    anything, and by then the only record of why is whatever this said.
    """
    if path not in SIGNAL_PATHS:
        raise ValueError(f"unknown signal path {path!r}; known: {list(SIGNAL_PATHS)}")
    requirement = ANALYSERS.get(analyser)
    if requirement is None:
        raise KeyError(f"unknown analyser {analyser!r}; known: {sorted(ANALYSERS)}")
    if requirement.allows(path):
        return
    raise ValueError(
        f"{analyser!r} may not be measured on a {path!r} signal path "
        f"(it needs {sorted(requirement.paths)}): {requirement.because}"
    )
