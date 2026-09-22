"""Where published radiometry stops backing the sea model, and how far outside a frame sits (SE.1).

:mod:`irsim.atmosphere.sea` computes an angular emissivity for every ray. This module records the
range of angles over which anyone has *checked* that quantity against the sea, which is a much
smaller range, and lets a frame report how much of itself lies outside it.

**The envelope.** In-situ angular radiometry -- a CE 312 four-channel radiometer in the 8-14 µm
interval, sea temperature from probes on oceanographic buoys, several view angles and wind speeds
-- validates the Masuda-class model (Fresnel emissivity integrated over a Cox-Munk slope
distribution, which is what :class:`~irsim.atmosphere.sea.SeaModel` computes) only to about **50°
from nadir**. Past that, multiple reflections between facets of the roughened surface produce
discrepancies and the Wu-Smith model is required for 8-13 µm. Emissivity falls roughly 2-3 % by a
55° view angle. See ``docs/research/2026-09-15-field-survey.md`` (medium confidence).

**The envelope has two axes, not one.** The published drop is a figure for a sea, so it carries the
wind that roughened it. Our chain reproduces 2-3 % at 55° for winds up to **7.3 m/s** and exceeds
3 % above that (4.3 % at 15 m/s), because the facet spread widens with wind and drags more of the
tilt distribution into the steep part of the Fresnel curve. Quoting 50° without a wind would make
the envelope look like a property of the geometry when half of it is a property of the sea state.
:data:`VALIDATED_WIND_M_S` carries the second axis.

**Why this matters more here than it would elsewhere.** A camera on a shore or a mast sees the sea
at grazing incidence almost everywhere in its frame: at a 20 m eye height the 50° limit is crossed
at **31 m of slant range**, so everything beyond a boat length is outside the validated envelope.
That is not a corner case to note, it is the entire maritime working band. An airborne camera
looking down is the opposite -- a 2 km UAV at -60° sees 17-44° and is inside the envelope
everywhere. The flag therefore separates two real deployments rather than firing on everything,
and :func:`envelope_report` is what says which one a frame is.

**What is *not* claimed.** This module does not correct anything. It records a limit and measures
a frame against it; the sea model past 50° is unbounded in error, not adjusted for it. ADR 0118
says why recording the envelope was preferred to implementing Wu-Smith from a one-line survey
claim, and what would be needed to close it.

docs/physics-model.md §5.3; roadmap SE.1; ADR 0078, ADR 0079, ADR 0118
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.sea import EARTH_RADIUS_M, horizon_depression_rad

__all__ = [
    "VALIDATED_ZENITH_DEG",
    "VALIDATED_WIND_M_S",
    "PUBLISHED_DROP_BY_55_DEG",
    "ENVELOPE_SOURCE",
    "EnvelopeReport",
    "view_zenith_rad",
    "depression_at_zenith_rad",
    "beyond_envelope",
    "envelope_report",
]

#: View zenith angle (degrees from nadir) past which no published in-situ radiometry backs the
#: Masuda-class sea emissivity. Not a limit of the code -- the code runs to 90° -- but of the
#: evidence for it.
VALIDATED_ZENITH_DEG = 50.0

#: Wind speed (m/s) up to which this chain reproduces the published 2-3 % emissivity drop at 55°.
#: Above it the facet spread takes the drop past 3 %, so the published figure no longer brackets
#: what we compute even at an angle the published figure covers. Measured, not cited.
VALIDATED_WIND_M_S = 7.3

#: The published fractional drop in band emissivity between nadir and a 55° view angle.
PUBLISHED_DROP_BY_55_DEG = (0.02, 0.03)

ENVELOPE_SOURCE = (
    "CE 312 in-situ angular radiometry over buoy-instrumented sea, 8-14 um; "
    "docs/research/2026-09-15-field-survey.md (medium confidence)"
)


def view_zenith_rad(camera_height_m: float, depression_rad: Any) -> NDArray[np.float64]:
    """Zenith angle of the view ray **at the sea surface**, for a ray at ``depression_rad``.

    This is the angle the published validation is indexed by, and it is not the complement of the
    depression: the surface normal at the patch is not parallel to the camera's local vertical.
    From the sine rule in the (centre, camera, patch) triangle,

    ``sin(theta) = (1 + h/R) cos(delta)``

    which is ``90° - delta`` in the flat-earth limit but gives **exactly 90° at the geometric
    horizon** for any height, where ``90° - delta`` is short by the horizon dip (0.14° at 20 m,
    3.2° at 10 km). Since the whole point of the envelope is how close a shore camera runs to
    grazing, getting the grazing end exactly right is the part that matters.

    Rays above the horizon never meet the sea and raise, as
    :func:`~irsim.atmosphere.sea.slant_range_m` does.
    """
    delta = np.asarray(depression_rad, dtype=np.float64)
    horizon = horizon_depression_rad(camera_height_m)
    if np.any(delta < horizon - 1e-12):
        raise ValueError(
            f"depression {np.min(delta):.6g} rad is above the horizon at {horizon:.6g} rad for a "
            f"camera {camera_height_m} m up: the ray never hits the sea, so it has no incidence "
            "angle on it"
        )
    scale = 1.0 + float(camera_height_m) / EARTH_RADIUS_M
    return np.asarray(np.arcsin(np.clip(scale * np.cos(delta), -1.0, 1.0)), dtype=np.float64)


def depression_at_zenith_rad(
    camera_height_m: float, zenith_rad: float = math.radians(VALIDATED_ZENITH_DEG)
) -> float:
    """The depression whose ray meets the sea at ``zenith_rad`` -- the inverse of the above.

    The number that makes the envelope concrete: pass the default and it answers "how far down
    must this camera look to be inside the validated envelope", which for a 20 m eye height is
    40.0002° and a slant range of 31 m.
    """
    scale = 1.0 + float(camera_height_m) / EARTH_RADIUS_M
    value = math.sin(float(zenith_rad)) / scale
    if not -1.0 <= value <= 1.0:
        raise ValueError(f"no ray from {camera_height_m} m reaches a zenith of {zenith_rad} rad")
    return math.acos(value)


def beyond_envelope(
    camera_height_m: float,
    depression_rad: Any,
    limit_deg: float = VALIDATED_ZENITH_DEG,
) -> NDArray[np.bool_]:
    """True where the ray meets the sea past ``limit_deg`` from nadir -- the flag itself."""
    zenith = view_zenith_rad(camera_height_m, depression_rad)
    return np.asarray(zenith > math.radians(float(limit_deg)), dtype=np.bool_)


@dataclass(frozen=True)
class EnvelopeReport:
    """How much of one frame's sea lies outside the validated angular envelope."""

    camera_height_m: float
    limit_deg: float
    n_samples: int
    n_beyond: int
    zenith_min_deg: float
    zenith_max_deg: float
    horizon_depression_deg: float
    envelope_range_m: float

    @property
    def fraction_beyond(self) -> float:
        """Fraction of the sea samples past the limit. 0.0 on a frame with no sea."""
        return 0.0 if self.n_samples == 0 else self.n_beyond / self.n_samples

    @property
    def inside(self) -> bool:
        """Whether the whole frame is backed by published radiometry."""
        return self.n_beyond == 0

    def summary(self) -> str:
        """One line, for a Tier 3 report to print."""
        return (
            f"sea angular envelope: {self.fraction_beyond:.4f} of {self.n_samples} sea samples "
            f"past {self.limit_deg:.0f} deg from nadir "
            f"(view zenith {self.zenith_min_deg:.3f}-{self.zenith_max_deg:.3f} deg; "
            f"camera {self.camera_height_m:.1f} m, envelope ends at "
            f"{self.envelope_range_m:.1f} m slant range)"
        )


def envelope_report(
    camera_height_m: float,
    depression_rad: Any,
    mask: Any | None = None,
    limit_deg: float = VALIDATED_ZENITH_DEG,
) -> EnvelopeReport:
    """Measure a frame's depression plane against the envelope.

    ``mask`` selects the sea pixels; everything else is ignored rather than clamped, because a
    maritime G-buffer carries depression ``0`` above the horizon and that is a sentinel, not a
    ray. A frame with no sea in it reports zero samples rather than raising -- the caller asked
    what fraction is outside, and "none of it, there is none" is an answer.
    """
    from irsim.atmosphere.sea import slant_range_m

    delta = np.asarray(depression_rad, dtype=np.float64)
    selected = delta.ravel() if mask is None else delta[np.asarray(mask, dtype=bool)]
    horizon = horizon_depression_rad(camera_height_m)
    envelope_delta = depression_at_zenith_rad(camera_height_m, math.radians(float(limit_deg)))
    common = dict(
        camera_height_m=float(camera_height_m),
        limit_deg=float(limit_deg),
        horizon_depression_deg=math.degrees(horizon),
        envelope_range_m=float(slant_range_m(camera_height_m, envelope_delta)),
    )
    if selected.size == 0:
        return EnvelopeReport(
            n_samples=0, n_beyond=0, zenith_min_deg=math.nan, zenith_max_deg=math.nan, **common
        )
    zenith = np.degrees(view_zenith_rad(camera_height_m, selected))
    return EnvelopeReport(
        n_samples=int(selected.size),
        n_beyond=int(np.count_nonzero(zenith > float(limit_deg))),
        zenith_min_deg=float(zenith.min()),
        zenith_max_deg=float(zenith.max()),
        **common,
    )
