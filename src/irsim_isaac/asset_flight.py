"""A flight path for an **imported** aircraft, and the one rotation that mounts it on the stage.

Two small pieces of geometry, both pure NumPy so they can be tested without Isaac Sim, and both
written down here rather than inside a render driver because getting either wrong produces a
picture that looks plausible and is not.

**The mount** (:func:`world_frame_to_stage`). A scene config declares its own world frame --
``world_frame: {up: [0,0,1], north: [0,1,0]}`` for an asset that came out of an FBX Z-up -- while
every stage this project authors is the renderer's Y-up convention: +X right, +Y up, -Z north. The
two are related by exactly one rotation, and that rotation is not a matter of taste: it is the
scene's own ``world_frame`` block rewritten as a matrix. Deriving it, rather than writing
``rotateX 90`` into a driver, is what stops the mount and the sun from disagreeing -- the dome, the
``DistantLight`` and the sky model all read the stage's Y as up and its -Z as north, so an asset
mounted by hand at the wrong sign gets lit from the wrong quarter and nothing complains.

**The path** (:class:`FigureEightTrack`). A lemniscate of Bernoulli, flown level in front of a
ground observer. A circular orbit *around* the observer is easier to write and is the wrong shape:
an aircraft circling you at a constant radius presents the same aspect for the whole pass, so the
frame shows a target that translates but never turns. A figure of eight crosses its own track, so
one circuit takes the observer's line of sight through broadside, head-on, broadside and tail-on,
and the range swings by more than 2:1 on the way.

**The other path** (:class:`StraightOutTrack`). The aircraft flies straight away from the
observer, near to far. This is the detection question rather than the aspect question: it asks at
what range a target stops being a target, and it is the one clip in which the answer is visible
rather than argued. `render_quad_outbound` has filmed it since `PT.9` for a **generated**
airframe, by moving the camera instead of the aircraft -- that scene's occluders are authored in
the world frame and the thermal core refuses to move them, so a flying airframe would lose its own
self-shadowing. An imported asset with a mesh field has no such constraint: its sky view is
analytic per cell (`self_occluding: false`, ADR 0104) and its scene authors no occluders, so here
the aircraft really flies and the camera holds station, which is also what a ground observer
actually sees.

**Range is spaced geometrically, not linearly.** A target's width in pixels goes as 1/R, so equal
steps in range spend most of the clip at the far end where nothing changes: from 4 m to 80 m,
half the linear frames sit beyond 42 m, where the aircraft is already under twenty pixels. Equal
steps in log R make it shrink at a constant rate -- the same fraction per frame for the whole
clip -- which is what makes the collapse legible instead of a jump at the start followed by a
long crawl.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["FigureEightTrack", "StraightOutTrack", "world_frame_to_stage"]

#: The stage's own axes, in the order the rotation's rows are built from: east, up, north.
STAGE_EAST = (1.0, 0.0, 0.0)
STAGE_UP = (0.0, 1.0, 0.0)
STAGE_NORTH = (0.0, 0.0, -1.0)


def world_frame_to_stage(up: object, north: object) -> NDArray[np.float64]:
    """The 3x3 rotation taking a scene's world coordinates into stage axes.

    ``up`` and ``north`` are the scene config's ``world_frame`` block. The returned matrix ``M``
    is applied as ``stage = M @ world`` and satisfies, by construction,

        ``M @ up = +Y``      ``M @ north = -Z``      ``M @ (north x up) = +X``

    which is the stage convention :func:`irsim_isaac.visible_sky.stage_direction` and
    :func:`irsim_isaac.stage.author_environment` already assume. East is derived rather than
    declared, exactly as :class:`irsim.config.scene.WorldFrameSpec` derives it, so a scene cannot
    hand out a left-handed triple here that it refused at load.

    Raises if the two axes are parallel or degenerate -- the case where "north" carries no
    information about the horizontal, and any answer would be a silent guess.
    """
    u = np.asarray(up, dtype=np.float64)
    n = np.asarray(north, dtype=np.float64)
    if u.shape != (3,) or n.shape != (3,):
        raise ValueError("up and north must each be three components")
    nu = float(np.linalg.norm(u))
    if nu < 1e-12:
        raise ValueError("the world frame's up axis has zero length")
    u = u / nu
    # Only the part of north that is perpendicular to up defines a bearing; the parallel part is
    # a tilt of the declared north out of the horizontal and is dropped rather than honoured.
    n = n - float(np.dot(n, u)) * u
    nn = float(np.linalg.norm(n))
    if nn < 1e-9:
        raise ValueError("the world frame's north axis is parallel to its up axis")
    n = n / nn
    east = np.cross(n, u)
    matrix = np.stack([east, u, -n])
    # A rotation, not merely a change of basis: the three rows are orthonormal and right-handed,
    # so `det == +1`. A -1 here would mirror the aircraft, which reads as a perfectly convincing
    # aircraft facing the wrong way.
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-9):
        raise ValueError("the world frame is left-handed: north x up does not give east")
    return matrix


@dataclass(frozen=True)
class StraightOutTrack:
    """A climbing run straight away from a ground observer, near to far, in **stage** axes.

    Shares :class:`FigureEightTrack`'s interface exactly -- ``position_m``, ``velocity``,
    ``yaw_deg``, ``range_m``, ``elevation_deg``, all on a phase in [0, 1] -- so a driver can fly
    either without knowing which it has.

    **The aircraft climbs, and the angle it climbs at is the parameter.** A target receding at a
    fixed height sinks toward the horizon: from 1.5 m of eye height, a drone holding 3 m is 14
    degrees up at 6 m and 1.1 degrees up at 80 m. This project's aerial scenes author no terrain
    (ADR 0060), so a ray leaving below the horizon samples the sky model at a negative elevation
    and comes back near air temperature -- a warm target on a warm background, which is not what
    an aircraft against the sky looks like, and the collapse would be an artefact of the track
    rather than a property of the range. Holding the elevation constant instead gives the target
    the *same* background at both ends of the run, so the contrast that survives is a statement
    about range. The aircraft's own temperature still moves, because the mission runs while it
    flies -- the readout carries it -- but the sky behind it does not.
    """

    #: The ground observer, and therefore the camera mount.
    observer_m: tuple[float, float, float] = (0.0, 1.5, 0.0)
    #: Slant range at the start and end of the run.
    near_m: float = 4.0
    far_m: float = 80.0
    #: Elevation above the observer's horizon, held for the whole run. 16 degrees is well clear
    #: of the horizon at both ends and still a plausible thing to watch from the ground. Named
    #: apart from :meth:`elevation_deg` on purpose: that method is part of the interface this
    #: shares with :class:`FigureEightTrack`, and a field of the same name would shadow it.
    hold_elevation_deg: float = 16.0

    def __post_init__(self) -> None:
        if not 0.0 < self.near_m < self.far_m:
            raise ValueError(f"need 0 < near_m < far_m, got {self.near_m} and {self.far_m}")
        if not 0.0 < self.hold_elevation_deg < 90.0:
            raise ValueError(
                f"hold_elevation_deg must be in (0, 90), got {self.hold_elevation_deg}: at "
                "or below zero the aircraft is on the horizon and its background stops being sky"
            )

    def _range(self, phase: object) -> NDArray[np.float64]:
        """Slant range at a phase, spaced **geometrically**: a fixed shrink rate per frame."""
        p = np.clip(np.asarray(phase, dtype=np.float64), 0.0, 1.0)
        return np.asarray(self.near_m * (self.far_m / self.near_m) ** p, dtype=np.float64)

    def position_m(self, phase: object) -> NDArray[np.float64]:
        """Aircraft position, ``(3,)`` for a scalar phase or ``(n, 3)`` for an array of them."""
        r = self._range(phase)
        theta = math.radians(self.hold_elevation_deg)
        origin = np.asarray(self.observer_m, dtype=np.float64)
        out = np.stack(
            [
                np.broadcast_to(origin[0], np.shape(r)),
                origin[1] + r * math.sin(theta),
                origin[2] - r * math.cos(theta),
            ],
            axis=-1,
        )
        return np.asarray(out, dtype=np.float64)

    def velocity(self, phase: object, *, delta: float = 1e-4) -> NDArray[np.float64]:
        """Unit velocity, by central difference on the phase. Direction only, not a speed."""
        p = np.asarray(phase, dtype=np.float64)
        step = self.position_m(np.clip(p + delta, 0.0, 1.0)) - self.position_m(
            np.clip(p - delta, 0.0, 1.0)
        )
        norm = np.linalg.norm(step, axis=-1, keepdims=True)
        return np.asarray(step / np.maximum(norm, 1e-12), dtype=np.float64)

    def yaw_deg(self, phase: object) -> NDArray[np.float64]:
        """Heading of the nose about +Y, degrees, in :meth:`FigureEightTrack.yaw_deg`'s sense.

        Zero for the whole run: the aircraft departs along -Z, which is north, and an aircraft
        flies where it points. The camera therefore sees its tail throughout, which is the honest
        aspect for a departing target and is *not* what the figure-eight clip shows -- that one
        sweeps the full circle of aspects and this one holds a single aspect so that range is the
        only variable. The two clips answer different questions and neither replaces the other.
        """
        return np.zeros(np.shape(np.asarray(phase, dtype=np.float64)), dtype=np.float64)

    def range_m(self, phase: object) -> NDArray[np.float64]:
        """Slant range from the observer to the aircraft."""
        return self._range(phase)

    def elevation_deg(self, phase: object) -> NDArray[np.float64]:
        """Elevation above the observer's horizon -- constant by construction, measured anyway.

        Measured from the position rather than returned from the field, so that a mistake in
        :meth:`position_m` shows up here instead of being confirmed by it.
        """
        delta = self.position_m(phase) - np.asarray(self.observer_m, dtype=np.float64)
        horizontal = np.hypot(delta[..., 0], delta[..., 2])
        return np.asarray(np.degrees(np.arctan2(delta[..., 1], horizontal)), dtype=np.float64)


@dataclass(frozen=True)
class FigureEightTrack:
    """One level circuit of a lemniscate, in **stage** axes (+X right, +Y up, -Z north).

    The figure is centred ``centre_range_m`` north of ``observer_m`` and lies in a horizontal
    plane whose height rises and falls once per circuit. Parameterised by a phase in [0, 1] rather
    than by time, so the same shape serves a 10 s pass and a 28 min mission.

    ``half_width_m`` is the lemniscate's ``a``: the figure reaches +-a across the line of sight and
    about +-0.354a along it, so the aircraft sweeps a wide arc and the camera has to slew.
    """

    #: The ground observer, and therefore the camera mount.
    observer_m: tuple[float, float, float] = (0.0, 1.5, 0.0)
    #: Distance from the observer to the crossing point of the eight. The defaults are chosen
    #: against the sensor rather than for looks: on a 640 x 512 Boson at 30.7 deg they put a
    #: 0.41 m aircraft between 46 and 104 px across, swing the slant range by 2.2:1, and keep the
    #: aircraft between 10 and 53 deg elevation -- never within ten degrees of the horizon.
    centre_range_m: float = 5.0
    half_width_m: float = 6.0
    altitude_low_m: float = 3.0
    altitude_high_m: float = 7.5

    def position_m(self, phase: object) -> NDArray[np.float64]:
        """Aircraft position, ``(3,)`` for a scalar phase or ``(n, 3)`` for an array of them."""
        p = np.asarray(phase, dtype=np.float64)
        u = 2.0 * math.pi * p
        denominator = 1.0 + np.sin(u) ** 2
        across = self.half_width_m * np.cos(u) / denominator
        along = self.half_width_m * np.sin(u) * np.cos(u) / denominator
        # (1 - cos) is zero at both ends of the circuit, so the height closes with the figure:
        # a lap that ended 8 m above where it started would make a looping video jump.
        height = self.altitude_low_m + (self.altitude_high_m - self.altitude_low_m) * 0.5 * (
            1.0 - np.cos(u)
        )
        origin = np.asarray(self.observer_m, dtype=np.float64)
        out = np.stack(
            [origin[0] + across, height, origin[2] - (self.centre_range_m + along)], axis=-1
        )
        return np.asarray(out, dtype=np.float64)

    def velocity(self, phase: object, *, delta: float = 1e-4) -> NDArray[np.float64]:
        """Unit velocity, by central difference on the phase. Direction only, not a speed."""
        p = np.asarray(phase, dtype=np.float64)
        step = self.position_m(p + delta) - self.position_m(p - delta)
        norm = np.linalg.norm(step, axis=-1, keepdims=True)
        return np.asarray(step / np.maximum(norm, 1e-12), dtype=np.float64)

    def yaw_deg(self, phase: object) -> NDArray[np.float64]:
        """Heading of the nose about +Y, in degrees, for an aircraft that flies where it points.

        Zero when the nose is along the stage's -Z (north), positive turning toward +X (east) --
        the same sense as :meth:`irsim_isaac.visible_sky.DomeSpec.sun_azimuth_in_stage_rad`, so a
        yaw and a sun azimuth quoted here mean the same thing.
        """
        v = self.velocity(phase)
        return np.asarray(np.degrees(np.arctan2(v[..., 0], -v[..., 2])), dtype=np.float64)

    def range_m(self, phase: object) -> NDArray[np.float64]:
        """Slant range from the observer to the aircraft."""
        delta = self.position_m(phase) - np.asarray(self.observer_m, dtype=np.float64)
        return np.asarray(np.linalg.norm(delta, axis=-1), dtype=np.float64)

    def elevation_deg(self, phase: object) -> NDArray[np.float64]:
        """Elevation of the aircraft above the observer's horizon.

        The one number that decides whether the background is sky or ground: this scene authors no
        terrain, so a ray that leaves below the horizon samples the sky model at a negative
        elevation and comes back at near-air temperature -- a bright background behind a warm
        target, which is not what an aircraft against the sky looks like.
        """
        delta = self.position_m(phase) - np.asarray(self.observer_m, dtype=np.float64)
        horizontal = np.hypot(delta[..., 0], delta[..., 2])
        return np.asarray(np.degrees(np.arctan2(delta[..., 1], horizontal)), dtype=np.float64)
