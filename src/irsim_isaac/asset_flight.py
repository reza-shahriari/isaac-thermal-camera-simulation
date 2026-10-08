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

**No path at all** (:class:`ScatterTrack`). A detector's training set is not a clip: consecutive
frames of a flight are near-duplicates, and a set built from them has seen one range, one aspect
and one patch of sky many times over. This one draws every frame's range, elevation, bearing and
heading independently from stated bands, behind the same interface, so the driver that films a
flight also films a dataset.

**Losing the aircraft** (:class:`LostLock`, EV.26). A real operator's mount sometimes loses the
drone, and the real sets' frames without one carry no box. A lost run swings the boresight off the
aircraft, sideways or up, holds it past the frame edge, and swings back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FigureEightTrack",
    "LostLock",
    "ScatterTrack",
    "StraightOutExitTrack",
    "StraightOutTrack",
    "WanderTrack",
    "HORIZON_MARGIN_DEG",
    "LOST_RAMP_FRAMES",
    "lost_lock_offset",
    "clear_exit_seconds",
    "horizon_safe_aim",
    "tilt_matrix",
    "lowest_framable_elevation_deg",
    "world_frame_to_stage",
]

#: EV.10: how far above the horizon a frame's bottom edge is kept, degrees. The aerial scenes
#: author no terrain (ADR 0060), so a frame reaching the horizon would show analytic ground.
HORIZON_MARGIN_DEG = 0.3


def horizon_safe_aim(
    target_elevation_deg: object,
    offset_elevation_deg: object,
    half_vfov_deg: float,
    margin_deg: float = HORIZON_MARGIN_DEG,
) -> NDArray[np.float64]:
    """The boresight's elevation offset from the target, raised where it would put the horizon in
    the frame (EV.10).

    A mount aimed at ``target + offset`` shows down to ``target + offset - half_vfov``. Where that
    falls below ``margin_deg`` the offset is raised until it does not, so the target sits lower in
    the frame -- what a real mount tracking a drone just above the horizon shows. Until EV.10 the
    driver refused any elevation band whose *worst* jittered frame could reach the horizon, which
    for the 50 mm lens and a 0.6 jitter stopped the band at 6 deg; real anti-UAV footage runs down
    to a fraction of a degree. Raises where no offset keeps the target inside the frame.
    """
    target = np.asarray(target_elevation_deg, dtype=np.float64)
    offset = np.asarray(offset_elevation_deg, dtype=np.float64)
    needed = margin_deg + float(half_vfov_deg) - target
    raised = np.maximum(offset, needed)
    # The target leaves the frame's bottom edge once the boresight is more than half the field
    # above it; keep a tenth of the half-field inside.
    if np.any(raised > 0.9 * float(half_vfov_deg)):
        lowest = float(np.min(target))
        raise ValueError(
            f"a target at {lowest:.2f} deg cannot be framed with the bottom edge "
            f"{margin_deg} deg above the horizon in a {2.0 * half_vfov_deg:.1f} deg field"
        )
    return np.asarray(raised, dtype=np.float64)


def lowest_framable_elevation_deg(
    half_vfov_deg: float, margin_deg: float = HORIZON_MARGIN_DEG
) -> float:
    """The lowest target elevation :func:`horizon_safe_aim` can frame, degrees."""
    return margin_deg + 0.1 * float(half_vfov_deg)


#: EV.25: the largest tilt a drawn attitude may reach, degrees. A camera multirotor's attitude
#: limit in normal flight is about 25-35 deg (DJI quotes 25 for the Phantom 4 in P-mode).
TILT_MAX_DEG = 25.0


def tilt_matrix(pitch_deg: float, roll_deg: float) -> NDArray[np.float64]:
    """Stage-axes rotation that pitches and rolls an aircraft whose nose points north (-Z) and
    whose right is east (+X) (EV.25).

    Pitch is nose-up positive, about the right axis; roll is right-wing-down positive, about the
    nose axis; roll is applied first, in the body. A driver composes ``spin(heading) @ tilt @
    spin(-nose)``, so the aircraft keeps its heading and leans about its own axes.
    """
    p, r = math.radians(float(pitch_deg)), math.radians(float(roll_deg))
    # Nose-up about +X: the nose (0, 0, -1) goes to (0, sin p, -cos p).
    pitch = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(p), -math.sin(p)], [0.0, math.sin(p), math.cos(p)]]
    )
    # Right-wing-down about the nose: +X goes to (cos r, -sin r, 0).
    roll = np.array(
        [[math.cos(r), math.sin(r), 0.0], [-math.sin(r), math.cos(r), 0.0], [0.0, 0.0, 1.0]]
    )
    return np.asarray(pitch @ roll, dtype=np.float64)


def _drawn_tilts(seed: int, count: int, sigma_deg: float) -> NDArray[np.float64]:
    """``(count, 2)`` pitch and roll, degrees: a tilt half-normal in magnitude with ``sigma_deg``,
    capped at :data:`TILT_MAX_DEG`, in a uniformly drawn direction. On its own stream, so adding
    it moved no other draw of a track."""
    rng = np.random.default_rng([int(seed), 25])
    magnitude = np.minimum(np.abs(rng.normal(0.0, sigma_deg, count)), TILT_MAX_DEG)
    direction = rng.uniform(0.0, 2.0 * math.pi, count)
    return np.stack([magnitude * np.cos(direction), magnitude * np.sin(direction)], axis=-1)


#: The stage's own axes, in the order the rotation's rows are built from: east, up, north.
STAGE_EAST = (1.0, 0.0, 0.0)
STAGE_UP = (0.0, 1.0, 0.0)
STAGE_NORTH = (0.0, 0.0, -1.0)


#: EV.26: frames the mount takes to swing off the aircraft before a lost run, and to come back
#: after it. One: a set's frames are seconds apart (5.6 s at the planner's default), so the swing
#: happens between two frames, and the frames outside the run are on the aircraft as usual. A
#: slower swing left the frame either side of a run empty too, so the run's empty frames came out
#: two more than its length.
LOST_RAMP_FRAMES = 1
#: EV.26: how far past the frame edge the boresight is held during a lost run, in half-fields,
#: beyond the wander's own reach: a drone at the dataset's ranges is under a tenth of a half-field
#: across (the real boxes are 34-83 px of 640), so half a half-field clears it with room.
LOST_CLEARANCE = 0.5


@dataclass(frozen=True)
class LostLock:
    """EV.26: a run of frames in which the camera has lost the aircraft.

    ``first`` and ``length`` are frame indices of the clip; ``direction_deg`` is where the mount
    swings, in the frame's own sense (0 right, 90 up, 180 left). Up or sideways only: a mount
    swung down would take the frame to the horizon, which the aerial scenes do not author, and
    :func:`horizon_safe_aim` would lift it back onto the aircraft.
    """

    first: int
    length: int
    direction_deg: float

    def __post_init__(self) -> None:
        if self.first < 0 or self.length < 1:
            raise ValueError(f"need first >= 0 and length >= 1, got {self.first}, {self.length}")
        if not 0.0 <= self.direction_deg <= 180.0:
            raise ValueError(f"direction must be in [0, 180] deg, got {self.direction_deg}")

    @classmethod
    def parse(cls, text: str) -> tuple[LostLock, ...]:
        """``"first:length:direction,..."`` as ``render_phantom4.py --lost-lock`` takes it."""
        runs = []
        for item in filter(None, (t.strip() for t in text.split(","))):
            first, length, direction = item.split(":")
            runs.append(cls(int(first), int(length), float(direction)))
        return tuple(sorted(runs, key=lambda r: r.first))

    def envelope(self, frame: float) -> float:
        """0 on the aircraft, 1 lost: a smoothstep over :data:`LOST_RAMP_FRAMES` either side of
        the run, flat at 1 for frames ``first`` to ``first + length - 1``."""
        ramp = float(LOST_RAMP_FRAMES)
        start, end = self.first - ramp, self.first + self.length - 1 + ramp
        if frame <= start or frame >= end:
            return 0.0
        x = min(frame - start, end - frame) / ramp
        x = min(max(x, 0.0), 1.0)
        return x * x * (3.0 - 2.0 * x)


def lost_lock_offset(
    runs: tuple[LostLock, ...], frame: float, aim_jitter: float
) -> NDArray[np.float64]:
    """The boresight's extra offset from the aircraft at ``frame``, in half-fields (x right, y up).

    At full envelope the larger component is ``1 + aim_jitter + LOST_CLEARANCE``: past the frame
    edge by more than the wander (up to ``aim_jitter`` half-fields, either way) can bring back,
    plus the aircraft's own size. The direction keeps its angle; it is scaled on its larger axis
    so a diagonal swing clears the frame as surely as a straight one.
    """
    out = np.zeros(2, dtype=np.float64)
    for run in runs:
        level = run.envelope(frame)
        if level > 0.0:
            a = math.radians(run.direction_deg)
            d = np.array([math.cos(a), math.sin(a)])
            out += level * (1.0 + aim_jitter + LOST_CLEARANCE) * d / float(np.abs(d).max())
    return out


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
class StraightOutExitTrack:
    """`EV.20`'s track: :class:`StraightOutTrack`, then the aircraft strafes out of the frame.

    The run is in **seconds**, not only phase: ``outbound_s`` of the climbing run from
    ``near_m`` to ``far_m``, then the mount **stops following** and holds the last boresight while
    the aircraft flies sideways at ``exit_speed_m_s``, then ``hold_s`` of the sky it left. The
    phase interface is kept (phase = clip seconds / :attr:`duration_s`), so a driver flies this
    as it flies the others and asks :meth:`aim_m` where to point.

    **The aircraft moves; the camera does not.** `EV.16`'s generated quad leaves the frame by a
    camera pan (ADR 0123: its occluders are authored in the world frame and cannot move). An
    imported asset with a mesh field has no such constraint (see :class:`StraightOutTrack`), so
    here the exit is what a ground observer actually sees: a fixed field and a drone crossing it,
    with the sky behind it never moving.

    **Sideways is -X, the camera's left, and the nose does not turn.** At the end of the run the
    boresight lies in the stage's YZ plane, so a displacement along X is perpendicular to it and
    the depth to the aircraft stays the slant range ``R`` at which the mount stopped: the
    aircraft's angle off the boresight is ``atan(x / R)`` exactly. A multirotor translates
    without yawing, so the heading stays north (tail to the camera) and the aspect does not change
    in the last second of the clip -- only the position does.
    """

    observer_m: tuple[float, float, float] = (0.0, 1.5, 0.0)
    near_m: float = 5.0
    far_m: float = 60.0
    hold_elevation_deg: float = 16.0
    outbound_s: float = 15.0
    exit_speed_m_s: float = 10.0
    #: How long the strafe lasts. :func:`clear_exit_seconds` solves the shortest that leaves the
    #: field; anything longer flies on out of it.
    exit_s: float = 2.0
    hold_s: float = 1.0

    def __post_init__(self) -> None:
        # The outbound part's own checks (near < far, elevation in (0, 90)) run here once.
        self._base()
        for name in ("outbound_s", "exit_speed_m_s", "exit_s"):
            if not float(getattr(self, name)) > 0.0:
                raise ValueError(f"{name} must be positive")
        if not float(self.hold_s) >= 0.0:
            raise ValueError("hold_s must not be negative")

    def _base(self) -> StraightOutTrack:
        return StraightOutTrack(
            observer_m=self.observer_m,
            near_m=self.near_m,
            far_m=self.far_m,
            hold_elevation_deg=self.hold_elevation_deg,
        )

    @property
    def duration_s(self) -> float:
        """The clip: outbound, strafe, hold."""
        return float(self.outbound_s + self.exit_s + self.hold_s)

    def time_s(self, phase: object) -> NDArray[np.float64]:
        """Clip seconds at a phase."""
        p = np.clip(np.asarray(phase, dtype=np.float64), 0.0, 1.0)
        return np.asarray(p * self.duration_s, dtype=np.float64)

    def position_m(self, phase: object) -> NDArray[np.float64]:
        """Aircraft position: on the run, then displaced along -X at the strafe speed.

        The aircraft keeps flying through the hold, so it is further out of the field, not
        parked on its edge.
        """
        t = self.time_s(phase)
        base = self._base()
        on_run = base.position_m(np.minimum(t / self.outbound_s, 1.0))
        lateral = self.exit_speed_m_s * np.maximum(t - self.outbound_s, 0.0)
        shift = np.stack([-lateral, np.zeros_like(lateral), np.zeros_like(lateral)], axis=-1)
        return np.asarray(on_run + shift, dtype=np.float64)

    def aim_m(self, phase: object) -> NDArray[np.float64]:
        """Where the mount points: at the aircraft on the run, then frozen where the run ended."""
        t = self.time_s(phase)
        return self._base().position_m(np.minimum(t / self.outbound_s, 1.0))

    def velocity(self, phase: object, *, delta: float = 1e-4) -> NDArray[np.float64]:
        """Unit velocity, by central difference on the phase. Direction only, not a speed."""
        p = np.asarray(phase, dtype=np.float64)
        step = self.position_m(np.clip(p + delta, 0.0, 1.0)) - self.position_m(
            np.clip(p - delta, 0.0, 1.0)
        )
        norm = np.linalg.norm(step, axis=-1, keepdims=True)
        return np.asarray(step / np.maximum(norm, 1e-12), dtype=np.float64)

    def yaw_deg(self, phase: object) -> NDArray[np.float64]:
        """Zero throughout: north on the run, and a strafe does not turn the nose."""
        return np.zeros(np.shape(np.asarray(phase, dtype=np.float64)), dtype=np.float64)

    def range_m(self, phase: object) -> NDArray[np.float64]:
        """Slant range from the observer, measured from the position."""
        delta = self.position_m(phase) - np.asarray(self.observer_m, dtype=np.float64)
        return np.asarray(np.linalg.norm(delta, axis=-1), dtype=np.float64)

    def elevation_deg(self, phase: object) -> NDArray[np.float64]:
        """Elevation above the observer's horizon, measured from the position."""
        delta = self.position_m(phase) - np.asarray(self.observer_m, dtype=np.float64)
        horizontal = np.hypot(delta[..., 0], delta[..., 2])
        return np.asarray(np.degrees(np.arctan2(delta[..., 1], horizontal)), dtype=np.float64)


def clear_exit_seconds(
    far_m: float,
    hfov_deg: float,
    ifov_mrad: float,
    extent_m: float,
    exit_speed_m_s: float,
    *,
    margin_px: float = 3.0,
) -> float:
    """The shortest strafe that leaves the whole aircraft ``margin_px`` beyond the field's edge.

    Closed form, because the strafe is perpendicular to a boresight that has stopped: the
    aircraft's centre is ``atan(x / far_m)`` off axis, so its near edge clears the field when
    ``x - extent_m / 2 = far_m * tan(hfov / 2 + margin)``. (`EV.16`'s pan had to be bisected,
    because there the camera turned and the aircraft did not.)
    """
    if not (far_m > 0.0 and exit_speed_m_s > 0.0 and extent_m >= 0.0):
        raise ValueError("far_m and exit_speed_m_s must be positive, extent_m not negative")
    if not 0.0 < hfov_deg < 180.0:
        raise ValueError("hfov_deg must be a real field of view")
    angle = math.radians(0.5 * float(hfov_deg)) + float(margin_px) * 1e-3 * float(ifov_mrad)
    if angle >= 0.5 * math.pi:
        raise ValueError("the margin puts the edge behind the camera")
    lateral = float(far_m) * math.tan(angle) + 0.5 * float(extent_m)
    return lateral / float(exit_speed_m_s)


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


@dataclass(frozen=True)
class ScatterTrack:
    """``count`` independent poses, one per frame, in **stage** axes -- a dataset, not a flight.

    Shares :class:`FigureEightTrack`'s interface, with the phase in [0, 1] selecting pose
    ``round(phase * (count - 1))``; a driver steps it with ``linspace(0, 1, count)``.

    **What is drawn, and why each one.** Slant range, log-uniform, because apparent size goes as
    1/R and a uniform draw would spend the set on small targets. Elevation, uniform over a band
    whose lower edge the *driver* must keep above half the vertical field of view, because the
    aerial scenes author no terrain (ADR 0060) and a horizon in frame is analytic ground. Bearing,
    uniform over the full circle: the cloud field is fixed in the sky, so turning the mount is
    what puts a different piece of it behind each target. Heading, uniform over the full circle
    and independent of bearing, so every aspect from head-on to tail-on is in the set.

    The published ablation this follows is the largest in the sim-to-real literature for drones:
    mAP@0.5 of 0.464 with a fixed camera pitch against 0.981 with a random one (roadmap EV.10).
    """

    count: int
    seed: int = 0
    observer_m: tuple[float, float, float] = (0.0, 1.5, 0.0)
    near_m: float = 4.0
    far_m: float = 40.0
    elevation_low_deg: float = 14.0
    elevation_high_deg: float = 60.0
    #: EV.25: each pose's tilt, half-normal with this sigma (degrees); 0 flies level.
    tilt_sigma_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"count must be >= 1, got {self.count}")
        if not 0.0 < self.near_m <= self.far_m:
            raise ValueError(f"need 0 < near_m <= far_m, got {self.near_m} and {self.far_m}")
        if not 0.0 < self.elevation_low_deg <= self.elevation_high_deg < 90.0:
            raise ValueError(
                "need 0 < elevation_low_deg <= elevation_high_deg < 90, got "
                f"{self.elevation_low_deg} and {self.elevation_high_deg}"
            )
        rng = np.random.default_rng(self.seed)
        n = self.count
        ranges = np.exp(rng.uniform(math.log(self.near_m), math.log(self.far_m), n))
        elevation = np.radians(rng.uniform(self.elevation_low_deg, self.elevation_high_deg, n))
        bearing = rng.uniform(0.0, 2.0 * math.pi, n)
        heading = rng.uniform(-180.0, 180.0, n)
        origin = np.asarray(self.observer_m, dtype=np.float64)
        horizontal = ranges * np.cos(elevation)
        # Bearing 0 is north (-Z), positive toward east (+X): the sense `yaw_deg` uses.
        positions = origin + np.stack(
            [
                horizontal * np.sin(bearing),
                ranges * np.sin(elevation),
                -horizontal * np.cos(bearing),
            ],
            axis=-1,
        )
        # Frozen dataclass: the draws are derived state, set once here and never again.
        object.__setattr__(self, "_positions", positions)
        object.__setattr__(self, "_heading", heading)
        object.__setattr__(self, "_tilts", _drawn_tilts(self.seed, n, self.tilt_sigma_deg))

    def _index(self, phase: object) -> NDArray[np.int64]:
        p = np.clip(np.asarray(phase, dtype=np.float64), 0.0, 1.0)
        return np.asarray(np.rint(p * (self.count - 1)), dtype=np.int64)

    def position_m(self, phase: object) -> NDArray[np.float64]:
        """Aircraft position, ``(3,)`` for a scalar phase or ``(n, 3)`` for an array of them."""
        positions: NDArray[np.float64] = self._positions  # type: ignore[attr-defined]
        return np.asarray(positions[self._index(phase)], dtype=np.float64)

    def yaw_deg(self, phase: object) -> NDArray[np.float64]:
        """Heading of the nose about +Y, degrees, in :meth:`FigureEightTrack.yaw_deg`'s sense."""
        heading: NDArray[np.float64] = self._heading  # type: ignore[attr-defined]
        return np.asarray(heading[self._index(phase)], dtype=np.float64)

    def attitude_deg(self, phase: object) -> NDArray[np.float64]:
        """``(pitch, roll)`` in degrees (:func:`tilt_matrix`'s sense) of the pose at ``phase``."""
        tilts: NDArray[np.float64] = self._tilts  # type: ignore[attr-defined]
        return np.asarray(tilts[self._index(phase)], dtype=np.float64)

    def range_m(self, phase: object) -> NDArray[np.float64]:
        """Slant range from the observer to the aircraft."""
        delta = self.position_m(phase) - np.asarray(self.observer_m, dtype=np.float64)
        return np.asarray(np.linalg.norm(delta, axis=-1), dtype=np.float64)

    def elevation_deg(self, phase: object) -> NDArray[np.float64]:
        """Elevation above the observer's horizon, measured from the position."""
        delta = self.position_m(phase) - np.asarray(self.observer_m, dtype=np.float64)
        horizontal = np.hypot(delta[..., 0], delta[..., 2])
        return np.asarray(np.degrees(np.arctan2(delta[..., 1], horizontal)), dtype=np.float64)


@dataclass(frozen=True)
class WanderTrack:
    """A seeded, smooth, never-repeating flight inside stated bands -- a clip that is also a set.

    :class:`ScatterTrack` gives a detector variety and gives a viewer nothing: its frames are
    unrelated, so it cannot be watched and the airframe's temperature cannot be seen to change.
    This track keeps the variety and makes it a flight. Range (in log R), elevation, bearing and
    heading are each a constant plus a sum of three sinusoids with seeded frequencies and phases,
    so the aircraft closes and recedes, climbs and sinks, circles the observer and turns through
    every aspect, continuously -- and a clip that spans the mission shows the airframe warm up
    while it does. The mount does not hold the target dead centre either: :meth:`aim_fraction`
    is the same kind of smooth wander, for the boresight.

    Same interface as :class:`FigureEightTrack`, phase in [0, 1]. ``cycles`` is the fastest
    component's number of periods per clip: frame-to-frame motion is about ``cycles / frames`` of
    a full swing, so a 300-frame clip at the default moves a few percent of its band per frame.
    """

    seed: int = 0
    observer_m: tuple[float, float, float] = (0.0, 1.5, 0.0)
    near_m: float = 4.0
    far_m: float = 40.0
    elevation_low_deg: float = 14.0
    elevation_high_deg: float = 60.0
    cycles: float = 6.0
    #: Full turns of bearing around the observer over the clip, on top of the wander: what takes
    #: the line of sight through every sun aspect.
    bearing_turns: float = 1.0
    #: EV.25: the tilt's scale, degrees: pitch and roll each wander smoothly over about
    #: +-1.5 sigma, capped at TILT_MAX_DEG. 0 flies level.
    tilt_sigma_deg: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.near_m <= self.far_m:
            raise ValueError(f"need 0 < near_m <= far_m, got {self.near_m} and {self.far_m}")
        if not 0.0 < self.elevation_low_deg <= self.elevation_high_deg < 90.0:
            raise ValueError(
                "need 0 < elevation_low_deg <= elevation_high_deg < 90, got "
                f"{self.elevation_low_deg} and {self.elevation_high_deg}"
            )
        if self.cycles < 1.0:
            raise ValueError(f"cycles must be >= 1, got {self.cycles}")
        rng = np.random.default_rng(self.seed)
        # Six channels (log range, elevation, bearing, heading, aim x, aim y), three sinusoids
        # each: frequencies spread over [1, cycles] and never equal, so no channel is periodic
        # over the clip and no two channels move in step.
        frequency = rng.uniform(1.0, self.cycles, size=(6, 3))
        phase = rng.uniform(0.0, 2.0 * math.pi, size=(6, 3))
        weight = rng.uniform(0.5, 1.0, size=(6, 3))
        weight /= weight.sum(axis=1, keepdims=True)
        object.__setattr__(self, "_frequency", frequency)
        object.__setattr__(self, "_phase", phase)
        object.__setattr__(self, "_weight", weight)
        object.__setattr__(self, "_start", rng.uniform(0.0, 360.0, size=2))
        # EV.25: two more channels on a stream of their own, so every earlier draw stands.
        tilt_rng = np.random.default_rng([self.seed, 25])
        object.__setattr__(self, "_tilt_frequency", tilt_rng.uniform(1.0, self.cycles, size=(2, 3)))
        object.__setattr__(self, "_tilt_phase", tilt_rng.uniform(0.0, 2.0 * math.pi, size=(2, 3)))

    def _wander(self, channel: int, phase: object) -> NDArray[np.float64]:
        """One channel's smooth signal in [-1, 1]."""
        p = np.asarray(phase, dtype=np.float64)[..., None]
        f = self._frequency[channel]  # type: ignore[attr-defined]
        return np.asarray(
            np.sum(
                self._weight[channel]  # type: ignore[attr-defined]
                * np.sin(2.0 * math.pi * f * p + self._phase[channel]),  # type: ignore[attr-defined]
                axis=-1,
            ),
            dtype=np.float64,
        )

    def _range(self, phase: object) -> NDArray[np.float64]:
        lo, hi = math.log(self.near_m), math.log(self.far_m)
        return np.exp(0.5 * (lo + hi) + 0.5 * (hi - lo) * self._wander(0, phase))

    def _elevation_rad(self, phase: object) -> NDArray[np.float64]:
        lo, hi = self.elevation_low_deg, self.elevation_high_deg
        return np.radians(0.5 * (lo + hi) + 0.5 * (hi - lo) * self._wander(1, phase))

    def position_m(self, phase: object) -> NDArray[np.float64]:
        """Aircraft position, ``(3,)`` for a scalar phase or ``(n, 3)`` for an array of them."""
        p = np.asarray(phase, dtype=np.float64)
        ranges, elevation = self._range(p), self._elevation_rad(p)
        bearing = np.radians(
            self._start[0]  # type: ignore[attr-defined]
            + 360.0 * self.bearing_turns * p
            + 40.0 * self._wander(2, p)
        )
        origin = np.asarray(self.observer_m, dtype=np.float64)
        horizontal = ranges * np.cos(elevation)
        out = origin + np.stack(
            [
                horizontal * np.sin(bearing),
                ranges * np.sin(elevation),
                -horizontal * np.cos(bearing),
            ],
            axis=-1,
        )
        return np.asarray(out, dtype=np.float64)

    def yaw_deg(self, phase: object) -> NDArray[np.float64]:
        """Heading of the nose about +Y, degrees, in :meth:`FigureEightTrack.yaw_deg`'s sense.

        A multirotor does not have to fly where it points, and this one does not: the heading
        turns on its own schedule, two and a half times round over the clip plus a wander, so
        the aspect seen from the ground is decoupled from the direction of travel.
        """
        p = np.asarray(phase, dtype=np.float64)
        heading = self._start[1] + 900.0 * p + 120.0 * self._wander(3, p)  # type: ignore[attr-defined]
        return np.asarray((heading + 180.0) % 360.0 - 180.0, dtype=np.float64)

    def attitude_deg(self, phase: object) -> NDArray[np.float64]:
        """``(pitch, roll)`` in degrees, each a smooth wander of about +-1.5 ``tilt_sigma_deg``
        (three sinusoids at seeded frequencies), capped at :data:`TILT_MAX_DEG`."""
        p = np.asarray(phase, dtype=np.float64)[..., None, None]
        f = self._tilt_frequency  # type: ignore[attr-defined]
        signal = np.sin(2.0 * math.pi * f * p + self._tilt_phase).mean(axis=-1)  # type: ignore[attr-defined]
        tilt = 1.5 * self.tilt_sigma_deg * np.sqrt(3.0) * signal
        return np.asarray(np.clip(tilt, -TILT_MAX_DEG, TILT_MAX_DEG), dtype=np.float64)

    def aim_fraction(self, phase: object) -> NDArray[np.float64]:
        """Where the mount points relative to the aircraft, ``(..., 2)`` in [-1, 1] per axis."""
        return np.stack([self._wander(4, phase), self._wander(5, phase)], axis=-1)

    def range_m(self, phase: object) -> NDArray[np.float64]:
        """Slant range from the observer to the aircraft."""
        delta = self.position_m(phase) - np.asarray(self.observer_m, dtype=np.float64)
        return np.asarray(np.linalg.norm(delta, axis=-1), dtype=np.float64)

    def elevation_deg(self, phase: object) -> NDArray[np.float64]:
        """Elevation above the observer's horizon, measured from the position."""
        delta = self.position_m(phase) - np.asarray(self.observer_m, dtype=np.float64)
        horizontal = np.hypot(delta[..., 0], delta[..., 2])
        return np.asarray(np.degrees(np.arctan2(delta[..., 1], horizontal)), dtype=np.float64)
