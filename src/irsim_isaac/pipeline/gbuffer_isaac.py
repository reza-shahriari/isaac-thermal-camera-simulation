"""Isaac Sim geometry AOVs -> the M0.6 :class:`~irsim.config.gbuffer.GBuffer` contract.

docs/physics-model.md §13.3 (AOV table), §13.6 (the Warp path), §5.3(a) (the reflected term);
roadmap M10.1; ADR 0014 (what this build actually transports) and ADR 0045 (the V_s definition).

The renderer is asked for **geometry and ids only**. Temperature never crosses a colour AOV on this
build (every one of them is float16 -- ADR 0014), so ``temperature_k`` and ``material_id`` are
looked up from float32 facet tables keyed by instance id and handed to :func:`to_gbuffer` by the
caller (the thermal bridge, M10.18, and the material-id transport, M10.2). What lives here is the
part that is purely geometric:

===================  ======================  =================================================
G-buffer plane       annotator (measured)    note
===================  ======================  =================================================
``distance_m``       ``DistanceToCameraSD``  Euclidean ray length, **not**
                                             ``DistanceToImagePlaneSD`` (z-depth; the two differ
                                             by 1/cos of the field angle, 4.7 % at the corner of
                                             a 50° field)
``normal_dot_view``  ``normals``             against the **per-pixel** ray direction
``normal_dot_up``    ``normals``             against the stage up axis; the AOV is world space
``sky_view_factor``  --                      ``occlusion * (1 + n.up)/2`` (ADR 0045); no ambient
                                             occlusion AOV delivers, so occlusion = 1
``motion_px``        --                      no motion AOV transports motion on this build, so the
                                             channel is **not attached** (``UNVERIFIED_CHANNELS``)
                                             and the plane is omitted -- it is optional in M0.6.
                                             The pipeline's motion is synthesised instead
                                             (:mod:`irsim.optics.motion`)
``semantic_id``      ``semantic_segmentation``
``sky_mask``         ``DistanceToCameraSD``  no geometry hit: the ray length is ``inf``
===================  ======================  =================================================

Those annotator choices are measurements, not guesses -- ``scripts/probe_isaac_geometry.py``
surveys every plausible name and the ADR 0014 addendum records the result. The trap worth naming
here: ``PtWorldNormal``, the obvious candidate and the one ADR 0014 originally recorded, attaches
without complaint and returns float16 at **half** the render product's resolution with every value
zero. Preferring it yields a G-buffer that validates cleanly and contains no surface orientation
at all, so :class:`AovReader` rejects an all-zero or wrong-shaped plane on a required channel.

**Why the ray direction and not the optical axis.** §13.5 assumed the normals AOV packed
``n.view`` in its ``.w`` channel; it does not exist in this build, so the cosine is computed here.
Doing it against the optical axis (or, equivalently, reading a view-space normal's z component)
is correct only at the principal point and drifts as the field angle grows -- on a sphere filling
half the frame the error exceeds 0.05, which is 5x the tolerance ``test_sphere_cos_theta`` allows.
The per-pixel ray direction comes from the position AOV, whose frame is a probed property of the
build rather than an assumption (:class:`PositionFrame`).

**Two-sided normals.** Meshes may be authored double-sided (the probe scenes are), so the stored
normal can point away from the camera. The surface facing the camera is the one radiating towards
it, so normals are flipped to the viewing hemisphere before either cosine is taken. A downward
facing surface seen from below therefore gets ``n.up = -1`` and ``V_s = 0``, which is what §5.3(a)
wants: it sees the ground, not the sky.

**Device residency.** Assembly is host-side NumPy in phase 1: it is the form the M0.6 schema test,
the CPU oracle (ADR 0018) and the M10.4 equivalence harness all consume, and it is a few hundred
microseconds at the resolutions in use. :class:`AovReader` still requests ``device="cuda"`` when
asked, and :class:`RawAovs` keeps the undecoded annotator payloads in ``device_handles``, so M10.9a
can bind the same buffers to Warp kernels without a host round trip when the whole frame moves onto
the device.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from irsim.config.gbuffer import GBuffer
from irsim.pipeline.environment import sky_view_factor

__all__ = [
    "PositionFrame",
    "MotionConvention",
    "UP_AXIS_VECTOR",
    "FORWARD_AXIS_VECTOR",
    "RawAovs",
    "GeometryPlanes",
    "AovReader",
    "camera_pose",
    "ray_directions",
    "orient_to_viewer",
    "geometry_planes",
    "to_gbuffer",
    "motion_px_per_frame",
    "AOV_NAMES",
    "UNVERIFIED_CHANNELS",
    "AOV_INIT_PARAMS",
]

#: Frame the position AOV is expressed in. **Measured on this build:** ``Camera3dPositionSD`` is
#: ``camera``, as its name says -- 9 mm of residual against the ray the distance AOV measures,
#: with the world reading out by the full camera offset (M2.4; ADR 0014's M2.4 addendum). ADR 0014
#: originally recorded *world* because a camera at the origin makes the two frames coincide, and
#: reading it as world on a tilted camera tilts every ray, elevation and view cosine together --
#: smooth, plausible and entirely wrong (M10.19). ``IrCamera`` therefore passes ``"camera"``.
#: The ``"world"`` option stays because the engine-free fixtures author world positions directly;
#: it is not what this renderer returns.
PositionFrame = Literal["world", "camera"]

#: How the motion AOV expresses image-plane velocity. ``"pixels"``: already px/frame.
#: ``"ndc"``: fraction of the full frame per frame, so (vx, vy) scale by (width, height).
#: ``"uv"``: same but with y pointing down. Probed, not assumed (M10.1).
MotionConvention = Literal["pixels", "ndc", "uv"]

UP_AXIS_VECTOR: dict[str, tuple[float, float, float]] = {
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}

#: A reference bearing on the horizon for each up axis, used where an azimuth has to be measured
#: from *something*. It must only be perpendicular to its own up vector: `sky_angles` states that
#: only stability matters for sampling, not where zero lands. Kept beside `UP_AXIS_VECTOR` because
#: the two are a pair -- passing an up axis without its companion forward is what made every Z-up
#: stage raise "forward must not be parallel to up" (AI.2).
FORWARD_AXIS_VECTOR: dict[str, tuple[float, float, float]] = {
    "Y": (0.0, 0.0, -1.0),
    "Z": (0.0, 1.0, 0.0),
}

#: Annotators to try per channel, in preference order, as measured by
#: ``scripts/probe_isaac_geometry.py`` on this build (ADR 0014 addendum). The order matters and is
#: not cosmetic: ``PtWorldNormal`` attaches happily, returns float16 at **half** the render
#: product's resolution, and is **all zero** -- a reader that preferred it would produce a
#: plausible-looking G-buffer with no surface orientation in it at all. ``normals`` is the one that
#: delivers float32 at full resolution. ``AmbientOcclusion`` and ``Motion2d`` return no data here;
#: they are kept as second choices in case a later build revives them.
#:
#: ``instance_id_segmentation`` before ``instance_segmentation`` is the other measured correction:
#: ``instance_segmentation`` gives a distinct id only to prims that carry a **semantic label**, and
#: collapses every unlabelled prim into a single ``UNLABELLED`` id. ADR 0014 recorded it as exact
#: per prim, but its ramp scene labelled all 64 quads, so the degeneracy never showed. An asset
#: with unlabelled prims would silently paint them all one material.
#: Channels whose annotator returns *something* on this build but whose meaning has never been
#: established, so :class:`AovReader` does not attach them for production use. A caller that wants
#: one has to ask for it by name (``unverified=("motion",)``), which is what the probe does.
#:
#: ``motion`` is the only member and the reason is IG.5. ADR 0014's addendum measured
#: ``motion_vectors`` sitting at a ~6e-5 floor after a **180 px** displacement -- it transports no
#: motion -- and ``Motion2d`` returns nothing. But ``_reject_reason`` applies its all-zero test
#: only to *required* channels, and 6e-5 is not zero anyway, so the plane was reaching the
#: G-buffer and :func:`irsim.optics.stage` was running the smear path on it. Numerically that is a
#: no-op; the hazard is the **convention**. Nothing has ever checked the sign, Replicator's own
#: documentation gives both signs opposite to this project's contract, and a plane that is noise
#: today becomes a plane that is backwards the day a build starts filling it in.
#:
#: The motion the pipeline actually uses is synthesised from rigid-body transforms instead
#: (:mod:`irsim.optics.motion`, M10.1b) -- which exists precisely because this AOV does not work,
#: and is verified in-sim to 0.1 px. Reinstating the AOV means asserting its sign against a known
#: displacement first, which is what ``tests/integration/test_motion_px_isaac.py`` already does for
#: the analytic path.
UNVERIFIED_CHANNELS = frozenset({"motion"})

AOV_NAMES: dict[str, tuple[str, ...]] = {
    "distance": ("DistanceToCameraSD",),
    "position": ("Camera3dPositionSD", "PtWorldPos"),
    "normal": ("normals", "PtWorldNormal", "SmoothNormal"),
    "occlusion": ("AmbientOcclusion",),
    "motion": ("motion_vectors", "Motion2d"),
    "instance": ("instance_id_segmentation", "instance_segmentation"),
    "semantic": ("semantic_segmentation",),
    #: The ordinary visible-light render from the *same* camera prim. It carries no infrared
    #: information whatever -- it is float16-adjacent 8-bit colour through the renderer's own tone
    #: mapping -- and is captured only as a companion image: same pose, same lens, same distortion,
    #: so an RGB/IR pair is registered by construction rather than by calibration.
    "rgb": ("rgb", "LdrColor"),
}

#: Init params per annotator. ``colorize=False`` keeps the segmentation channels as uint32 ids
#: rather than an RGBA palette image -- a colorized id cannot be looked up, only guessed at.
AOV_INIT_PARAMS: dict[str, dict[str, Any]] = {
    "instance": {"colorize": False},
    "semantic": {"colorize": False},
}

_SKY_NORMAL_DOT_VIEW = np.float32(1.0)
_SKY_NORMAL_DOT_UP = np.float32(1.0)
_SKY_SKY_VIEW_FACTOR = np.float32(1.0)


def _as_f64_plane(name: str, value: Any, channels: int | None) -> NDArray[np.float64]:
    """Validate one float AOV. **No plane may arrive as float16** (CLAUDE.md #2, roadmap IG.8).

    This used to carve out normals, occlusion and motion, on the stated grounds that "the renderer
    on this build delivers the normals AOV as float16 whether we like it or not (measured;
    ADR 0014 addendum)" -- so banning fp16 "would have meant no normals at all".

    **That claim was wrong, and it was the whole justification.** ADR 0014's own addendum table
    records ``normals`` as **float32 x4 at full resolution** and marks it *use*; this build's
    Replicator registry agrees, registering
    ``"normals": AnnotatorParams("NormalSD", np.float32, 4, ...)``. Nothing was being rescued. The
    fp16 plane the addendum *does* record is ``PtWorldNormal``, which is half-resolution and all
    zero, and which :class:`AovReader` rejects for those reasons rather than for its dtype.

    So the carve-out bought no physics and spent the one rule that has no cheap symptom: fp16's
    0.5 % relative spacing is invisible in a rendered frame and fatal in a radiometric one. If a
    future build really does deliver a half-precision plane the pipeline needs, that is a
    measurement and a decision to record -- not a default to inherit from a sentence.
    """
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"{name} must be a float AOV, got {arr.dtype}")
    if arr.dtype == np.float16:
        raise TypeError(
            f"{name} arrived as float16, which no plane of the G-buffer may be: half precision "
            "spaces values 0.5 % apart, which is the silent-failure mode CLAUDE.md non-negotiable "
            "#2 exists to stop. Register the annotator with output_data_type=np.float32 "
            "(ADR 0014; the registry already does for every channel this pipeline reads)."
        )
    if channels is None:
        if arr.ndim != 2:
            raise ValueError(f"{name} must be (H, W), got {arr.shape}")
    else:
        if arr.ndim != 3 or arr.shape[2] < channels:
            raise ValueError(f"{name} must be (H, W, >={channels}), got {arr.shape}")
        arr = arr[:, :, :channels]
    return arr.astype(np.float64)


@dataclass(frozen=True)
class RawAovs:
    """One frame of renderer output, still in the renderer's own conventions.

    ``device_handles`` keeps whatever :meth:`AovReader.read` was given before it became NumPy
    (Warp arrays when ``device="cuda"``), so a later all-device path can bind them directly. It is
    never read by the host assembly below.
    """

    distance_m: NDArray[np.floating[Any]]
    normal: NDArray[np.floating[Any]]
    position: NDArray[np.floating[Any]]
    occlusion: NDArray[np.floating[Any]] | None = None
    motion: NDArray[np.floating[Any]] | None = None
    instance_id: NDArray[np.integer[Any]] | None = None
    semantic_id: NDArray[np.integer[Any]] | None = None
    rgb: NDArray[np.uint8] | None = None
    device_handles: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.distance_m.shape[0]), int(self.distance_m.shape[1]))


@dataclass(frozen=True)
class GeometryPlanes:
    """The geometry half of the G-buffer, float32 and finite everywhere."""

    distance_m: NDArray[np.float32]
    normal_dot_view: NDArray[np.float32]
    normal_dot_up: NDArray[np.float32]
    sky_view_factor: NDArray[np.float32]
    sky_mask: NDArray[np.bool_]
    motion_px: NDArray[np.float32] | None = None
    semantic_id: NDArray[np.uint32] | None = None
    instance_id: NDArray[np.uint32] | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.distance_m.shape[0]), int(self.distance_m.shape[1]))


def camera_pose(
    camera_path: str, *, stage: Any = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """A camera prim's ``(translation, rotation)`` in the convention :func:`ray_directions` wants.

    USD matrices are row-vector -- ``p_world = p_camera @ M`` -- while :func:`ray_directions`
    applies its rotation as ``vec @ rot.T``, so what it wants is the **transpose** of the upper
    left 3x3. That transpose is the single place the two conventions meet, and getting it
    backwards rotates every ray by twice the camera's tilt without raising anything, so it is
    written down once here and imported rather than repeated at each call site.
    """
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = stage if stage is not None else omni.usd.get_context().get_stage()
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(camera_path))
    matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = np.asarray(matrix.ExtractTranslation(), dtype=np.float64)
    return translation, np.asarray(matrix, dtype=np.float64)[:3, :3].T


def ray_directions(
    position: Any,
    *,
    frame: PositionFrame,
    camera_position: Any = None,
    camera_to_world: Any = None,
) -> NDArray[np.float64]:
    """Unit vector from the camera to each surface point, in **world** axes.

    ``frame="world"``: the AOV holds world positions and ``camera_position`` is subtracted.
    ``frame="camera"``: the AOV holds camera-space positions, whose direction is already the ray;
    it is rotated into world axes by ``camera_to_world`` so it can be dotted with a world normal.
    Degenerate rays (a point at the camera, or a sky pixel whose position is 0/inf) come back as
    ``(0, 0, 0)`` and are masked out by the caller.
    """
    pos = _as_f64_plane("position", position, 3)
    if frame == "world":
        if camera_position is None:
            raise ValueError("frame='world' needs camera_position")
        cam = np.asarray(camera_position, dtype=np.float64).reshape(3)
        vec = pos - cam
    elif frame == "camera":
        vec = pos
        if camera_to_world is not None:
            rot = np.asarray(camera_to_world, dtype=np.float64)
            if rot.shape == (4, 4):
                rot = rot[:3, :3]
            if rot.shape != (3, 3):
                raise ValueError(f"camera_to_world must be 3x3 or 4x4, got {rot.shape}")
            vec = vec @ rot.T
    else:  # pragma: no cover - Literal is checked by mypy
        raise ValueError(f"unknown position frame {frame!r}")
    finite = np.asarray(np.isfinite(vec).all(axis=2))
    vec = np.where(finite[..., None], vec, 0.0)
    norm = np.linalg.norm(vec, axis=2, keepdims=True)
    return np.asarray(np.divide(vec, norm, out=np.zeros_like(vec), where=norm > 0.0))


def orient_to_viewer(
    normal: Any, ray_dir: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unit normals flipped into the viewing hemisphere, with ``cos(theta) = n . (-d)``.

    Returns ``(normal, normal_dot_view)``. The cosine is non-negative by construction; a normal
    that is exactly perpendicular to the ray (a silhouette pixel) gives 0.
    """
    n = _as_f64_plane("normal", normal, 3)
    norm = np.linalg.norm(n, axis=2, keepdims=True)
    n = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0.0)
    cos_theta = -np.sum(n * ray_dir, axis=2)
    flip = cos_theta < 0.0
    n = np.where(flip[..., None], -n, n)
    return n, np.abs(cos_theta)


def motion_px_per_frame(
    motion: Any, *, convention: MotionConvention, shape: tuple[int, int]
) -> NDArray[np.float32]:
    """Image-plane velocity as (H, W, 2) float32 px/frame, y positive downwards.

    The G-buffer's ``motion_px`` is px/frame in image coordinates (tests/conftest.py's
    ``gbuffer_moving_edge``). ``"ndc"`` output covers the frame over [-1, 1] with y up, so it
    scales by half the resolution and flips; ``"uv"`` covers [0, 1] with y already down.
    """
    height, width = shape
    raw = _as_f64_plane("motion", motion, 2)
    if convention == "pixels":
        out = raw
    elif convention == "ndc":
        out = raw * np.array([0.5 * width, -0.5 * height], dtype=np.float64)
    elif convention == "uv":
        out = raw * np.array([width, height], dtype=np.float64)
    else:  # pragma: no cover - Literal is checked by mypy
        raise ValueError(f"unknown motion convention {convention!r}")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def geometry_planes(
    aovs: RawAovs,
    *,
    up_axis: str = "Y",
    position_frame: PositionFrame = "world",
    camera_position: Any = None,
    camera_to_world: Any = None,
    motion_convention: MotionConvention | None = None,
    default_occlusion: float = 1.0,
) -> GeometryPlanes:
    """Turn one frame of raw AOVs into the geometry half of the G-buffer (ADR 0014, ADR 0045).

    Sky pixels -- where the renderer hit nothing and ``DistanceToCameraSD`` is ``inf`` -- get
    ``distance_m = 0`` so a consumer that ignores ``sky_mask`` still sees tau = 1, and neutral
    cosines; the kernels treat them as blackbody-equivalent from the sky model (M10.18), so the
    values under the mask are placeholders, not physics.

    ``default_occlusion`` is used when the build returns no ambient-occlusion AOV, making
    ``V_s = (1 + n.up)/2``: the unoccluded sky fraction of a tilted plane. That is exact for the
    open-sky aerial scenes of phase 1 and optimistic for a cluttered ground scene.
    """
    if up_axis not in UP_AXIS_VECTOR:
        raise ValueError(f"up_axis must be one of {sorted(UP_AXIS_VECTOR)}, got {up_axis!r}")
    up = np.asarray(UP_AXIS_VECTOR[up_axis], dtype=np.float64)

    distance = _as_f64_plane("distance_m", aovs.distance_m, None)
    sky = ~np.isfinite(distance)
    if aovs.instance_id is not None:
        # Background id 0 is the other half of the same fact (ADR 0014); either marks sky.
        ids = np.asarray(aovs.instance_id)
        if ids.shape != distance.shape:
            raise ValueError(f"instance_id has shape {ids.shape}, expected {distance.shape}")
        sky |= ids == 0
    if np.any(distance[~sky] < 0.0):
        raise ValueError("distance_m has negative values on geometry pixels")

    ray = ray_directions(
        aovs.position,
        frame=position_frame,
        camera_position=camera_position,
        camera_to_world=camera_to_world,
    )
    normal, n_view = orient_to_viewer(aovs.normal, ray)
    n_up = np.sum(normal * up, axis=2)

    if aovs.occlusion is None:
        occlusion = np.full(distance.shape, float(default_occlusion), dtype=np.float64)
    else:
        occ = np.asarray(aovs.occlusion)
        occ = occ[:, :, 0] if occ.ndim == 3 else occ
        occlusion = np.clip(_as_f64_plane("occlusion", occ, None), 0.0, 1.0)

    v_s = sky_view_factor(np.clip(n_up, -1.0, 1.0), occlusion)

    distance = np.where(sky, 0.0, np.nan_to_num(distance, posinf=0.0, neginf=0.0))
    n_view = np.where(sky, _SKY_NORMAL_DOT_VIEW, np.clip(n_view, 0.0, 1.0))
    n_up = np.where(sky, _SKY_NORMAL_DOT_UP, np.clip(n_up, -1.0, 1.0))
    v_s = np.where(sky, _SKY_SKY_VIEW_FACTOR, v_s)

    motion = None
    if aovs.motion is not None:
        if motion_convention is None:
            raise ValueError(
                "a motion AOV was supplied with no motion_convention. There is no safe default: "
                "the three conventions differ by a factor of the resolution and by the sign of y, "
                "and Replicator's documentation gives both signs opposite to this project's "
                "contract, so a guess produces smear in the wrong direction rather than an error "
                "(IG.5). Name the convention, or use irsim.optics.motion's synthesised plane."
            )
        # `.shape` is `tuple[int, ...]`; the plane is 2-D by construction and the callee wants
        # that stated, so it is narrowed here rather than widened there.
        plane_shape = (int(distance.shape[0]), int(distance.shape[1]))
        motion = motion_px_per_frame(aovs.motion, convention=motion_convention, shape=plane_shape)
        motion = np.where(sky[..., None], np.float32(0.0), motion).astype(np.float32)

    return GeometryPlanes(
        distance_m=distance.astype(np.float32),
        normal_dot_view=n_view.astype(np.float32),
        normal_dot_up=n_up.astype(np.float32),
        sky_view_factor=v_s.astype(np.float32),
        sky_mask=sky,
        motion_px=motion,
        semantic_id=None if aovs.semantic_id is None else np.asarray(aovs.semantic_id, np.uint32),
        instance_id=None if aovs.instance_id is None else np.asarray(aovs.instance_id, np.uint32),
    )


def to_gbuffer(
    planes: GeometryPlanes,
    *,
    temperature_k: Any,
    material_id: Any,
    sky_temperature_k: Any | None = None,
) -> GBuffer:
    """Join the geometry planes to the facet-table lookups and validate the M0.6 contract.

    ``temperature_k`` and ``material_id`` are per-pixel planes the caller built by indexing the
    float32 facet table with ``planes.instance_id`` (M10.2, M10.18) -- they never come from a
    colour AOV on this build (ADR 0014). ``sky_temperature_k`` overwrites the masked pixels with
    the sky model's apparent temperature T_sky(theta) (MS.2); without it the caller is asserting
    that ``temperature_k`` already carries the sky.
    """
    t = np.asarray(temperature_k)
    if t.dtype == np.float16:
        raise TypeError(
            "temperature_k arrived as float16: 0.25 K spacing at 300 K against a 50 mK NETD "
            "(CLAUDE.md non-negotiable #2). The facet table must be float32."
        )
    t = t.astype(np.float32)
    if t.shape != planes.shape:
        raise ValueError(f"temperature_k has shape {t.shape}, expected {planes.shape}")
    if sky_temperature_k is not None:
        sky_t = np.asarray(sky_temperature_k, dtype=np.float32)
        t = np.where(planes.sky_mask, np.broadcast_to(sky_t, planes.shape), t).astype(np.float32)

    mat = np.asarray(material_id)
    if not np.issubdtype(mat.dtype, np.integer):
        raise TypeError(f"material_id must be an integer plane, got {mat.dtype}")

    out: dict[str, Any] = {
        "temperature_k": t,
        "normal_dot_view": planes.normal_dot_view,
        "distance_m": planes.distance_m,
        "material_id": mat.astype(np.int32),
        "sky_view_factor": planes.sky_view_factor,
        "sky_mask": planes.sky_mask,
    }
    if planes.motion_px is not None:
        out["motion_px"] = planes.motion_px
    if planes.semantic_id is not None:
        out["semantic_id"] = planes.semantic_id
    return GBuffer.from_dict(out)


# --- the engine side ---------------------------------------------------------------------------


def _to_numpy(data: Any) -> NDArray[Any] | None:
    """Annotator payload -> NumPy, without assuming which of the three shapes it took."""
    if data is None:
        return None
    if isinstance(data, dict):
        data = data.get("data", data)
    if hasattr(data, "numpy"):  # a Warp array read back to the host
        data = data.numpy()
    arr = np.asarray(data)
    return None if arr.size == 0 else arr


class AovReader:
    """Attach the M10.1 annotator set to one render product and read a frame at a time.

    Annotators are attached once and reused: attaching per frame costs a render each and would
    make a moving-target readout meaningless. Each logical channel is resolved against the
    candidate names in :data:`AOV_NAMES`, because which of them a build answers to is a measured
    property, not a documented one (ADR 0014 found the colour AOVs renamed and several
    ground-truth ones silent). :attr:`resolved` records what was actually found so a test can
    assert on it instead of quietly running with a missing plane.

    ``device="cuda"`` asks Replicator for Warp arrays; they are kept in
    :attr:`RawAovs.device_handles` and also copied to the host for the phase-1 assembly.
    """

    def __init__(
        self,
        render_product_path: str,
        *,
        device: str = "cpu",
        names: dict[str, tuple[str, ...]] | None = None,
        required: tuple[str, ...] = ("distance", "position", "normal"),
        expected_shape: tuple[int, int] | None = None,
        unverified: Sequence[str] = (),
    ) -> None:
        self.render_product_path = render_product_path
        self.device = device
        # A channel in `UNVERIFIED_CHANNELS` is attached only when the caller names it (IG.5). It
        # is dropped rather than rejected later, because `_reject_reason` cannot tell a plane whose
        # *values* are meaningless from one whose values are fine.
        wanted = set(unverified)
        unknown = wanted - UNVERIFIED_CHANNELS
        if unknown:
            raise ValueError(
                f"{sorted(unknown)} are not unverified channels; "
                f"the unverified set is {sorted(UNVERIFIED_CHANNELS)}"
            )
        self.names = {
            channel: candidates
            for channel, candidates in dict(names or AOV_NAMES).items()
            if channel not in UNVERIFIED_CHANNELS or channel in wanted
        }
        self.skipped = tuple(sorted((set(dict(names or AOV_NAMES)) & UNVERIFIED_CHANNELS) - wanted))
        blocked = set(required) & set(self.skipped)
        if blocked:
            raise ValueError(
                f"{sorted(blocked)} cannot be required while unverified: pass "
                f"unverified={tuple(sorted(blocked))!r} to attach it deliberately"
            )
        self.required = required
        self.expected_shape = expected_shape
        self.resolved: dict[str, str] = {}
        self.failures: dict[str, str] = {}
        self._annotators: dict[str, Any] = {}

    def attach(self, *, settle_frames: int = 4, rt_subframes: int = 1) -> AovReader:
        """Attach every channel whose annotator can be created, then settle the renderer.

        A channel is kept only once it has returned data: ADR 0014 measured several annotators
        that attach happily and never produce anything, and a kernel fed a silently missing plane
        is exactly the failure this class exists to prevent.
        """
        import omni.replicator.core as rep

        for channel, candidates in self.names.items():
            init_params = AOV_INIT_PARAMS.get(channel)
            for name in candidates:
                try:
                    anno = rep.AnnotatorRegistry.get_annotator(
                        name, init_params=init_params, device=self.device
                    )
                    anno.attach(self.render_product_path)
                except Exception as exc:  # noqa: BLE001 - any failure means "try the next name"
                    self.failures[f"{channel}:{name}"] = f"{type(exc).__name__}: {exc}"
                    continue
                self._annotators[channel] = anno
                self.resolved[channel] = name
                break

        for _ in range(settle_frames):
            rep.orchestrator.step(rt_subframes=rt_subframes)

        for channel, anno in list(self._annotators.items()):
            reason = self._reject_reason(channel, _to_numpy(anno.get_data()))
            if reason is not None:
                self.failures[f"{channel}:{self.resolved[channel]}"] = reason
                with contextlib.suppress(Exception):
                    anno.detach(self.render_product_path)
                del self._annotators[channel]
                del self.resolved[channel]

        missing = [c for c in self.required if c not in self._annotators]
        if missing:
            raise RuntimeError(
                f"required G-buffer channels {missing} produced no data on this build; "
                f"tried {[self.names[c] for c in missing]}, failures: {self.failures}"
            )
        return self

    def _reject_reason(self, channel: str, arr: NDArray[Any] | None) -> str | None:
        """Why this annotator cannot be used, or None if it can.

        Two of the three rejections come from measured behaviour on this build: an annotator that
        attaches and returns nothing, and one that returns an all-zero buffer (``PtWorldNormal``).
        ADR 0014 records the same hazard on the SPG side -- a kernel that fails to load still
        produces a zero-filled output with status ok -- so silence and zeros are both treated as
        failure for the channels the physics cannot do without.
        """
        if arr is None:
            return "attached but returned no data"
        if channel not in self.required:
            return None
        if np.issubdtype(arr.dtype, np.floating) and not np.any(arr != 0.0):
            return f"returned an all-zero {arr.dtype} buffer of shape {arr.shape}"
        if self.expected_shape is not None and tuple(arr.shape[:2]) != self.expected_shape:
            return (
                f"returned shape {arr.shape} at a render product of {self.expected_shape}: "
                "an AOV at the renderer's internal resolution misaddresses every pixel lookup"
            )
        return None

    def step(self, *, frames: int = 1, rt_subframes: int = 1) -> None:
        import omni.replicator.core as rep

        for _ in range(frames):
            rep.orchestrator.step(rt_subframes=rt_subframes)

    def read(self) -> RawAovs:
        """One frame of raw AOVs. Call :meth:`step` first to advance a moving scene."""
        handles = {c: a.get_data() for c, a in self._annotators.items()}
        planes = {c: _to_numpy(v) for c, v in handles.items()}
        for channel in self.required:
            if planes.get(channel) is None:
                raise RuntimeError(f"channel {channel!r} returned no data this frame")
        ids = planes.get("instance")
        semantic = planes.get("semantic")
        colour = planes.get("rgb")
        return RawAovs(
            distance_m=np.asarray(planes["distance"]),
            normal=np.asarray(planes["normal"]),
            position=np.asarray(planes["position"]),
            occlusion=None if planes.get("occlusion") is None else np.asarray(planes["occlusion"]),
            motion=None if planes.get("motion") is None else np.asarray(planes["motion"]),
            instance_id=None if ids is None else np.asarray(ids).astype(np.uint32),
            semantic_id=None if semantic is None else np.asarray(semantic).astype(np.uint32),
            rgb=None if colour is None else np.asarray(colour).astype(np.uint8),
            device_handles=handles,
        )

    def detach(self) -> None:
        for anno in self._annotators.values():
            with contextlib.suppress(Exception):
                anno.detach(self.render_product_path)
        self._annotators.clear()

    def __enter__(self) -> AovReader:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.detach()
