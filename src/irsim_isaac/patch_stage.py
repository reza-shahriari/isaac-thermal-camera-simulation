"""A patch scene on the stage: every solved surface becomes the prim its field is drawn on (IG.20).

Phase P's reference scenes -- a wall half in sun (`PT.20`), a road wet on one half (`PH.2`) -- are
**thermal** scenes: a scene config of world-frame patches, each with a ``prim_path``, solved and
tested engine-free on a synthetic G-buffer. None had a stage, so none had ever been rendered;
their rows said the frame "is IG.2's". This module is the step from a patch to a prim, for any
such scene, so a new one needs a camera placement and nothing else.

**The config is rotated into the stage frame, not the stage into the config's.** The patch
scenes are authored in ENU (x east, y north, z up); the Isaac glue is Y-up with -Z forward
(`sun_direction_stage`, the dome, the camera helpers), and the point bridge compares each
pixel's *stage* position with the patch's *own* coordinates. So :func:`to_stage_frame` rewrites
every world coordinate in the config -- patch origins and axes, occluders, plume origins and
directions -- through one rotation, and declares ``world_frame: {up: +Y, north: -Z}`` so the sun
is still placed from the site and the clock (PT.18). A rotation changes no physics: every cell
solves to the same temperature, which `tests/unit/test_patch_stage.py` asserts, and that
equality is what licenses the whole approach.

**One prim per surface.** Two surfaces may share a ``prim_path`` -- the wall's concrete and
insulated-render halves are one face -- but a prim has one infrared material, so a shared path is
split into ``<prim_path>/<surface>`` children and the bindings follow. Each prim carries
``thermal:material`` from its own surface (ADR 0047), so emissivity and temperature come from the
same line of YAML.

**A water film is drawn for the visible companion only.** The road's film is a *thermal* state
of its cells -- latent cooling on the wet half -- and the infrared picture of it is the solved
field. The visible band needs to see it too, or the pair disagrees about where the road is wet:
:func:`patch_prims` emits a thin, dark, glossy sheet over the film's region, which the driver
hands to ``IrCamera(companion_only_prim_paths=...)`` (AT.31) so it never reaches the G-buffer.
The infrared emissivity of the wet cells stays the dry surface's -- ~0.93 against a water film's
~0.96, a flagged approximation (the camera has one material per prim).

**Nothing below is optional physics.** A config with mesh surfaces or solved objects is refused
rather than half-rotated: those carry frames of their own and are not what this driver is for.

docs/physics-model.md §6.13, §13.4; ADR 0087 (the field), ADR 0095 (occluders, world frame),
ADR 0172 (this driver)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from irsim.config.scene import SceneConfig

__all__ = [
    "ENU_TO_STAGE",
    "FALLBACK_NODE",
    "PatchPrim",
    "QUAD_INSET_M",
    "author_patch_prims",
    "binding_path",
    "enu_to_stage",
    "patch_bindings",
    "patch_prims",
    "to_stage_frame",
    "with_fallback_node",
]

#: ENU -> stage: stage X = east, stage Y = up, stage Z = -north. A proper rotation (det +1), so
#: every patch keeps its outward normal.
ENU_TO_STAGE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)

#: The per-prim node every patch prim falls back to where its field does not reach. The driver
#: runs with ``strict_patch_coverage`` so a pixel that lands on it raises; it exists because the
#: bridge needs a node per prim, not because any pixel should read it.
FALLBACK_NODE = "patch_fallback"

#: How far inside its patch each quad is drawn, metres, on every edge. The bridge's containment
#: test is exact at the boundary, and a point the renderer reports on a quad's very edge lands a
#: rounding error outside the patch -- measured: one pixel of the neighbour's west face on the
#: first wall render, which strict coverage rightly refused. A millimetre against 0.25-0.5 m
#: cells moves no pixel's temperature and puts every rendered point inside its field.
QUAD_INSET_M = 1e-3

#: Visible-band film: height above the surface it wets, metres. Far above the depth buffer's
#: resolution at these ranges, far below anything the eye resolves.
FILM_LIFT_M = 0.002


def enu_to_stage(v: Any) -> tuple[float, float, float]:
    """One ENU point or direction in stage coordinates."""
    out = ENU_TO_STAGE @ np.asarray(v, dtype=np.float64)
    return (float(out[0]), float(out[1]), float(out[2]))


def _is_enu(config: SceneConfig) -> bool:
    wf = config.scene.world_frame
    return tuple(wf.up) == (0.0, 0.0, 1.0) and tuple(wf.north) == (0.0, 1.0, 0.0)


def to_stage_frame(config: SceneConfig) -> SceneConfig:
    """``config`` with every world coordinate rotated into the Y-up stage frame.

    A config already declaring a non-ENU ``world_frame`` is returned unchanged: it was authored
    in some stage frame already (the car scenes are), and rotating it again would be wrong.
    """
    if not _is_enu(config):
        return config
    spec = config.scene
    thermal = spec.thermal
    if thermal is not None:
        if thermal.objects:
            raise ValueError("to_stage_frame does not rotate solved objects; author them Y-up")
        for srf in thermal.surfaces:
            if getattr(srf, "mesh", None) is not None:
                raise ValueError(f"surface {srf.name!r} is a mesh surface; author it Y-up")
            if srf.patch is not None and srf.patch.frame != "world":
                raise ValueError(f"surface {srf.name!r} is in frame {srf.patch.frame!r}")
        surfaces = [
            srf
            if srf.patch is None
            else srf.model_copy(
                update={
                    "patch": srf.patch.model_copy(
                        update={
                            "origin_m": enu_to_stage(srf.patch.origin_m),
                            "u_axis": enu_to_stage(srf.patch.u_axis),
                            "v_axis": enu_to_stage(srf.patch.v_axis),
                        }
                    )
                }
            )
            for srf in thermal.surfaces
        ]
        occluders = [
            occ.model_copy(
                update={
                    "centre_m": enu_to_stage(occ.centre_m),
                    "u_axis": enu_to_stage(occ.u_axis),
                    "v_axis": enu_to_stage(occ.v_axis),
                }
            )
            for occ in thermal.occluders
        ]
        thermal = thermal.model_copy(update={"surfaces": surfaces, "occluders": occluders})
    targets = [
        t
        if t.plume is None
        else t.model_copy(
            update={
                "plume": t.plume.model_copy(
                    update={
                        "origin_m": enu_to_stage(t.plume.origin_m),
                        "direction": enu_to_stage(t.plume.direction),
                    }
                )
            }
        )
        for t in spec.targets
    ]
    frame = spec.world_frame.model_copy(update={"up": (0.0, 1.0, 0.0), "north": (0.0, 0.0, -1.0)})
    return config.model_copy(
        update={
            "scene": spec.model_copy(
                update={"thermal": thermal, "targets": targets, "world_frame": frame}
            )
        }
    )


def with_fallback_node(config: SceneConfig, t_air_offset_k: float = 0.0) -> SceneConfig:
    """``config`` plus :data:`FALLBACK_NODE`, a per-prim node at the air temperature."""
    from irsim.config.scene import TargetSpec

    spec = config.scene
    if any(t.name == FALLBACK_NODE for t in spec.targets):
        return config
    node = TargetSpec(name=FALLBACK_NODE, solver="airframe", offset_k=float(t_air_offset_k))
    return config.model_copy(
        update={"scene": spec.model_copy(update={"targets": [*spec.targets, node]})}
    )


@dataclass(frozen=True)
class PatchPrim:
    """One quad on the stage: its path, its four corners (stage metres), what it is made of.

    ``surface`` is the scene surface whose field it carries, or ``None`` for a visible-only
    sheet (a water film); ``companion_only`` marks the latter.
    """

    path: str
    corners_m: tuple[tuple[float, float, float], ...]
    material: str
    surface: str | None
    companion_only: bool = False


def _quad(
    origin: Any, u: Any, v: Any, u_len: float, v_len: float, lift: Any = None
) -> tuple[tuple[float, float, float], ...]:
    """Four corners of the ``u_len`` x ``v_len`` rectangle, inset by :data:`QUAD_INSET_M`."""
    ua = np.asarray(u, dtype=np.float64)
    va = np.asarray(v, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64) + QUAD_INSET_M * (ua + va)
    if lift is not None:
        o = o + np.asarray(lift, dtype=np.float64)
    uu = ua * (float(u_len) - 2.0 * QUAD_INSET_M)
    vv = va * (float(v_len) - 2.0 * QUAD_INSET_M)
    pts = (o, o + uu, o + uu + vv, o + vv)
    return tuple((float(p[0]), float(p[1]), float(p[2])) for p in pts)


def patch_prims(config: SceneConfig) -> list[PatchPrim]:
    """The quads a stage-frame config needs: one per patched surface, plus one per water film.

    A ``prim_path`` two surfaces share is split into ``<prim_path>/<surface>``; the driver
    rebinds with :func:`binding_path` so each field lands on its own quad.
    """
    thermal = config.scene.thermal
    if thermal is None:
        return []
    patched = [s for s in thermal.surfaces if s.patch is not None and s.patch.prim_path]
    shared: dict[str, int] = {}
    for srf in patched:
        key = str(srf.patch.prim_path) if srf.patch is not None else ""
        shared[key] = shared.get(key, 0) + 1
    out: list[PatchPrim] = []
    for srf in patched:
        p = srf.patch
        assert p is not None and p.prim_path
        path = binding_path(p.prim_path, srf.name, shared[p.prim_path] > 1)
        out.append(
            PatchPrim(
                path=path,
                corners_m=_quad(p.origin_m, p.u_axis, p.v_axis, p.n_u * p.du_m, p.n_v * p.dv_m),
                material=srf.material,
                surface=srf.name,
            )
        )
        film = getattr(srf, "film", None)
        if film is not None:
            u0, u1, v0, v1 = (
                film.region_m
                if film.region_m is not None
                else (0.0, p.n_u * p.du_m, 0.0, p.n_v * p.dv_m)
            )
            u = np.asarray(p.u_axis, dtype=np.float64)
            v = np.asarray(p.v_axis, dtype=np.float64)
            normal = np.cross(u, v)
            origin = np.asarray(p.origin_m, dtype=np.float64) + u0 * u + v0 * v
            out.append(
                PatchPrim(
                    path=f"/World/Films/{srf.name}",
                    corners_m=_quad(origin, u, v, u1 - u0, v1 - v0, lift=FILM_LIFT_M * normal),
                    material="water_film",
                    surface=None,
                    companion_only=True,
                )
            )
    return out


def binding_path(prim_path: str, surface: str, is_shared: bool) -> str:
    """The prim a surface's field is bound to: its own path, or a child when the path is shared."""
    return f"{prim_path.rstrip('/')}/{surface}" if is_shared else prim_path


def patch_bindings(scene: Any) -> list[tuple[str, Any]]:
    """``(prim path, field)`` per patched surface, on the prims :func:`patch_prims` authors.

    `Scene.surface_bindings` with the shared paths split exactly as :func:`patch_prims` splits
    them, so the two cannot disagree about which quad a field is drawn on.
    """
    counts: dict[str, int] = {}
    for name in scene.surface_fields:
        if name in scene.patch_prims:
            counts[scene.patch_prims[name]] = counts.get(scene.patch_prims[name], 0) + 1
    return [
        (binding_path(scene.patch_prims[name], name, counts[scene.patch_prims[name]] > 1), fld)
        for name, fld in scene.surface_fields.items()
        if name in scene.patch_prims
    ]


def author_patch_prims(stage: Any, prims: list[PatchPrim]) -> dict[str, str]:
    """Write every quad; returns prim path -> :data:`FALLBACK_NODE` for the thermal ones.

    Double-sided, with the winding of the patch's own (u, v), so the geometric normal is the
    patch's outward one. A visible-only film gets a dark glossy look and no thermal material.
    """
    from pxr import Gf, Sdf, UsdGeom

    from irsim_isaac.stage import bind_visible_look

    prim_to_node: dict[str, str] = {}
    for prim in prims:
        parent = prim.path.rsplit("/", 1)[0]
        if parent and not stage.GetPrimAtPath(parent):
            UsdGeom.Xform.Define(stage, parent)
        mesh = UsdGeom.Mesh.Define(stage, prim.path)
        mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in prim.corners_m])
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        mesh.CreateDoubleSidedAttr(True)
        pts = np.asarray(prim.corners_m, dtype=np.float64)
        mesh.CreateExtentAttr([Gf.Vec3f(*pts.min(axis=0)), Gf.Vec3f(*pts.max(axis=0))])
        if not prim.companion_only:
            mesh.GetPrim().CreateAttribute("thermal:material", Sdf.ValueTypeNames.String).Set(
                prim.material
            )
            prim_to_node[prim.path] = FALLBACK_NODE
        bind_visible_look(stage, mesh, prim.material)
    return prim_to_node
