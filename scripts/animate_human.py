#!/usr/bin/env python3
"""A walking body for HU.10's probe: key a gait on the rig, export a skinned USD and the answer.

    blender -b --python scripts/animate_human.py -- \\
        --source outputs/humans/man_adult/man_adult_rigged.glb \\
        --out outputs/humans/man_adult/walk/man_adult_walk.usdc

Runs **inside Blender**. The source is ``prep_human.py --keep-rig`` output: the seventeen
``skin_<Segment>`` objects, still skinned to MPFB's game-engine rig. What it does:

1. **A walk cycle, procedural.** :func:`gait_angles` gives hip, knee, shoulder and elbow flexion
   over one stride as smooth periodic curves with the ranges of a normal adult walk (hip
   -10..+25 degrees, knee 5..60 with the swing peak at 72 % of the cycle, arms swinging +/-15
   against the legs; Winter 2009, *Biomechanics and Motor Control of Human Movement*, ch. 3,
   normative gait). The body walks on the spot: the probe asks whether the renderer's ids and
   positions follow
   a deforming surface, and a treadmill walk deforms it as much as a travelling one. A captured
   clip (CMU, Mixamo) is the realism upgrade and is never committed (owner, HU.1: animations are
   not shipped).
2. **Keyed about the body's lateral axis.** Each joint rotates about world X in the rig's rest
   frame (the body faces -Y, so flexion carries a foot or hand toward -Y). In a chain whose joints
   all turn about one axis the local keys compose exactly, so a thigh and its calf need no IK.
3. **The answer, from Blender.** For every frame, each skin segment's evaluated (skinned) mesh:
   its world vertices and triangles (``<out>.vertices.npz``) and its area-weighted centroid
   (``<out>.expected.json``). The renderer is checked against these, never against itself.
4. **A skinned USD**: ``wm.usd_export`` with armatures and animation -- one ``Skeleton``, each
   segment mesh bound to it with its own weights, the joint rotations time-sampled.

Nothing here touches temperature: a segment's temperature is keyed by its prim, and the probe's
question is whether the prim's pixels go where the prim goes.

docs/physics-model.md §6.1, §13.3; roadmap HU.10; ADR 0199.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

#: One stride, seconds (cadence about 109 steps a minute, a normal adult walk).
STRIDE_S = 1.1
FPS = 30
STRIDES = 2


def _bump(phase: float, centre: float, width: float) -> float:
    """A periodic Gaussian bump on the unit cycle."""
    d = (phase - centre + 0.5) % 1.0 - 0.5
    return math.exp(-0.5 * (d / width) ** 2)


def gait_angles(phase: float) -> dict[str, float]:
    """Flexion in degrees for the left side at ``phase`` (0 = left heel strike, 1 = the next).

    Positive is flexion: the hip carries the thigh forward, the knee folds the shank back, the
    shoulder carries the arm forward, the elbow bends. The right side is the same curve half a
    cycle later; the arms swing against the legs of their own side.
    """
    hip = 7.5 + 17.5 * math.cos(2.0 * math.pi * phase)
    knee = 5.0 + 12.0 * _bump(phase, 0.15, 0.06) + 55.0 * _bump(phase, 0.72, 0.10)
    shoulder = -15.0 * math.cos(2.0 * math.pi * phase)
    elbow = 15.0 + 10.0 * (1.0 - math.cos(2.0 * math.pi * phase)) / 2.0
    return {"hip": hip, "knee": knee, "shoulder": shoulder, "elbow": elbow}


#: Game-engine bones each joint turns, and the sign that makes positive flexion carry the limb
#: toward -Y (forward) about world +X: R_x(theta) carries a downward limb (0, 0, -1) to
#: (0, sin theta, -cos theta), i.e. backward, so forward flexion is a negative rotation.
JOINTS: dict[str, tuple[str, float]] = {
    "thigh": ("hip", -1.0),
    "calf": ("knee", +1.0),
    "upperarm": ("shoulder", -1.0),
    "lowerarm": ("elbow", -1.0),
}


def _animate(rig, frames: int) -> None:  # type: ignore[no-untyped-def]
    from mathutils import Matrix, Vector

    axis = rig.matrix_world.inverted().to_3x3() @ Vector((1.0, 0.0, 0.0))
    axis.normalize()
    for frame in range(frames + 1):
        t = frame / FPS
        for side, offset in (("l", 0.0), ("r", 0.5)):
            phase = (t / STRIDE_S + offset) % 1.0
            angles = gait_angles(phase)
            for bone_stem, (joint, sign) in JOINTS.items():
                name = f"{bone_stem}_{side}"
                pb = rig.pose.bones.get(name)
                if pb is None:
                    raise SystemExit(f"the rig has no bone {name!r}")
                rest = pb.bone.matrix_local.to_3x3()
                world_rot = Matrix.Rotation(math.radians(sign * angles[joint]), 3, axis)
                pb.rotation_mode = "QUATERNION"
                pb.rotation_quaternion = (rest.inverted() @ world_rot @ rest).to_quaternion()
                pb.keyframe_insert("rotation_quaternion", frame=frame + 1)


def _centroids(
    frames: int, vertices: dict[str, list] | None = None
) -> dict[str, list[list[float]]]:
    """Each segment's area-weighted world centroid per frame; world vertices into ``vertices``."""
    import bpy

    out: dict[str, list[list[float]]] = {}
    segs = [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("skin_")]
    for frame in range(frames + 1):
        bpy.context.scene.frame_set(frame + 1)
        dg = bpy.context.evaluated_depsgraph_get()
        for obj in segs:
            ev = obj.evaluated_get(dg)
            me = ev.to_mesh()
            mw = obj.matrix_world
            area = 0.0
            acc = [0.0, 0.0, 0.0]
            for poly in me.polygons:
                c = mw @ poly.center
                a = poly.area
                area += a
                for k in range(3):
                    acc[k] += a * c[k]
            if vertices is not None:
                vertices.setdefault(obj.name, []).append([tuple(mw @ v.co) for v in me.vertices])
                if frame == 0:  # the topology is the same every frame
                    me.calc_loop_triangles()
                    vertices[f"{obj.name}.faces"] = [
                        tuple(tri.vertices) for tri in me.loop_triangles
                    ]
            ev.to_mesh_clear()
            out.setdefault(obj.name, []).append([round(v / area, 6) for v in acc])
    return out


def main(argv: list[str]) -> None:
    import bpy

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=pathlib.Path, required=True, help="prep_human --keep-rig glb")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="the skinned .usdc to write")
    a = ap.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(a.source))
    rigs = [o for o in bpy.data.objects if o.type == "ARMATURE"]
    if len(rigs) != 1:
        raise SystemExit(f"expected one rig in {a.source}, found {len(rigs)} (was --keep-rig set?)")
    rig = rigs[0]
    frames = round(STRIDES * STRIDE_S * FPS)
    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start, scene.frame_end = 1, frames + 1
    _animate(rig, frames)
    vertices: dict[str, list] = {}
    expected = _centroids(frames, vertices)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.usd_export(
        filepath=str(a.out),
        export_animation=True,
        export_armatures=True,
        only_deform_bones=True,
        export_shapekeys=False,
        export_materials=True,
        evaluation_mode="RENDER",
    )
    record = {
        "source": str(a.source),
        "usd": str(a.out),
        "fps": FPS,
        "stride_s": STRIDE_S,
        "frame_start": 1,
        "frames": frames + 1,
        "centroids_m": expected,
    }
    a.out.with_suffix(".expected.json").write_text(json.dumps(record) + "\n")
    # the skinned surface itself, per frame: the probe puts every rendered pixel on it
    import numpy as np

    np.savez_compressed(
        a.out.with_suffix(".vertices.npz"),
        **{
            name: np.asarray(v, dtype=np.int32 if name.endswith(".faces") else np.float32)
            for name, v in vertices.items()
        },
    )
    print(f"[animate_human] {frames + 1} frames, {len(expected)} segments -> {a.out}")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
