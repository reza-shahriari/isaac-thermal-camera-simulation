#!/usr/bin/env python3
"""Write the Mixamo skeleton templates Make-It-Animatable reads, without the gated dataset (HU.9).

    ~/miniconda3/envs/mia/bin/python scripts/mixamo_template.py \\
        --mia-root ~/tools/Make-It-Animatable

Make-It-Animatable builds its kinematic tree at import from ``data/Mixamo/bones.fbx`` (and, for
its optional ears-and-tail model, ``bones_vroid.fbx``), which its README fetches from the
Hugging Face dataset ``jasongzy/Mixamo`` -- a dataset that requires the owner's approval. For
inference the tool reads **only bone names and parent links** from that file
(``util/dataset_mixamo.py: build_skeleton``); the rest pose matters only to its own FBX export,
which ``autorig_mia.py`` never calls. So this writes armatures carrying exactly the tool's bone
list (``MIXAMO_JOINTS``) in the standard Mixamo hierarchy -- Hips at the root, the spine to the
head, shoulders off Spine2, three-bone fingers -- at plausible T-pose positions, with Blender's
``bpy`` module from the tool's own environment.

A skeleton's hierarchy is a fact about a bone-naming convention, not Adobe's character data; no
Mixamo asset is copied.

docs/physics-model.md §6.1; roadmap HU.9; ADR 0201.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

P = "mixamorig:"
#: (bone, parent, head xyz in metres, Blender Z-up, character facing -Y, its left at +X)
_SPINE = [
    ("Hips", None, (0.0, 0.0, 1.00)),
    ("Spine", "Hips", (0.0, 0.0, 1.10)),
    ("Spine1", "Spine", (0.0, 0.0, 1.22)),
    ("Spine2", "Spine1", (0.0, 0.0, 1.34)),
    ("Neck", "Spine2", (0.0, 0.0, 1.50)),
    ("Head", "Neck", (0.0, 0.0, 1.60)),
]
_FINGERS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


def _side(sign: float, side: str) -> list[tuple[str, str | None, tuple[float, float, float]]]:
    s = sign
    bones = [
        (f"{side}Shoulder", "Spine2", (0.05 * s, 0.0, 1.45)),
        (f"{side}Arm", f"{side}Shoulder", (0.18 * s, 0.0, 1.45)),
        (f"{side}ForeArm", f"{side}Arm", (0.45 * s, 0.0, 1.45)),
        (f"{side}Hand", f"{side}ForeArm", (0.70 * s, 0.0, 1.45)),
    ]
    for k, finger in enumerate(_FINGERS):
        parent = f"{side}Hand"
        for n in (1, 2, 3):
            name = f"{side}Hand{finger}{n}"
            bones.append((name, parent, ((0.76 + 0.03 * n) * s, -0.03 + 0.015 * k, 1.45)))
            parent = name
    bones += [
        (f"{side}UpLeg", "Hips", (0.09 * s, 0.0, 0.95)),
        (f"{side}Leg", f"{side}UpLeg", (0.09 * s, 0.0, 0.52)),
        (f"{side}Foot", f"{side}Leg", (0.09 * s, 0.0, 0.08)),
        (f"{side}ToeBase", f"{side}Foot", (0.09 * s, -0.12, 0.02)),
    ]
    return bones


def skeleton(extra: bool = False) -> list[tuple[str, str | None, tuple[float, float, float]]]:
    bones = _SPINE + _side(1.0, "Left") + _side(-1.0, "Right")
    if extra:  # the tool's ears-and-tail joints (dataset_mixamo_additional.ADDITIONAL_JOINTS)
        bones += [
            ("LRabbitEar2", "Head", (0.05, 0.0, 1.85)),
            ("RRabbitEar2", "Head", (-0.05, 0.0, 1.85)),
            ("FoxTail1", "Hips", (0.0, 0.10, 0.95)),
        ]
        bones += [
            (f"FoxTail{i}", f"FoxTail{i - 1}", (0.0, 0.10 + 0.08 * i, 0.95)) for i in range(2, 6)
        ]
    return bones


def _write(
    path: pathlib.Path, bones: list[tuple[str, str | None, tuple[float, float, float]]]
) -> None:
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    arm = bpy.data.armatures.new("Armature")
    obj = bpy.data.objects.new("Armature", arm)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    made = {}
    children = {b: [c for c, p, _ in bones if p == b] for b, _, _ in bones}
    pos = {b: h for b, _, h in bones}
    for name, parent, head in bones:
        eb = arm.edit_bones.new(P + name)
        eb.head = head
        kids = children[name]
        if kids:  # point at the first child; a leaf gets a short stub along its parent
            eb.tail = pos[kids[0]]
        else:
            ph = pos[parent] if parent else (head[0], head[1], head[2] - 0.1)
            d = [head[k] - ph[k] for k in range(3)]
            n = max(sum(x * x for x in d) ** 0.5, 1e-6)
            eb.tail = tuple(head[k] + 0.05 * d[k] / n for k in range(3))
        if parent is not None:
            eb.parent = made[parent]
        made[name] = eb
    bpy.ops.object.mode_set(mode="OBJECT")
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(filepath=str(path), add_leaf_bones=False, object_types={"ARMATURE"})
    print(f"[mixamo_template] {len(bones)} bones -> {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mia-root", type=pathlib.Path, required=True)
    a = ap.parse_args(argv)
    out = a.mia_root.expanduser() / "data" / "Mixamo"
    _write(out / "bones.fbx", skeleton())
    _write(out / "bones_vroid.fbx", skeleton(extra=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
