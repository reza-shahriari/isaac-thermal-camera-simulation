#!/usr/bin/env python3
"""Generate a human from MakeHuman's MPFB2 in headless Blender, as a rigged glTF (HU.3).

    blender -b --python scripts/make_human.py -- --name man_adult --gender 1.0 --age 0.58 \\
        --skin mindfront_aksel_skin --out outputs/humans/man_adult/man_adult.glb

Runs **inside Blender** (the ``bpy`` interpreter), with the MPFB extension installed from
extensions.blender.org (``blender --command extension install mpfb -s -e``) and the CC0 asset
packs it needs unpacked into MPFB's user data directory (skins02, makehuman_system_assets, …;
``static.makehumancommunity.org/assets/assetpacks.html``). Everything here is CPU work.

What it does, in order:

1. **A body from the macro sliders.** ``HumanService.create_human`` with MakeHuman's eleven
   macros -- gender (0 female … 1 male), age (0.0 baby, 0.1875 child, 0.5 young adult, 1.0 old),
   height, weight, muscle, proportions -- at ``scale=0.1`` so the decimetre base mesh lands in
   metres, feet on the ground. The helper geometry (joints, skirt and tights helpers) is masked
   and the export applies the mask, so what leaves is the body alone.
2. **Skin and body parts.** A skin from a named CC0 ``.mhmat`` through the MAKESKIN node tree --
   a Principled BSDF with the diffuse, normal and specular maps, which glTF can carry; the SSS
   tree cannot be exported. Eyes, eyebrows and eyelashes from the system packs when present.
3. **A rig.** ``add_builtin_rig(basemesh, <rig>)``; ``game_engine`` by default, the Unreal
   mannequin naming the body schema maps (``configs/humans/body_schema.yaml``, rig
   ``mpfb_game_engine``). The weights are MPFB's own, so every vertex has a bone.
4. **Canonical names.** The body object is ``body``, its material ``skin``; eyes ``eyes``,
   eyebrows and eyelashes ``eyebrows`` / ``eyelashes``, with materials of the same names -- the
   names ``prep_human.py`` and the asset config key on, so nothing downstream reads MPFB's.
5. **Export** a binary glTF with skins, and a sidecar JSON beside it: the measured stature (the
   body's vertical extent, metres), the macro values, the skin, the rig and the MPFB build -- the
   phenotype record the asset config is authored from. Mass is **not** measured here: the asset
   config estimates it from stature and a stated BMI, flagged ESTIMATED.

docs/physics-model.md §6.1, §16.2; roadmap HU.3; ADR 0192.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys

import bpy

MPFB_PACKAGE = "bl_ext.blender_org.mpfb"

#: MakeHuman's age macro at a few ages (MPFB macro docs: 0.0 baby, 0.1875 child, 0.5 young adult,
#: 1.0 old; MakeHuman's own mapping is 1 y → 0, 11 y → 0.1875, 25 y → 0.5, 90 y → 1.0).
AGE_ANCHORS_Y = ((0.0, 1.0), (0.1875, 11.0), (0.5, 25.0), (1.0, 90.0))


def age_macro_to_years(a: float) -> float:
    """Piecewise-linear inverse of MakeHuman's age slider, years."""
    for (m0, y0), (m1, y1) in zip(AGE_ANCHORS_Y, AGE_ANCHORS_Y[1:], strict=False):
        if a <= m1:
            return y0 + (y1 - y0) * (a - m0) / (m1 - m0)
    return 90.0


def years_to_age_macro(years: float) -> float:
    for (m0, y0), (m1, y1) in zip(AGE_ANCHORS_Y, AGE_ANCHORS_Y[1:], strict=False):
        if years <= y1:
            return m0 + (m1 - m0) * (years - y0) / (y1 - y0)
    return 1.0


def _mpfb(module: str):  # type: ignore[no-untyped-def]
    if MPFB_PACKAGE not in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_enable(module=MPFB_PACKAGE)
    return importlib.import_module(f"{MPFB_PACKAGE}.{module}")


def _clear_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.armatures, bpy.data.images):
        for block in list(coll):
            if block.users == 0:
                coll.remove(block)


def _find_asset(location_service, kind: str, name: str | None, exts: tuple[str, ...]) -> str | None:  # type: ignore[no-untyped-def]
    """``<user data>/<kind>/<name>/<name>.<ext>``, or the first entry of that kind when ``name`` is
    None, or None when the pack is not installed."""
    root = pathlib.Path(location_service.get_user_data(kind))
    if not root.is_dir():
        return None
    candidates = sorted(root.iterdir()) if name is None else [root / name]
    for d in candidates:
        for ext in exts:
            f = d / f"{d.name}{ext}"
            if f.is_file():
                return str(f)
    return None


def _stature_m(obj: bpy.types.Object) -> float:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(depsgraph)
    mesh = ev.to_mesh()
    try:
        zs = [(obj.matrix_world @ v.co).z for v in mesh.vertices]
    finally:
        ev.to_mesh_clear()
    return max(zs) - min(zs)


def _rename(obj: bpy.types.Object, name: str) -> None:
    obj.name = name
    if obj.data is not None:
        obj.data.name = name
    for slot in obj.material_slots:
        if slot.material is not None:
            slot.material.name = name


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True, help="the asset's name, e.g. man_adult")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="the .glb to write")
    ap.add_argument("--gender", type=float, default=0.5, help="0 female … 1 male")
    ap.add_argument("--age", type=float, default=None, help="MakeHuman age macro, 0..1")
    ap.add_argument("--age-years", type=float, default=None, help="age in years (sets --age)")
    ap.add_argument("--height", type=float, default=0.5, help="height macro, 0..1")
    ap.add_argument("--weight", type=float, default=0.5, help="weight macro, 0..1")
    ap.add_argument("--muscle", type=float, default=0.5)
    ap.add_argument("--proportions", type=float, default=0.5)
    ap.add_argument("--skin", default=None, help="a skin's name under MPFB's user data skins/")
    ap.add_argument("--rig", default="game_engine", help="an MPFB built-in rig")
    ap.add_argument("--no-eyes", action="store_true")
    a = ap.parse_args(argv)
    if a.age is None:
        a.age = years_to_age_macro(a.age_years) if a.age_years is not None else 0.5

    human_service = _mpfb("services.humanservice").HumanService
    target_service = _mpfb("services.targetservice").TargetService
    location_service = _mpfb("services.locationservice").LocationService
    object_service = _mpfb("services.objectservice").ObjectService

    _clear_scene()
    macro = target_service.get_default_macro_info_dict()
    macro.update(
        gender=a.gender,
        age=a.age,
        height=a.height,
        weight=a.weight,
        muscle=a.muscle,
        proportions=a.proportions,
    )
    basemesh = human_service.create_human(
        mask_helpers=True,
        detailed_helpers=True,
        extra_vertex_groups=True,
        feet_on_ground=True,
        scale=0.1,
        macro_detail_dict=macro,
    )

    skin_file = _find_asset(location_service, "skins", a.skin, (".mhmat",))
    if skin_file is None:
        raise SystemExit(
            f"no skin {a.skin!r} under {location_service.get_user_data('skins')}: unpack a CC0 "
            "skins pack (skins01 female, skins02 male) into MPFB's user data"
        )
    human_service.set_character_skin(skin_file, basemesh, skin_type="MAKESKIN")

    added: dict[str, str] = {}
    if not a.no_eyes:
        for kind, asset_type in (
            ("eyes", "Eyes"),
            ("eyebrows", "Eyebrows"),
            ("eyelashes", "Eyelashes"),
        ):
            mhclo = _find_asset(location_service, kind, None, (".mhclo",))
            if mhclo is None:
                print(f"[make_human] no {kind} pack installed; skipping")
                continue
            human_service.add_mhclo_asset(
                mhclo, basemesh, asset_type=asset_type, subdiv_levels=0, material_type="MAKESKIN"
            )
            added[kind] = pathlib.Path(mhclo).stem

    rig = human_service.add_builtin_rig(basemesh, a.rig)
    if rig is None:
        raise SystemExit(f"MPFB has no built-in rig {a.rig!r}")

    stature = _stature_m(basemesh)
    _rename(basemesh, "body")
    for slot in basemesh.material_slots:
        if slot.material is not None:
            slot.material.name = "skin"
    for kind, asset_type in (
        ("eyes", "Eyes"),
        ("eyebrows", "Eyebrows"),
        ("eyelashes", "Eyelashes"),
    ):
        part = object_service.find_object_of_type_amongst_nearest_relatives(basemesh, asset_type)
        if part is not None:
            _rename(part, kind)
    rig.name = "rig"

    a.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(a.out),
        export_format="GLB",
        use_selection=False,
        export_apply=True,
        export_skins=True,
        export_yup=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_extras=True,
    )
    build = getattr(bpy.context.preferences.addons[MPFB_PACKAGE].module, "__version__", None)
    sidecar = {
        "name": a.name,
        "generator": "MakeHuman MPFB2 in Blender",
        "blender": bpy.app.version_string,
        "mpfb_package": MPFB_PACKAGE,
        "mpfb_version": build,
        "macro": {k: v for k, v in macro.items() if k != "race"},
        "race": macro["race"],
        "age_years": round(age_macro_to_years(a.age), 1),
        "sex": "male" if a.gender >= 0.5 else "female",
        "stature_m": round(stature, 4),
        "skin": pathlib.Path(skin_file).stem,
        "rig": a.rig,
        "body_parts": added,
        "objects": sorted(o.name for o in bpy.data.objects),
    }
    a.out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"[make_human] wrote {a.out} ({stature:.3f} m tall) and {a.out.with_suffix('.json')}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    main(argv)
