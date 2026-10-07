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
   tree cannot be exported. One material for the whole body (no per-region instances): a
   second material on a segment becomes a second mesh prim under it in USD, named after the
   material, and the renderer then cannot find the segment by its prim's name. Eyes, eyebrows
   and eyelashes from the system packs when present.
3. **A rig.** ``add_builtin_rig(basemesh, <rig>)``; ``game_engine`` by default, the Unreal
   mannequin naming the body schema maps (``configs/humans/body_schema.yaml``, rig
   ``mpfb_game_engine``). The weights are MPFB's own, so every vertex has a bone.
4. **Canonical names.** The body object is ``body``, its material ``skin``; eyes ``eyes``,
   eyebrows and eyelashes ``eyebrows`` / ``eyelashes``, with materials of the same names -- the
   names ``prep_human.py`` and the asset config key on, so nothing downstream reads MPFB's.
5. **Garments** (``--garment slot=asset``): a MakeHuman clothes asset per body-schema slot,
   fitted and rigged to the body by MPFB, its object and material renamed ``garment_<slot>``;
   the slot is the one key that joins the mesh to the asset config's ``garments:`` (HU.5).
6. **Export** a binary glTF with skins, and a sidecar JSON beside it: the measured stature (the
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


#: Bisection steps for the height macro: 2^-14 of the slider is about 0.1 mm of an adult.
_HEIGHT_STEPS = 14


def _solve_height(basemesh, macro: dict, stature_m: float, target_service) -> float:  # type: ignore[no-untyped-def]
    """The height macro that gives ``stature_m``, by bisection on MakeHuman's own body.

    Stature rises monotonically with the height slider at a fixed age and sex (measured: an
    8-year-old girl is 0.969 / 1.085 / 1.272 m at 0.3 / 0.5 / 0.7), so a bracket on [0, 1]
    cannot fail; a stature outside what the slider reaches for this age is refused, not clipped.
    """
    objprops = importlib.import_module(f"{MPFB_PACKAGE}.entities.objectproperties")
    props = objprops.HumanObjectProperties

    def stature_at(h: float) -> float:
        props.set_value("height", h, entity_reference=basemesh)
        target_service.reapply_macro_details(basemesh)
        return _stature_m(basemesh)

    lo, hi = 0.0, 1.0
    s_lo, s_hi = stature_at(lo), stature_at(hi)
    if not s_lo <= stature_m <= s_hi:
        raise SystemExit(
            f"stature {stature_m:.3f} m is outside {s_lo:.3f}-{s_hi:.3f} m, the range the height "
            f"slider reaches at age macro {macro['age']:.3f}"
        )
    for _ in range(_HEIGHT_STEPS):
        mid = 0.5 * (lo + hi)
        if stature_at(mid) < stature_m:
            lo = mid
        else:
            hi = mid
    h = 0.5 * (lo + hi)
    stature_at(h)
    print(f"[make_human] height macro {h:.5f} for {stature_m:.4f} m")
    return h


#: The vest (HU.7), as fractions of stature and metres. A duty vest over soft armour stands 20-30
#: mm off the chest; it runs from the waist to the shoulders. Two 50 mm retroreflective bands
#: (EN ISO 20471 class 2 minimum width) around the torso; a 45 mm duty belt at the waist, 35 mm
#: out so it sits over the pullover's hem and the trousers' waistband.
VEST_BOTTOM, VEST_TOP = 0.578, 0.84
VEST_OFFSET_M = 0.025
TAPE_BANDS = ((0.625, 0.654), (0.672, 0.701))
TAPE_OFFSET_M = 0.0015
BELT_BAND = (0.552, 0.578)
BELT_OFFSET_M = 0.035
#: Game-engine bones that carry the torso a vest is cut from (the arms and neck are excluded).
TORSO_BONES = ("pelvis", "spine_01", "spine_02", "spine_03", "clavicle_l", "clavicle_r")


def _cut_shell(
    body: bpy.types.Object,
    bones: tuple[str, ...],
    z_ranges: list[tuple[float, float]],
    offset_m: float,
    name: str,
    colour: tuple[float, float, float],
) -> bpy.types.Object:
    """A copy of ``body`` keeping the skin faces whose dominant bone is in ``bones`` and that lie in
    one of ``z_ranges`` (metres), pushed ``offset_m`` out along its normals, with the body's vertex
    groups and armature modifier so it deforms with the body. The horizontal edges are cut by exact
    planes (``bisect_plane``), so a band is straight rather than the staircase of whole faces. One
    material of ``name`` with a flat base colour."""
    import bmesh

    shell = body.copy()
    shell.data = body.data.copy()
    for mod in list(shell.modifiers):
        if mod.type != "ARMATURE":
            shell.modifiers.remove(mod)
    bpy.context.collection.objects.link(shell)
    if shell.data.shape_keys is not None:
        # MakeHuman's macros are shape keys: the body one sees is the basis plus their mix, so the
        # shell is cut from the mix, not from the unmorphed base mesh 2-5 cm away
        for o in bpy.context.selected_objects:
            o.select_set(False)
        shell.select_set(True)
        bpy.context.view_layer.objects.active = shell
        bpy.ops.object.shape_key_remove(all=True, apply_mix=True)
    groups = {g.index: g.name for g in shell.vertex_groups}
    rig = next(
        (mod.object for mod in shell.modifiers if mod.type == "ARMATURE" and mod.object), None
    )
    if rig is None:
        raise SystemExit("the vest is cut by bone, so the body must be rigged first")
    rig_bones = {b.name for b in rig.data.bones}
    body_group = shell.vertex_groups.get("body")
    me = shell.data
    dominant = []
    in_body = []
    for v in me.vertices:
        weights = [g for g in v.groups if groups.get(g.group, "") in rig_bones]
        best = max(weights, key=lambda g: g.weight, default=None)
        dominant.append(groups.get(best.group, "") if best is not None else "")
        # MakeHuman's helper geometry (tights, skirt, joint cubes) carries torso weights too;
        # only the faces of the `body` group are skin
        in_body.append(body_group is None or any(g.group == body_group.index for g in v.groups))
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    # cut in world space: once parented to the rig the mesh's own frame is not the stage's
    world = shell.matrix_world.copy()
    bmesh.ops.transform(bm, matrix=world, verts=bm.verts)
    lo_all, hi_all = min(a for a, _ in z_ranges), max(b for _, b in z_ranges)
    gone = []
    for f in bm.faces:
        zs = [v.co.z for v in f.verts]
        votes = [dominant[v.index] for v in f.verts]
        bone = max(sorted(set(votes)), key=votes.count)
        keep = all(in_body[v.index] for v in f.verts) and bone in bones
        if not keep or max(zs) < lo_all or min(zs) > hi_all:
            gone.append(f)
    bmesh.ops.delete(bm, geom=gone, context="FACES")
    # exact horizontal edges: split at every range edge, then drop what lies outside the ranges
    for z in sorted({e for pair in z_ranges for e in pair}):
        geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
        bmesh.ops.bisect_plane(bm, geom=geom, plane_co=(0.0, 0.0, z), plane_no=(0.0, 0.0, 1.0))
    outside = [
        f for f in bm.faces if not any(a <= f.calc_center_median().z <= b for a, b in z_ranges)
    ]
    bmesh.ops.delete(bm, geom=outside, context="FACES")
    bm.normal_update()
    for v in bm.verts:
        v.co += v.normal * offset_m
    bmesh.ops.transform(bm, matrix=world.inverted(), verts=bm.verts)
    bm.to_mesh(me)
    bm.free()
    me.materials.clear()
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    me.materials.append(mat)
    _rename(shell, name)
    return shell


def _vest_tape_and_belt(body: bpy.types.Object, stature_m: float) -> list[str]:
    """The procedural vest (garment_torso), its tape (equipment_tape) and a belt (equipment_belt).

    MakeHuman's CC0 packs have no hi-vis vest, so it is cut from the body itself: the torso faces
    between the waist and the shoulders, by their dominant bone, offset outward. Deterministic,
    CC0 like the body it comes from, and rigged by construction.
    """
    h = stature_m
    belt_bones = ("pelvis", "spine_01")
    vest = [(VEST_BOTTOM * h, VEST_TOP * h)]
    tape = [(a * h, b * h) for a, b in TAPE_BANDS]
    belt = [(BELT_BAND[0] * h, BELT_BAND[1] * h)]
    _cut_shell(body, TORSO_BONES, vest, VEST_OFFSET_M, "garment_torso", (1.0, 1.0, 1.0))
    _cut_shell(
        body, TORSO_BONES, tape, VEST_OFFSET_M + TAPE_OFFSET_M, "equipment_tape", (0.72, 0.72, 0.72)
    )
    _cut_shell(body, belt_bones, belt, BELT_OFFSET_M, "equipment_belt", (0.02, 0.02, 0.02))
    return ["belt", "tape"]


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
    ap.add_argument(
        "--stature-m",
        type=float,
        default=None,
        help="solve the height macro for this standing height, metres (e.g. a WHO median); "
        "overrides --height",
    )
    ap.add_argument("--weight", type=float, default=0.5, help="weight macro, 0..1")
    ap.add_argument("--muscle", type=float, default=0.5)
    ap.add_argument("--proportions", type=float, default=0.5)
    ap.add_argument("--skin", default=None, help="a skin's name under MPFB's user data skins/")
    ap.add_argument("--rig", default="game_engine", help="an MPFB built-in rig")
    ap.add_argument("--no-eyes", action="store_true")
    ap.add_argument(
        "--vest",
        action="store_true",
        help="a hi-vis vest cut from the body's own torso (garment_torso, 25 mm out: armour and "
        "vest), two retroreflective bands on it (equipment_tape) and a duty belt at the waist "
        "(equipment_belt), all rigged with the body's weights (HU.7)",
    )
    ap.add_argument(
        "--garment",
        action="append",
        default=[],
        metavar="SLOT=ASSET",
        help="dress a body-schema slot with a MakeHuman clothes asset (user data clothes/<ASSET>), "
        "e.g. torso=toigo_basic_tucked_t-shirt legs=toigo_wool_pants feet=shoes01; the object "
        "and its material are named garment_<SLOT> (HU.5)",
    )
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

    if a.stature_m is not None:
        macro["height"] = _solve_height(basemesh, macro, float(a.stature_m), target_service)

    skin_file = _find_asset(location_service, "skins", a.skin, (".mhmat",))
    if skin_file is None:
        raise SystemExit(
            f"no skin {a.skin!r} under {location_service.get_user_data('skins')}: unpack a CC0 "
            "skins pack (skins01 female, skins02 male) into MPFB's user data"
        )
    human_service.set_character_skin(
        skin_file, basemesh, skin_type="MAKESKIN", material_instances=False
    )

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

    garments: dict[str, str] = {}
    for item in a.garment:
        slot, _, asset_name = item.partition("=")
        if not slot or not asset_name:
            raise SystemExit(f"--garment takes SLOT=ASSET, got {item!r}")
        mhclo = _find_asset(location_service, "clothes", asset_name, (".mhclo",))
        if mhclo is None:
            raise SystemExit(
                f"no clothes asset {asset_name!r} under {location_service.get_user_data('clothes')}"
            )
        before = set(bpy.data.objects.keys())
        human_service.add_mhclo_asset(
            mhclo, basemesh, asset_type="Clothes", subdiv_levels=0, material_type="MAKESKIN"
        )
        fresh = [o for o in bpy.data.objects if o.name not in before]
        meshes = [o for o in fresh if o.type == "MESH"]
        if len(meshes) != 1:
            raise SystemExit(f"garment {asset_name!r} added {len(meshes)} meshes, expected one")
        piece = meshes[0]
        if not any(s.material for s in piece.material_slots):
            # some CC0 garments (the newsboy cap) ship no material; give one, named for the slot,
            # so a colour can be dyed into it and the asset map can bind it
            mat = bpy.data.materials.new(f"garment_{slot}")
            mat.use_nodes = True
            piece.data.materials.clear()
            piece.data.materials.append(mat)
        _rename(piece, f"garment_{slot}")
        garments[slot] = asset_name

    # MPFB masks the body under a garment that declares a delete group (shoes hide the feet),
    # which is right for a game and wrong here: the body schema wants every segment present, and
    # the garment prim is what hides the skin from the camera. Keep only the helper mask.
    for mod in list(basemesh.modifiers):
        if mod.type == "MASK" and mod.name != "Hide helpers":
            basemesh.modifiers.remove(mod)

    rig = human_service.add_builtin_rig(basemesh, a.rig)
    if rig is None:
        raise SystemExit(f"MPFB has no built-in rig {a.rig!r}")

    stature = _stature_m(basemesh)
    equipment: list[str] = []
    if a.vest:
        if "torso" in garments:
            raise SystemExit("--vest is the torso garment; do not also pass --garment torso=...")
        equipment = _vest_tape_and_belt(basemesh, stature)
        garments["torso"] = "irsim_hi_vis_vest"  # generated, not a MakeHuman asset
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
        "garments": garments,
        "equipment": equipment,
        "objects": sorted(o.name for o in bpy.data.objects),
    }
    a.out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"[make_human] wrote {a.out} ({stature:.3f} m tall) and {a.out.with_suffix('.json')}")


if __name__ == "__main__":
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    main(argv)
