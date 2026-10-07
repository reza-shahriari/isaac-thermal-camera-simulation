#!/usr/bin/env python3
"""Label a rigged human mesh onto the body schema and split it into one object per segment (HU.3).

    python scripts/prep_human.py --source outputs/humans/man_adult/man_adult.glb \\
        --name man_adult --rig mpfb_game_engine --sidecar outputs/humans/man_adult/man_adult.json \\
        --out outputs/humans/man_adult/man_adult_labelled.glb --write-config

The human half of the asset prep path. An aircraft's parts are discovered by connected component
(ADR 0138); a human's are known in advance, and the skeleton says which is which (ADR 0192). So
this does, for a body mesh skinned to a known rig:

1. **A segment per vertex** -- the deforming bone with the largest weight, looked up in the body
   schema's map for the rig (``configs/humans/body_schema.yaml``), so a vertex owned by
   ``lowerarm_l`` is ``LArm``. Non-deforming and control bones are ignored, as the schema says.
2. **A segment per face** -- the vertex labels' weighted vote. A face that lands on the ``torso``
   region is resolved to ``Chest`` or ``Back`` by where its centre sits relative to the torso's
   centre along the body's forward axis (``--forward-axis``, the asset's ``human.forward_axis``).
3. **A sanity check on the forward axis** from the feet: a foot's centre lies forward of the shin
   above it (toes point forward), so the declared axis must agree with that displacement or the
   run stops -- a body labelled with Chest and Back swapped renders a plausible person.
4. **One object per segment**, named ``skin_<Segment>`` (an underscore, because a USD prim name
   cannot carry a dot and the renderer finds a segment by its prim's name). Eyes, eyebrows,
   eyelashes and hair objects are kept whole and renamed to their layer; garments and equipment
   are kept whole. The result is **static**: the rest pose, with no skeleton -- unless
   ``--keep-rig``, which keeps the skin segments skinned to the rig for HU.10's moving body.
   These names are what the asset config's ``parts:``
   block selects by (``granularity: object``, AI.16/AI.18), so ``prep_asset.py`` needs nothing
   new.
5. **Export** the labelled glTF, a report JSON (faces and area per segment, left/right ratios,
   stature, the forward check) and, with ``--write-config``, the asset YAML skeleton with
   ``kind: human``, the phenotype from the generator's sidecar, the material map and the parts.

**The file runs twice**, as ``prep_asset.py`` does: as the *driver* under the project interpreter
(pydantic, PyYAML, the body schema) it writes the rig's map to JSON and re-invokes itself inside
Blender; with ``bpy`` importable it is the *worker* and does the geometry. CPU only.

docs/physics-model.md §6.1, §16.2; roadmap HU.3; ADR 0192; ADR 0138 (what this is not).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Objects that are not the body, and the layer each becomes.
LAYER_BY_OBJECT = {"eyes": "eyes", "eyebrows": "hair", "eyelashes": "hair", "hair": "hair"}
AXES = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
}


# ---------------------------------------------------------------------------------------------
# worker (inside Blender)
# ---------------------------------------------------------------------------------------------


def _worker(argv: list[str]) -> None:
    import bpy

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=pathlib.Path, required=True)
    ap.add_argument("--rig-json", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--report", type=pathlib.Path, required=True)
    ap.add_argument("--forward-axis", default="-y")
    ap.add_argument("--body-object", default="body")
    ap.add_argument("--garments-json", type=pathlib.Path, default=None)
    ap.add_argument("--keep-rig", action="store_true")
    a = ap.parse_args(argv)
    spec = json.loads(a.rig_json.read_text())
    garments = json.loads(a.garments_json.read_text()) if a.garments_json else {}

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.import_scene.gltf(filepath=str(a.source))
    # the importer's bone display shape (an icosphere in `glTF_not_exported`) is not geometry
    shapes = bpy.data.collections.get("glTF_not_exported")
    if shapes is not None:
        for obj in list(shapes.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(shapes)

    body = bpy.data.objects.get(a.body_object)
    if body is None or body.type != "MESH":
        # glTF's importer splits a skinned node that has children into an empty of the node's
        # name and a mesh `<name>.001`; the mesh is the body, whatever the importer called it
        import re

        meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.vertex_groups]
        named = [o for o in meshes if re.sub(r"\.\d{3}$", "", o.name) == a.body_object]
        if len(named) == 1:
            if body is not None:
                body.name = f"{a.body_object}_node"
            body = named[0]
            body.name = a.body_object
        elif len(meshes) == 1:
            body = meshes[0]
        else:
            raise SystemExit(
                f"no object {a.body_object!r}; skinned meshes are {[m.name for m in meshes]} -- "
                "name the body with --body-object"
            )

    # --- 1-3. the labelling arithmetic is irsim.io.human_labels (NumPy only, tested without
    # Blender); this worker only gathers the arrays and acts on the answer.
    import importlib.util

    import numpy as np

    spec_hl = importlib.util.spec_from_file_location(
        "human_labels", REPO_ROOT / "src" / "irsim" / "io" / "human_labels.py"
    )
    assert spec_hl and spec_hl.loader
    hl = importlib.util.module_from_spec(spec_hl)
    spec_hl.loader.exec_module(hl)
    face_labels, forward_check = hl.face_labels, hl.forward_check
    split_region, vertex_labels = hl.split_region, hl.vertex_labels

    def target_of(group: str) -> str | None:
        name = group
        for p in spec["strip_prefixes"]:
            if name.startswith(p):
                name = name[len(p) :]
                break
        if any(group.startswith(p) for p in spec["ignore_prefixes"]):
            return None
        base = name.rsplit(".", 1)[0] if name[-4:-3] == "." and name[-3:].isdigit() else name
        for cand in (name, base):
            if cand in spec["non_deforming"]:
                return None
            if cand in spec["bones"]:
                return spec["bones"][cand]
        import fnmatch

        for pat, tgt in spec["patterns"].items():
            if fnmatch.fnmatchcase(name, pat):
                return tgt
        raise SystemExit(f"vertex group {group!r} is not a bone the rig map knows")

    labels = sorted(set(spec["segments"]) | set(spec["regions"]))
    index_of = {lab: i for i, lab in enumerate(labels)}
    mesh = body.data
    groups = list(body.vertex_groups)
    group_label = np.array(
        [index_of[t] if (t := target_of(g.name)) is not None else -1 for g in groups],
        dtype=np.int64,
    )
    weights = np.zeros((len(mesh.vertices), len(groups)), dtype=np.float64)
    for v in mesh.vertices:
        for g in v.groups:
            weights[v.index, g.group] = g.weight
    vertex_label = vertex_labels(weights, group_label)
    mw = body.matrix_world
    face_centre = np.array([tuple(mw @ p.center) for p in mesh.polygons], dtype=np.float64)
    faces = [list(p.vertices) for p in mesh.polygons]
    unlabelled = int(sum(1 for f in faces if all(vertex_label[i] < 0 for i in f)))
    face_idx = face_labels(faces, vertex_label, face_centre)
    forward = np.array(AXES[a.forward_axis], dtype=np.float64)
    for region, split in spec["regions"].items():
        face_idx = split_region(
            face_idx,
            face_centre,
            index_of[region],
            index_of[split["positive"]],
            index_of[split["negative"]],
            forward,
        )
    pairs = [(index_of[f], index_of[s_]) for f, s_ in (("LFoot", "LLeg"), ("RFoot", "RLeg"))]
    check = forward_check(face_idx, face_centre, pairs, forward)
    forward_check_m = {labels[k]: round(v, 4) for k, v in check.items()}
    if forward_check_m and not all(v > 0.0 for v in forward_check_m.values()):
        raise SystemExit(
            f"the feet point against --forward-axis {a.forward_axis}: foot − shin along it is "
            f"{forward_check_m}; the body faces the other way"
        )
    face_label = [labels[i] for i in face_idx]
    present = sorted(set(face_label))

    # --- 4. one object per segment ---------------------------------------------------------------
    attr = mesh.attributes.new("irsim_segment", "INT", "FACE")
    present_index = {lab: i for i, lab in enumerate(present)}
    attr.data.foreach_set("value", [present_index[lab] for lab in face_label])

    bpy.ops.object.select_all(action="DESELECT")
    body.select_set(True)
    bpy.context.view_layer.objects.active = body
    made: dict[str, str] = {}
    remaining = body
    for li, label in enumerate(present[:-1]):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="DESELECT")
        bpy.ops.object.mode_set(mode="OBJECT")
        seg = remaining.data.attributes["irsim_segment"].data
        for poly in remaining.data.polygons:
            poly.select = seg[poly.index].value == li
        before = set(bpy.data.objects.keys())
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.separate(type="SELECTED")
        bpy.ops.object.mode_set(mode="OBJECT")
        fresh = sorted(set(bpy.data.objects.keys()) - before)
        assert len(fresh) == 1, fresh
        piece = bpy.data.objects[fresh[0]]
        piece.name = f"skin_{label}"
        piece.data.name = piece.name
        made[label] = piece.name
        bpy.ops.object.select_all(action="DESELECT")
        remaining.select_set(True)
        bpy.context.view_layer.objects.active = remaining
    remaining.name = f"skin_{present[-1]}"
    remaining.data.name = remaining.name
    made[present[-1]] = remaining.name
    for obj in bpy.data.objects:
        if obj.type == "MESH" and obj.name.lower() in LAYER_BY_OBJECT:
            layer = LAYER_BY_OBJECT[obj.name.lower()]
            obj.name = obj.name.lower() if layer == "eyes" else obj.name.lower()
            obj.data.name = obj.name

    # --- 4c. garments: a colour from the asset config dyes the garment's texture -----------------
    # The tint is baked into a copy of the garment's diffuse image, pixel by pixel, because the
    # USD the renderer reads (UsdPreviewSurface) carries a texture or a constant, never a multiply
    # node; the glTF exporter would keep the node, Blender's USD exporter drops it. The same
    # colour dyes the material's alpha_sol and NIR in irsim.materials.colour.
    import numpy as np

    tinted = []
    for slot, g in garments.items():
        rgb = g.get("colour_rgb")
        mat = bpy.data.materials.get(f"garment_{slot}")
        if rgb is None or mat is None or not mat.use_nodes:
            continue
        tree = mat.node_tree
        bsdf = next((n for n in tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            continue
        base = bsdf.inputs["Base Color"]
        tex = next(
            (link.from_node for link in base.links if link.from_node.type == "TEX_IMAGE"), None
        )
        code = "".join(f"{int(round(c * 255)):02x}" for c in rgb)
        if tex is None or tex.image is None:
            base.default_value = (*rgb, 1.0)
            tinted.append(slot)
            continue
        src = tex.image
        if src.get("irsim_tint_source"):
            src = bpy.data.images[src["irsim_tint_source"]]  # re-dye from the original
        w, h = src.size
        px = np.empty(w * h * 4, dtype=np.float32)
        src.pixels.foreach_get(px)
        px = px.reshape(-1, 4)
        px[:, :3] *= np.asarray(rgb, dtype=np.float32)
        dyed = bpy.data.images.new(f"garment_{slot}_rgb{code}", w, h, alpha=True)
        dyed.pixels.foreach_set(px.reshape(-1))
        dyed["irsim_tint_source"] = src.name
        dyed.filepath_raw = str(a.out.parent / f"garment_{slot}_rgb{code}.png")
        dyed.file_format = "PNG"
        dyed.save()
        tex.image = dyed
        tinted.append(slot)

    # --- 4d. static: the labelled asset is the rest pose, with no skeleton ---------------------
    # MakeHuman's garments arrive unskinned while the body and anything cut from it are skinned.
    # Exported under one skeleton, the USD binds the garments to it with no real weights, and the
    # renderer's skinning collapses them (HU.7: an officer drawn without trousers). No step renders
    # a deforming person yet -- that is HU.10, which takes the rig from make_human.py's own output
    # -- so the labelled asset is static: every mesh keeps its rest-pose world transform, the
    # armature modifiers and the rig go.
    # HU.10: `--keep-rig` skips this, for a body that moves (scripts/animate_human.py). Its
    # garments must then be skinned first, which MakeHuman's are not; the probe body is bare.
    for obj in [o for o in bpy.data.objects if o.type == "MESH" and not a.keep_rig]:
        for mod in [m for m in obj.modifiers if m.type == "ARMATURE"]:
            obj.modifiers.remove(mod)
        if obj.parent is not None:
            world_matrix = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = world_matrix
    for obj in [o for o in bpy.data.objects if o.type in ("ARMATURE", "EMPTY") and not a.keep_rig]:
        bpy.data.objects.remove(obj, do_unlink=True)

    # --- 5. report and export --------------------------------------------------------------------
    stats = {}
    for label in present:
        o = bpy.data.objects[made[label]]
        area = sum(p.area for p in o.data.polygons)
        s = o.matrix_world.to_scale()
        stats[label] = {"faces": len(o.data.polygons), "area_m2": round(area * s.x * s.y, 6)}
    zs = [
        (o.matrix_world @ v.co).z
        for o in bpy.data.objects
        if o.type == "MESH" and o.name.startswith("skin_")
        for v in o.data.vertices
    ]
    report = {
        "source": str(a.source),
        "segments": stats,
        "stature_m": round(max(zs) - min(zs), 4),
        "forward_axis": a.forward_axis,
        "forward_check_m": forward_check_m,
        "unlabelled_faces_resolved_by_nearest": unlabelled,
        "garments": sorted(
            o.name[len("garment_") :]
            for o in bpy.data.objects
            if o.type == "MESH" and o.name.startswith("garment_")
        ),
        "tinted": tinted,
        "objects": sorted(o.name for o in bpy.data.objects if o.type == "MESH"),
    }
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(
        filepath=str(a.out),
        export_format="GLB",
        use_selection=False,
        export_apply=False,
        export_skins=True,
        export_yup=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
    )
    print(f"[prep_human] {len(present)} segments -> {a.out}; report {a.report}")


# ---------------------------------------------------------------------------------------------
# driver (project interpreter)
# ---------------------------------------------------------------------------------------------


def _rig_json(rig: str) -> dict:
    from irsim.config.humans import load_body_schema

    schema = load_body_schema()
    if rig not in schema.rigs:
        raise SystemExit(f"unknown rig {rig!r}; the body schema has {sorted(schema.rigs)}")
    r = schema.rigs[rig]
    return {
        "rig": rig,
        "bones": dict(r.bones),
        "patterns": dict(r.patterns),
        "non_deforming": list(r.non_deforming),
        "strip_prefixes": list(r.strip_prefixes),
        "ignore_prefixes": list(r.ignore_prefixes),
        "regions": {
            k: {"positive": v.positive, "negative": v.negative} for k, v in schema.regions.items()
        },
        "segments": list(schema.segment_names),
    }


def _garments(name: str, sidecar: dict) -> dict[str, dict]:
    """The `garments:` block: an existing config's, else defaults for the slots the mesh wears."""
    import yaml

    from irsim.config.humans import DEFAULT_GARMENT_CLO, DEFAULT_GARMENT_MATERIAL

    cfg = REPO_ROOT / "configs" / "assets" / f"{name}.yaml"
    worn = dict(sidecar.get("garments") or {})
    if cfg.is_file():
        existing = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        kept = ((existing.get("asset") or {}).get("human") or {}).get("garments") or {}
        kept = {slot: dict(g) for slot, g in kept.items() if not worn or slot in worn}
    else:
        kept = {}
    for slot in worn:
        if slot not in kept:
            kept[slot] = {
                "material": DEFAULT_GARMENT_MATERIAL.get(slot, "cotton_clothing"),
                "clo": DEFAULT_GARMENT_CLO.get(slot, 0.1),
            }
    return kept


def _equipment(name: str, objects: list[str]) -> dict[str, dict]:
    """The `equipment:` block: an existing config's entries, one for every `equipment_<x>` object.

    Equipment has no default: a belt and a tape are different materials on different slots, and
    a guess would be a material nobody chose. An object without an entry stops the run.
    """
    import yaml

    cfg = REPO_ROOT / "configs" / "assets" / f"{name}.yaml"
    have: dict[str, dict] = {}
    if cfg.is_file():
        existing = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        have = dict(((existing.get("asset") or {}).get("human") or {}).get("equipment") or {})
    worn = sorted(o[len("equipment_") :] for o in objects if o.startswith("equipment_"))
    missing = [w for w in worn if w not in have]
    if missing:
        raise SystemExit(
            f"equipment {missing} has no entry in configs/assets/{name}.yaml `human.equipment` "
            "(material and the slot it sits on): seed the config before running"
        )
    return {w: dict(have[w]) for w in worn}


def _asset_config(
    name: str,
    source_rel: str,
    sidecar: dict,
    report: dict,
    rig: str,
    forward: str,
    bmi: float,
    garments: dict[str, dict],
    bmi_source: str = "ESTIMATED",
    equipment: dict[str, dict] | None = None,
) -> str:
    import yaml

    from irsim.config.humans import JOS3_SEGMENTS

    stature = float(report["stature_m"])
    mass = round(bmi * stature * stature, 1)
    parts = [{"name": f"skin_{s}", "select": {"objects": [f"skin_{s}"]}} for s in JOS3_SEGMENTS]
    for layer_obj in sorted(o for o in report["objects"] if not o.startswith("skin_")):
        parts.append({"name": layer_obj, "select": {"objects": [layer_obj]}})
    doc = {
        "schema_version": 1,
        "asset": {
            "name": name,
            "kind": "human",
            "source_file": source_rel,
            "scale_to_metres": 1.0,
            "materials": {
                "skin": "human_skin",
                "eyes": "human_skin",
                "eyebrows": "human_skin",
                "eyelashes": "human_skin",
                "hair": "hair",
                **{f"garment_{slot}": g["material"] for slot, g in garments.items()},
                **{f"equipment_{n}": e["material"] for n, e in (equipment or {}).items()},
            },
            "human": {
                "phenotype": {
                    "sex": sidecar.get("sex", "male"),
                    "age_y": float(sidecar.get("age_years", 30.0)),
                    "height_m": round(stature, 3),
                    "mass_kg": mass,
                },
                "rig": rig,
                "forward_axis": forward,
                "garments": garments,
                **({"equipment": equipment} if equipment else {}),
            },
            "parts": {"granularity": "object", "parts": parts},
        },
    }
    head = (
        f"# {name}: a human generated by MakeHuman's MPFB2 (CC0 base mesh, skin and body parts)\n"
        f"# and labelled onto the body schema by its {rig} skeleton with scripts/prep_human.py\n"
        f"# (HU.3, ADR 0192). Schema: irsim.materials.mapping.AssetConfig with `kind: human`.\n#\n"
        f"# Phenotype: sex and age from the generator's macros ({sidecar.get('macro', {})});\n"
        f"# stature {stature:.3f} m MEASURED on the mesh; mass from a BMI of {bmi} ({mass} kg),\n"
        f"# {bmi_source}.\n"
        "# Materials: eyes, eyebrows and eyelashes carry skin's optics (ESTIMATED). Garments: one\n"
        "# object per slot, `garment_<slot>`, each with a library material, an insulation in clo\n"
        "# (ISO 9920 garment values, ESTIMATED by slot unless edited) and optionally a colour:\n"
        "# edit `colour_rgb` and re-run prep_human.py -- the tint reaches the RGB companion and\n"
        "# the dyed variant's alpha_sol and NIR reach the infrared (HU.5, ADR 0195).\n"
    )
    return head + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=pathlib.Path, required=True, help="the rigged .glb")
    ap.add_argument("--name", required=True, help="the asset's name")
    ap.add_argument("--rig", default="mpfb_game_engine", help="a rig of the body schema")
    ap.add_argument(
        "--forward-axis", default="-y", choices=sorted(AXES), help="write it as --forward-axis=-y"
    )
    ap.add_argument("--body-object", default="body")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="the labelled .glb to write")
    ap.add_argument("--sidecar", type=pathlib.Path, default=None, help="make_human.py's JSON")
    ap.add_argument("--write-config", action="store_true", help="write configs/assets/<name>.yaml")
    ap.add_argument(
        "--source-rel",
        default=None,
        help="source_file for the config (default: --out, repo-relative)",
    )
    ap.add_argument("--bmi", type=float, default=23.0, help="for the ESTIMATED mass")
    ap.add_argument(
        "--bmi-source", default="ESTIMATED", help="where the BMI comes from, for the config header"
    )
    ap.add_argument(
        "--keep-rig",
        action="store_true",
        help="keep the skeleton and the skinning (HU.10, scripts/animate_human.py); the default "
        "is the static rest-pose asset every render uses",
    )
    ap.add_argument("--blender", default="blender")
    a = ap.parse_args(argv)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    rig_json = a.out.with_suffix(".rig.json")
    rig_json.write_text(json.dumps(_rig_json(a.rig), indent=1))
    sidecar = json.loads(a.sidecar.read_text()) if a.sidecar and a.sidecar.is_file() else {}
    garments = _garments(a.name, sidecar)
    garments_json = a.out.with_suffix(".garments.json")
    garments_json.write_text(json.dumps(garments, indent=1))
    report = a.out.with_suffix(".report.json")
    cmd = [
        a.blender, "-b", "--python", str(pathlib.Path(__file__).resolve()), "--",
        "--source", str(a.source), "--rig-json", str(rig_json), "--out", str(a.out),
        "--report", str(report), f"--forward-axis={a.forward_axis}", "--body-object", a.body_object,
        "--garments-json", str(garments_json), *(["--keep-rig"] if a.keep_rig else []),
    ]  # fmt: skip
    r = subprocess.run(cmd, cwd=REPO_ROOT)
    if r.returncode or not report.is_file():
        raise SystemExit(f"the Blender worker failed ({r.returncode})")
    rep = json.loads(report.read_text())
    segs = rep["segments"]
    missing = [s for s in _rig_json(a.rig)["segments"] if s not in segs]
    if missing:
        raise SystemExit(f"the body has no faces for segments {missing}")
    pairs = [
        ("LShoulder", "RShoulder"),
        ("LArm", "RArm"),
        ("LHand", "RHand"),
        ("LThigh", "RThigh"),
        ("LLeg", "RLeg"),
        ("LFoot", "RFoot"),
    ]
    print(
        f"[prep_human] {a.name}: stature {rep['stature_m']} m, "
        f"forward check {rep['forward_check_m']}"
    )
    for s in segs:
        print(f"  {s:10s} {segs[s]['faces']:6d} faces  {segs[s]['area_m2']:.4f} m²")
    for l_, r_ in pairs:
        ratio = segs[l_]["area_m2"] / segs[r_]["area_m2"]
        flag = "" if 0.9 <= ratio <= 1.1 else "  <-- asymmetric"
        print(f"  {l_}/{r_} area ratio {ratio:.3f}{flag}")
    if rep.get("garments"):
        print(f"  garments: {rep['garments']} (tinted: {rep.get('tinted') or 'none'})")
    if a.write_config:
        rel = a.source_rel or str(a.out.resolve().relative_to(REPO_ROOT))
        cfg = REPO_ROOT / "configs" / "assets" / f"{a.name}.yaml"
        equipment = _equipment(a.name, rep["objects"])
        cfg.write_text(
            _asset_config(
                a.name,
                rel,
                sidecar,
                rep,
                a.rig,
                a.forward_axis,
                a.bmi,
                garments,
                a.bmi_source,
                equipment,
            )
        )
        print(f"[prep_human] wrote {cfg}")
    return 0


def _in_blender() -> bool:
    try:
        import bpy  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    if _in_blender():
        _worker(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    else:
        sys.exit(main(sys.argv[1:]))
