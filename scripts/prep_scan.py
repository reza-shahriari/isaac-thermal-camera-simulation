#!/usr/bin/env python3
"""Any downloaded human, unrigged, to a labelled ``kind: human`` asset (HU.9).

    python scripts/prep_scan.py --source 3d_models/dennis/source/scan.glb --name dennis \\
        --sex male --age-years 30 --write-config

A scan is one mesh -- skin, clothes and hair fused -- with no skeleton and no part names.
``prep_human.py`` labels a person whose rig an artist made; this labels one whose rig nobody
made, in five steps, each reusing what the project already tests:

1. **Rig** it: ``autorig_mia.py`` runs Make-It-Animatable (MIT) in its own environment and
   returns Mixamo bone weights per vertex, the joints, and the bone hierarchy.
2. **Colour** it: ``scan_colours.py`` (Blender) samples each triangle's texture.
3. **Segments**, from the weights, by ``prep_human.py``'s own arithmetic
   (:func:`irsim.io.human_scan.label_faces_from_weights`); the facing comes from the predicted
   joints, never a guess (:func:`~irsim.io.human_scan.forward_from_joints`).
4. **Skin, hair or garment**, from the person's own skin colour
   (:func:`~irsim.io.human_scan.classify_surfaces`); a triangle colour cannot decide is decided
   anyway and **listed** in ``<out>.review.json`` for a person (the Blender add-on's job, B14).
5. **Split and write**: the worker (this file, in Blender) separates the scan into
   ``skin_<Segment>``, ``garment_<slot>`` and ``hair`` objects -- the names every other human
   asset uses, so ``prep_asset.py`` and the scene need nothing new -- and, with
   ``--write-config``, the asset YAML: the phenotype you state (sex, age; stature measured, mass
   from a BMI), each garment slot with a default material and clo and **its measured colour**.

With ``--truth`` (a ``fuse_human.py`` answer file) it also scores itself against the answer.
The tools that touched the asset and their licences are written into the config's header; the
asset's own licence is the provenance file's (``register_local_asset.py``).

docs/physics-model.md §6.1, §16.2; roadmap HU.9; ADR 0192, ADR 0201.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MIA_PYTHON = pathlib.Path.home() / "miniconda3" / "envs" / "mia" / "bin" / "python"
MIA_ROOT = pathlib.Path.home() / "tools" / "Make-It-Animatable"
TOOLS = (
    "Make-It-Animatable (Guo et al. 2025, github.com/jasongzy/Make-It-Animatable, MIT): bone "
    "weights; irsim.io.human_scan: segments and skin / hair / garment by the person's own skin "
    "colour"
)


# ---------------------------------------------------------------------------------------------
# worker (Blender)
# ---------------------------------------------------------------------------------------------


def _worker(argv: list[str]) -> None:
    import bpy
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=pathlib.Path, required=True)
    ap.add_argument("--labels", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--report", type=pathlib.Path, required=True)
    a = ap.parse_args(argv)
    lab = np.load(a.labels)
    centres_ref = lab["centres_gltf"].astype(np.float64)
    names = lab["labels"].astype(str)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(a.source))
    for obj in [o for o in bpy.data.objects if o.type != "MESH"]:
        bpy.data.objects.remove(obj, do_unlink=True)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    for obj in meshes:
        for mod in list(obj.modifiers):
            obj.modifiers.remove(mod)
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    scan = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(scan.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.to_mesh(scan.data)
    bm.free()
    me = scan.data
    c = np.array([tuple(p.center) for p in me.polygons], dtype=np.float64)
    c_gltf = np.c_[c[:, 0], c[:, 2], -c[:, 1]]
    # nearest labelled triangle, brute force in chunks (the labels come from the same mesh)
    ref_sq = np.einsum("ij,ij->i", centres_ref, centres_ref)
    pick = np.empty(len(c_gltf), dtype=np.int64)
    worst = 0.0
    for i in range(0, len(c_gltf), 1024):
        q = c_gltf[i : i + 1024]
        d2 = np.einsum("ij,ij->i", q, q)[:, None] - 2.0 * q @ centres_ref.T + ref_sq[None, :]
        j = np.argmin(d2, axis=1)
        pick[i : i + 1024] = j
        worst = max(worst, float(np.sqrt(max(d2[np.arange(len(q)), j].max(), 0.0))))
    face_label = names[pick]

    made = {}
    for label in sorted(set(face_label.tolist())):
        obj = scan.copy()
        obj.data = scan.data.copy()
        bpy.context.collection.objects.link(obj)
        bm = bmesh.new()
        bm.from_mesh(obj.data)
        bm.faces.ensure_lookup_table()
        gone = [f for f in bm.faces if face_label[f.index] != label]
        bmesh.ops.delete(bm, geom=gone, context="FACES")
        bm.to_mesh(obj.data)
        bm.free()
        obj.name = obj.data.name = label
        # every material on the object becomes the object's own copy, named for its label: the
        # asset map binds by material name, and a scan's one texture (or a fused person's several)
        # must not carry skin's name onto a garment or a garment's onto skin
        for i, slot in enumerate(obj.material_slots):
            if slot.material is not None:
                mat = slot.material.copy()
                mat.name = f"{label}__{i}"
                slot.material = mat
        made[label] = obj
    bpy.data.objects.remove(scan, do_unlink=True)

    stats = {
        k: {
            "faces": len(o.data.polygons),
            "area_m2": round(sum(p.area for p in o.data.polygons), 6),
        }
        for k, o in made.items()
    }
    zs = [
        (o.matrix_world @ v.co).z
        for k, o in made.items()
        if k.startswith(("skin_", "garment_feet", "hair"))
        for v in o.data.vertices
    ]
    used = {m.name for o in made.values() for m in o.data.materials if m is not None}
    report = {
        "source": str(a.source),
        "objects": sorted(made),
        "materials": {k: sorted(n for n in used if n.startswith(f"{k}__")) for k in made},
        "segments": {k[5:]: v for k, v in stats.items() if k.startswith("skin_")},
        "areas": stats,
        "stature_m": round(max(zs) - min(zs), 4),
        "worst_label_match_m": worst,
    }
    a.report.write_text(json.dumps(report, indent=2) + "\n")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(a.out),
        export_format="GLB",
        use_selection=False,
        export_skins=False,
        export_yup=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
    )
    print(f"[prep_scan] {len(made)} objects -> {a.out}")


# ---------------------------------------------------------------------------------------------
# driver (project interpreter)
# ---------------------------------------------------------------------------------------------


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _config(
    name: str, source_rel: str, report: dict, garments: dict, a: argparse.Namespace, review: float
) -> str:
    import yaml

    from irsim.config.humans import JOS3_SEGMENTS

    stature = float(report["stature_m"])
    mass = round(a.bmi * stature * stature, 1)
    parts = [{"name": f"skin_{s}", "select": {"objects": [f"skin_{s}"]}} for s in JOS3_SEGMENTS]
    parts += [
        {"name": o, "select": {"objects": [o]}}
        for o in sorted(report["objects"])
        if not o.startswith("skin_")
    ]
    doc = {
        "schema_version": 1,
        "asset": {
            "name": name,
            "kind": "human",
            "source_file": source_rel,
            "scale_to_metres": 1.0,
            "materials": {
                mat: (
                    "human_skin"
                    if obj.startswith("skin_")
                    else "hair"
                    if obj == "hair"
                    else garments[obj[len("garment_") :]]["material"]
                )
                for obj, mats in sorted(report["materials"].items())
                for mat in mats
            },
            "human": {
                "phenotype": {
                    "sex": a.sex,
                    "age_y": float(a.age_years),
                    "height_m": round(stature, 3),
                    "mass_kg": mass,
                },
                "rig": "mixamo",
                "forward_axis": "-y",
                "garments": garments,
            },
            "parts": {"granularity": "object", "parts": parts},
        },
    }
    head = (
        f"# {name}: a downloaded human scan, labelled onto the body schema with no rig of its own\n"
        "# by scripts/prep_scan.py (HU.9, ADR 0201). Schema: irsim.materials.mapping.AssetConfig.\n"
        f"# Tools: {TOOLS}.\n"
        f"# Phenotype: sex {a.sex} and age {a.age_years} STATED by whoever ran it; stature\n"
        f"# {stature:.3f} m MEASURED on the mesh; mass from a BMI of {a.bmi} ({mass} kg), "
        f"{a.bmi_source}.\n"
        "# Garments: one object per slot the scan's non-skin surface falls on, a default material\n"
        "# and clo (ESTIMATED by slot; edit them), and the colour MEASURED on the scan, linear.\n"
        f"# {review:.1%} of the surface was decided by colour near its threshold and is listed in\n"
        f"# outputs/humans/{name}/{name}.review.json for a person to confirm.\n"
    )
    return head + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def main(argv: list[str]) -> int:
    import numpy as np

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from irsim.config.humans import (
        DEFAULT_GARMENT_CLO,
        DEFAULT_GARMENT_MATERIAL,
        load_body_schema,
    )
    from irsim.io.human_scan import (
        classify_surfaces,
        forward_from_joints,
        label_faces_from_weights,
        nearest,
        score,
        score_classes,
    )

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=pathlib.Path, required=True, help="the scan (.glb)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--sex", choices=("male", "female"), required=True)
    ap.add_argument("--age-years", type=float, required=True)
    ap.add_argument("--bmi", type=float, default=23.0)
    ap.add_argument("--bmi-source", default="ESTIMATED (a mid-range adult BMI)")
    ap.add_argument("--out-dir", type=pathlib.Path, default=None)
    ap.add_argument("--write-config", action="store_true")
    ap.add_argument("--source-rel", default=None)
    ap.add_argument("--truth", type=pathlib.Path, default=None, help="fuse_human.py's answer")
    ap.add_argument("--cpu", action="store_true", help="run the rigger on the CPU")
    ap.add_argument("--mia-python", type=pathlib.Path, default=MIA_PYTHON)
    ap.add_argument("--mia-root", type=pathlib.Path, default=MIA_ROOT)
    ap.add_argument("--blender", default="blender")
    a = ap.parse_args(argv)
    out_dir = a.out_dir or REPO_ROOT / "outputs" / "humans" / a.name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / a.name

    # 1. rig
    mia = stem.with_suffix(".mia.npz")
    cmd = [str(a.mia_python), str(REPO_ROOT / "scripts" / "autorig_mia.py"), "--mia-root",
           str(a.mia_root), "--input", str(a.source.resolve()), "--out", str(mia),
           *(["--cpu"] if a.cpu else [])]  # fmt: skip
    # the project interpreter (Isaac's python.sh) exports its own PYTHONPATH / PYTHONHOME; the
    # rigger's interpreter must not inherit them, or it loads Isaac's standard library
    import os

    clean = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONNOUSERSITE", "LD_PRELOAD")
    }
    if subprocess.run(cmd, cwd=REPO_ROOT, env=clean).returncode or not mia.is_file():
        raise SystemExit("the rigger failed (is Make-It-Animatable installed? see autorig_mia.py)")
    # 2. colour
    colours = stem.with_suffix(".colours.npz")
    cmd = [a.blender, "-b", "--python", str(REPO_ROOT / "scripts" / "scan_colours.py"), "--",
           "--source", str(a.source), "--out", str(colours)]  # fmt: skip
    if subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True).returncode or not colours.is_file():
        raise SystemExit("colour sampling failed")

    # 3-4. segments, then skin / hair / garment, on the rigger's (merged-vertex) mesh
    schema = load_body_schema()
    m = np.load(mia)
    verts, tris = m["vertices"].astype(np.float64), m["faces"].astype(np.int64)
    bones = [str(b) for b in m["bones"]]
    fwd = forward_from_joints(bones, m["joints_head"])
    at = {b: i for i, b in enumerate(bones)}
    up = m["joints_head"][at["mixamorig:Head"]] - m["joints_head"][at["mixamorig:Hips"]]
    segments = label_faces_from_weights(
        schema, "mixamo", bones, m["weights"], verts, tris, forward=fwd
    )
    centres = verts[tris].mean(axis=1)
    e1, e2 = verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    col = np.load(colours)
    idx, _ = nearest(centres, col["centres_gltf"])
    rgb = col["rgb"][idx]
    classes, record = classify_surfaces(
        schema, segments, rgb, centres, areas, fwd, up, faces=tris, n_vertices=len(verts)
    )
    review = np.asarray(record.pop("review"), dtype=bool)
    labels_npz = stem.with_suffix(".labels.npz")
    np.savez_compressed(
        labels_npz,
        centres_gltf=centres.astype(np.float32),
        labels=np.array(classes),
        segments=np.array(segments),
        review=review,
        areas=areas.astype(np.float32),
    )
    review_area = float(areas[review].sum() / areas.sum())
    listed = {}
    for k in sorted(set(np.array(classes)[review].tolist())):
        sel = review & (np.array(classes) == k)
        listed[k] = {"triangles": int(sel.sum()), "area_m2": round(float(areas[sel].sum()), 5)}
    stem.with_suffix(".review.json").write_text(
        json.dumps(
            {
                "what": "triangles whose skin / garment / hair call lies within REVIEW_BAND of the "
                "colour threshold: decided, but a person should confirm",
                "fraction_of_area": round(review_area, 4),
                "skin_reference_lab": {k: round(v, 2) for k, v in record.items()},
                "rigger_seconds": float(m["seconds"]),
                "rigger_device": str(m["device"]),
                "listed": listed,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[prep_scan] rigged in {float(m['seconds']):.1f} s on {m['device']}; skin reference "
        f"Lab {record['L']:.1f} {record['a']:.1f} {record['b']:.1f}; {review_area:.1%} for review"
    )

    if a.truth is not None:
        t = np.load(a.truth)
        seg_score = score(t["labels"], t["centres_gltf"], t["areas"], segments, centres)
        j, _ = nearest(t["centres_gltf"], centres)
        cls_score = score_classes(t["labels"], t["areas"], [classes[k] for k in j], review[j])
        print(seg_score.render())
        print(cls_score.render())
        stem.with_suffix(".score.json").write_text(
            json.dumps(
                {
                    "segment_agreement": seg_score.agreement,
                    "per_segment": seg_score.per_segment,
                    "class_accuracy": cls_score.accuracy,
                    "class_confident_accuracy": cls_score.confident_accuracy,
                    "review_fraction": cls_score.review_fraction,
                    "rigger_seconds": float(m["seconds"]),
                },
                indent=2,
            )
            + "\n"
        )

    # 5. split and write
    labelled = stem.with_name(f"{a.name}_labelled.glb")
    report = stem.with_suffix(".report.json")
    cmd = [a.blender, "-b", "--python", str(pathlib.Path(__file__).resolve()), "--",
           "--source", str(a.source), "--labels", str(labels_npz), "--out", str(labelled),
           "--report", str(report)]  # fmt: skip
    if subprocess.run(cmd, cwd=REPO_ROOT).returncode or not report.is_file():
        raise SystemExit("the Blender worker failed")
    rep = json.loads(report.read_text())
    missing = [s for s in schema.segment_names if s not in rep["segments"]]
    if missing:
        print(f"[prep_scan] WARNING: no skin faces for {missing} (covered, or mislabelled)")
    if a.write_config:
        garments = {}
        cls = np.array(classes)
        for obj in rep["objects"]:
            if not obj.startswith("garment_"):
                continue
            slot = obj[len("garment_") :]
            sel = cls == obj
            med = np.median(rgb[sel], axis=0) if sel.any() else np.array([0.5, 0.5, 0.5])
            garments[slot] = {
                "material": DEFAULT_GARMENT_MATERIAL.get(slot, "cotton_clothing"),
                "clo": DEFAULT_GARMENT_CLO.get(slot, 0.1),
                "colour_rgb": [round(_srgb_to_linear(float(v)), 4) for v in med],
            }
        rel = a.source_rel or str(labelled.resolve().relative_to(REPO_ROOT))
        cfg = REPO_ROOT / "configs" / "assets" / f"{a.name}.yaml"
        cfg.write_text(_config(a.name, rel, rep, garments, a, review_area))
        print(f"[prep_scan] wrote {cfg}")
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
