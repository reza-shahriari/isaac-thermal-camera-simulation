#!/usr/bin/env python3
"""Turn an asset-library entry into a Thermal Model Zoo bundle, ready for the zoo's publish tool.

    $PYTHON scripts/zoo_bundle.py dji_mini_3_pro --category drones --subcategory Multirotor
    python3 ../ThermalModelZoo/tools/publish.py zoo_bundles/dji-mini-3-pro

The zoo (https://github.com/reza-shahriari/ThermalModelZoo) is a static catalogue of part-split,
material-mapped models. Its bundle format is `entry.json`, `images/`, `files/`, `ATTRIBUTION.md`;
this script writes one from what the ingest-asset skill already produced:

* **entry.json** -- author, links, licence and the list of changes from `<asset>.provenance.yaml`;
  one row per functional part from the `_parts` .blend, its library materials resolved through
  `configs/assets/<asset>.yaml` (`AssetMapping.lookup`, plus a part's asserted `material:` from
  `part_assignment.json`), and an **area-weighted 8-14 um emissivity** per part from
  `Material.band_properties` over an 8-14 um top-hat (the band the zoo states; the registry's
  `lwir` is 7.5-13.5 um). A part's source is its weakest material's (`literature` -> reference).
  Mirror-image parts with the same materials collapse into one row ("Arm (x4)").
* **images/** -- rendered by Blender, headless: `beauty` (Eevee), `wireframe` (one flat colour per
  part) and `emissivity` (each surface flat grey at its LWIR emissivity, with a 0-1 scale). No
  thermal still exists for library assets until one is rendered through Isaac; pass it with
  `--thermal frame.png` when there is one -- nothing fakes it.
* **files/** -- `<id>.blend` (textures packed), `<id>.usdc.zip` (USD + textures),
  `<id>.fbx.zip` (FBX + its .fbm folder): the `_parts` exports, which are what the zoo is for.

Only CC0 / CC-BY / CC-BY-SA sources are accepted (the zoo's rule); anything else stops here.
Needs Blender (`$BLENDER`, else `blender` on PATH) and Pillow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Provenance licence slug -> the zoo's licence id. Anything else is not redistributable there.
ZOO_LICENCE = {"cc0": "CC0-1.0", "by": "CC-BY-4.0", "by-sa": "CC-BY-SA-4.0"}
#: MaterialSpec.source -> the zoo's vocabulary, weakest last.
ZOO_SOURCE = {"measured": "measured", "literature": "reference", "estimated": "estimated"}
SOURCE_RANK = ["measured", "reference", "estimated"]
#: Trailing name tokens that mark one of several mirror-image copies of a part.
SIDE_TOKENS = (
    r"(front|rear|back|left|right|port|starboard|bow|stern"
    r"|upper|lower|top|bottom|inner|outer|fore|aft|\d+)"
)
BAND_UM = (8.0, 14.0)
RES = (1200, 900)  # the zoo's stage is 4:3
LEGEND_H = 80  # a strip under every view; the emissivity map draws its scale there
RENDER_SCALE = (
    200  # render at 2x, crop to the model, scale down: framing independent of the camera fit
)
BG = (17, 19, 24)  # the zoo's dark surface


# --------------------------------------------------------------------------------------------
# pure helpers (unit-tested in tests/unit/test_zoo_bundle.py)
# --------------------------------------------------------------------------------------------
def zoo_id(asset: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", asset.lower()).strip("-")


def zoo_licence(slug: str) -> str:
    try:
        return ZOO_LICENCE[slug.lower()]
    except KeyError:
        raise SystemExit(
            f"licence {slug!r} is not one the zoo accepts (CC0, CC-BY, CC-BY-SA); not bundling"
        ) from None


def part_base(name: str) -> str:
    """`motor_front_left` -> `motor`; `arm_2` -> `arm`; `fuselage` -> `fuselage`."""
    base = re.sub(rf"(_{SIDE_TOKENS})+$", "", name)
    return base or name


def part_colours(names: Sequence[str]) -> dict[str, tuple[float, float, float]]:
    """One colour per PART: the parts image's colour key.

    Every part is its own piece of hardware and must read as one: each propeller distinct from the
    other three, and from the motor and duct beside it. The n parts take n hues spaced evenly round
    the wheel, in the order -- a stride through the sorted names -- that puts the pairs that must
    differ most furthest apart: copies of one kind (`propeller_front_left` / `_rear_left`) and parts
    of one station (`propeller_front_left` / `duct_front_left`). Hue-adjacent slots cycle through
    three shades (bright, pale, deep), so neighbours on the wheel still differ in lightness. On the
    DJI Avata 2's 22 parts those pairs are at least 49 deg apart. Two earlier schemes failed the
    owner: a golden-angle hue per part index put each propeller 20 deg from its duct, and one hue
    per kind gave all four propellers one colour.
    """
    import colorsys
    import itertools

    ordered = sorted(names)
    n = len(ordered)
    if n == 0:
        return {}
    side = {name: name[len(part_base(name)) :] for name in ordered}
    related = [
        (a, b)
        for a, b in itertools.combinations(ordered, 2)
        if part_base(a) == part_base(b) or (side[a] and side[a] == side[b])
    ]

    def gaps(stride: int) -> tuple[float, float]:
        slot = {name: (i * stride) % n for i, name in enumerate(ordered)}

        def d(a: str, b: str) -> float:
            k = abs(slot[a] - slot[b])
            return min(k, n - k) / n

        everything = itertools.combinations(ordered, 2)
        return (
            min((d(a, b) for a, b in related), default=1.0),
            min((d(a, b) for a, b in everything), default=1.0),
        )

    stride = max((s for s in range(1, n) if math.gcd(s, n) == 1), key=gaps, default=1)
    shades = ((0.65, 0.95), (0.40, 0.80), (0.70, 0.62))
    out = {}
    for i, name in enumerate(ordered):
        k = (i * stride) % n
        out[name] = colorsys.hsv_to_rgb(k / n, *shades[k % len(shades)])
    return out


def part_label(name: str) -> str:
    """`propeller_front_left` -> `Propeller front left`: a part's row name in the zoo table."""
    return name.replace("_", " ").capitalize()


def area_weighted(values: Mapping[str, float], areas: Mapping[str, float]) -> float:
    total = sum(areas.values())
    if total <= 0:
        raise ValueError("a part with no surface area")
    return sum(values[k] * a for k, a in areas.items()) / total


def weakest(sources: Sequence[str]) -> str:
    return max(sources, key=SOURCE_RANK.index)


def group_parts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse mirror-image parts that share a base name into one row, summing material areas.

    Copies need not match material for material -- one motor of four may carry the status LED --
    so the row's areas are the union, which keeps its area-weighted emissivity exact.
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault(part_base(r["name"]), []).append(r)
    out = []
    for base, rs in groups.items():
        areas: dict[str, float] = {}
        for r in rs:
            for k, a in r["areas"].items():
                areas[k] = areas.get(k, 0.0) + a
        label = base.replace("_", " ").capitalize()
        out.append(
            {
                "name": f"{label} (x{len(rs)})" if len(rs) > 1 else label,
                "areas": areas,
                "target": rs[0].get("target", ""),
            }
        )
    return out


def material_text(areas: Mapping[str, float], top: int = 3) -> str:
    total = sum(areas.values())
    ranked = sorted(areas.items(), key=lambda kv: -kv[1])
    if len(ranked) == 1:
        return ranked[0][0].replace("_", " ")
    shown = [f"{k.replace('_', ' ')} {100 * a / total:.0f} %" for k, a in ranked[:top]]
    return ", ".join(shown) + (f", +{len(ranked) - top} more" if len(ranked) > top else "")


# --------------------------------------------------------------------------------------------
# worker -- runs inside Blender
# --------------------------------------------------------------------------------------------
def _mesh_objects() -> list[Any]:
    import bpy

    return [o for o in bpy.context.scene.objects if o.type == "MESH" and o.visible_get()]


def worker_stats(out: pathlib.Path) -> None:
    """Per object: triangles and world-space area per material; the whole asset's bounding box."""
    import bmesh
    import bpy

    deps = bpy.context.evaluated_depsgraph_get()
    objs, lo, hi = {}, [math.inf] * 3, [-math.inf] * 3
    for o in _mesh_objects():
        bm = bmesh.new()
        bm.from_object(o, deps)
        bm.transform(o.matrix_world)
        areas: dict[str, float] = {}
        tris = 0
        for f in bm.faces:
            slot = o.material_slots[f.material_index].material if o.material_slots else None
            name = slot.name if slot else ""
            areas[name] = areas.get(name, 0.0) + f.calc_area()
            tris += len(f.verts) - 2
        for v in bm.verts:
            for i in range(3):
                lo[i], hi[i] = min(lo[i], v.co[i]), max(hi[i], v.co[i])
        bm.free()
        objs[o.name] = {"triangles": tris, "areas": areas}
    out.write_text(json.dumps({"objects": objs, "bbox": [lo, hi]}))


def _frame(bbox: Sequence[Sequence[float]]) -> None:
    """A three-quarter-view camera fitted to the bounding-box corners, then backed off 8 %."""
    import itertools

    import bpy
    from mathutils import Vector

    s = bpy.context.scene
    s.render.resolution_x, s.render.resolution_y = RES
    lo, hi = Vector(bbox[0]), Vector(bbox[1])
    c, r = (lo + hi) / 2, (hi - lo).length / 2
    cam = bpy.data.objects.new("zoo_cam", bpy.data.cameras.new("zoo_cam"))
    s.collection.objects.link(cam)
    cam.data.lens = 50.0
    d = Vector((1.0, -1.25, 0.75)).normalized()
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    corners = [v for xyz in itertools.product(*zip(lo, hi, strict=True)) for v in xyz]
    loc, _ = cam.camera_fit_coords(bpy.context.evaluated_depsgraph_get(), corners)
    cam.location = Vector(loc) + d * (Vector(loc) - c).length * 0.08  # back off along the view axis
    cam.data.clip_start, cam.data.clip_end = r * 0.01, r * 100
    s.camera = cam


def _render(path: pathlib.Path) -> None:
    import bpy

    s = bpy.context.scene
    s.render.resolution_x, s.render.resolution_y, s.render.resolution_percentage = (
        *RES,
        RENDER_SCALE,
    )
    s.render.film_transparent = True
    s.render.image_settings.file_format, s.render.image_settings.color_mode = "PNG", "RGBA"
    s.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def worker_render(out_dir: pathlib.Path, spec_path: pathlib.Path) -> None:
    import bpy

    spec = json.loads(spec_path.read_text())
    s = bpy.context.scene
    _frame(spec["bbox"])
    s.view_settings.view_transform = "Standard"

    # beauty: Eevee, a key sun, a soft fill and a grey studio world
    s.render.engine = "BLENDER_EEVEE"
    if hasattr(s, "eevee"):
        s.eevee.taa_render_samples = 64
    world = bpy.data.worlds.new("zoo_world")
    world.use_nodes = True
    bgnode = world.node_tree.nodes.get("Background")
    bgnode.inputs[0].default_value, bgnode.inputs[1].default_value = (0.55, 0.57, 0.6, 1), 0.8
    s.world = world
    for name, kind, energy, rot in (
        ("key", "SUN", 3.0, (0.8, 0.2, 0.6)),
        ("fill", "SUN", 0.8, (1.1, 0.0, -2.4)),
    ):
        light = bpy.data.objects.new(name, bpy.data.lights.new(name, kind))
        light.data.energy, light.rotation_euler = energy, rot
        s.collection.objects.link(light)
    _render(out_dir / "beauty.png")

    # parts: Workbench, flat studio light, one colour per part (`part_colours`)
    s.render.engine = "BLENDER_WORKBENCH"
    sh = s.display.shading
    sh.light, sh.color_type, sh.show_cavity = "STUDIO", "OBJECT", True
    objects = _mesh_objects()
    colours = part_colours([o.name for o in objects])
    for o in objects:
        o.color = (*colours[o.name], 1.0)
    _render(out_dir / "wireframe.png")

    # emissivity: FLAT light, every material's viewport colour set to its LWIR emissivity as grey
    sh.light, sh.color_type, sh.show_cavity = "FLAT", "MATERIAL", False
    s.view_settings.view_transform = "Standard"
    s.display_settings.display_device = "sRGB"
    by_src, by_obj, lo = spec["emissivity_by_source"], spec["emissivity_by_object"], spec["eps_lo"]
    grey = lambda e: (*([_srgb_to_linear((e - lo) / (1.0 - lo))] * 3), 1.0)  # noqa: E731
    for m in bpy.data.materials:
        e = by_src.get(m.name)
        if e is not None:
            m.diffuse_color = grey(e)
    for o in _mesh_objects():
        if o.name in by_obj:  # a part with an asserted material: one colour for all its slots
            m = bpy.data.materials.new(f"zoo_eps_{o.name}")
            m.diffuse_color = grey(by_obj[o.name])
            for slot in o.material_slots:
                slot.link, slot.material = "OBJECT", m
    _render(out_dir / "emissivity.png")


def _srgb_to_linear(v: float) -> float:
    """So that a stored emissivity e displays as sRGB grey level e after the view transform."""
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def run_worker(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stats", "render"])
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--spec", type=pathlib.Path)
    a = ap.parse_args(argv)
    if a.mode == "stats":
        worker_stats(a.out)
    else:
        worker_render(a.out, a.spec)
    return 0


# --------------------------------------------------------------------------------------------
# driver -- runs under the project interpreter
# --------------------------------------------------------------------------------------------
def _blender(blend: pathlib.Path, *args: str) -> None:
    exe = os.environ.get("BLENDER") or shutil.which("blender")
    if not exe:
        raise SystemExit("Blender not found; set $BLENDER")
    cmd = [
        exe,
        "-b",
        str(blend),
        "--python",
        str(pathlib.Path(__file__).resolve()),
        "--",
        "--worker",
        *args,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode or "Traceback" in r.stderr + r.stdout:
        raise SystemExit(f"Blender failed:\n{(r.stdout + r.stderr)[-3000:]}")


def _emissivity_table(
    asset: str, names: set[str], asserted: Mapping[str, str]
) -> tuple[dict, dict]:
    """Source material -> library id, and library id -> (8-14 um emissivity, zoo source)."""
    import numpy as np

    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import load_asset_mapping
    from irsim.radiometry.spectral_response import SpectralResponse

    mapping = load_asset_mapping(asset)
    lib = MaterialLibrary.load()
    band = SpectralResponse(
        wavelength_um=np.array(BAND_UM),
        response=np.array([1.0, 1.0]),
        source_path="<top-hat 8-14 um>",
        sha256="",
    )
    to_lib = {}
    for n in names:
        lib_id = mapping.lookup(n)
        if lib_id is None:
            raise SystemExit(
                f"source material {n!r} has no library mapping in configs/assets/{asset}.yaml"
            )
        to_lib[n] = lib_id
    props = {}
    for lib_id in set(to_lib.values()) | set(asserted.values()):
        m = lib[lib_id]
        props[lib_id] = (
            float(m.band_properties("lwir", response=band).emissivity),
            ZOO_SOURCE[m.spec.source],
        )
    return to_lib, props


def eps_scale_floor(values: Sequence[float]) -> float:
    """Bottom of the emissivity map's grey scale: 0.1 below the lowest value, floored to 0.1.

    Most models are all paint and plastic (0.85-0.95), which on a 0-1 scale is one flat white;
    the scale is stretched to the model's own range instead, and the legend says so.
    """
    return max(0.0, math.floor((min(values) - 0.1) * 10) / 10)


def crop_window(
    box: Sequence[int], size: Sequence[int], aspect: float, margin: float = 0.06
) -> tuple[int, ...]:
    """The smallest window of `aspect` (w/h) holding `box` plus `margin` on every side, centred
    on the box and kept inside the image where it fits."""
    x0, y0, x1, y1 = box
    w, h = (x1 - x0) * (1 + 2 * margin), (y1 - y0) * (1 + 2 * margin)
    w, h = max(w, h * aspect), max(h, w / aspect)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(cx - w / 2, 0), max(size[0] - w, 0))
    top = min(max(cy - h / 2, 0), max(size[1] - h, 0))
    return round(left), round(top), round(left + w), round(top + h)


def _finish_image(
    src: pathlib.Path, dst: pathlib.Path, window: Sequence[int], legend: float | None = None
) -> None:
    """Crop the transparent render to `window`, set it on the zoo's dark surface above the legend
    strip, and draw the emissivity scale (`legend` is its bottom value) if asked."""
    from PIL import Image, ImageDraw, ImageFont

    w, h = RES
    im = (
        Image.open(src).convert("RGBA").crop(tuple(window)).resize((w, h - LEGEND_H), Image.LANCZOS)
    )
    bg = Image.new("RGBA", RES, (*BG, 255))
    bg.alpha_composite(im)
    if legend is not None:
        d = ImageDraw.Draw(bg)
        try:
            font, um = ImageFont.truetype("DejaVuSans.ttf", 18), "µm"
        except OSError:  # Pillow's built-in face has no micro sign
            font, um = ImageFont.load_default(size=18), "um"
        x0, x1, y0 = int(w * 0.62), int(w * 0.95), h - 52
        for x in range(x0, x1):
            g = round(255 * (x - x0) / (x1 - x0 - 1))
            d.line([(x, y0), (x, y0 + 16)], fill=(g, g, g, 255))
        d.rectangle([x0 - 1, y0 - 1, x1, y0 + 17], outline=(120, 124, 130, 255))
        for v, x in ((legend, x0), ((legend + 1) / 2, (x0 + x1) // 2), (1.0, x1)):
            d.text((x, y0 + 22), f"{v:.2f}", fill=(200, 204, 210, 255), font=font, anchor="ma")
        d.text(
            (x0, y0 - 8),
            f"LWIR emissivity, 8-14 {um}",
            fill=(200, 204, 210, 255),
            font=font,
            anchor="ls",
        )
    bg.convert("RGB").save(dst)


def _zip(dst: pathlib.Path, entries: Sequence[pathlib.Path], base: pathlib.Path) -> None:
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for e in entries:
            for p in [e] if e.is_file() else sorted(q for q in e.rglob("*") if q.is_file()):
                z.write(p, p.relative_to(base))


def bundle(a: argparse.Namespace) -> pathlib.Path:
    import yaml

    asset = a.asset
    prov = yaml.safe_load((ROOT / "configs" / "assets" / f"{asset}.provenance.yaml").read_text())[
        "provenance"
    ]
    licence = zoo_licence(prov["license"]["slug"])
    if not prov.get("shareable", False):
        raise SystemExit(f"{asset} is marked not shareable in its provenance; not bundling")
    pdir = ROOT / "data" / "assets" / f"{asset}_parts"
    blend = pdir / f"{asset}_parts.blend"
    if not blend.is_file():
        raise SystemExit(
            f"{blend.relative_to(ROOT)} is missing: the zoo publishes the part-split export, "
            "which the ingest-asset skill writes -- finish that first"
        )
    assign_p = ROOT / "data" / "assets" / asset / f"{asset}.part_assignment.json"
    assign = json.loads(assign_p.read_text()) if assign_p.is_file() else {}
    asserted: dict[str, str] = assign.get("materials", {})

    mid = zoo_id(asset)
    out = (a.out or ROOT / "zoo_bundles") / mid
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "files").mkdir()

    with tempfile.TemporaryDirectory() as tmp:
        t = pathlib.Path(tmp)
        print(f"[{asset}] measuring parts in Blender")
        _blender(blend, "stats", "--out", str(t / "stats.json"))
        stats = json.loads((t / "stats.json").read_text())
        names = {n for o in stats["objects"].values() for n in o["areas"] if n}
        to_lib, props = _emissivity_table(asset, names, asserted)

        rows = []
        for obj, o in sorted(stats["objects"].items()):
            if obj in asserted:
                areas = {asserted[obj]: sum(o["areas"].values())}
            else:
                areas = {}
                for src, ar in o["areas"].items():
                    areas[to_lib[src]] = areas.get(to_lib[src], 0.0) + ar
            rows.append({"name": obj, "areas": areas})
        parts = []
        # one row per part (the owner's choice, 2026-10-04); --group-mirrors folds copies together
        table = (
            group_parts(rows)
            if a.group_mirrors
            else [{"name": part_label(r["name"]), "areas": r["areas"]} for r in rows]
        )
        for g in table:
            eps = area_weighted({k: props[k][0] for k in g["areas"]}, g["areas"])
            n = len(g["areas"])
            parts.append(
                {
                    "name": g["name"],
                    "material": material_text(g["areas"]),
                    "emissivity": round(eps, 2),
                    "source": weakest([props[k][1] for k in g["areas"]]),
                    "note": f"area-weighted over {n} materials" if n > 1 else "",
                }
            )

        print(f"[{asset}] rendering beauty, parts and emissivity views")
        eps_all = [props[k][0] for k in set(to_lib.values()) | set(asserted.values())]
        spec = {
            "bbox": stats["bbox"],
            "eps_lo": eps_scale_floor(eps_all),
            "emissivity_by_source": {s: props[lib_id][0] for s, lib_id in to_lib.items()},
            "emissivity_by_object": {
                o: props[m][0] for o, m in asserted.items() if o in stats["objects"]
            },
        }
        (t / "spec.json").write_text(json.dumps(spec))
        _blender(blend, "render", "--out", str(t), "--spec", str(t / "spec.json"))
        from PIL import Image

        with Image.open(t / "beauty.png") as im:
            box = im.getchannel("A").getbbox()
            window = crop_window(box, im.size, RES[0] / (RES[1] - LEGEND_H))
        _finish_image(t / "beauty.png", out / "images" / "beauty.png", window)
        _finish_image(t / "wireframe.png", out / "images" / "wireframe.png", window)
        _finish_image(
            t / "emissivity.png", out / "images" / "emissivity.png", window, legend=spec["eps_lo"]
        )
    if a.thermal:
        shutil.copy(a.thermal, out / "images" / f"thermal{pathlib.Path(a.thermal).suffix.lower()}")

    print(f"[{asset}] packing files")
    shutil.copy(blend, out / "files" / f"{mid}.blend")
    usd = pdir / f"{asset}_parts.usdc"
    if usd.is_file():
        _zip(
            out / "files" / f"{mid}.usdc.zip",
            [usd] + [p for p in [pdir / "textures"] if p.is_dir()],
            pdir,
        )
    fbx = pdir / f"{asset}_parts.fbx"
    if fbx.is_file():
        _zip(
            out / "files" / f"{mid}.fbx.zip",
            [fbx] + [p for p in [pdir / f"{asset}_parts.fbm"] if p.is_dir()],
            pdir,
        )
    attribution = ROOT / "3d_models" / asset / "ATTRIBUTION.md"
    if attribution.is_file():
        shutil.copy(attribution, out / "ATTRIBUTION.md")
    else:
        (out / "ATTRIBUTION.md").write_text(
            f"# {prov.get('real_name') or prov['title']}\n\n"
            f'Original: "{prov["title"]}" by {prov["author"]}, '
            f"{prov['model_url']}, licensed {licence}.\n\nChanges: {prov['modifications']}\n"
        )

    lo, hi = stats["bbox"]
    real = prov.get("real_name") or prov["title"]
    n_parts = len(stats["objects"])
    entry = {
        "id": mid,
        "name": real,
        "real_name": prov["title"] if prov["title"] != real else "",
        "category": a.category,
        "subcategory": a.subcategory or "",
        "summary": a.summary
        or (
            f"{real}, split into {n_parts} functional parts, each with its library material "
            f"and an area-weighted LWIR (8-14 µm) emissivity."
        ),
        "licence": licence,
        "author": prov["author"],
        "author_url": prov.get("author_url", ""),
        "source_url": prov["model_url"],
        "changes": prov["modifications"],
        "dimensions_m": [round(top - bot, 3) for bot, top in zip(lo, hi, strict=True)],
        "triangles": sum(o["triangles"] for o in stats["objects"].values()),
        "parts": parts,
        "captions": {
            "emissivity": "LWIR emissivity per surface, 8-14 µm "
            f"(black {spec['eps_lo']:.1f}, white 1.0)",
            "wireframe": f"Functional parts, one colour each ({n_parts} parts)",
        },
    }
    if a.featured:
        entry["featured"] = True
    (out / "entry.json").write_text(json.dumps(entry, indent=2, ensure_ascii=False) + "\n")
    print(f"[{asset}] bundle ready: {out}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" in argv:  # inside Blender: everything after `--` is ours
        argv = argv[argv.index("--") + 1 :]
    if argv[:1] == ["--worker"]:
        return run_worker(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("asset", help="library asset name, e.g. dji_mini_3_pro")
    ap.add_argument(
        "--category",
        required=True,
        help="zoo category id: drones, aircraft, ground, ships, animals, infrastructure, "
        "vegetation, environments, props",
    )
    ap.add_argument("--subcategory", default="")
    ap.add_argument(
        "--summary", default="", help="one-sentence card text (a generated one otherwise)"
    )
    ap.add_argument(
        "--thermal", type=pathlib.Path, help="a rendered LWIR still to show as the thermal image"
    )
    ap.add_argument("--featured", action="store_true", help="show it on the zoo's home page")
    ap.add_argument(
        "--group-mirrors",
        action="store_true",
        help="one table row per kind of part ('Propeller (x4)') instead of one per part",
    )
    ap.add_argument("--out", type=pathlib.Path, help="bundle root (default zoo_bundles/)")
    bundle(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
