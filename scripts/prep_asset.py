#!/usr/bin/env python3
"""Prepare a third-party 3D asset for the simulator: rescale, convert to USD, audit its materials.

    python scripts/prep_asset.py --asset phantom4
    python scripts/prep_asset.py --asset phantom4 --source 3d_models/phantom4.fbx \
        --out-dir data/assets

Everything here runs on the **CPU**. Blender is the USD toolchain -- it ships a complete ``pxr``
(OpenUSD 26.03 on the 5.2 LTS build) -- so no part of this boots Kit, starts a renderer or touches
CUDA. That is the point: the inspect -> map -> audit loop is the loop an operator runs dozens of
times per asset, and paying a 15-35 s Kit boot for it (ADR 0014) made it a loop nobody would run.

**The file runs twice.** Invoked normally it is the *driver*: it reads the asset config (needing
pydantic and PyYAML, which Blender's interpreter does not have), then re-invokes itself inside
Blender for the geometry work, then audits the result. Invoked with ``bpy`` importable it is the
*worker*. The split exists because neither interpreter has the other's dependencies, not because
the work is naturally two pieces.

What the worker does, in order:

1. **Import** the source file. FBX goes through ``wm.fbx_import`` -- the C++/ufbx importer that is
   the default from Blender 5.0 -- which preserves material names and per-face material indices.
2. **Rescale** by the asset's ``scale_to_metres``. An FBX carries no unit, so this is the one fact
   about the asset that cannot be recovered from the file and must be authored. Unapplied, a
   Phantom 4 enters the stage 60 m wide and still renders a plausible image.
3. **Export** USD with ``UsdPreviewSurface`` (Isaac Sim supports it; Blender has had no MDL
   exporter since the Omniverse connector was discontinued).
4. **Walk** the stage into engine-free prim records -- the same ``{path, material_name,
   semantic_class, override}`` shape ``irsim_isaac.pipeline.materials_usd`` produces inside Kit,
   so the JSON is interchangeable and `irsim` still never imports USD.

The worker also refuses a mesh that carries ``materialBind`` subsets *and* a direct mesh-level
binding without reporting it: Blender binds the first material slot to the mesh as well as making
subsets (a documented Hydra workaround), so reading the mesh binding alone would map a whole
multi-material building to slot 0 -- silently, at 100 % coverage. See ADR 0128.

docs/physics-model.md §13.3; ADR 0047 (precedence, the UNMAPPED sentinel, the coverage gate),
ADR 0128 (per-asset mapping, the Blender prep path).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from collections.abc import Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Source formats the worker knows how to import.
IMPORTERS: dict[str, str] = {
    ".fbx": "fbx",
    ".obj": "obj",
    ".gltf": "gltf",
    ".glb": "gltf",
    ".usd": "usd",
    ".usda": "usd",
    ".usdc": "usd",
}


# --------------------------------------------------------------------------------------------
# worker -- runs inside Blender
# --------------------------------------------------------------------------------------------


def _import_source(path: pathlib.Path) -> str:
    import bpy

    kind = IMPORTERS[path.suffix.lower()]
    if kind == "fbx":
        # The C++/ufbx importer (default from Blender 5.0). `use_existing` keeps two assets that
        # share a material name from becoming `Foo` and `Foo.001`, which would defeat the map.
        try:
            bpy.ops.wm.fbx_import(filepath=str(path))
            return "wm.fbx_import (C++/ufbx)"
        except AttributeError:
            bpy.ops.import_scene.fbx(filepath=str(path))
            return "import_scene.fbx (legacy Python)"
    if kind == "obj":
        bpy.ops.wm.obj_import(filepath=str(path))
        return "wm.obj_import"
    if kind == "gltf":
        bpy.ops.import_scene.gltf(filepath=str(path))
        return "import_scene.gltf"
    bpy.ops.wm.usd_import(filepath=str(path))
    return "wm.usd_import"


def _rescale(factor: float) -> None:
    """Scale every root object about the world origin, then apply it into the mesh data.

    Applied rather than left on the transform so the exported USD carries metres in its point
    data. A scale left on an Xform survives export but is then one more thing a downstream reader
    has to honour, and the mesh bridge (ADR 0110) queries points, not transforms.
    """
    import bpy

    if factor == 1.0:
        return
    for obj in bpy.data.objects:
        if obj.parent is None:
            obj.scale = tuple(s * factor for s in obj.scale)
            obj.location = tuple(c * factor for c in obj.location)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=True)
    bpy.ops.object.select_all(action="DESELECT")


def _prim_records(usd_path: pathlib.Path) -> list[dict[str, object]]:
    """Walk the exported stage into engine-free records, subsets included.

    Mirrors ``irsim_isaac.pipeline.materials_usd.prim_records``: visible ``UsdGeom.Gprim`` only,
    the bound material's *name* rather than its path (names survive re-parenting; paths do not),
    and the ``thermal:material`` override read straight off the prim.
    """
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"could not open the stage this script just wrote: {usd_path}")
    records: list[dict[str, object]] = []
    shadowed: list[str] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Gprim):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable and imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        binding = UsdShade.MaterialBindingAPI(prim)
        subsets = binding.GetMaterialBindSubsets()
        if subsets:
            # The multi-material case. Each subset is its own record; the mesh-level binding is
            # deliberately ignored, and reported, because Blender writes slot 0 there as well.
            if binding.ComputeBoundMaterial()[0]:
                shadowed.append(str(prim.GetPath()))
            for subset in subsets:
                material = UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
                records.append(_record(subset.GetPrim(), material))
            continue
        records.append(_record(prim, binding.ComputeBoundMaterial()[0]))
    if shadowed:
        print(
            f"note: {len(shadowed)} mesh(es) carry materialBind subsets AND a direct binding; "
            "the subsets win (ADR 0128). First: " + shadowed[0]
        )
    return records


def _record(prim: object, material: object) -> dict[str, object]:
    name = None
    if material is not None and material.GetPrim().IsValid():  # type: ignore[attr-defined]
        name = str(material.GetPrim().GetName())  # type: ignore[attr-defined]
    override = None
    attr = prim.GetAttribute("thermal:material")  # type: ignore[attr-defined]
    if attr and attr.IsValid():
        override = attr.Get()
    return {
        "path": str(prim.GetPath()),  # type: ignore[attr-defined]
        "material_name": name,
        "semantic_class": None,
        "override": None if override is None else str(override),
    }


def run_worker(argv: Sequence[str]) -> int:
    import bpy

    ap = argparse.ArgumentParser(prog="prep_asset (worker)")
    ap.add_argument("--source", type=pathlib.Path, required=True)
    ap.add_argument("--out-usd", type=pathlib.Path, required=True)
    ap.add_argument("--out-prims", type=pathlib.Path, required=True)
    ap.add_argument("--scale", type=float, required=True)
    args = ap.parse_args(list(argv))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    how = _import_source(args.source)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in meshes)
    print(f"imported {args.source.name} via {how}: {len(meshes)} meshes, {tris:,} triangles")

    _rescale(args.scale)
    args.out_usd.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.usd_export(
        filepath=str(args.out_usd),
        export_materials=True,
        generate_preview_surface=True,
        root_prim_path="/World",
    )
    records = _prim_records(args.out_usd)
    args.out_prims.parent.mkdir(parents=True, exist_ok=True)
    args.out_prims.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print(f"wrote {args.out_usd}")
    print(f"wrote {args.out_prims} ({len(records)} prim records)")
    return 0


# --------------------------------------------------------------------------------------------
# driver -- runs under the project interpreter
# --------------------------------------------------------------------------------------------


def blender_command(
    blender: str,
    source: pathlib.Path,
    out_usd: pathlib.Path,
    out_prims: pathlib.Path,
    scale: float,
) -> list[str]:
    """The exact argv the driver runs. Split out so a test can check it without Blender."""
    return [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(pathlib.Path(__file__).resolve()),
        "--",
        "--source",
        str(source),
        "--out-usd",
        str(out_usd),
        "--out-prims",
        str(out_prims),
        "--scale",
        repr(float(scale)),
    ]


def run_driver(argv: Sequence[str] | None = None) -> int:
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import (
        MaterialResolver,
        PrimRecord,
        audit,
        load_asset_mapping,
        load_mapping_rules,
    )
    from irsim.materials.table import MaterialTable

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--asset", required=True, help="name in configs/assets, or a path to one")
    ap.add_argument(
        "--source", type=pathlib.Path, default=None, help="override the asset's source_file"
    )
    ap.add_argument(
        "--out-dir", type=pathlib.Path, default=None, help="default: data/assets/<name>"
    )
    ap.add_argument("--blender", default="blender", help="the Blender executable")
    ap.add_argument("--band", default="lwir", help="band whose packed table fixes the ids")
    ap.add_argument("--threshold", type=float, default=None, help="coverage gate override")
    ap.add_argument(
        "--skip-convert",
        action="store_true",
        help="audit an existing prims.json in --out-dir without re-running Blender",
    )
    args = ap.parse_args(argv)

    library = MaterialLibrary.load()
    asset = load_asset_mapping(args.asset, known_materials=library.names)
    source = args.source or (None if asset.source_file is None else REPO_ROOT / asset.source_file)
    out_dir = args.out_dir or (REPO_ROOT / "data" / "assets" / asset.name)
    out_usd = out_dir / f"{asset.name}.usdc"
    out_prims = out_dir / f"{asset.name}.prims.json"

    if not args.skip_convert:
        if source is None:
            ap.error("no --source and the asset config has no source_file")
        if not source.exists():
            ap.error(f"source not found: {source}")
        if source.suffix.lower() not in IMPORTERS:
            ap.error(f"unsupported source {source.suffix!r}; know {sorted(IMPORTERS)}")
        cmd = blender_command(args.blender, source, out_usd, out_prims, asset.scale_to_metres)
        print(f"$ {' '.join(cmd)}")
        try:
            done = subprocess.run(cmd, check=False)
        except FileNotFoundError:
            ap.error(f"{args.blender!r} not found; pass --blender /path/to/blender")
        if done.returncode != 0:
            print(f"blender exited {done.returncode}", file=sys.stderr)
            return done.returncode
    if not out_prims.exists():
        print(f"no prim records at {out_prims}", file=sys.stderr)
        return 2

    names = MaterialTable.from_library(library, args.band).names
    rules = load_mapping_rules(known_materials=library.names)
    records = [PrimRecord.from_dict(d) for d in json.loads(out_prims.read_text(encoding="utf-8"))]
    report = audit(records, MaterialResolver(rules, names, asset=asset), args.threshold)
    print(report.render())
    return 0 if report.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        import bpy  # noqa: F401
    except ImportError:
        return run_driver(argv)
    inner = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return run_worker(inner)


if __name__ == "__main__":
    sys.exit(main())
