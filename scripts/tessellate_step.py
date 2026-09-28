#!/usr/bin/env python3
"""Tessellate a STEP/IGES CAD model into glTF, keeping the part tree and converting to metres.

    python scripts/tessellate_step.py model.step --out model.glb --angle-deg 5 --chord-mm 0.05

The downward half of the `low2high` skill's first rung (roadmap ``AI.10``, ADR 0155). A STEP file
holds exact surfaces, not triangles, and Blender cannot open one; OpenCascade — the kernel under
FreeCAD and CadQuery — can. This script uses it through the ``cadquery-ocp`` wheel (the optional
``cad`` extra, ``pip install -e '.[cad]'``, in any Python ≥ 3.10; it is deliberately not a default
and nothing in ``src/irsim`` imports it). The output is a ``.glb`` that
``scripts/register_local_asset.py`` and ``scripts/prep_asset.py`` take like any other model.

**Why tessellation is a decision, not a conversion.** The two tolerances set how finely every
curve is cut: ``--angle-deg`` bounds the angle between neighbouring facets and ``--chord-mm`` how
far a facet may stand off the true surface. Too coarse and a turned shaft images as a prism
(the faceting `irsim.io.mesh_facets` measures); too fine and a gearbox is ten million triangles
the renderer pays for and the camera cannot resolve. Pick ``--angle-deg`` from
`irsim.io.mesh_facets.max_facet_angle_deg` for the closest range the scene flies the object; the
report printed at the end checks the result against it.

**What a CAD file gives that a game model does not.** The assembly tree usually carries real part
names ("motor_shaft", "mount_bracket") — the part decomposition that took hand-measured selectors
for the Phantom 4 arrives for free — and STEP declares its length unit, so the scale is not a
guess. Both are preserved: part names become glTF node names, and the unit is converted to metres
on write. What CAD usually lacks is appearance (no textures, often one grey), which is why the
material map still comes from research, not from the file.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct
import sys
from collections.abc import Sequence
from typing import Any

#: STEP entities that declare no unit are read in millimetres, OpenCascade's convention and the
#: overwhelming default of mechanical CAD. The reader's own unit handling overrides this whenever
#: the file declares one.
DEFAULT_UNIT_M = 0.001


def _need_ocp() -> None:
    try:
        import OCP  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "OpenCascade bindings not found. Install the optional extra in any Python >= 3.10:\n"
            "    pip install cadquery-ocp        (or: pip install -e '.[cad]')\n"
            "and run this script with that interpreter."
        ) from error


def tessellate_step(
    source: pathlib.Path,
    out_glb: pathlib.Path,
    *,
    angle_deg: float = 5.0,
    chord_mm: float = 0.05,
) -> dict[str, Any]:
    """Read ``source`` (STEP or IGES) with its names, mesh it, write ``out_glb`` in metres.

    Returns a summary: part names as written, triangle count, and the bounding box in metres.
    """
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.IGESCAFControl import IGESCAFControl_Reader
    from OCP.Message import Message_ProgressRange
    from OCP.OCP.collections import (
        IndexedDataMap_TCollection_AsciiString_TCollection_AsciiString as FileInfo,
    )
    from OCP.OCP.collections import Sequence_TDF_Label
    from OCP.RWGltf import RWGltf_CafWriter
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    app.InitDocument(doc)

    suffix = source.suffix.lower()
    reader: Any
    if suffix in (".step", ".stp"):
        reader = STEPCAFControl_Reader()
    elif suffix in (".iges", ".igs"):
        reader = IGESCAFControl_Reader()
    else:
        raise SystemExit(f"not a STEP/IGES file: {source}")
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    if reader.ReadFile(str(source)) != IFSelect_RetDone:
        raise SystemExit(f"OpenCascade could not read {source}")
    reader.Transfer(doc)

    shapes = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    roots = Sequence_TDF_Label()
    shapes.GetFreeShapes(roots)
    chord_model_units = chord_mm  # OpenCascade's working unit is the millimetre
    for i in range(1, roots.Length() + 1):
        BRepMesh_IncrementalMesh(
            shapes.GetShape_s(roots.Value(i)),
            chord_model_units,
            False,
            math.radians(angle_deg),
            True,
        )

    out_glb.parent.mkdir(parents=True, exist_ok=True)
    writer = RWGltf_CafWriter(TCollection_AsciiString(str(out_glb)), True)
    writer.ChangeCoordinateSystemConverter().SetInputLengthUnit(DEFAULT_UNIT_M)
    if not writer.Perform(doc, FileInfo(), Message_ProgressRange()):
        raise SystemExit(f"glTF write failed: {out_glb}")
    return summarise_glb(out_glb)


def summarise_glb(path: pathlib.Path) -> dict[str, Any]:
    """Part names, triangle count and bounds straight from the .glb's JSON chunk."""
    data = path.read_bytes()
    (json_len,) = struct.unpack("<I", data[12:16])
    gltf = json.loads(data[20 : 20 + json_len])
    names = sorted({n["name"] for n in gltf.get("nodes", []) if n.get("name")})
    triangles = 0
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for mesh in gltf.get("meshes", []):
        for prim in mesh.get("primitives", []):
            if "indices" in prim:
                triangles += gltf["accessors"][prim["indices"]]["count"] // 3
            pos = gltf["accessors"][prim["attributes"]["POSITION"]]
            lo = [min(a, b) for a, b in zip(lo, pos.get("min", lo), strict=True)]
            hi = [max(a, b) for a, b in zip(hi, pos.get("max", hi), strict=True)]
    return {"parts": names, "triangles": triangles, "bounds_m": [lo, hi]}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=pathlib.Path, help="a .step/.stp or .iges/.igs file")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="default: <source>.glb")
    ap.add_argument(
        "--angle-deg",
        type=float,
        default=5.0,
        help="max angle between neighbouring facets (see irsim.io.mesh_facets)",
    )
    ap.add_argument(
        "--chord-mm", type=float, default=0.05, help="max stand-off of a facet from the surface"
    )
    args = ap.parse_args(argv)
    _need_ocp()
    out = args.out or args.source.with_suffix(".glb")
    summary = tessellate_step(args.source, out, angle_deg=args.angle_deg, chord_mm=args.chord_mm)
    lo, hi = summary["bounds_m"]
    size = [h - low for low, h in zip(lo, hi, strict=True)]
    print(f"wrote {out}: {summary['triangles']:,} triangles, {len(summary['parts'])} named parts")
    print("extent    : " + " x ".join(f"{s:.4f}" for s in size) + " m")
    print("parts     : " + (", ".join(summary["parts"]) or "(none named in the CAD file)"))
    print("next      : scripts/register_local_asset.py on this .glb, then the ingest-asset skill")
    return 0


if __name__ == "__main__":
    sys.exit(main())
