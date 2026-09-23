#!/usr/bin/env python3
"""Probe: does a prepared third-party asset load in Isaac Sim, and does its mapping resolve there?

    IRSIM_GPU=0 python.sh scripts/probe_isaac_asset.py --asset phantom4

ADR 0128 built the asset pipeline entirely on the CPU, through Blender's OpenUSD. Everything it
claims about the resulting stage -- that the prims are there, that `ComputeBoundMaterial` finds
the names the asset map keys on, that coverage is 100 % -- was measured with *Blender's* `pxr`.
This probe re-measures the same three things with **Kit's**, which is the reader that matters,
and is the in-engine half `AI.2` left open.

It renders nothing and asserts nothing about pixels. It opens the stage, walks it with the
production `prim_records`, resolves with the production resolver, and prints what it found.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", default="phantom4")
    ap.add_argument("--usd", type=pathlib.Path, default=None)
    ap.add_argument("--band", default="lwir")
    args = ap.parse_args(argv)

    usd = args.usd or (REPO / "data" / "assets" / args.asset / f"{args.asset}.usdc")
    if not usd.exists():
        print(
            f"no prepared asset at {usd}\n  run: python scripts/prep_asset.py --asset {args.asset}"
        )
        return 2

    from isaacsim import SimulationApp

    from irsim_isaac.env import simulation_app_config

    app = SimulationApp(simulation_app_config())
    status = 0
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        from irsim.materials.library import MaterialLibrary
        from irsim.materials.mapping import (
            MaterialResolver,
            audit,
            load_asset_mapping,
            load_mapping_rules,
        )
        from irsim.materials.table import MaterialTable
        from irsim_isaac.pipeline.materials_usd import prim_records

        ctx = omni.usd.get_context()
        result = ctx.open_stage(str(usd))
        if not (result[0] if isinstance(result, tuple) else bool(result)):
            print(f"Kit could NOT open {usd}", file=sys.stderr)
            app.close(exit_code=3)
        stage = ctx.get_stage()

        gprims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Gprim)]
        meshes = [p for p in gprims if p.IsA(UsdGeom.Mesh)]
        tris = 0
        for prim in meshes:
            counts = UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get()
            if counts:
                tris += int(sum(int(c) - 2 for c in counts))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"], useExtentsHint=False)
        size = (
            cache.ComputeWorldBound(stage.GetPrimAtPath("/World")).ComputeAlignedRange().GetSize()
        )

        print(f"\n--- Kit opened {usd.name} ---")
        print(f"  stage metersPerUnit : {UsdGeom.GetStageMetersPerUnit(stage)}")
        print(f"  stage upAxis        : {UsdGeom.GetStageUpAxis(stage)}")
        print(f"  gprims / meshes     : {len(gprims)} / {len(meshes)}")
        print(f"  triangles           : {tris:,}")
        print(f"  world bbox (m)      : {size[0]:.4f} x {size[1]:.4f} x {size[2]:.4f}")

        records = prim_records(stage, root="/")
        named = sum(1 for r in records if r.material_name)
        print(f"\n  prim_records        : {len(records)} ({named} with a bound material name)")

        library = MaterialLibrary.load()
        names = MaterialTable.from_library(library, args.band).names
        rules = load_mapping_rules(known_materials=library.names)
        asset = load_asset_mapping(args.asset, known_materials=library.names)

        bare = audit(records, MaterialResolver(rules, names))
        with_asset = audit(records, MaterialResolver(rules, names, asset=asset))
        print("\n  global rules only:")
        print("    " + bare.render().replace("\n", "\n    "))
        print(f"\n  with configs/assets/{args.asset}.yaml:")
        print("    " + with_asset.render().replace("\n", "\n    "))
        status = 0 if with_asset.passed else 1
    finally:
        app.close(exit_code=status)
    return status


if __name__ == "__main__":
    sys.exit(main())
