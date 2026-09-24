#!/usr/bin/env python3
"""Audit the thermal-material coverage of an asset from an engine-free prim dump (M7.17).

    python scripts/audit_materials.py prims.json [--rules configs/materials/mapping.yaml]
                                                 [--asset phantom4]
                                                 [--threshold 0.95] [--band lwir]

``--asset`` layers a per-asset material map (``configs/assets/<name>.yaml``) above the global
globs, which is how an imported asset states its own truth without changing every other scene
(ADR 0128).

``prims.json`` is a list of records the Isaac adapter writes without any physics:

    [{"path": "/World/Car/Body", "material_name": "Car_Paint_Red",
      "semantic_class": "car_body", "override": null}, ...]

A USD asset can be audited directly, which boots Isaac Sim because ``pxr`` is a Kit extension and
is not importable outside a running Kit application on this build (ADR 0014 addendum):

    python.sh scripts/audit_materials.py --stage asset.usd [--root /World] [--dump-prims p.json]

Use ``--dump-prims`` once inside Kit and then audit that JSON anywhere: the audit itself is
engine-free and takes milliseconds, where the Kit boot costs 15-35 s.

Prints every prim that fell through to UNMAPPED, grouped and counted, plus the coverage; exits
non-zero below the threshold (CI gate, ADR 0047).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
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
    ap.add_argument(
        "prims", type=pathlib.Path, nargs="?", help="JSON list of prim records (or use --stage)"
    )
    ap.add_argument(
        "--stage", type=pathlib.Path, default=None, help="USD asset to walk (needs Isaac Sim)"
    )
    ap.add_argument("--root", default="/", help="prim path to walk under (with --stage)")
    ap.add_argument(
        "--dump-prims", type=pathlib.Path, default=None, help="write the records as JSON"
    )
    ap.add_argument(
        "--rules", type=pathlib.Path, default=None, help="mapping.yaml (default: configs/materials)"
    )
    ap.add_argument(
        "--asset",
        default=None,
        help="per-asset mapping: a name in configs/assets or a path to one (ADR 0128)",
    )
    ap.add_argument(
        "--threshold", type=float, default=None, help="coverage threshold (default: from the rules)"
    )
    ap.add_argument("--band", default="lwir", help="band whose packed table fixes the ids")
    ap.add_argument(
        "--materials",
        type=pathlib.Path,
        default=None,
        help="material directory (default: configs/materials)",
    )
    args = ap.parse_args(argv)
    if (args.prims is None) == (args.stage is None):
        ap.error("give exactly one of prims.json or --stage")

    library = MaterialLibrary.load(args.materials)
    names = MaterialTable.from_library(library, args.band).names
    rules = load_mapping_rules(args.rules, known_materials=library.names)
    asset = (
        None
        if args.asset is None
        else load_asset_mapping(args.asset, known_materials=library.names)
    )
    app = None
    if args.stage is not None:
        records, app = _open_stage_and_walk(args.stage, args.root)
    else:
        records = [
            PrimRecord.from_dict(d) for d in json.loads(args.prims.read_text(encoding="utf-8"))
        ]
    if args.dump_prims is not None:
        from irsim_isaac.pipeline.materials_usd import dump_prim_records

        print(f"wrote {dump_prim_records(records, args.dump_prims)}")
    # AT.18: the audit also fails when a *glob* reaches a mirror, so it needs the band
    # emissivity of every material the rules can name.
    emissivity = {
        name: float(library[name].band_properties(args.band).emissivity) for name in library.names
    }
    report = audit(
        records,
        MaterialResolver(rules, names, asset=asset),
        args.threshold,
        emissivity=emissivity,
    )
    print(report.render())
    status = 0 if report.passed else 1
    if app is not None:
        # SimulationApp.close() ends in os._exit (ADR 0014), so it must come after every print
        # and it carries the exit status itself -- the `return` below is never reached.
        app.close(exit_code=status)
    return status


def _open_stage_and_walk(stage_path: pathlib.Path, root: str) -> tuple[list, object]:  # type: ignore[type-arg]
    """Boot Kit, open the asset, walk it, and hand the app back so the caller closes it last.

    Separated so the JSON path never touches Isaac Sim, and so the report is printed before the
    app is closed: ``SimulationApp.close()`` does not return.
    """
    from isaacsim import SimulationApp

    from irsim_isaac.env import simulation_app_config

    app = SimulationApp(simulation_app_config())
    import omni.usd

    from irsim_isaac.pipeline.materials_usd import prim_records

    ctx = omni.usd.get_context()
    # open_stage returns a bare bool on this build (6.1.0-rc.26), not the (ok, error) pair the
    # older docs show; unpacking it raises inside Kit, which then reports exit status 0 anyway.
    result = ctx.open_stage(str(stage_path))
    ok = result[0] if isinstance(result, tuple) else bool(result)
    if not ok:
        print(f"could not open {stage_path}", file=sys.stderr)
        app.close(exit_code=2)
    return prim_records(ctx.get_stage(), root=root), app


if __name__ == "__main__":
    sys.exit(main())
