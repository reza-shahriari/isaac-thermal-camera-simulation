#!/usr/bin/env python3
"""Check that every contact an asset authors joins two parts that actually touch.

    python scripts/check_asset_contacts.py dji_inspire_3
    python scripts/check_asset_contacts.py dji_matrice_300_rtk --cell-m 0.02 --gap-m 0.03
    python scripts/check_asset_contacts.py phantom4_parts --archive phantom4_parts

A contact (`ContactSpec`, AI.11) gives two parts a conductance ``h_c · A``. The solver spreads it
over the cell pairs within a gap of each other
(:func:`irsim.thermal.coupling.proximity_contactor`, TC.16), and a scene refuses to build when
there are none. This tool runs that test when the parts are authored, not when a scene first
binds the asset. It reads the part-split archive a scene reads (``--archive``: by default
``<asset>_parts``, which ``prep_asset.py --emit-parts`` writes, else the asset's own), so run it
after prep. It coarsens every part at ``--cell-m`` (ADR 0165, long faces bisected first per
ADR 0176), and reports for each contact between two shown parts:
* the nearest pair of cell centres;
* how many pairs fall within ``--gap-m``.

Contacts with a hidden part are placed by their box and are skipped. The defaults are a solved
scene's: 2 cm cells and a 3 cm gap (`configs/scenes/phantom4_solved.yaml`). Exit status 1 if
any contact finds no pair.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def main(argv: Sequence[str] | None = None) -> int:
    import numpy as np
    from scipy.spatial import cKDTree

    from irsim.io.assets import load_asset_meshes
    from irsim.materials.mapping import load_asset_mapping
    from irsim.thermal.mesh_coarse import coarsen_mesh

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("asset", help="asset name, as in configs/assets/<name>.yaml")
    parser.add_argument("--cell-m", type=float, default=0.02, help="coarse cell size (m)")
    parser.add_argument("--gap-m", type=float, default=0.03, help="contact gap (m)")
    parser.add_argument("--archive", help="part-split archive (default: <asset>_parts if present)")
    args = parser.parse_args(argv)

    parts = load_asset_mapping(args.asset).parts
    if parts is None or not parts.contacts:
        print(f"{args.asset}: no contacts authored")
        return 0
    archive = args.archive
    if archive is None:
        split = f"{args.asset}_parts"
        archive = split if (REPO_ROOT / "data" / "assets" / split).is_dir() else args.asset
    meshes = load_asset_meshes(archive).meshes
    missing = sorted({p.name for p in parts.parts} - set(meshes))
    if missing:
        print(f"{archive} has no mesh for parts {missing}: re-run prep_asset.py --emit-parts")
        return 1
    hidden = {h.name for h in parts.hidden_parts}
    centres: dict[str, np.ndarray] = {}

    def cells(name: str) -> np.ndarray:
        if name not in centres:
            m = meshes[name]
            centres[name] = np.asarray(
                coarsen_mesh(m.vertices_m, m.faces, args.cell_m).cell_centres()
            )
        return centres[name]

    print(
        f"{args.asset} ({archive}): {len(meshes)} parts, {args.cell_m} m cells, {args.gap_m} m gap"
    )
    refused = 0
    for c in parts.contacts:
        if c.a in hidden or c.b in hidden:
            continue
        a, b = cells(c.a), cells(c.b)
        nearest = float(cKDTree(b).query(a)[0].min())
        pairs = int(cKDTree(a).count_neighbors(cKDTree(b), args.gap_m))
        refused += pairs == 0
        print(
            f"  {'ok ' if pairs else 'NO '} {c.a:24s} {c.b:24s} nearest {nearest * 100:5.1f} cm"
            f"  {pairs:5d} pairs within the gap  {c.area_m2 * 1e4:5.1f} cm2"
        )
    if refused:
        print(f"{refused} contact(s) join parts that do not touch: a scene would refuse them")
        return 1
    print("every contact touches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
