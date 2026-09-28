#!/usr/bin/env python3
"""Report whether a model's curves are fine enough for the camera: facet angles per mesh.

    python scripts/mesh_facets.py model.glb
    python scripts/mesh_facets.py x.meshes.npz --range-m 2 --ifov-mrad 1.36

The measuring half of the `low2high` skill (roadmap ``AI.10``, ADR 0155), run before and after
every rung so an improvement is a number, not an impression. Reads a binary glTF (what
``tessellate_step.py`` and Sketchfab produce) or a prepared ``.meshes.npz`` archive. For the
render geometry, prepare that archive with ``prep_asset.py --emit-mesh --dissolve-deg 0``:
the default planar dissolve simplifies the *thermal* copy (ADR 0132) and is not what the camera
sees.

**Give it the scene.** With ``--range-m`` (the closest the scene flies the object) and
``--ifov-mrad`` (the sensor's pixel), an edge is faceted when its own silhouette error — the
distance its facets stand off the curve they approximate — exceeds half a pixel there. That is
the physical test, and it lets sub-pixel detail stay as coarse as it likes. Without the scene the
report falls back to a scale-free angle band (``--smooth-max-deg``, default
``irsim.io.mesh_facets.DEFAULT_SMOOTH_MAX_DEG``), which also flags small coarse details the
camera may never resolve — read it as a list of candidates, not a verdict. Exit status 1 if any
mesh is faceted.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Sequence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from irsim.io.mesh_facets import (  # noqa: E402
    DEFAULT_CREASE_MIN_DEG,
    DEFAULT_SMOOTH_MAX_DEG,
    facet_report,
    glb_meshes,
)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", type=pathlib.Path, help="a .glb or a prepared .meshes.npz")
    ap.add_argument("--smooth-max-deg", type=float, default=None)
    ap.add_argument("--crease-min-deg", type=float, default=DEFAULT_CREASE_MIN_DEG)
    ap.add_argument("--allowed-fraction", type=float, default=0.01)
    ap.add_argument("--range-m", type=float, default=None, help="closest range in the scene")
    ap.add_argument("--ifov-mrad", type=float, default=None, help="the sensor's pixel IFOV")
    args = ap.parse_args(argv)

    if (args.range_m is None) != (args.ifov_mrad is None):
        ap.error("--range-m and --ifov-mrad go together")
    max_sagitta = None
    smooth = args.smooth_max_deg if args.smooth_max_deg is not None else DEFAULT_SMOOTH_MAX_DEG
    smooth = min(smooth, args.crease_min_deg - 1e-9)
    if args.range_m is not None:
        max_sagitta = 0.5 * args.range_m * args.ifov_mrad * 1e-3
        rule = (
            f"faceted where an edge's silhouette error exceeds half a pixel: "
            f"{1e3 * max_sagitta:.3g} mm at {args.range_m} m, {args.ifov_mrad} mrad"
        )
    else:
        rule = f"no scene given: faceted between {smooth:.2f} and {args.crease_min_deg:g} deg"

    if args.model.suffix.lower() == ".glb":
        meshes = glb_meshes(args.model)
    elif args.model.name.endswith(".meshes.npz"):
        from irsim.io.assets import load_asset_meshes

        meshes = [(m.name, m.vertices_m, m.faces) for m in load_asset_meshes(args.model).values()]
    else:
        ap.error("expected a .glb or a .meshes.npz")

    print(rule)
    reports = [
        facet_report(
            name,
            verts,
            faces,
            smooth_max_deg=smooth,
            crease_min_deg=args.crease_min_deg,
            allowed_fraction=args.allowed_fraction,
            max_sagitta_m=max_sagitta,
        )
        for name, verts, faces in meshes
    ]
    for report in sorted(reports, key=lambda r: -r.faceted_fraction):
        print(report.render())
    faceted = [r for r in reports if not r.passed]
    total = sum(r.faces for r in reports)
    print(f"{len(reports)} meshes, {total:,} faces, {len(faceted)} faceted")
    return 1 if faceted else 0


if __name__ == "__main__":
    raise SystemExit(main())
