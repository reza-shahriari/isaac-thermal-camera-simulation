#!/usr/bin/env python3
"""WX.22: the cloudscape's cover pattern from above over three weather tiles, before and after.

    python scripts/cloud_field_repeat.py [--out outputs/cloud_field_repeat] [--genus cumulus]

weather-fx's weather map is one 32 km tile and a camera sees 80 km, so until WX.22 the same cover
pattern stood two or three times in a low frame. This draws the cloudy columns (vertical visible
optical depth past the cloudscape's own "cloudy" threshold) over a 96 km square, once with the
plain wrapped lookup and once with the baked aperiodic field (``Cloudscape.weather_at``), and
measures the masks' correlation at the old tile period two ways: column by column, and as the
cover pattern (block averages, 3 km).
A patch cloudscape's clouds are hashed per lattice cell, so column by column they never repeated;
the weather map's grouping of them did, and the block measure is the one that sees it. It also
reports the cloudy share of each, which the aperiodic blend must not move.
Engine-free: weather-fx's core is NumPy.

Writes ``cloud_field_repeat.png`` and ``cloud_field_repeat.json``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "outputs" / "cloud_field_repeat"))
    parser.add_argument("--genus", default="cumulus")
    parser.add_argument("--cover", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument(
        "--columns", type=int, default=384, help="columns per side (a multiple of 3: whole tiles)"
    )
    parser.add_argument(
        "--block-km", type=float, default=3.0, help="the cover pattern's scale: block averages"
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from irsim.atmosphere.weather_fx import ensure_weather_fx_on_path

    ensure_weather_fx_on_path()
    from weather_fx.core import cloudscape as cs

    patches = cs.load_patches()
    scape = cs.Cloudscape(
        cover=args.cover,
        base_m=1200.0,
        profile=cs.CLOUDSCAPE_TYPES[args.genus],
        seed=args.seed,
        patches=patches or None,
    )
    tile = scape.weather_tile_m
    span = 3.0 * tile
    axis = (np.arange(args.columns) + 0.5) / args.columns * span - 0.5 * span
    x, z = np.meshgrid(axis, axis, indexing="xy")
    bias = scape.patch_cover if scape.patches else scape.coverage_bias

    def cloudy() -> np.ndarray:
        od = scape._column_optical_depth(x, z, bias, 12)
        return np.asarray(od > scape._cloudy_optical_depth)

    after = cloudy()
    aperiodic = scape.weather_at
    scape.weather_at = lambda u, v: cs.sample_wrapped(scape.weather, u, v)  # type: ignore[method-assign]
    try:
        before = cloudy()
    finally:
        scape.weather_at = aperiodic  # type: ignore[method-assign]

    if args.columns % 3:
        raise SystemExit("--columns must be a multiple of 3, so one tile is a whole number of them")
    lag = args.columns // 3
    block = max(1, int(round(args.block_km * 1000.0 / (span / args.columns))))

    def correlation(field: np.ndarray) -> float:
        a, b = field[:, :-lag].ravel().astype(float), field[:, lag:].ravel().astype(float)
        return float(np.corrcoef(a, b)[0, 1])

    def blocks(mask: np.ndarray) -> np.ndarray:
        n = (mask.shape[0] // block) * block
        m = mask[:n, :n].astype(float)
        return m.reshape(n // block, block, n // block, block).mean(axis=(1, 3))

    def block_correlation(mask: np.ndarray) -> float:
        b = blocks(mask)
        k = lag // block
        return float(np.corrcoef(b[:, :-k].ravel(), b[:, k:].ravel())[0, 1])

    report = {
        "genus": args.genus,
        "cover_asked": args.cover,
        "weather_tile_km": tile / 1000.0,
        "span_km": span / 1000.0,
        "columns": args.columns,
        "cloudy_share": {"wrapped": float(before.mean()), "aperiodic": float(after.mean())},
        # Column by column: a patch cloudscape's clouds are hashed per lattice cell, so they never
        # repeated one by one; the weather map's grouping -- which cells hold cloud, the clear
        # lanes between -- did, and that is the block average at the cover pattern's scale.
        "correlation_at_tile_lag": {
            "wrapped": correlation(before),
            "aperiodic": correlation(after),
        },
        "block_km": block * span / args.columns / 1000.0,
        "cover_pattern_correlation_at_tile_lag": {
            "wrapped": block_correlation(before),
            "aperiodic": block_correlation(after),
        },
    }
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cloud_field_repeat.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    extent = (axis[0] / 1000, axis[-1] / 1000, axis[0] / 1000, axis[-1] / 1000)
    for ax, mask, name in ((axes[0], before, "wrapped"), (axes[1], after, "aperiodic")):
        ax.imshow(mask, origin="lower", cmap="Greys_r", extent=extent, interpolation="nearest")
        for k in (-1.0, 0.0, 1.0):  # the tile borders: whole multiples of the tile
            ax.axhline(k * tile / 1000, color="tab:red", lw=0.6, alpha=0.6)
            ax.axvline(k * tile / 1000, color="tab:red", lw=0.6, alpha=0.6)
        r = report["cover_pattern_correlation_at_tile_lag"][name]
        share = report["cloudy_share"][name]
        title = "before: one tile, repeated" if name == "wrapped" else "after: every tile its own"
        ax.set_title(
            f"{title}\ncover pattern one tile away: correlation {r:.2f}; cloudy {share:.1%}",
            fontsize=10,
        )
        ax.set_xlabel("km")
    axes[0].set_ylabel("km")
    fig.suptitle(
        f"{args.genus} cloudscape from above over {span / 1000:.0f} km; red lines: the "
        f"{tile / 1000:.0f} km weather tile",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "cloud_field_repeat.png", dpi=110)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
