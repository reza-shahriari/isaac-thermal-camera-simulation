#!/usr/bin/env python3
"""Plot the M6.13 facet scene's 24 h diurnal curves, for the manual half of the Tier 3 pass.

    python scripts/validate_thermal_diurnal.py --out outputs/thermal_diurnal.png

`tests/unit/test_tier3_thermal.py` asserts everything that can be asserted. This produces the
picture for everything that cannot: whether the curves have the *shape* a thermal engineer expects,
whether the ordering near dawn is a smooth convergence or a numerical artefact, and whether the
shaded and sunlit twins separate the way shadow actually behaves. The Tier 3 checklist under
`docs/validation/` lists what to look for.

Writes a PNG if matplotlib is importable and a CSV either way, because the CSV is the thing that can
be diffed between runs and the plot is the thing that can be looked at.

Every curve here is an **absolute** temperature, so the run also prints what fraction of the
emissivity behind it is extrapolated rather than measured (`AT.7`): at the 300 K these scenes
evaluate eps_hemi at, 0.607 of Planck's weight lies outside the configured bands, four fifths of
it in the tail beyond 13.5 um. A curve that sits a kelvin off is not necessarily a solver error.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

SCENE = "configs/scenes/thermal_facet_scene.yaml"
LOCAL_OFFSET_H = 2.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=SCENE)
    parser.add_argument("--out", default="outputs/thermal_diurnal.png")
    parser.add_argument("--minutes", type=int, default=5)
    args = parser.parse_args(argv)

    from irsim.scene import Scene

    scene = Scene.from_file(args.scene)
    scene.thermal.advance_to(scene.t0_s + 24 * 3600.0)
    hours = np.arange(0.0, 24.0, args.minutes / 60.0)
    temps = np.array(
        [scene.thermal.temperature_at(scene.t0_s + h * 3600.0) for h in hours], dtype=np.float64
    )
    names = list(scene.thermal_surfaces)
    air = np.array(
        [float(np.asarray(scene.weather.at(scene.t0_s + h * 3600.0).t_air_k)) for h in hours]
    )
    local = (hours + LOCAL_OFFSET_H) % 24.0

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    csv = out.with_suffix(".csv")
    header = "hour_utc,hour_local,t_air_c," + ",".join(f"{n}_c" for n in names)
    rows = np.column_stack([hours, local, air - 273.15, temps - 273.15])
    np.savetxt(csv, rows, delimiter=",", header=header, comments="", fmt="%.4f")
    print(f"wrote {csv}")

    for i, name in enumerate(names):
        column = temps[:, i]
        peak = int(np.argmax(column))
        print(
            f"  {name:14s} max {column[peak] - 273.15:6.2f} C at {local[peak]:5.2f} local, "
            f"swing {float(np.ptp(column)):5.2f} K"
        )
    keep = [i for i, n in enumerate(names) if n not in ("hood_moving", "asphalt_shade")]
    spread = temps[:, keep].std(axis=1)
    print(
        f"  scene spread: max {spread.max():.2f} K at {local[int(np.argmax(spread))]:.2f} local, "
        f"min {spread.min():.2f} K at {local[int(np.argmin(spread))]:.2f} local"
    )
    # AT.7: this report quotes absolute surface temperatures, so it carries how much of the
    # emissivity that produced them is an extension of the nearest band rather than data.
    print(f"  {scene.emissivity_extrapolation().summary()}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; CSV written, plot skipped")
        return 0

    order = np.argsort(local)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 9), sharex=True, height_ratios=[3, 1])
    top.plot(local[order], air[order] - 273.15, "k--", lw=1.5, label="air")
    for i, name in enumerate(names):
        top.plot(local[order], temps[order, i] - 273.15, lw=1.2, label=name)
    top.set_ylabel("surface temperature (°C)")
    top.legend(ncol=3, fontsize=8)
    top.grid(alpha=0.3)
    top.set_title(f"{scene.spec.name}: 24 h from {scene.spec.start_utc:%Y-%m-%d %H:%MZ}")
    bottom.plot(local[order], spread[order], lw=1.5, color="crimson")
    bottom.set_ylabel("scene spread (K)")
    bottom.set_xlabel("local solar time (h)")
    bottom.grid(alpha=0.3)
    bottom.set_xlim(0, 24)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
