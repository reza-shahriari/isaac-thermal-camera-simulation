#!/usr/bin/env python3
"""Tier 4 sky-flat bench: the radial bowl of a featureless sky, rendered and real (SC.20).

A clear sky gives the AGC nothing to stretch but the camera, so the smooth radial shading an
uncooled core leaves -- its housing seen through the field (§8.2, ADR 0145) and the housing's drift
since the last shutter event (§11.2, ADR 0148) -- fills the display. This script renders
`configs/scenes/sky_only.yaml` through a sensor with its full chain and flat field, takes the
shutter at t = 0, lets the housing drift by ``--housing-drift-k``, renders again, and measures the
bowl with :func:`irsim.validation.radial.radial_fit`.

Rendered frames are measured in the camera's own flat-fielded counts, before the AGC: plateau
equalisation is non-linear, and a sky seen across a 30° field has a curvature of its own that the
equalisation would mix with the camera's. The frame at the shutter event is the sky alone (the
shutter has just flattened the camera); the difference ``drift_bowl`` = after − at shutter is the
camera alone, and that is what the prediction is checked on. Any ``--real`` frames (a public
clear-sky frame, say) are measured the same way, so sign, radial share and profile shape can be
compared directly.

What §8.2 and §11.2 predict, and the report checks: right after the shutter the frame has no
bowl; a housing that has cooled since then leaves a bright centre (k < 0) on a white-hot display, a
housing that has warmed a dark one.

    python scripts/validate_sky_flat.py                               # rendered only
    python scripts/validate_sky_flat.py --real path/to/clear_sky.png  # and a real frame
    python scripts/validate_sky_flat.py --housing-drift-k 1.0         # a warming housing

Writes ``report.json`` and ``profiles.png`` under ``--out`` (default
``outputs/validation/sky_flat``). The real frames are read, never copied: a published report
carries their numbers and their profile, not the pixels.

Needs no GPU and no Isaac Sim: this is the engine-free CPU pipeline (ADR 0018).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SENSOR = REPO / "configs" / "sensors" / "flir_boson_640_lwir.yaml"
DEFAULT_SCENE = REPO / "configs" / "scenes" / "sky_only.yaml"
DEFAULT_OUT = REPO / "outputs" / "validation" / "sky_flat"


def _read_image(path: pathlib.Path) -> np.ndarray:
    """A real frame as a float grey plane. PIL lives outside the engine-free core."""
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float64)


def _fit_dict(fit: Any) -> dict[str, Any]:
    return {
        "k": fit.k,
        "sign": fit.sign,
        "radial_share": round(fit.radial_share, 4),
        "relative_depth": round(fit.relative_depth, 4),
        "monotonic": fit.monotonic,
        "outliers": fit.outliers,
        "profile_r": [round(float(v), 4) for v in fit.profile_r],
        "profile": [float(v) for v in fit.profile],
    }


def render_pair(
    sensor_path: pathlib.Path,
    scene_path: pathlib.Path,
    boresight_deg: float,
    drift_k: float,
    dt_s: float,
) -> dict[str, np.ndarray]:
    """Flat-fielded counts and display grey at the shutter event and ``dt_s`` later."""
    from irsim.config.loader import load_sensor_config
    from irsim.config.sensor import SensorConfig
    from irsim.materials import MaterialLibrary, MaterialTable
    from irsim.pipeline import PipelineConfig, PipelineState, attach_sensor_chain, run_frame
    from irsim.radiometry.lut import BandLUT
    from irsim.radiometry.spectral_response import load_spectral_response
    from irsim.scene import Scene
    from irsim.validation.aerial_scene import build_aerial_gbuffer

    cfg = load_sensor_config(sensor_path)
    doc = cfg.model_dump(mode="json")
    # The housing is the air, so the drift since the shutter is exactly what we set; no noise, so
    # the bowl is not hidden under a frame of temporal noise the AGC would also stretch.
    doc["sensor"]["optics"]["housing_temp_mode"] = "ambient"
    sensor = SensorConfig.model_validate(doc)
    band = sensor.sensor.band.band_id
    lut = BandLUT.build(load_spectral_response(sensor.sensor.band.spectral_response), n=4001)
    scene = Scene.from_file(scene_path, {band: lut}, quantity=sensor.sensor.quantity)
    materials = MaterialTable.from_library(MaterialLibrary.load(), band)
    config = PipelineConfig.from_sensor(
        sensor,
        materials,
        lut,
        sky=scene.sky_models[band],
        atmosphere=scene.layered,
        noise_enabled=False,
        flat_field_enabled=True,
    )
    t_cal = config.t_housing_cal_k
    config = attach_sensor_chain(
        config,
        ambient_provider=lambda t: t_cal + (drift_k if t > 0.0 else 0.0),
        t0_s=scene.t0_s,
        defects_enabled=False,
        residual_enabled=False,
    )
    built = build_aerial_gbuffer(
        sensor.sensor,
        scene.sky_models[band],
        materials,
        t_s=scene.t0_s,
        boresight_elevation_deg=boresight_deg,
        supersample=sensor.sensor.optics.supersample_factor,
    )
    if not built.sky_mask.all():
        raise SystemExit(f"the horizon is in frame at {boresight_deg} deg; raise --boresight")
    state = PipelineState(t_s=0.0)
    frames = {}
    for name, t in (("at_shutter", 0.0), ("after_drift", dt_s)):
        state.t_s = t
        out = run_frame(built.planes, config, state)
        assert out.display8 is not None and out.dn16 is not None and config.chain is not None
        nuc = config.chain.display_nuc
        assert nuc is not None, "a shuttered camera flat-fields at power-up (SC.18)"
        frames[name] = np.asarray(nuc.apply(out.dn16), dtype=np.float64)
        frames[f"{name}_display8"] = np.asarray(out.display8)[..., 0].astype(np.float64)
    frames["drift_bowl"] = frames["after_drift"] - frames["at_shutter"]
    return frames


def _plot(fits: dict[str, Any], path: pathlib.Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=110)
    for name, fit in fits.items():
        if name.endswith("_display8") or name == "at_shutter":
            continue
        prof = np.asarray(fit.profile)
        span = float(np.ptp(prof)) or 1.0
        ax.plot(
            fit.profile_r, (prof - prof[0]) / span, marker="o", label=f"{name} (k={fit.k:+.1f})"
        )
    ax.set_xlabel("radius, fraction of the corner")
    ax.set_ylabel("plane-removed grey, normalised")
    ax.set_title("Radial shading of a featureless sky")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sensor", type=pathlib.Path, default=DEFAULT_SENSOR)
    ap.add_argument("--scene", type=pathlib.Path, default=DEFAULT_SCENE)
    ap.add_argument("--boresight", type=float, default=45.0, help="deg above the horizon")
    ap.add_argument("--housing-drift-k", type=float, default=-1.5, help="since the shutter event")
    ap.add_argument("--dt-s", type=float, default=120.0, help="time from shutter to second frame")
    ap.add_argument("--real", type=pathlib.Path, action="append", default=[])
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    from irsim.validation.radial import radial_fit

    frames = render_pair(args.sensor, args.scene, args.boresight, args.housing_drift_k, args.dt_s)
    fits = {name: radial_fit(img) for name, img in frames.items()}
    for path in args.real:
        fits[f"real:{path.name}"] = radial_fit(_read_image(path))

    predicted = -1 if args.housing_drift_k < 0 else 1
    report = {
        "sensor": str(
            args.sensor.relative_to(REPO) if args.sensor.is_relative_to(REPO) else args.sensor
        ),
        "scene": str(
            args.scene.relative_to(REPO) if args.scene.is_relative_to(REPO) else args.scene
        ),
        "boresight_deg": args.boresight,
        "housing_drift_since_shutter_k": args.housing_drift_k,
        "predicted_sign_after_drift": predicted,
        "fits": {name: _fit_dict(fit) for name, fit in fits.items()},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not args.no_plot:
        _plot(fits, args.out / "profiles.png")

    print(f"{'frame':32s} {'k':>9s} {'sign':>5s} {'radial':>7s} {'depth':>6s} monotonic")
    for name, fit in fits.items():
        print(
            f"{name:32s} {fit.k:+9.2f} {fit.sign:+5d} {fit.radial_share:7.3f} "
            f"{fit.relative_depth:6.3f} {fit.monotonic}"
        )
    ok = fits["drift_bowl"].sign == predicted and fits["drift_bowl"].monotonic
    print(
        f"after a {args.housing_drift_k:+.1f} K housing drift §8.2/§11.2 predict sign "
        f"{predicted:+d}: "
        + ("the rendered drift bowl matches" if ok else "the rendered drift bowl DOES NOT MATCH")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
