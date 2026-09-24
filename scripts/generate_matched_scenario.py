#!/usr/bin/env python3
"""Roadmap M12.1: render clips matched to the reference set, ready for the ME.6 comparison.

    python scripts/generate_matched_scenario.py --clips 8 --frames 30 --out outputs/matched
    python scripts/generate_matched_scenario.py --priors          # what is matched, and what is not

A Tier 4 comparison is only as good as the match between what was rendered and what was filmed. If
the synthetic clips are clear sky at 100 m and the real ones are half cloud at 200 m, the report
measures the difference in *scenario* and the physics is never tested. So the camera is the
reference set's own (`configs/sensors/halmstad_boson_320.yaml`), the scenarios are drawn from
`irsim.validation.scenario`'s priors, and each clip is **encoded through the same codec the
publication used** before anything is measured on it -- because ME.2b showed x264 at CRF 18 removes
95 % of a clip's temporal noise, so comparing an uncoded render against a coded clip would measure
the encoder.

**What is matched and what is not is printed with every run.** ME.5 could not measure the sky or
the target distributions on the published set -- the sky statistics need a labelled region and the
target ones need boxes that are MATLAB MCOS objects with no Python reader -- so those priors are
ESTIMATED, and `--priors` says so parameter by parameter. A Tier 4 number derived from these clips
is a comparison against an assumed scenario, and that has to travel with it.

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
DEFAULT_SENSOR = REPO / "configs" / "sensors" / "halmstad_boson_320.yaml"
DEFAULT_SCENE = REPO / "configs" / "scenes" / "sky_target_clear_day.yaml"

#: The publication's own container settings (ME.5 measured them with ffprobe): h.264 Main profile
#: in mp4 at ~128 kb/s, stored at 30 fps although the core runs at 60.
PUBLISHED_FPS = 30.0
PUBLISHED_CRF = 23  # ESTIMATED from the measured ~128 kb/s at 320x256x30; the encoder is unstated


def _render_clip(
    scenario: Any,
    *,
    sensor: Any,
    scene: Any,
    lut: Any,
    materials: Any,
    frames: int,
    supersample: int,
) -> tuple[np.ndarray, list[tuple[float, float, float, float]]]:
    """One clip: (T, H, W) uint8 display frames and one box per frame."""
    from irsim.pipeline import PipelineConfig, PipelineState, run_frame
    from irsim.validation.aerial_scene import SceneTarget, build_aerial_gbuffer

    spec = sensor.sensor
    sky = scene.sky_models[spec.band.band_id]
    config = PipelineConfig.from_sensor(sensor, materials, lut, sky=sky, atmosphere=scene.layered)
    state = PipelineState(t_s=scene.t0_s)
    height, width = spec.fpa_shape
    t_air = scene.weather.at(scene.t0_s).t_air_k

    images: list[np.ndarray] = []
    boxes: list[tuple[float, float, float, float]] = []
    # A straight track across the frame: the motion is what the smear and the static-clip gate
    # both key on, and a stationary target would make half the Tier 4 statistics vacuous.
    start_x = 0.2 * width
    end_x = 0.8 * width
    for index in range(frames):
        fraction = index / max(frames - 1, 1)
        position = (start_x + fraction * (end_x - start_x), 0.45 * height)
        target = SceneTarget(
            material="aircraft_aluminium_painted",
            temperature_k=float(t_air + scenario["target_delta_t_k"]),
            size_m=float(scenario["target_size_m"]),
            range_m=float(scenario["range_m"]),
            position_px=position,
        )
        built = build_aerial_gbuffer(
            spec,
            sky,
            materials,
            t_s=state.t_s,
            boresight_elevation_deg=float(scenario["boresight_elevation_deg"]),
            targets=[target],
            cloud_seed=scenario.index if scenario["cloud_fraction"] > 0.05 else None,
            supersample=supersample,
        )
        out = run_frame(built.planes, config, state, built.point_targets)
        if out.display8 is None:
            raise SystemExit("the matched camera must emit display_8 for a DN8 comparison")
        images.append(np.asarray(out.display8)[..., 0].astype(np.uint8))
        if built.resolved:
            # `resolved` reports the box on the **supersampled** grid the G-buffer was built at;
            # the frames written out are the native detector grid the optics stage boxed down to.
            # A box in the wrong grid is off by the supersample factor and lands outside the image,
            # which is a silent, total failure of every box-dependent statistic downstream.
            _, (y0, x0, y1, x1) = built.resolved[0]
            k = float(built.supersample)
            boxes.append((float(x0) / k, float(y0) / k, float(x1 - x0) / k, float(y1 - y0) / k))
        else:
            extent = max(target.extent_px(spec), 1.0)
            boxes.append((position[0] - extent / 2, position[1] - extent / 2, extent, extent))
    return np.stack(images), boxes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--out", default=str(REPO / "outputs" / "matched"))
    parser.add_argument("--sensor", default=str(DEFAULT_SENSOR))
    parser.add_argument("--scene", default=str(DEFAULT_SCENE))
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument(
        "--supersample",
        type=int,
        default=None,
        help="G-buffer oversampling; default is the sensor config's own optics.supersample_factor, "
        "which is what run_frame's optics stage expects to box-filter down from",
    )
    parser.add_argument("--crf", type=int, default=PUBLISHED_CRF)
    parser.add_argument("--no-encode", action="store_true", help="skip the codec round trip")
    parser.add_argument("--priors", action="store_true", help="print the priors and exit")
    args = parser.parse_args(argv)

    from irsim.validation.scenario import ScenarioSampler

    sampler = ScenarioSampler(seed=args.seed)
    if args.priors:
        summary = sampler.summary()
        print(json.dumps(summary, indent=2))
        counts = summary["provenance_counts"]
        print(
            f"\n{counts['measured']} of {summary['n_parameters']} parameters are measured on the "
            f"published set; {counts['stated']} come from the publication's text and "
            f"{counts['estimated']} are judgements. Any Tier 4 number from these clips is a "
            "comparison against an assumed scenario.",
            file=sys.stderr,
        )
        return 0

    from irsim.config.loader import config_hash, load_sensor_config
    from irsim.materials import MaterialLibrary, MaterialTable
    from irsim.radiometry.lut import BandLUT
    from irsim.radiometry.spectral_response import load_spectral_response
    from irsim.scene import Scene
    from irsim_eval.data import Box, Frame, Sequence, write_sequence

    sensor = load_sensor_config(args.sensor)
    supersample = args.supersample or sensor.sensor.optics.supersample_factor
    band = sensor.sensor.band.band_id
    lut = BandLUT.build(load_spectral_response(sensor.sensor.band.spectral_response), n=4001)
    scene = Scene.from_file(args.scene, {band: lut}, quantity=sensor.sensor.quantity)
    materials = MaterialTable.from_library(MaterialLibrary.load(), band)

    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for scenario in sampler.draw(args.clips):
        images, boxes = _render_clip(
            scenario,
            sensor=sensor,
            scene=scene,
            lut=lut,
            materials=materials,
            frames=args.frames,
            supersample=supersample,
        )
        coded = images
        if not args.no_encode:
            from irsim_eval.transcode import h264_round_trip

            # Through the publication's own container before anything is measured: ME.2b showed
            # x264 at CRF 18 removes 95 % of a clip's temporal noise, so an uncoded render put
            # beside a coded clip would measure the encoder and call it physics.
            coded = h264_round_trip(images, crf=args.crf, fps=PUBLISHED_FPS)
        name = f"matched_{scenario.index:03d}"
        sequence = Sequence(
            name=name,
            shape=(int(coded.shape[1]), int(coded.shape[2])),
            frames=[
                Frame(
                    index=i,
                    image=coded[i],
                    boxes=(Box(*boxes[i], label="drone"),),
                    attributes={"scenario": scenario.values, "provenance": scenario.provenance},
                )
                for i in range(coded.shape[0])
            ],
            source_dataset="synthetic:matched_halmstad",
            # These frames are `out.display8` -- this simulator's own AGC, not a recorder's
            # conversion. The set they are matched to (`halmstad_drone_detection`) is `recorder`,
            # so the pair differs by a signal path before it differs by any physics. That is not
            # a defect introduced here: M12.2's acceptance run already found `noise_scale` to be
            # the discriminator's heaviest feature. Recording the path is how the next reader
            # sees it without re-deriving it (XD.2).
            signal_path="display",
        )
        write_sequence(out_root / name, sequence)
        written.append(
            {
                "name": name,
                "frames": int(coded.shape[0]),
                "scenario": scenario.values,
                "measured_fraction": scenario.measured_fraction,
            }
        )
        print(f"wrote {out_root / name} ({coded.shape[0]} frames)", file=sys.stderr)

    (out_root / "index.json").write_text(
        json.dumps(
            {
                "sensor": sensor.sensor.name,
                "config_hash": config_hash(sensor),
                "encoded": not args.no_encode,
                "crf": args.crf,
                "fps": PUBLISHED_FPS,
                "priors": sampler.summary(),
                "clips": written,
            },
            indent=2,
            sort_keys=True,
        ),
        "utf-8",
    )
    print(f"index: {out_root / 'index.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
