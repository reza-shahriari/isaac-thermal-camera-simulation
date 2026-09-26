"""Re-display a finished render's saved float32 planes through a corrected display chain (SC.24).

A render writes its radiometry as float32 planes (``frame_*_apparent_t.npy``) beside the 8-bit
clips, so a display defect can be fixed and *shown* without re-rendering and without the GPU. This
script re-derives each saved frame's display and writes ``FIXED_*`` clips next to the originals,
plus ``FIXED_*_vs_old`` clips with the original frame on the left and the corrected one on the
right.
It never touches the planes.

Two corrections are applied, both from the 2026-09-26 diagnosis (spec issues S52, S53):

* **ADC floor (S53, preview of `SC.23`).** The camera's ``dn16`` put DN 0 at −40 °C and clipped
  up to 79 % of a cold sky to one code. The planes are unclipped, so the DN is re-derived from
  them over L_B(200 K) … L_B(473.15 K) -- the band LUT's own floor -- at the camera's bit depth.
  This is an affine map of band radiance, which is all the display branch sees of the ADC.
* **AGC (S52).** ``FIXED_agc`` runs the camera's own display branch (`run_display_branch`) in
  ``agc: information_based`` with ``linear_percent`` (ADR 0147), per frame, as the camera would. The
  old clip rebuilt plateau equalisation over 65 536 float bins, of which none reached the plateau.
  ``FIXED_ir`` is the linear white-hot clip of *apparent* temperature. Its floor is the render's
  own (the target's 1st percentile) and its top is the hottest apparent target pixel of the run.
  The old clip stopped at the 99th percentile, so the motors saturated to one code.

**The colour bar and the gauges answer different questions.** The gauges are each part's
*kinetic* temperature from the thermal model, for that frame. The picture and its bar are
*apparent* temperature, which is what the camera measures: a part with ε < 1 also reflects the
−45 °C sky, so it reads colder than it is. At frame 96 the parts are 15 … 38 °C kinetic and
−26 … +34 °C apparent. The bar is fixed for the whole clip, which is what lets a warm-up be seen.
Under the AGC the mapping is rebuilt every frame, so that clip has gauges but no bar.

**The AGC is selectable** (``--agc``, SC.25): linear, global plateau equalisation, local tiled
equalisation (CLAHE) and detail-weighted equalisation, each a vendor-neutral family that a given
camera parameterises (ADR 0149), with the shared controls ``--linear-percent``,
``--clip-limit-low`` and ``--max-gain``. ``--agc all`` writes one clip per mode and a 2×2 mosaic.

Usage (CPU only, no Isaac Sim)::

    python.sh scripts/redisplay_planes.py outputs/phantom4_perpart
    python.sh scripts/redisplay_planes.py outputs/phantom4_perpart --agc all
    python.sh scripts/redisplay_planes.py outputs/phantom4_perpart --agc plateau_equalization \
        --clip-limit-low 1e-3 --max-gain 4
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: The band LUT's floor (ADR 0011) and the pipeline's ceiling (`RADIOMETRIC_RANGE_K`).
FLOOR_K = 200.0
CEILING_K = 473.15
#: The readout's footprint: the caption block is ~90 px tall and the gauge panel ~380 px wide.
#: Both even, so the encoded frame keeps even dimensions.
MARGIN_TOP = 96
MARGIN_LEFT = 400
#: Every AGC the display branch can run (`isp.agc`, §11.3). The camera's own config chooses one;
#: this script lets the viewer choose, and compare them on the same frames.
AGC_MODES = ("linear", "plateau_equalization", "plateau_local", "information_based")
#: One line per mode for the frame caption: which vendor family each one stands for.
AGC_LABEL = {
    "linear": "linear, percentile tails (every vendor's 'linear' preset)",
    "plateau_equalization": "global plateau HEQ (Boson plateau / Lepton HEQ)",
    "plateau_local": "local tiled HEQ (CLAHE)",
    "information_based": "detail-weighted HEQ (Boson IBE)",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("run", type=pathlib.Path, help="a render's output directory")
    parser.add_argument("--prefix", default="FIXED_")
    parser.add_argument("--fps", type=float, default=3.0, help="saved planes are every Nth frame")
    parser.add_argument("--plateau", type=float, default=0.07, help="FLIR's default [R51]")
    parser.add_argument(
        "--linear-percent",
        type=float,
        default=0.3,
        help="Linear Percent blend (FLIR's worked example uses 30 %%; its default is unpublished)",
    )
    parser.add_argument(
        "--agc",
        default="information_based",
        help="the AGC mode(s) for the AGC clip: one of "
        f"{', '.join(AGC_MODES)}, a comma-separated list, or 'all'. More than one writes a clip "
        "per mode plus FIXED_agc_modes.mp4, every mode side by side on the same frames",
    )
    parser.add_argument(
        "--clip-limit-low",
        type=float,
        default=0.0,
        help="Lepton-family low clip, fraction of N per occupied bin (SC.25; 0 = off)",
    )
    parser.add_argument(
        "--max-gain",
        type=float,
        default=0.0,
        help="cap on display codes per DN, Boson/Xenics max gain (SC.25; 0 = off)",
    )
    args = parser.parse_args(argv)
    modes = list(AGC_MODES) if args.agc == "all" else [m.strip() for m in args.agc.split(",")]
    unknown = [m for m in modes if m not in AGC_MODES]
    if unknown:
        parser.error(f"unknown AGC mode(s) {unknown}; choose from {', '.join(AGC_MODES)}")
    # FIXED_agc and its side-by-side stay the detail-weighted mode whenever it was asked for, so
    # `--agc all` adds clips rather than changing what the existing ones show.
    main_mode = "information_based" if "information_based" in modes else modes[0]

    import numpy as np
    from PIL import Image

    from irsim.config.loader import load_sensor_config
    from irsim.io.png import write_png
    from irsim.isp.display import run_display_branch
    from irsim.isp.palette import palette_table, quantise_display
    from irsim.radiometry.lut_files import load_band_lut_for_config
    from irsim_eval.video import (
        annotate,
        encode_mp4,
        ffmpeg_available,
        overlay_readout,
        target_span_k,
    )

    run: pathlib.Path = args.run
    metas = sorted(run.glob("frame_*.json"))
    if not metas:
        print(
            f"{run}: no frame_*.json sidecars -- was the render run with planes?", file=sys.stderr
        )
        return 1
    frames = [json.loads(m.read_text()) for m in metas]
    sensor = load_sensor_config(REPO / "configs" / "sensors" / frames[0]["sensor"])
    lut = load_band_lut_for_config(sensor, REPO / "data" / "lut")
    bits = sensor.sensor.fpa.bit_depth
    top = float(2**bits - 1)
    l_lo = float(lut.lookup(FLOOR_K)[()])
    l_hi = float(lut.lookup(CEILING_K)[()])
    isps = {
        mode: sensor.sensor.isp.model_copy(
            update={
                "agc": mode,
                "plateau": args.plateau,
                "linear_percent": args.linear_percent,
                "clip_limit_low": args.clip_limit_low,
                "max_gain": args.max_gain,
            }
        )
        for mode in modes
    }
    nodes = [{k: float(v) for k, v in f["node_temperatures_k"].items()} for f in frames]
    # The gauges carry the parts' *kinetic* temperatures (the thermal model's), which fix the
    # gauge scale. The picture is *apparent* temperature: each part also reflects the −45 °C sky,
    # so at frame 96 the parts read −26 … +34 °C apparent against 15 … 38 °C kinetic. The image
    # span therefore comes from the target's apparent pixels: the render's own floor (1st
    # percentile over the run) and the hottest apparent target pixel it recorded, so nothing on
    # the aircraft clips. The old span stopped at the 99th percentile and clipped the motors.
    gauge_span = target_span_k(nodes)
    summary = (
        json.loads((run / "summary.json").read_text()) if (run / "summary.json").is_file() else {}
    )
    rows = summary.get("rows", [])
    if rows and summary.get("display_span_c"):
        span = (
            float(summary["display_span_c"][0]) + 273.15,
            max(float(r["target_max_c"]) for r in rows if r.get("target_px")) + 273.15,
        )
    else:
        span = gauge_span
    gray = palette_table("gray")

    def with_margin(image: Any) -> Any:
        """Put the frame below and right of an empty margin, so the readout covers no scene.

        `overlay_readout` draws the caption top-left and the part gauges bottom-left, and on
        `phantom4_perpart` the gauges sat over half of the aircraft -- in the old clip too. The
        margin is where they land now; the frame itself is untouched.
        """
        height, width = image.shape[:2]
        canvas = np.zeros((height + MARGIN_TOP, width + MARGIN_LEFT, 3), dtype=np.uint8)
        canvas[MARGIN_TOP:, MARGIN_LEFT:] = image[..., :3]
        return canvas

    out_dir = run / "frames"
    stats: dict[str, Any] = {
        "agc_modes": modes,
        "floor_k": FLOOR_K,
        "span_k": span,
        "gauge_span_k": gauge_span,
        "frames": [],
    }
    for meta, values in zip(frames, nodes, strict=True):
        index = int(meta["frame_index"])
        t_k = np.load(run / meta["planes"]["apparent_t"]["file"])
        if t_k.dtype != np.float32:
            raise TypeError(f"{meta['name']}: apparent_t is {t_k.dtype}, expected float32")
        radiance = lut.lookup(np.clip(t_k, FLOOR_K, CEILING_K)).astype(np.float64)
        dn = np.clip(np.round((radiance - l_lo) / (l_hi - l_lo) * top), 0, top).astype(np.uint16)
        dn16_file = meta["planes"].get("dn16", {}).get("file")
        old_clipped = (
            float(np.mean(np.asarray(Image.open(run / dn16_file)) == 0)) if dn16_file else 0.0
        )

        minutes, seconds = divmod(int(rows[index]["t_rel_s"]) if rows else 0, 60)
        head = (
            f"T+{minutes:02d}:{seconds:02d}   range {meta['range_m']:5.1f} m   "
            f"el {meta['elevation_deg']:4.1f} deg"
            + (f"   span {rows[index]['px_across']:5.1f} px" if rows else "")
        )
        shown = {
            mode: run_display_branch(dn, isp, bits).display8[..., :3] for mode, isp in isps.items()
        }
        per_mode = {
            mode: overlay_readout(
                with_margin(image),
                [
                    head,
                    f"FIXED: {AGC_LABEL[mode]}, linear {args.linear_percent:.0%}, per frame",
                    f"ADC floor {FLOOR_K:.0f} K (was 233 K: {old_clipped:.0%} of frame at DN 0)",
                ],
                values,
                gauge_span,
            )
            for mode, image in shown.items()
        }
        agc = shown[main_mode]
        agc_frame = per_mode[main_mode]
        if len(modes) > 1:
            for mode, image in per_mode.items():
                write_png(out_dir / f"{args.prefix}agc-{mode}_{index:05d}.png", image)
            tiles = [annotate(image, [head, AGC_LABEL[mode]]) for mode, image in shown.items()]
            while len(tiles) % 2:
                tiles.append(np.zeros_like(tiles[0]))
            rows_of_two = [
                np.concatenate(tiles[i : i + 2], axis=1) for i in range(0, len(tiles), 2)
            ]
            write_png(
                out_dir / f"{args.prefix}modes_{index:05d}.png",
                np.ascontiguousarray(np.concatenate(rows_of_two, axis=0)[..., :3]),
            )
        linear = gray[quantise_display((t_k - span[0]) / (span[1] - span[0]))]
        ir_frame = overlay_readout(
            with_margin(linear),
            [
                head,
                "FIXED: linear white-hot, apparent T, top = hottest target pixel of the run",
                "bar = apparent T, fixed for the clip; gauges = kinetic T, this frame",
            ],
            values,
            span,
            palette=gray,
        )
        for kind, image in (("agc", agc_frame), ("ir", ir_frame)):
            write_png(out_dir / f"{args.prefix}{kind}_{index:05d}.png", image)
            old = out_dir / f"{kind}_{index:05d}.png"
            if old.is_file():
                raw = np.asarray(Image.open(old).convert("RGB"))
                before = np.zeros((image.shape[0], raw.shape[1], 3), dtype=np.uint8)
                before[image.shape[0] - raw.shape[0] :, :] = raw  # bottom-aligned with the frame
                pair = np.ascontiguousarray(np.concatenate([before, image], axis=1))
                # `vsold`, not `_vs_old`: `FIXED_agc_*.png` must not also glob the pairs.
                write_png(out_dir / f"{args.prefix}{kind}vsold_{index:05d}.png", pair)
        codes = agc[..., 0].astype(int)
        hot = t_k > 287.0
        stats["frames"].append(
            {
                "frame_index": index,
                "old_sky_clipped_fraction": old_clipped,
                "target_codes_fixed_agc": int(codes[hot].max() - codes[hot].min())
                if hot.any()
                else 0,
            }
        )

    (run / f"{args.prefix}summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if not ffmpeg_available():
        print("ffmpeg not available: stills written, no video", file=sys.stderr)
        return 0
    for kind, name in (
        ("agc", "agc"),
        ("ir", "ir"),
        ("agcvsold", "agc_vs_old"),
        ("irvsold", "ir_vs_old"),
        ("modes", "agc_modes"),
        *((f"agc-{mode}", f"agc_{mode}") for mode in modes if len(modes) > 1),
    ):
        pattern = out_dir / f"{args.prefix}{kind}_*.png"
        if not any(out_dir.glob(pattern.name)):
            continue
        path = run / f"{args.prefix}{name}.mp4"
        encode_mp4(str(pattern), path, fps=args.fps)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
