"""One aircraft, every band, every AGC: a grid of what each camera would show (SC.25, ADR 0149).

Rows are the bands of a multiband render (``render_multiband.py`` output: one directory per band,
each with the camera's raw ``dn16`` and a registered RGB companion). Columns are the RGB companion
and a set of AGC scenarios. Every scenario is the band's **own** ISP from its sensor YAML with only
the AGC swapped, run on the **whole frame** as the camera runs it, and only then cropped around the
aircraft -- so a tile shows what the camera would put on a screen, not what a stretch chosen for
the crop would. The crop follows the aircraft, which is found in the RGB companion (the one image in
which the target is always visible against the sky), and is the same size for a band across the
whole clip so the aircraft's motion is real.

Nothing is rendered: it reads ``dn16`` PNGs, so it runs on the CPU in minutes and needs neither
Isaac Sim nor the GPU. It writes a still (``<scene>_agc_grid.png``) at the frame where the motors
peak and a clip (``<scene>_agc_grid.mp4``) over the run.

    python.sh scripts/agc_band_grid.py outputs/multiband/drone
    python.sh scripts/agc_band_grid.py outputs/multiband/drone --stride 3 --tile 280
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BANDS = ("lwir", "mwir", "swir", "nir")

#: (column label, isp overrides). Each is a family of ADR 0149 with the shared controls; the last
#: is the tuned combination the sky lane asks for (floor of shades for a small target, and a cap
#: on how far a bland sky is stretched).
SCENARIOS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("linear (percentile)", {"agc": "linear"}),
    ("global plateau HEQ", {"agc": "plateau_equalization", "plateau": 0.07}),
    ("local HEQ (CLAHE)", {"agc": "plateau_local", "plateau": 0.07}),
    ("detail-weighted HEQ", {"agc": "information_based", "plateau": 0.07, "linear_percent": 0.3}),
    (
        "HEQ + low clip + max gain",
        {"agc": "plateau_equalization", "plateau": 0.07, "clip_limit_low": 1e-3, "max_gain": 4.0},
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("scene", type=pathlib.Path, help="a render_multiband.py scene directory")
    parser.add_argument("--stride", type=int, default=3, help="use every Nth frame for the clip")
    parser.add_argument("--tile", type=int, default=300, help="tile edge in pixels")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image, ImageDraw

    from irsim.config.loader import load_sensor_config
    from irsim.io.png import write_png
    from irsim.isp.display import run_display_branch
    from irsim_eval.video import _font, encode_mp4, ffmpeg_available

    scene: pathlib.Path = args.scene
    bands = [b for b in BANDS if (scene / b / "summary.json").is_file()]
    if not bands:
        print(
            f"{scene}: no <band>/summary.json -- not a render_multiband.py scene", file=sys.stderr
        )
        return 1
    frames = sorted(
        set.intersection(
            *({int(p.name[6:12]) for p in (scene / b).glob("frame_*_dn16.png")} for b in bands)
        )
    )
    summary = json.loads((scene / bands[0] / "summary.json").read_text())
    peak_frame = int(round(float(summary.get("motor_peak_at_s", 0.0)) / summary["interval_s"]))
    peak_frame = min(frames, key=lambda f: abs(f - peak_frame))

    sensors = {}
    for band in bands:
        meta = json.loads((scene / band / f"frame_{frames[0]:06d}.json").read_text())
        sensors[band] = load_sensor_config(REPO / "configs" / "sensors" / f"{meta['sensor']}.yaml")

    def find_target(rgb: Any) -> tuple[float, float, float, float] | None:
        """Bounding box of the aircraft in the RGB companion: pixels far from the smooth sky."""
        lum = rgb[..., :3].astype(np.float64).mean(axis=2)
        small = np.asarray(
            Image.fromarray(lum.astype(np.uint8)).resize((lum.shape[1] // 16, lum.shape[0] // 16))
        )
        sky = np.asarray(
            Image.fromarray(small).resize((lum.shape[1], lum.shape[0]), Image.BILINEAR)
        ).astype(np.float64)
        ys, xs = np.nonzero(np.abs(lum - sky) > 18.0)
        if ys.size < 20:
            return None
        return (
            float(np.percentile(ys, 1)),
            float(np.percentile(ys, 99)),
            float(np.percentile(xs, 1)),
            float(np.percentile(xs, 99)),
        )

    # One crop size per band for the whole clip (the largest the aircraft needs, plus margin), and
    # a per-frame centre smoothed over the run so the tile does not jitter with the detector.
    boxes: dict[str, dict[int, tuple[float, float, float, float]]] = {b: {} for b in bands}
    for band in bands:
        for f in frames[:: args.stride] + [peak_frame]:
            rgb = np.asarray(Image.open(scene / band / f"frame_{f:06d}_rgb.png"))
            box = find_target(rgb)
            if box is not None:
                boxes[band][f] = box
    side = {
        b: int(1.5 * max(max(y1 - y0, x1 - x0) for y0, y1, x0, x1 in boxes[b].values()))
        for b in bands
        if boxes[b]
    }

    font = _font(15)
    footer_font = _font(17)

    def label(img: Any, text: str, height: int = 26, face: Any = None) -> Any:
        canvas = Image.new("RGB", (img.shape[1], img.shape[0] + height), (0, 0, 0))
        canvas.paste(Image.fromarray(img), (0, height))
        ImageDraw.Draw(canvas).text((6, 5), text, fill=(235, 235, 235), font=face or font)
        return np.asarray(canvas)

    def crop(image: Any, band: str, frame: int) -> Any:
        y0, y1, x0, x1 = boxes[band][frame]
        cy, cx, s = (y0 + y1) / 2.0, (x0 + x1) / 2.0, side[band]
        h, w = image.shape[:2]
        top = int(min(max(cy - s / 2, 0), max(h - s, 0)))
        left = int(min(max(cx - s / 2, 0), max(w - s, 0)))
        tile = image[top : top + s, left : left + s]
        return np.asarray(Image.fromarray(tile).resize((args.tile, args.tile), Image.NEAREST))

    out_dir = scene / "_agc_grid"
    out_dir.mkdir(exist_ok=True)
    written = []
    for f in sorted(set(frames[:: args.stride]) | {peak_frame}):
        if any(f not in boxes[b] for b in bands):
            continue
        rows = []
        for band in bands:
            sensor = sensors[band].sensor
            dn = np.asarray(Image.open(scene / band / f"frame_{f:06d}_dn16.png"))
            rgb = np.asarray(Image.open(scene / band / f"frame_{f:06d}_rgb.png"))[..., :3]
            tiles = [label(crop(rgb, band, f), f"{band.upper()} · RGB companion")]
            for name, overrides in SCENARIOS:
                isp = sensor.isp.model_copy(update=overrides)
                shown = run_display_branch(dn, isp, sensor.fpa.bit_depth).display8[..., :3]
                tiles.append(label(crop(shown, band, f), f"{band.upper()} · {name}"))
            rows.append(np.concatenate(tiles, axis=1))
        grid = np.concatenate(rows, axis=0)
        meta = json.loads((scene / bands[0] / f"frame_{f:06d}.json").read_text())
        nodes = "   ".join(
            f"{k} {v - 273.15:4.1f}C" for k, v in sorted(meta["node_temperatures_k"].items())
        )
        minutes, seconds = divmod(int(f * summary["interval_s"]), 60)
        footer = (
            f"T+{minutes:02d}:{seconds:02d}   range {meta['range_m']:.0f} m   "
            f"throttle {meta.get('throttle', 0.0):.0%}   {nodes}   "
            "each tile: the band's own ISP, only the AGC swapped, run on the whole frame"
        )
        grid = label(grid, footer, height=30, face=footer_font)
        path = out_dir / f"grid_{f:05d}.png"
        write_png(path, np.ascontiguousarray(grid))
        written.append(path)
        if f == peak_frame:
            write_png(scene / f"{scene.name}_agc_grid.png", np.ascontiguousarray(grid))
    print(f"wrote {len(written)} grid frames; still at frame {peak_frame} (motor peak)")
    if ffmpeg_available() and written:
        clip = scene / f"{scene.name}_agc_grid.mp4"
        encode_mp4(str(out_dir / "grid_*.png"), clip, fps=args.fps)
        print(f"wrote {clip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
