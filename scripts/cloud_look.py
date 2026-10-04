#!/usr/bin/env python3
"""The visible cloud as a camera on the ground sees it, through a day: `isaac-weather-fx`'s own sky.

A look-development tool for the weather lane (`WX.3`–`WX.9`): it renders what weather-fx's dome
paints for a perspective camera, frame by frame through the hours of one day, with no engine --
`weather_fx.core` is numpy. It is how a change to the cloud's shape or light is judged by eye
before it is judged in Isaac Sim, and how two versions of weather-fx are put side by side:

    # one version: the submodule (default) or any other weather-fx checkout
    python scripts/cloud_look.py render --out outputs/cloud_look/after
    python scripts/cloud_look.py render --out outputs/cloud_look/before \\
        --weather-fx-root /path/to/an/older/isaac-weather-fx
    # the two, side by side, as a video
    python scripts/cloud_look.py compose outputs/cloud_look/before \\
        outputs/cloud_look/after --out outputs/cloud_look/cloud_look_before_after.mp4

Two versions cannot be imported into one Python process (both are the package `weather_fx`), so
`render` draws one version's frames and `compose` joins two renders that share a scene.

**What the pixels are.** Each pixel is `weather_fx.core.sky.sky_radiance_rgb` along that pixel's
ray: the clear sky, the cloud marched and composed in front of it (`T · L_sky + added`), the
ground below the horizon -- exactly what the real-time dome is baked from, at the camera's own
resolution rather than the dome's texels. It is exposed by weather-fx's own meter
(`dome_exposure` on a 128-row environment map of the same conditions) and mapped to the screen by
a stand-in tone curve, `1 − exp(−0.7 L·e / 217.5)` then sRGB, so a frame is legible on a page.
That curve is **not** the RTX tonemapper: compare frames from this script with each other, not
with an Isaac Sim viewport.

`summary.json` records, per frame, the hour, the sun's elevation, the exposure and the cover the
field measured, and the weather-fx commit the frames came from.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
import subprocess
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: The meter's highlight target, weather-fx's `DOME_HIGHLIGHT_TARGET`; the tone curve is scaled
#: by it so that the metered highlight sits at about half of white.
TONE_GAIN = 0.7


def _use_weather_fx(root: pathlib.Path | None) -> str:
    """Put a weather-fx checkout's package first on the path; return its commit."""
    if root is None:
        root = ROOT / "third_party" / "isaac-weather-fx"
    package = root / "exts" / "weather.fx"
    if not package.is_dir():
        raise SystemExit(f"no weather-fx package at {package}")
    sys.path.insert(0, str(package))
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def camera_directions(
    width: int, height: int, hfov_deg: float, elevation_deg: float, azimuth_deg: float
) -> np.ndarray:
    """Unit rays of a pinhole camera, in weather-fx's frame: +Y up, -Z at azimuth 0, +X at 90.

    The same convention as ``BodyPosition.direction`` and ``irsim_isaac.visible_sky``; a camera
    at compass azimuth 270 looks west.
    """
    focal = 0.5 * width / math.tan(math.radians(hfov_deg) / 2.0)
    xs = (np.arange(width) + 0.5 - width / 2.0) / focal
    ys = -(np.arange(height) + 0.5 - height / 2.0) / focal
    x, y = np.meshgrid(xs, ys)
    el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
    forward = np.array([math.cos(el) * math.sin(az), math.sin(el), -math.cos(el) * math.cos(az)])
    right = np.array([math.cos(az), 0.0, math.sin(az)])
    up = np.cross(right, forward)
    rays = forward[None, None, :] + x[..., None] * right + y[..., None] * up
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def tone(radiance: np.ndarray, exposure: float, target: float) -> np.ndarray:
    """The stand-in display curve: exponential shoulder, then sRGB. 8-bit, for viewing only."""
    x = radiance * exposure / target
    y = 1.0 - np.exp(-TONE_GAIN * x)
    curve = 1.055 * np.power(np.clip(y, 0.0, None), 1 / 2.4) - 0.055
    srgb = np.where(y <= 0.0031308, 12.92 * y, curve)
    return (np.clip(srgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _pixel_radiance(sky, conditions, rays: np.ndarray, stride: int, exposure: float) -> np.ndarray:
    """Each output pixel's radiance: one ray per pixel at ``stride`` 1; otherwise the pixel
    integrated over a ``stride``-times grid by marching its cloud edges again
    (``irsim.atmosphere.cloud_deck.march_on_native_grid``, the infrared camera's own rule, WX.4)
    and box-averaging, as a detector does."""
    if stride <= 1:
        return sky.sky_radiance_rgb(rays.reshape(-1, 3), conditions).reshape(rays.shape)
    sys.path.insert(0, str(ROOT / "src"))
    from irsim.atmosphere.cloud_deck import march_on_native_grid

    luma = np.array([0.2126, 0.7152, 0.0722])
    scale = exposure / sky.DOME_HIGHLIGHT_TARGET
    el = np.arcsin(np.clip(rays[..., 1], -1.0, 1.0))
    az = np.arctan2(rays[..., 0], -rays[..., 2])

    def evaluate(e, a):
        e, a = np.asarray(e), np.asarray(a)
        d = np.stack([np.cos(e) * np.sin(a), np.sin(e), -np.cos(e) * np.cos(a)], axis=-1)
        rgb = sky.sky_radiance_rgb(d.reshape(-1, 3), conditions).reshape(e.shape + (3,))
        # The first plane is what decides an edge: luminance on the display's scale, where the
        # metered highlight is about 1, so 0.02 is a just-visible step.
        return (rgb @ luma * scale, rgb[..., 0], rgb[..., 1], rgb[..., 2])

    _, r, g, b = march_on_native_grid(evaluate, el, az, stride, threshold=0.02)
    rgb = np.stack([r, g, b], axis=-1)
    rows, cols = el.shape[0] // stride, el.shape[1] // stride
    return rgb.reshape(rows, stride, cols, stride, 3).mean(axis=(1, 3))


def render(args: argparse.Namespace) -> None:
    commit = _use_weather_fx(args.weather_fx_root)
    from PIL import Image
    from weather_fx.core import sky
    from weather_fx.core.state import WeatherState

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(args.stride))
    rays = camera_directions(
        args.width * stride, args.height * stride, args.hfov, args.elevation, args.azimuth
    )
    hours = np.linspace(args.start_hour, args.end_hour, args.frames)
    frames = []
    for index, hour in enumerate(hours):
        started = time.time()
        state = WeatherState.from_dict(
            {
                "general": {"time_source": "manual"},
                "sky": {
                    "latitude_deg": args.latitude,
                    "longitude_deg": args.longitude,
                    "date_utc": args.date,
                    "hour_utc": float(hour),
                    "turbidity": args.turbidity,
                },
                "clouds": {
                    "enabled": True,
                    "cover": args.cover,
                    "genus": args.genus,
                    "seed": args.seed,
                    "temperature_c": args.temperature,
                    "dewpoint_c": args.dewpoint,
                },
            },
            strict=False,
        )
        conditions = sky.conditions_from_state(state)
        if args.drift_mps:
            # The wind carries the field along +X at the given speed from the first frame.
            elapsed = (float(hour) - args.start_hour) * 3600.0
            conditions = dataclasses.replace(
                conditions, cloud_offset_m=(args.drift_mps * elapsed, 0.0, 0.0)
            )
        exposure = sky.dome_exposure(sky.environment_map(conditions, height=128), conditions)
        radiance = _pixel_radiance(sky, conditions, rays, stride, exposure)
        image = tone(radiance, exposure, sky.DOME_HIGHLIGHT_TARGET)
        Image.fromarray(image).save(out / f"frame_{index:04d}.png")
        frames.append(
            {
                "index": index,
                "hour_utc": round(float(hour), 4),
                "sun_elevation_deg": round(float(conditions.sun.elevation_deg), 3),
                "exposure": float(exposure),
                "measured_cover": round(float(conditions.cloud.measured_cover), 4)
                if conditions.cloud is not None
                else 0.0,
                "seconds": round(time.time() - started, 2),
            }
        )
        print(
            f"frame {index + 1}/{len(hours)} hour {hour:5.2f} UTC "
            f"sun {conditions.sun.elevation_deg:5.1f} deg  {frames[-1]['seconds']:.1f}s",
            flush=True,
        )
    summary = {
        "weather_fx_commit": commit,
        "scene": {
            k: getattr(args, k)
            for k in (
                "latitude",
                "longitude",
                "date",
                "turbidity",
                "cover",
                "genus",
                "seed",
                "temperature",
                "dewpoint",
            )
        },
        "camera": {
            k: getattr(args, k) for k in ("width", "height", "hfov", "elevation", "azimuth")
        },
        "stride": stride,
        "drift_mps": args.drift_mps,
        "tone": f"1 - exp(-{TONE_GAIN} L e / DOME_HIGHLIGHT_TARGET), sRGB; not the RTX tonemapper",
        "frames": frames,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


def compose(args: argparse.Namespace) -> None:
    from PIL import Image, ImageDraw

    sys.path.insert(0, str(ROOT / "src"))
    from irsim_eval.video import encode_mp4

    left, right = pathlib.Path(args.left), pathlib.Path(args.right)
    left_summary = json.loads((left / "summary.json").read_text())
    right_summary = json.loads((right / "summary.json").read_text())
    if left_summary["scene"] != right_summary["scene"] or (
        left_summary["camera"] != right_summary["camera"]
    ):
        raise SystemExit("the two renders are of different scenes or cameras")
    out = pathlib.Path(args.out)
    frames_dir = out.parent / (out.stem + "_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels = (
        args.left_label or f"weather-fx {left_summary['weather_fx_commit']}",
        args.right_label or f"weather-fx {right_summary['weather_fx_commit']}",
    )
    for a, b in zip(left_summary["frames"], right_summary["frames"], strict=True):
        one = Image.open(left / f"frame_{a['index']:04d}.png")
        two = Image.open(right / f"frame_{b['index']:04d}.png")
        gap = 8
        sheet = Image.new("RGB", (one.width + two.width + gap, one.height), (24, 24, 24))
        sheet.paste(one, (0, 0))
        sheet.paste(two, (one.width + gap, 0))
        draw = ImageDraw.Draw(sheet)
        hour = a["hour_utc"]
        clock = f"{int(hour):02d}:{round((hour % 1) * 60) % 60:02d} UTC"
        stamp = f"{clock}  sun {a['sun_elevation_deg']:+.1f} deg"
        for x, text in ((6, labels[0]), (one.width + gap + 6, labels[1])):
            # A dark box under each label: a cloud is white, and so is the text.
            for y, line in ((6, text), (one.height - 16, stamp)):
                box = draw.textbbox((x, y), line)
                draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(20, 20, 20))
                draw.text((x, y), line, fill=(255, 255, 255))
        sheet.save(frames_dir / f"pair_{a['index']:04d}.png")
    encode_mp4(str(frames_dir / "pair_*.png"), out, fps=args.fps)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("render", help="render one weather-fx version's frames")
    one.add_argument("--out", required=True)
    one.add_argument("--weather-fx-root", type=pathlib.Path, default=None)
    one.add_argument("--frames", type=int, default=40)
    one.add_argument("--start-hour", type=float, default=8.0)
    one.add_argument("--end-hour", type=float, default=19.0)
    one.add_argument("--width", type=int, default=480)
    one.add_argument("--height", type=int, default=270)
    one.add_argument("--hfov", type=float, default=70.0)
    one.add_argument("--elevation", type=float, default=14.0)
    one.add_argument("--azimuth", type=float, default=260.0)
    one.add_argument("--latitude", type=float, default=48.14)
    one.add_argument("--longitude", type=float, default=11.58)
    one.add_argument("--date", default="2024-06-21")
    one.add_argument("--turbidity", type=float, default=2.8)
    one.add_argument("--cover", type=float, default=0.35)
    one.add_argument("--genus", default="cumulus")
    one.add_argument("--seed", type=int, default=17)
    one.add_argument("--temperature", type=float, default=22.0)
    one.add_argument("--dewpoint", type=float, default=12.0)
    one.add_argument(
        "--stride",
        type=int,
        default=1,
        help="integrate each pixel over a stride x stride grid, marching only its cloud edges "
        "again (WX.4); 1 is one ray per pixel",
    )
    one.add_argument(
        "--drift-mps", type=float, default=0.0, help="wind carrying the field along +X, m/s"
    )
    pair = sub.add_parser("compose", help="put two renders of one scene side by side as a video")
    pair.add_argument("left")
    pair.add_argument("right")
    pair.add_argument("--out", required=True)
    pair.add_argument("--left-label", default=None)
    pair.add_argument("--right-label", default=None)
    pair.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args()
    if args.command == "render":
        render(args)
    else:
        compose(args)


if __name__ == "__main__":
    main()
