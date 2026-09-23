#!/usr/bin/env python3
"""OC.3 -- two cubes, one in focus: the smallest thing that makes a defocused IR frame.

No scene config and no Isaac Sim. A synthetic two-depth scene is built at the supersampled grid,
each layer is blurred with the kernel its own range earns (`OC.2`, ADR 0129), the layers are
composited back to front, and the box filter samples the result -- the same order the real pipeline
uses (`irsim.optics.stage.apply_optics`), with the pieces `OC.5` will wire in properly.

Two things it does *not* pretend to solve. The cubes do not overlap each other, and the only thing
behind them is sky, whose radiance here is a constant and is therefore known exactly -- so the
partial-occlusion problem `OC.6`-`OC.8` exists for cannot arise in this scene by construction.

**Blurring happens in radiance, never in Kelvin** (CLAUDE.md non-negotiable #3): dL/dT rises with
temperature, so averaging Kelvin across an edge puts the edge in the wrong place. The band radiance
here is a top-hat integral of Planck over the band, which is a demo's approximation of the
camera's own R(lambda)-weighted LUT and is stated as such.

    python scripts/focus_demo.py --out outputs/focus_demo
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from irsim.optics.defocus import blur_circle_um, defocus_w020_um, hyperfocal_distance_m
from irsim.optics.psf import apply_psf, defocus_psf
from irsim.optics.sampling import box_downsample
from irsim.radiometry.planck import band_radiance_tophat

#: FLIR Boson 640 optics, configs/sensors/flir_boson_640_lwir.yaml.
FOCAL_MM, F_NUMBER, PITCH_UM, SIGMA_ABERR_UM = 14.0, 1.0, 12.0, 1.654
BAND_UM = (7.5, 13.5)
T_GRID_K = np.arange(200.0, 400.0 + 1e-9, 0.05)


@dataclass(frozen=True)
class Layer:
    """One depth in the scene. ``distance_m`` of ``None`` is sky, which never defocuses."""

    name: str
    distance_m: float | None
    temperature_k: float
    mask: NDArray[np.float64]


def radiance_table() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(T, L_band) on a 0.05 K grid -- the demo's stand-in for the camera's band LUT."""
    lb = np.array([band_radiance_tophat(*BAND_UM, float(t)) for t in T_GRID_K], dtype=np.float64)
    return T_GRID_K, lb


def two_cube_scene(
    width: int, height: int, supersample: int, near_m: float, far_m: float
) -> list[Layer]:
    """Sky, a far cube and a near cube -- the near one lower-left, the far one upper-right."""
    h, w = height * supersample, width * supersample
    sky = np.ones((h, w), dtype=np.float64)
    near = np.zeros((h, w), dtype=np.float64)
    far = np.zeros((h, w), dtype=np.float64)
    near[int(0.45 * h) : int(0.85 * h), int(0.10 * w) : int(0.42 * w)] = 1.0
    far[int(0.18 * h) : int(0.50 * h), int(0.58 * w) : int(0.88 * w)] = 1.0
    return [
        Layer("sky", None, 250.0, sky),
        Layer("far cube", far_m, 300.0, far),
        Layer("near cube", near_m, 315.0, near),
    ]


def kernel_for(distance_m: float | None, focus_distance_m: float | None, supersample: int):
    """The PSF a layer at ``distance_m`` earns when the lens is focused at ``focus_distance_m``."""
    if distance_m is None:  # sky is at infinity
        c_um = (
            0.0
            if focus_distance_m is None
            else float(blur_circle_um(1e9, FOCAL_MM, F_NUMBER, focus_distance_m))
        )
    else:
        c_um = float(blur_circle_um(distance_m, FOCAL_MM, F_NUMBER, focus_distance_m))
    w020 = float(defocus_w020_um(c_um, F_NUMBER))
    lam = 0.5 * (BAND_UM[0] + BAND_UM[1])
    return defocus_psf(lam, F_NUMBER, w020, SIGMA_ABERR_UM, PITCH_UM, supersample), c_um


def render(
    layers: list[Layer], focus_distance_m: float | None, supersample: int
) -> tuple[NDArray[np.float64], dict[str, float]]:
    """Composite back to front in radiance and box-sample. Returns (native L, blur circles µm)."""
    t_grid, l_grid = radiance_table()
    out = np.zeros_like(layers[0].mask)
    blur_um: dict[str, float] = {}
    for layer in layers:
        kernel, c_um = kernel_for(layer.distance_m, focus_distance_m, supersample)
        blur_um[layer.name] = c_um
        radiance = float(np.interp(layer.temperature_k, t_grid, l_grid))
        # premultiplied over: blur the layer's radiance and its coverage with the same kernel, so
        # a defocused edge is semi-transparent rather than a hard cut
        premult = apply_psf(radiance * layer.mask, kernel)
        alpha = apply_psf(layer.mask, kernel)
        out = premult + out * (1.0 - np.clip(alpha, 0.0, 1.0))
    return box_downsample(out, supersample), blur_um


def apparent_temperature(radiance: NDArray[np.float64]) -> NDArray[np.float64]:
    t_grid, l_grid = radiance_table()
    return np.asarray(np.interp(radiance, l_grid, t_grid))


def white_hot(t_k: NDArray[np.float64], span_k: tuple[float, float]) -> NDArray[np.uint8]:
    """Grayscale white-hot over a FIXED span, so frames in a sweep are comparable."""
    lo, hi = span_k
    return np.clip((t_k - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("outputs/focus_demo"))
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--supersample", type=int, default=4)
    ap.add_argument("--near", type=float, default=3.0, help="near cube range, m")
    ap.add_argument("--far", type=float, default=80.0, help="far cube range, m")
    ap.add_argument("--frames", type=int, default=72, help="focus sweep length")
    args = ap.parse_args()

    from PIL import Image

    args.out.mkdir(parents=True, exist_ok=True)
    layers = two_cube_scene(args.width, args.height, args.supersample, args.near, args.far)
    span = (248.0, 318.0)
    h_m = hyperfocal_distance_m(FOCAL_MM, F_NUMBER, PITCH_UM)

    for name, focus in (
        ("near", args.near),
        ("far", args.far),
        ("hyperfocal", h_m),
        ("infinity", None),
    ):
        radiance, blur = render(layers, focus, args.supersample)
        Image.fromarray(white_hot(apparent_temperature(radiance), span), mode="L").save(
            args.out / f"focus_{name}.png"
        )
        where = "infinity" if focus is None else f"{focus:.1f} m"
        print(f"focus {where:>10}: " + ", ".join(f"{k} {v:6.1f} µm" for k, v in blur.items()))

    sweep = np.geomspace(args.near * 0.6, args.far * 4.0, args.frames)
    for i, focus in enumerate(sweep):
        radiance, _ = render(layers, float(focus), args.supersample)
        Image.fromarray(white_hot(apparent_temperature(radiance), span), mode="L").save(
            args.out / f"sweep_{i:04d}.png"
        )
    print(f"wrote {args.frames} sweep frames to {args.out}")

    from irsim_eval.video import encode_mp4, ffmpeg_available

    if ffmpeg_available():
        path = encode_mp4(str(args.out / "sweep_*.png"), args.out / "focus_sweep.mp4", fps=12.0)
        print(f"wrote {path}")
    else:
        print("ffmpeg not on PATH -- PNG sequence kept, no video")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
