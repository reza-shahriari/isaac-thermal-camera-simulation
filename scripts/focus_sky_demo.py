#!/usr/bin/env python3
"""OC.12 -- a cube against a cloudy sky, with the focus pulled from the sky onto the cube.

`OC.3` put two cubes at two ranges and focused on either. This one answers the question that
demo cannot: what a *background* looks like when the lens leaves it. The scene is one near cube
against sky, the sweep starts focused far away and ends focused on the cube, and the sky goes soft
as the cube comes sharp.

**A clear sky cannot show this, and that is physics rather than a shortcoming of the demo.**
Defocus is a low-pass filter, so it can only remove detail that was there. A clear LWIR sky is a
smooth ramp in elevation -- `cloud_airmass`'s 1/sin(theta) through the column -- and a smooth ramp
convolved with any normalised kernel is very nearly itself. Blurring it changes nothing you can
see. The high spatial frequencies in a real sky are **cloud edges**, so the sky here carries the
project's own sky-fixed cloud field (`irsim.atmosphere.cloud`), and the thing that visibly blurs is
the cloud.

The compositing is the shipped `OC.6`/`OC.7` path, not a demo's own: `layered_defocus` bins the
cube's own depth range into layers, and the analytic sky radiance along **every** ray -- including
the rays the cube hides -- is handed to it as `background_radiance`. That is the case `OC.7` exists
for, and it is exact here rather than bounded, because the sky's radiance is a function of ray
direction with no geometry in it. The gap opened behind the cube's defocused silhouette is filled
with the cloud that is really behind it.

Two approximations are the demo's own and are not the pipeline's. The band radiance is a top-hat
integral of Planck rather than the camera's R(lambda)-weighted LUT, as in `OC.3`. And the sky is
one clear column with cloud blended over it by the cloud's own optical depth, where the real
`SkyModel` splits the column at the cloud base and carries the below-cloud air in front of it --
worth about a kelvin here, against the 38 K cloud-to-clear contrast the picture is made of.

**Blurring happens in radiance, never in Kelvin** (CLAUDE.md non-negotiable #3).

    python scripts/focus_sky_demo.py --out outputs/focus_sky_demo
"""

from __future__ import annotations

import argparse
import math
import pathlib
from dataclasses import dataclass

import numpy as np
from focus_demo import (
    BAND_UM,
    F_NUMBER,
    FOCAL_MM,
    PITCH_UM,
    SIGMA_ABERR_UM,
    apparent_temperature,
    radiance_table,
    white_hot,
)
from numpy.typing import NDArray

from irsim.atmosphere.cloud import (
    cloud_airmass,
    cloud_base_temperature_k,
    generate_sky_cloud,
    lifting_condensation_level_m,
    sky_angles,
)
from irsim.optics.autofocus import focus_measure
from irsim.optics.defocus import blur_circle_um
from irsim.optics.layered import layered_defocus
from irsim.optics.psf import DefocusKernelBank
from irsim.optics.sampling import box_downsample

#: Clear-sky emissivity at the zenith **in the 8-14 um window**, not broadband. The two differ by
#: a factor of three and using the broadband 0.7-0.8 here renders a sky at air temperature: the
#: window is transparent precisely where this camera looks. 0.25 is a moderately dry column;
#: docs/physics-model.md §6.3. The slant dependence is Beer-Lambert on the plane-parallel airmass.
EPS_ZENITH_WINDOW = 0.25
T_AIR_K, RH, LAPSE_K_PER_M = 288.0, 0.60, 0.0065
#: 1/f^beta structure and coverage. beta comes from configs/environments/scattered_cumulus.yaml,
#: which cites `MS.3`; it is the project's value for cloud everywhere and is not re-chosen here.
CLOUD_BETA, CLOUD_FRACTION, CLOUD_SEED = 1.8, 0.45, 20260923
#: Half-degree cells are what `generate_sky_cloud` defaults to, sized for wide-field clutter
#: statistics: ten Boson pixels across. This demo refines them to an eighth of a degree -- 2.6
#: pixels, so the finest cloud structure is about five pixels and the blur circle the cube's range
#: earns is 5.4. A cloud whose smallest feature is twenty pixels wide barely registers the blur.
CLOUD_GRID = (720, 2880)
#: Faces of an isometric cube: top coolest (it radiates into a cold sky), one side sunlit.
FACE_T_K = {"top": 296.0, "left": 314.0, "right": 304.0}


@dataclass(frozen=True)
class SkyCubeScene:
    """A supersampled scene: what the camera sees, how far away it is, and what is behind it."""

    radiance_ss: NDArray[np.float64]
    distance_m: NDArray[np.float64]
    sky_mask: NDArray[np.bool_]
    background_radiance: NDArray[np.float64]
    cube_mask: NDArray[np.bool_]
    supersample: int
    near_m: float
    far_m: float
    #: (row slice, column slice) of a patch wholly inside the cloud, and one inside the cube.
    sky_patch: tuple[slice, slice]
    cube_patch: tuple[slice, slice]


def band_radiance(t_k: object) -> NDArray[np.float64]:
    """Band radiance at a temperature, on `focus_demo`'s shared table."""
    t_grid, l_grid = radiance_table()
    return np.asarray(np.interp(np.asarray(t_k, dtype=np.float64), t_grid, l_grid))


def ray_directions(
    width: int, height: int, supersample: int, boresight_deg: float
) -> NDArray[np.float64]:
    """Unit ray per supersample cell, in stage axes (up +y, forward -z), pitched up by boresight.

    A pinhole at the lens centre: the cell at sensor offset (x, y) looks along (x, y, -f). The
    sensor is taken upright rather than inverted, which costs the demo nothing and saves the
    reader a flip.
    """
    h, w = height * supersample, width * supersample
    cell_mm = PITCH_UM * 1e-3 / supersample
    xs = (np.arange(w, dtype=np.float64) - 0.5 * (w - 1)) * cell_mm
    ys = -(np.arange(h, dtype=np.float64) - 0.5 * (h - 1)) * cell_mm  # row 0 is the top of frame
    gx, gy = np.meshgrid(xs, ys)
    d = np.stack([gx, gy, np.full_like(gx, -FOCAL_MM)], axis=-1)
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    a = math.radians(boresight_deg)
    # rotate about +x (the right axis), which carries forward (0,0,-1) to (0, sin a, -cos a)
    rot = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]]
    )
    return np.asarray(d @ rot.T)


def sky_radiance_field(directions: NDArray[np.float64], cloud) -> NDArray[np.float64]:
    """Band radiance along every ray: a clear column with cloud blended on by its optical depth.

    This is the function of ray direction that makes `OC.7` exact for an aerial scene. It is
    defined for *every* ray, not only the ones where sky is visible, which is exactly what the
    layered composite needs to fill in behind a defocused silhouette.
    """
    elevation, azimuth = sky_angles(directions)
    # Beer-Lambert along the slant path: tau(theta) = tau_zenith^airmass, so the column's
    # emissivity rises toward the horizon and the sky warms as it goes down the frame.
    emissivity = 1.0 - (1.0 - EPS_ZENITH_WINDOW) ** cloud_airmass(elevation)
    l_clear = emissivity * band_radiance(T_AIR_K)
    base_m = lifting_condensation_level_m(T_AIR_K, RH)
    l_cloud = band_radiance(cloud_base_temperature_k(T_AIR_K, base_m, LAPSE_K_PER_M))
    depth = cloud.density(elevation, azimuth)
    return np.asarray(depth * l_cloud + (1.0 - depth) * l_clear)


def _face(
    shape: tuple[int, int],
    centre: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Mask and (u + v) for the parallelogram ``centre + u*a + v*b``, u, v in [0, 1].

    ``u + v`` is what carries depth: the projection of a cube edge onto the view axis is the same
    for both edges of an isometric face, so range is affine in it.
    """
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w]
    px, py = xs - centre[0], ys - centre[1]
    det = a[0] * b[1] - a[1] * b[0]
    u = (px * b[1] - py * b[0]) / det
    v = (a[0] * py - a[1] * px) / det
    inside = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    return inside, np.asarray(u + v)


def sky_cube_scene(  # noqa: PLR0913
    width: int,
    height: int,
    supersample: int,
    distance_m: float,
    side_m: float,
    boresight_deg: float,
    cloud_grid: tuple[int, int] = CLOUD_GRID,
    cloud_fraction: float = CLOUD_FRACTION,
) -> SkyCubeScene:
    """One near cube, drawn isometrically, against a cloudy sky.

    The cube's near corner is at ``distance_m`` and its faces recede from there, so the cube has a
    depth range of its own -- ``side_m * 2 / sqrt(3)`` deep -- and the layered path bins it rather
    than treating it as one plane. At 3 m with a 0.6 m cube that is 0.69 m, worth a pixel of blur
    across the cube when the lens is focused on its front corner.
    """
    h, w = height * supersample, width * supersample
    directions = ray_directions(width, height, supersample, boresight_deg)
    cloud = generate_sky_cloud(
        CLOUD_BETA, cloud_fraction, CLOUD_SEED, n_elevation=cloud_grid[0], n_azimuth=cloud_grid[1]
    )
    background = sky_radiance_field(directions, cloud)

    radius = 0.30 * min(h, w)
    centre = (0.46 * w, 0.56 * h)
    corner = {
        "top": (
            (-0.5 * math.sqrt(3.0) * radius, -0.5 * radius),
            (0.5 * math.sqrt(3.0) * radius, -0.5 * radius),
        ),
        "left": ((-0.5 * math.sqrt(3.0) * radius, -0.5 * radius), (0.0, radius)),
        "right": ((0.5 * math.sqrt(3.0) * radius, -0.5 * radius), (0.0, radius)),
    }
    #: One cube edge projects to side/sqrt(3) along the view axis in an isometric view.
    edge_depth_m = side_m / math.sqrt(3.0)
    radiance = background.copy()
    distance = np.zeros((h, w), dtype=np.float64)
    cube = np.zeros((h, w), dtype=bool)
    # painter's order is irrelevant -- the three faces of a convex solid do not overlap in an
    # isometric projection -- but the top face is laid first so a shared edge resolves to a side
    for name in ("top", "left", "right"):
        a, b = corner[name]
        inside, uv = _face((h, w), centre, a, b)
        new = inside & ~cube
        radiance[new] = band_radiance(FACE_T_K[name])[()]
        distance[new] = distance_m + edge_depth_m * uv[new]
        cube |= inside
    far_m = float(distance[cube].max()) if np.any(cube) else distance_m

    # Patches for the sharpness readout. The sky one sits in cloud clear of the cube. The cube one
    # straddles the **vertical edge where its two side faces meet** -- a 10 K step that is there in
    # every frame -- because a focus measure over flat faces has nothing to measure and would
    # report a demo that works as one that does nothing.
    ph, pw = int(0.16 * h), int(0.16 * w)
    sky_patch = (slice(int(0.04 * h), int(0.04 * h) + ph), slice(int(0.66 * w), int(0.66 * w) + pw))
    cube_patch = (
        slice(int(centre[1] + 0.10 * radius), int(centre[1] + 0.70 * radius)),
        slice(int(centre[0] - 0.25 * radius), int(centre[0] + 0.25 * radius)),
    )
    native = tuple(slice(s.start // supersample, s.stop // supersample) for s in sky_patch)
    native_cube = tuple(slice(s.start // supersample, s.stop // supersample) for s in cube_patch)
    if np.any(cube[sky_patch]):
        raise AssertionError("the sky patch overlaps the cube")
    if not np.all(cube[cube_patch]):
        raise AssertionError("the cube patch runs off the cube")
    return SkyCubeScene(
        radiance_ss=radiance,
        distance_m=distance,
        sky_mask=~cube,
        background_radiance=background,
        cube_mask=cube,
        supersample=supersample,
        near_m=distance_m,
        far_m=far_m,
        sky_patch=native,  # type: ignore[arg-type]
        cube_patch=native_cube,  # type: ignore[arg-type]
    )


def render(
    scene: SkyCubeScene,
    focus_distance_m: float | None,
    bank: DefocusKernelBank,
    max_layers: int = 4,
) -> NDArray[np.float64]:
    """The shipped layered path, then the box filter. Returns native-resolution band radiance."""
    blurred = layered_defocus(
        scene.radiance_ss,
        scene.distance_m,
        bank,
        FOCAL_MM,
        F_NUMBER,
        focus_distance_m,
        sky_mask=scene.sky_mask,
        max_layers=max_layers,
        background_radiance=scene.background_radiance,
    )
    return np.asarray(box_downsample(blurred, scene.supersample))


def blur_circles_um(scene: SkyCubeScene, focus_distance_m: float | None) -> dict[str, float]:
    """Blur circle at the sky and at the cube's near corner, for the readout."""
    sky = (
        0.0
        if focus_distance_m is None
        else float(blur_circle_um(1e9, FOCAL_MM, F_NUMBER, focus_distance_m))
    )
    return {
        "sky": sky,
        "cube": float(blur_circle_um(scene.near_m, FOCAL_MM, F_NUMBER, focus_distance_m)),
    }


def sharpness(scene: SkyCubeScene, native_t_k: NDArray[np.float64]) -> dict[str, float]:
    """`OC.9`'s focus measure over the sky patch and the cube patch, separately.

    Two numbers, not one: a whole-frame measure cannot tell "focused on the cube" from "focused on
    the sky", because both are sharp somewhere. Split by region, the crossover is the demonstration.
    """
    return {
        "sky": focus_measure(native_t_k[scene.sky_patch]),
        "cube": focus_measure(native_t_k[scene.cube_patch]),
    }


def focus_schedule(near_m: float, far_m: float, hold: int, ramp: int) -> list[float]:
    """Hold on the sky, pull to the cube, hold, pull back -- a sequence that loops.

    The ramp is linear in **dioptres**, not in metres. Blur circle goes as |1/s - 1/s_f| for any
    subject well outside the focal length, so a linear-in-dioptre pull changes the blur at a
    constant rate; linear in metres would spend nine tenths of the sweep with nothing moving.
    """
    out = [far_m] * hold
    dioptres = np.linspace(1.0 / far_m, 1.0 / near_m, ramp)
    out += [float(1.0 / q) for q in dioptres]
    out += [near_m] * hold
    out += [float(1.0 / q) for q in dioptres[::-1]]
    return out


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("outputs/focus_sky_demo"))
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--supersample", type=int, default=4)
    ap.add_argument(
        "--distance", type=float, default=3.0, help="range of the cube's near corner, m"
    )
    ap.add_argument("--side", type=float, default=0.6, help="cube side, m")
    ap.add_argument("--boresight", type=float, default=20.0, help="camera elevation, degrees")
    ap.add_argument(
        "--sky-focus", type=float, default=2000.0, help="the 'focused on the sky' end, m"
    )
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--ramp", type=int, default=30)
    ap.add_argument("--fps", type=float, default=12.0)
    args = ap.parse_args()

    from PIL import Image

    from irsim_eval.video import annotate, encode_mp4, ffmpeg_available

    args.out.mkdir(parents=True, exist_ok=True)
    scene = sky_cube_scene(
        args.width, args.height, args.supersample, args.distance, args.side, args.boresight
    )
    bank = DefocusKernelBank(
        0.5 * (BAND_UM[0] + BAND_UM[1]), F_NUMBER, SIGMA_ABERR_UM, PITCH_UM, args.supersample
    )
    span = (235.0, 320.0)
    print(f"cube {scene.near_m:.2f}-{scene.far_m:.2f} m, sky patch {scene.sky_patch}")

    measured = {}
    for name, focus in (("sky", args.sky_focus), ("cube", args.distance)):
        t_k = apparent_temperature(render(scene, focus, bank))
        Image.fromarray(white_hot(t_k, span), mode="L").save(args.out / f"focus_{name}.png")
        c, m = blur_circles_um(scene, focus), sharpness(scene, t_k)
        measured[name] = m
        print(
            f"focus on the {name:4} ({focus:7.1f} m): blur sky {c['sky']:5.1f} um,"
            f" cube {c['cube']:5.1f} um  |  contrast sky {m['sky']:.3e}, cube {m['cube']:.3e}"
        )
    # The demonstration in two numbers: each region is sharpest when the lens is on *it*. A single
    # whole-frame measure cannot show this, because something is in focus either way.
    for region in ("sky", "cube"):
        on, off = measured[region][region], measured["cube" if region == "sky" else "sky"][region]
        print(f"  the {region:4} loses {on / off:4.1f}x of its contrast when the lens leaves it")

    schedule = focus_schedule(args.distance, args.sky_focus, args.hold, args.ramp)
    for i, focus in enumerate(schedule):
        t_k = apparent_temperature(render(scene, focus, bank))
        frame = np.repeat(white_hot(t_k, span)[..., None], 3, axis=2)
        c = blur_circles_um(scene, focus)
        frame = annotate(
            frame,
            [
                f"focus {focus:7.1f} m",
                f"blur  sky {c['sky']:5.1f} um",
                f"      cube {c['cube']:5.1f} um",
            ],
            corner="tl",
        )
        Image.fromarray(frame, mode="RGB").save(args.out / f"sweep_{i:04d}.png")
    print(f"wrote {len(schedule)} sweep frames to {args.out}")

    if not schedule:
        print("no sweep frames requested -- stills only, no video")
    elif ffmpeg_available():
        path = encode_mp4(str(args.out / "sweep_*.png"), args.out / "focus_pull.mp4", fps=args.fps)
        print(f"wrote {path}")
    else:
        print("ffmpeg not on PATH -- PNG sequence kept, no video")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
