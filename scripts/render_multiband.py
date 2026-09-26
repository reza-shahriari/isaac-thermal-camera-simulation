#!/usr/bin/env python3
"""Render each demo scene in all four bands, with a registered visible companion.

    python.sh scripts/render_multiband.py                 # everything
    python.sh scripts/render_multiband.py --scene drone --band nir
    python.sh scripts/render_multiband.py --contact-sheet-only

One scene, four cameras, one command. The point is the comparison: the *same* geometry, the same
weather, the same sun and the same instant, through four spectral bands whose physics could hardly
be more different. A quadrotor is four hot spots against a cold sky in LWIR and a black silhouette
against a bright sky in NIR, and those are not two renderings of one picture -- they are two
different detection problems, which is the argument for modelling bands as data.

**Exposure is per band and is physics, not taste.** A daylight reflective-band scene saturates a
low-light exposure by around 120x (measured: a 0.3-albedo surface in full sun puts 1.17e6
photoelectrons into a NIR pixel in 16 ms against a 1e4 well). The NIR file is authored at its
daylight exposure; SWIR keeps its own file -- its §9.4 NETD of 976 K is quoted in three places --
and is exposed here with `--integration-ms`, which changes the config hash exactly as it should.

Needs Isaac Sim and a GPU. Everything lands under ``outputs/multiband/<scene>/<band>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
#: The interpreter the *child* renders run under. Isaac Sim's `python.sh` sets up an environment
#: that its own `sys.executable` does not reproduce when re-invoked bare, so it is taken from
#: $IRSIM_PYTHON when set (ADR 0002) and only falls back to this process's interpreter.
DEFAULT_PYTHON = os.environ.get("IRSIM_PYTHON") or sys.executable

#: scene -> (render script, extra arguments, frame count). Every scene is filmed, so every count
#: has to be long enough to *watch*: the ship's 8 predates the maritime script encoding video at
#: all, and at 30 fps it is a quarter-second clip. The maritime camera is static where the two
#: aerial ones fly, so its 90 frames buy less motion -- but the sea state, the AGC and the noise
#: all move, and those are what a still frame cannot show.
SCENES: dict[str, tuple[str, list[str], int]] = {
    "drone": ("render_quad_flight.py", [], 150),
    "airplane": ("render_aircraft_pass.py", [], 150),
    "ship": ("render_maritime_demo.py", [], 90),
    # IG.13: the sweep covered three of the project's eight scene configs, so five of them had
    # never been filmed in any band but the one their own driver defaults to -- including the
    # whole aerial point-target lane, which is the one the owner ranked first.
    "sky_target": ("render_aerial_demo.py", [], 90),
    # PT.9's rendered point-wise aerial scenes (ADR 0123, ADR 0124). Worth filming in every band
    # for the same reason the point-target lane is: the sunlit skin's excess over air comes from
    # absorbed sunlight, so how much survives into the picture is a question about the band, and
    # a reflective one reads the same airframe by what it reflects rather than by what it emits.
    # Both are named explicitly rather than left to the driver's default, because one driver now
    # flies two aircraft and a default is not a statement about which.
    "drone_outbound": (
        "render_quad_outbound.py",
        [
            "--airframe",
            "heavy_lift",
            "--scene",
            str(REPO / "configs/scenes/quad_outbound_pointwise.yaml"),
        ],
        150,
    ),
    # The Phantom is the band question at its sharpest: white ABS is a near-blackbody in LWIR and
    # a strong *reflector* in NIR/SWIR, so the shell that is barely above air in the thermal
    # bands is the brightest thing in frame in the reflective ones.
    "phantom3": (
        "render_quad_outbound.py",
        [
            "--airframe",
            "phantom3",
            "--scene",
            str(REPO / "configs/scenes/phantom3_outbound_pointwise.yaml"),
        ],
        150,
    ),
    "vessel_departure": ("render_vessel_departure.py", [], 90),
    # The two car scenes differ only in cloud cover, and that is the point of filming both: an
    # overcast night sky radiates near air temperature and a clear one is 30 K colder, so the
    # same bonnet reads a different contrast against it. Both are *night* scenes -- their SWIR
    # and NIR frames are near-black by construction, which is the phenomenology and not a
    # failure, since neither band has a source once the sun is down.
    # 30 frames, not the driver's own 26: at 10 fps that is exactly the 3 s a clip needs to be
    # watchable, and at the 60 s frame period the last capture lands at T+1740 s -- inside the
    # 1800 s the scene's load schedules are authored over, which a 31st frame would leave.
    "car_overcast": ("render_car_ignition.py", [], 30),
    "car_clear": (
        "render_car_ignition.py",
        ["--scene", str(REPO / "configs/scenes/car_ignition_clear_night.yaml")],
        30,
    ),
}

#: Scene configs deliberately outside the sweep, with the reason. `test_render_multiband` checks
#: this accounts for every file in `configs/scenes/` that `SCENES` does not name, so a new scene
#: cannot be added and quietly left unfilmed.
UNSWEPT_SCENES: dict[str, str] = {
    "sky_only.yaml": (
        "SC.20's Tier 4 reference: nothing but sky, so the camera's own radial shading is all "
        "a frame holds. It is measured, not filmed: `scripts/validate_sky_flat.py` renders it "
        "through one LWIR camera with a controlled housing drift and fits the bowl. A band sweep "
        "of an empty sky would be four pictures of four housings and say nothing about bands."
    ),
    "phantom4_parts.yaml": (
        "AI.5's reference scene: the Phantom 4 solved on its **functional parts** rather than its "
        "material prims (ADR 0138), with four motors, four ESC mounts and a battery as separate "
        "thermal nodes. Unswept for the same reason as `phantom4_pointwise.yaml` and one more: it "
        "needs a *part-split* archive and USD that `scripts/prep_asset.py --asset phantom4 "
        "--emit-parts` generates and git does not carry, so a sweep on a fresh checkout would "
        "fail for a missing file. It is filmed by `scripts/render_phantom4.py --asset "
        "phantom4_parts`, which is a single-asset driver rather than a band sweep."
    ),
    "phantom4_pointwise.yaml": (
        "AI.2's reference scene: the first in this project whose geometry was not authored in "
        "Python. It needs a mesh archive that is generated rather than committed -- "
        "`scripts/prep_asset.py --asset phantom4 --emit-mesh` writes 18 MB from a 62 MB source "
        "FBX that is not in git either -- so a sweep that ran it on a fresh checkout would fail "
        "for a missing file rather than a missing camera. It also has no camera or prims yet: "
        "authoring the asset onto an Isaac stage is the in-engine half and is still open."
    ),
    "car_exhaust_plume.yaml": (
        "PH.6's reference scene: a tailpipe at cruise load and the gas cone it blows, one "
        "exhaust target and no camera or prims yet. Its numbers are tests/unit/test_plume.py's "
        "-- tau 0.866 in MWIR against 0.979 in LWIR from one authored plume -- measured on a "
        "synthetic G-buffer; the rendered frames are IG.2's, and when a driver places the car "
        "and its camera it joins the sweep."
    ),
    "vessel_pointwise_clear_day.yaml": (
        "PT.10's reference scene: a weather-deck field and two faces of one deckhouse prim on a "
        "vessel held still, with no driver yet. `irsim_isaac.vessel_pointwise` authors the prims "
        "the patches name, but nothing places a camera on the water in front of them -- the "
        "maritime drivers all film the *moving* vessel of `vessel_departure_clear_day.yaml`, "
        "which carries no patches. Its numbers are tests/unit/test_vessel_pointwise.py's, "
        "measured engine-free; the rendered frames are IG.2's, and it joins the sweep when a "
        "driver stands a camera off the beam."
    ),
    "thermal_facet_scene.yaml": (
        "the §6.13 facet bench: seven surfaces, no camera and no prims. It is driven by "
        "scripts/validate_thermal_diurnal.py, which produces a diurnal curve, not a frame."
    ),
    "wall_half_in_sun.yaml": (
        "PT.20's reference scene: seven patches on one building and its ground, no camera and "
        "no prims yet. scripts/wall_half_in_sun.py writes the engine-free frame from a synthetic "
        "G-buffer; the rendered one is IG.2's."
    ),
    "quad_flight_mesh.yaml": (
        "WM.7's reference scene: the same mission with the two arms as meshed tubes, no camera "
        "and no prims yet. scripts/quad_flight_mesh.py writes the unrolled frames and a video "
        "from a synthetic G-buffer through the mesh bridge; the rendered ones are IG.2's, and "
        "binding a mesh field to a real asset's triangles is the follow-on to WM.7."
    ),
    "quad_flight_pointwise.yaml": (
        "PT.9's regenerated aerial scene: the deck, the belly and two arms of a quadrotor as "
        "patched surfaces, no camera and no prims yet. scripts/quad_flight_pointwise.py writes "
        "the engine-free plan frames from a synthetic G-buffer; the rendered ones are IG.2's, "
        "and the curved-prim binding a fuselage needs is WM.3's."
    ),
    "parked_car_cabin.yaml": (
        "PT.15's reference scene: five panels of a schematic saloon and the cabin behind four "
        "of them, no camera and no prims yet. scripts/parked_car_cabin.py writes the "
        "engine-free plan frame from a synthetic G-buffer; the rendered one is IG.2's."
    ),
    "wet_road_noon.yaml": (
        "PH.2's reference field: one road patch, half wet, part shaded, no camera and no prims "
        "yet. Its numbers are tests/unit/test_wet_road.py's; the frame is IG.2's, and when a "
        "driver authors the road it joins the sweep."
    ),
}

#: band -> (sensor config, extra arguments). `--integration-ms` appears only where the camera's own
#: default would saturate in daylight, and the number is the measured 60 %-of-well exposure.
BANDS: dict[str, tuple[str, list[str]]] = {
    "lwir": ("flir_boson_640_lwir.yaml", []),
    "mwir": ("example_mwir_insb_640.yaml", []),
    "swir": ("example_swir_ingaas_640.yaml", ["--integration-ms", "0.08"]),
    "nir": ("example_nir_si_1280.yaml", []),
}


def _out_dir(root: pathlib.Path, scene: str, band: str) -> pathlib.Path:
    return root / scene / band


def render(
    scene: str,
    band: str,
    root: pathlib.Path,
    frames: int | None,
    subframes: int,
    python: str = DEFAULT_PYTHON,
) -> bool:
    script, scene_args, default_frames = SCENES[scene]
    sensor, band_args = BANDS[band]
    out = _out_dir(root, scene, band)
    command = [
        python,
        str(REPO / "scripts" / script),
        "--out", str(out),
        "--frames", str(frames if frames is not None else default_frames),
        "--rt-subframes", str(subframes),
        "--rgb",
        "--sensor", str(REPO / "configs" / "sensors" / sensor),
        *scene_args,
        *band_args,
    ]  # fmt: skip
    print(f"\n=== {scene} / {band} ===\n{' '.join(command)}", flush=True)
    started = time.time()
    result = subprocess.run(command, cwd=REPO, check=False)
    print(f"--- {scene}/{band}: exit {result.returncode} in {time.time() - started:.0f} s")
    return result.returncode == 0


#: Where in the clip the contact sheet samples. **Not frame 0**: these are time-lapses of a
#: process, the manual display span covers the whole flight, and at t = 0 the motors are still at
#: ambient -- so the first frame is the least informative one in the sequence.
SHEET_FRACTION = 0.7


def _first_frame(
    video: pathlib.Path, destination: pathlib.Path, fraction: float = SHEET_FRACTION
) -> bool:
    """Pull one frame out of an encoded video. The PNGs themselves are deleted after encoding."""
    if not video.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
         str(video)],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    seek: list[str] = []
    try:
        seek = ["-ss", f"{float(probe.stdout.strip()) * fraction:.3f}"]
    except ValueError:
        seek = []
    return (
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", *seek, "-i", str(video), "-vframes", "1",
             str(destination)],
            check=False,
        ).returncode
        == 0
    )  # fmt: skip


def contact_sheet(scene: str, root: pathlib.Path) -> pathlib.Path | None:
    """One PNG per scene: the visible companion above the four bands, all at one instant.

    Each band is resampled to a common width. The bands have different arrays and different fields
    of view -- that is the honest thing to show, since it is what a real four-camera mast looks
    like -- so the sheet is labelled with each camera's own resolution rather than pretending they
    are registered to one another. The *visible* companion **is** registered, pixel for pixel, to
    the band it was filmed with (ADR 0073).
    """
    import numpy as np

    from irsim.io.png import write_png
    from irsim_eval.data import _decode_png
    from irsim_eval.video import annotate

    scratch = root / scene / "_sheet"

    def sample(directory: pathlib.Path, video_suffix: str, png_suffix: str, tag: str):  # type: ignore[no-untyped-def]
        """One representative frame, from a video **or** from per-frame PNGs.

        The two aerial scripts encode videos and the maritime one writes a file per frame
        (`frame_000000_display8.png`), because a four-frame static demo does not want a container.
        The sheet has to read both or the ship column comes out empty.
        """
        videos = sorted(directory.glob(f"*{video_suffix}"))
        if videos:
            out = scratch / f"{tag}.png"
            return np.asarray(_decode_png(out))[..., :3] if _first_frame(videos[0], out) else None
        stills = sorted(directory.glob(f"frame_*{png_suffix}"))
        if not stills:
            return None
        index = min(int(len(stills) * SHEET_FRACTION), len(stills) - 1)
        return np.asarray(_decode_png(stills[index]))[..., :3]

    tiles: list[tuple[str, np.ndarray]] = []
    for band in BANDS:
        tile = sample(_out_dir(root, scene, band), "_ir.mp4", "_display8.png", band)
        if tile is not None:
            tiles.append((band.upper(), tile))
    if not tiles:
        return None
    visible = sample(_out_dir(root, scene, "lwir"), "_rgb.mp4", "_rgb.png", "rgb")
    if visible is not None:
        tiles.insert(0, ("VISIBLE", visible))

    # One common tile size by nearest-neighbour resampling, **not** by cropping: the NIR camera is
    # 1280x1024 and the rest are 640x512, and cropping would quietly cut the taller frame in half
    # and show a different part of the scene beside the others.
    height = min(tile.shape[0] for _, tile in tiles)
    width = min(tile.shape[1] for _, tile in tiles)
    columns = []
    for label, tile in tiles:
        rows_idx = (np.arange(height) * tile.shape[0] // height).clip(0, tile.shape[0] - 1)
        cols_idx = (np.arange(width) * tile.shape[1] // width).clip(0, tile.shape[1] - 1)
        small = tile[rows_idx][:, cols_idx]
        # The label goes on its **own strip** above the tile. Annotating over the frame put it on
        # top of the render's own readout and made both unreadable.
        strip = annotate(
            np.zeros((40, width, 3), dtype=np.uint8),
            [f"{label}   {tile.shape[1]}x{tile.shape[0]}"],
        )[..., :3]
        columns.append(np.concatenate([strip, small], axis=0))
    sheet = np.concatenate(columns, axis=1)
    path = root / scene / f"{scene}_contact_sheet.png"
    write_png(path, np.ascontiguousarray(sheet))
    return path


def merged_index(
    root: pathlib.Path, status: dict[str, str], sheets: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Fold this run's results into whatever `index.json` already records.

    The manifest describes the *output tree*, not the invocation that last touched it, and the
    two diverge the moment anyone passes `--scene` or `--band`: re-rendering the ship alone used
    to leave an index.json claiming the directory held four ship renders and nothing else, while
    twelve videos sat next to it. Entries this run produced win -- a scene re-rendered from ok to
    failed must say failed -- and entries it did not touch are carried through.

    A missing or unreadable index is treated as an empty one. It is a convenience file that any
    run can rebuild, so refusing to start because it is corrupt would trade a real render for a
    bookkeeping error.
    """
    previous: dict[str, dict[str, str]] = {}
    try:
        loaded = json.loads((root / "index.json").read_text("utf-8"))
        if isinstance(loaded, dict):
            previous = loaded
    except (OSError, ValueError):
        previous = {}

    def fold(key: str, fresh: dict[str, str]) -> dict[str, str]:
        kept = previous.get(key)
        merged = dict(kept) if isinstance(kept, dict) else {}
        merged.update(fresh)
        return merged

    return {"renders": fold("renders", status), "contact_sheets": fold("contact_sheets", sheets)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", choices=sorted(SCENES))
    parser.add_argument("--band", action="append", choices=sorted(BANDS))
    parser.add_argument("--out", default=str(REPO / "outputs" / "multiband"))
    parser.add_argument("--frames", type=int, default=None, help="override every scene's count")
    parser.add_argument("--rt-subframes", type=int, default=4)
    parser.add_argument("--contact-sheet-only", action="store_true")
    parser.add_argument(
        "--python",
        default=DEFAULT_PYTHON,
        help="interpreter for the child renders; default $IRSIM_PYTHON or this one",
    )
    args = parser.parse_args(argv)

    root = pathlib.Path(args.out)
    scenes = args.scene or list(SCENES)
    bands = args.band or list(BANDS)
    status: dict[str, str] = {}
    if not args.contact_sheet_only:
        for scene in scenes:
            for band in bands:
                ok = render(scene, band, root, args.frames, args.rt_subframes, args.python)
                status[f"{scene}/{band}"] = "ok" if ok else "failed"
    sheets = {}
    for scene in scenes:
        path = contact_sheet(scene, root)
        if path is not None:
            sheets[scene] = str(path)
            print(f"contact sheet: {path}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.json").write_text(
        json.dumps(merged_index(root, status, sheets), indent=2, sort_keys=True),
        "utf-8",
    )
    failed = [k for k, v in status.items() if v != "ok"]
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
