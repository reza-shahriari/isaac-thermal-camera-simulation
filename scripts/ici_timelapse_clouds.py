#!/usr/bin/env python3
"""XD.6 part 2: calibrated radiance of real low cloud, read off ARM's ICI deployment time-lapse.

The 92 ICI images fetched for part 1 (ADR 0183) hold almost no low cloud: ERA5 puts their cloudy
minutes in cirrus and mid-level layers. The deployment's time-lapse (``time_lapse.mp4`` in the same
release, doi:10.5439/3001561) is one frame a minute from May to December 2023, and every frame
draws the **calibrated radiance** on a fixed colour scale (``jet``, -5 to 35 W/(m^2 sr), its colour
bar drawn beside it) next to the instrument's own cloud mask. Inverting the colour bar gives the
radiance per pixel back -- quantised to about 0.15 W/(m^2 sr) and through the video codec -- for
any minute of the deployment, daytime cumulus included. Above 35 W/(m^2 sr) (about 0 C in this
band) the scale saturates, and the script counts those pixels instead of guessing them.

The minutes it reads are chosen by ERA5 (Open-Meteo's archive, no account): hours whose low cloud
cover is 25-85 % with mid and high cover under 10 %, so the cloud in the frame is low cloud. For
each such hour it reads six frames ten minutes apart and records, per elevation band, the clear
pixels' median radiance and the cloud pixels' 50th/90th/99th percentiles, plus the surface
temperature and PWV the frame's own title prints and ERA5's dew point (which the lifting
condensation level needs and the title does not carry).

    curl -s "https://archive-api.open-meteo.com/v1/archive?latitude=36.605&longitude=-97.485\\
&start_date=2023-05-01&end_date=2023-12-31&hourly=temperature_2m,dew_point_2m,cloud_cover_low,\\
cloud_cover_mid,cloud_cover_high,total_column_integrated_water_vapour,is_day&timezone=UTC" \\
        > datasets/arm_icii23/era5_sgp_2023.json
    python scripts/ici_timelapse_clouds.py datasets/arm_icii23 \\
        --era5 datasets/arm_icii23/era5_sgp_2023.json \\
        --out data/validation/ici_sgp2023_low_cloud.csv

Needs ``ffmpeg`` on PATH, OpenCV and ``h5py``; the committed CSV is what everything downstream
reads.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as dt
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]

FRAME_W, FRAME_H = 1500, 800
#: Inner pixel boxes (x0, y0, x1, y1, exclusive end) of the panels in the 1500 x 800 frame,
#: measured off the axes' frame lines; the ICI image is 640 x 512 drawn at 0.598.
RADIANCE_BOX = (27, 77, 410, 383)
MASK_BOX = (534, 432, 917, 736)
#: The radiance colour bar: a column inside it, the rows of its 35 and -5 ends.
CBAR_X, CBAR_TOP, CBAR_BOTTOM = 435, 77, 382
CBAR_MAX, CBAR_MIN = 35.0, -5.0
SATURATED = CBAR_MAX - 0.3
ICI_W, ICI_H = 640, 512

ELEVATIONS_DEG = (15, 20, 30, 45, 60, 75, 88)
HALF_WIDTH_DEG = 2.0
MINUTES = (0, 10, 20, 30, 40, 50)
#: Frames between entries of the coarse clock index (a frame is about a minute).
INDEX_STRIDE = 500


def _title_reader() -> object:
    spec = importlib.util.spec_from_file_location(
        "ici_met_from_video", REPO / "scripts" / "ici_met_from_video.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grab(video: pathlib.Path, index: int, fps: float) -> np.ndarray:
    """One frame as RGB uint8, ``(800, 1500, 3)``."""
    raw = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            f"{index / fps:.4f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(FRAME_H, FRAME_W, 3)


def colour_bar(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The bar's colours and their radiance, read off the frame itself (not a textbook ``jet``)."""
    rows = np.arange(CBAR_TOP + 1, CBAR_BOTTOM)
    colours = frame[rows, CBAR_X - 3 : CBAR_X + 4].astype(np.float64).mean(axis=1)
    values = CBAR_MAX + (rows - CBAR_TOP) / (CBAR_BOTTOM - CBAR_TOP) * (CBAR_MIN - CBAR_MAX)
    return colours, values


def invert(panel: np.ndarray, colours: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Nearest colour-bar entry per pixel; NaN where no bar colour is near (text, sun ring)."""
    px = panel.reshape(-1, 3).astype(np.float64)
    out = np.empty(px.shape[0])
    dist = np.empty(px.shape[0])
    for s in range(0, px.shape[0], 20000):
        d = ((px[s : s + 20000, None, :] - colours[None, :, :]) ** 2).sum(axis=2)
        k = d.argmin(axis=1)
        out[s : s + 20000] = values[k]
        dist[s : s + 20000] = np.sqrt(d[np.arange(k.size), k])
    out[dist > 40.0] = np.nan
    return out.reshape(panel.shape[:2])


def ici_coordinates(box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Row and column of the ICI image under each pixel of a panel."""
    x0, y0, x1, y1 = box
    cols = ((np.arange(x0, x1) - x0 + 0.5) * ICI_W / (x1 - x0)).astype(int).clip(0, ICI_W - 1)
    rows = ((np.arange(y0, y1) - y0 + 0.5) * ICI_H / (y1 - y0)).astype(int).clip(0, ICI_H - 1)
    return np.meshgrid(rows, cols, indexing="ij")


def candidate_hours(era5: dict) -> list[dict]:
    """ERA5 hours whose cloud is low: low 25-85 %, mid and high under 10 %."""
    h = era5["hourly"]
    out = []
    for i, t in enumerate(h["time"]):
        lo, mid, hi = h["cloud_cover_low"][i], h["cloud_cover_mid"][i], h["cloud_cover_high"][i]
        if lo is None or mid is None or hi is None:
            continue
        if 25 <= lo <= 85 and mid < 10 and hi < 10:
            out.append(
                {
                    "utc": t,
                    "t2_c": h["temperature_2m"][i],
                    "td_c": h["dew_point_2m"][i],
                    "low": lo,
                    "mid": mid,
                    "high": hi,
                    "is_day": h["is_day"][i],
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="the unpacked ICI release: time_lapse.mp4, the elevation map")
    parser.add_argument("--era5", required=True, help="Open-Meteo archive JSON (see the docstring)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0, help="first N hours only (a quick look)")
    args = parser.parse_args()
    import h5py

    root = pathlib.Path(args.root)
    video = root / "time_lapse.mp4"
    met = _title_reader()
    reader = met.TitleReader(video)  # type: ignore[attr-defined]
    fps = float(met.FPS)  # type: ignore[attr-defined]
    n_frames = int(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "csv=p=0",
                str(video),
            ],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    )

    geo = h5py.File(root / "sky_azimuth_elevation_map.nc", "r")
    elevation = np.asarray(geo["elevation"][:], dtype=np.float64)
    ground = np.asarray(geo["gnd_mask"][:]) != 0
    r_rad, c_rad = ici_coordinates(RADIANCE_BOX)
    r_msk, c_msk = ici_coordinates(MASK_BOX)
    el_rad = elevation[r_rad, c_rad]
    sky_rad = ~ground[r_rad, c_rad]
    # The two panels are drawn at the same scale; the mask is resampled onto the radiance panel.
    mh, mw = r_msk.shape
    rh, rw = r_rad.shape
    my = (np.arange(rh) * mh / rh).astype(int)
    mx = (np.arange(rw) * mw / rw).astype(int)

    times: dict[int, float] = {}

    def epoch(i: int) -> tuple[int, float]:
        for k in (0, *(s * d for d in range(1, 61) for s in (1, -1))):
            j = min(max(i + k, 0), n_frames - 1)
            if j in times:
                return j, times[j]
            p = reader.read(j)  # type: ignore[attr-defined]
            if p:
                times[j] = met._epoch(p[0])  # type: ignore[attr-defined]
                return j, times[j]
        raise RuntimeError(f"no readable title near frame {i}")

    # One coarse index of the video's clock (a title every INDEX_STRIDE frames, ~1 min of reads),
    # then a local search inside the bracket: a binary search over all 224k frames per minute
    # read ~18 titles each and took hours.
    index = [epoch(i) for i in range(0, n_frames, INDEX_STRIDE)]
    index_t = [t for _, t in index]

    def frame_at(t: float) -> int | None:
        k = bisect.bisect_right(index_t, t)
        if k == 0 or k == len(index):
            return None  # before the deployment's first frame or after its last
        lo, hi = index[k - 1][0], index[k][0]
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if epoch(mid)[1] <= t:
                lo = mid
            else:
                hi = mid
        return min((epoch(lo), epoch(hi)), key=lambda q: abs(q[1] - t))[0]

    hours = candidate_hours(json.loads(pathlib.Path(args.era5).read_text(encoding="utf-8")))
    if args.limit:
        hours = hours[: args.limit]
    rows = []
    for n, hour in enumerate(hours):
        print(f"hour {n + 1}/{len(hours)} {hour['utc']}", flush=True)
        start = dt.datetime.strptime(hour["utc"], "%Y-%m-%dT%H:%M").replace(tzinfo=dt.timezone.utc)
        for minute in MINUTES:
            t = start.timestamp() + 60.0 * minute
            j = frame_at(t)
            if j is None:
                continue
            stamp, t_surface, pwv = reader.read(j)  # type: ignore[attr-defined]
            if abs(met._epoch(stamp) - t) > 180.0:  # type: ignore[attr-defined]
                continue  # a gap in the deployment
            frame = grab(video, j, fps)
            colours, values = colour_bar(frame)
            x0, y0, x1, y1 = RADIANCE_BOX
            radiance = invert(frame[y0:y1, x0:x1], colours, values)
            x0, y0, x1, y1 = MASK_BOX
            mask_panel = frame[y0:y1, x0:x1].astype(np.float64).mean(axis=2)
            cloud = (mask_panel > 128.0)[my][:, mx]
            sky = sky_rad & np.isfinite(radiance)
            row = {
                "utc": stamp.replace(" ", "T") + "Z",
                "t_surface_c": t_surface,
                "pwv_cm": pwv,
                "era5_t2_c": hour["t2_c"],
                "era5_td_c": hour["td_c"],
                "era5_low": hour["low"],
                "era5_mid": hour["mid"],
                "era5_high": hour["high"],
                "is_day": hour["is_day"],
                "cloud_fraction": round(float(cloud[sky].mean()), 4) if sky.any() else "",
            }
            for e in ELEVATIONS_DEG:
                band = sky & (np.abs(el_rad - e) < HALF_WIDTH_DEG)
                clear = radiance[band & ~cloud]
                cl = radiance[band & cloud]
                row[f"clear_el{e}"] = round(float(np.median(clear)), 2) if clear.size > 30 else ""
                for q in (50, 90, 99):
                    row[f"cloud_p{q}_el{e}"] = (
                        round(float(np.percentile(cl, q)), 2) if cl.size > 30 else ""
                    )
                row[f"cloud_sat_el{e}"] = (
                    round(float((cl >= SATURATED).mean()), 3) if cl.size > 30 else ""
                )
            rows.append(row)
            print(row["utc"], row["t_surface_c"], row["cloud_fraction"], flush=True)

    out = pathlib.Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write(
            "# ARM Infrared Cloud Imager, SGP 2023 (NWB Sensors; doi:10.5439/3001561; ARM data, "
            "no use constraints, cite the DOI). Low-cloud minutes.\n"
            "# Derived by scripts/ici_timelapse_clouds.py from the deployment time-lapse: radiance "
            "W/(m^2 sr) inverted from the frame's colour bar (-5..35, saturating above;\n"
            "# cloud_sat_elNN is the cloud pixels' saturated share), the instrument's own cloud "
            f"mask, +-{HALF_WIDTH_DEG} deg elevation bands. Surface T and PWV from the frame's "
            "title; dew point and cloud layers from ERA5 (Open-Meteo archive, hourly).\n"
        )
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} frames to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
