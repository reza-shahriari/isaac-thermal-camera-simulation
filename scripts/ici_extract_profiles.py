#!/usr/bin/env python3
"""XD.6: reduce ARM's Infrared Cloud Imager images to the clear-sky profiles the model is held to.

The ICI (NWB Sensors, ARM SGP, May-Dec 2023; doi:10.5439/3001561) is an upward-looking full-sky
LWIR imager whose NetCDF files carry calibrated sky radiance in W/(m^2 sr) per pixel, a cloud mask,
and a separate file mapping every pixel to its sky azimuth and elevation. Those images are 2.3 MB
each and are not in git. What the model comparison needs from each one is small: the median
radiance of the **clear** pixels in a handful of elevation bands, the image's cloud fraction, and
the surface air temperature and precipitable water vapour at that minute.

The last two are not in the image files. They are in the deployment's own time-lapse, printed on
every frame by the instrument's processing ("Surface (C): 28.5  PWV (cm): 4.35"), and
`scripts/ici_met_from_video.py` reads them off it into a JSON this script joins on. That is the
instrument provider's own surface met and GNSS-derived PWV, which is the input its own clear-sky
model ran on.

    python scripts/ici_extract_profiles.py datasets/arm_icii23 \\
        --met datasets/arm_icii23/ici_met_from_video.json \\
        --era5 datasets/arm_icii23/era5_sgp_2023.json \\
        --out data/validation/ici_sgp2023_clear_sky.csv

AT.37 adds the surface **dew point**, which the title does not print and the water column's
shape needs: with it, the surface humidity is measured and the water scale height is the one the
measured PWV implies (``H_w = PWV / w0``), instead of a humidity invented to fit the preset's
2 km. It is ERA5's hourly 2 m dew point at the site (Open-Meteo's archive, no account; the query
is in ``scripts/ici_timelapse_clouds.py``), the hour the image falls in.

Needs ``h5py`` (the files are NetCDF-4 / HDF5), which the project does not otherwise depend on;
the committed CSV is what everything downstream reads.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys

import numpy as np

#: Elevation bands, degrees, each +-1.5 deg. The ICI's sky mask ends near 7 deg, and the zenith band
#: is 88 so that the bin stays a ring rather than a few pixels at the pole.
ELEVATIONS_DEG = (10, 15, 20, 30, 45, 60, 75, 88)
HALF_WIDTH_DEG = 1.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", help="the unpacked ICI request: sky_azimuth_elevation_map.nc, data/"
    )
    parser.add_argument("--met", required=True, help="scripts/ici_met_from_video.py's JSON")
    parser.add_argument(
        "--era5", default=None, help="Open-Meteo ERA5 hourly JSON with dew_point_2m (AT.37)"
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        import h5py
    except ImportError:
        print("this script reads NetCDF-4 and needs h5py: pip install h5py", file=sys.stderr)
        return 2

    root = pathlib.Path(args.root)
    geo = h5py.File(root / "sky_azimuth_elevation_map.nc", "r")
    elevation = np.asarray(geo["elevation"][:], dtype=np.float64)
    ground = np.asarray(geo["gnd_mask"][:]) != 0
    met = {r["file"]: r for r in json.loads(pathlib.Path(args.met).read_text(encoding="utf-8"))}
    dew: dict[str, float] = {}
    if args.era5:
        hourly = json.loads(pathlib.Path(args.era5).read_text(encoding="utf-8"))["hourly"]
        dew = {
            t: float(v)
            for t, v in zip(hourly["time"], hourly["dew_point_2m"], strict=True)
            if v is not None
        }

    rows = []
    for path in sorted(root.glob("data/*/ici_*.nc")):
        f = h5py.File(path, "r")
        radiance = np.asarray(f["sky_radiance_wpm2sr"][:], dtype=np.float64)
        cloud = np.asarray(f["cloud_mask"][:]) != 0
        model = np.asarray(f["modeled_clear_sky_radiance_wpm2sr"][:], dtype=np.float64)
        sky = ~ground & np.isfinite(radiance)
        clear = sky & ~cloud
        m = met.get(path.name)
        if m is None or abs(m["dt_s"]) > 120.0:
            print(f"{path.name}: no met within 2 min, skipped", file=sys.stderr)
            continue
        when = dt.datetime.fromtimestamp(int(f["img_time"][()]), dt.timezone.utc)
        row = {
            "utc": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "t_surface_c": m["t_surface_c"],
            "pwv_cm": m["pwv_cm"],
            "cloud_fraction": round(float(cloud[sky].mean()), 4),
            "td_era5_c": dew.get(when.strftime("%Y-%m-%dT%H:00"), ""),
            "fpa_temp_c": round(float(f["fpa_temp_c"][()]), 2),
            "lens_temp_c": round(float(f["lens_temp_c"][()]), 2),
        }
        for e in ELEVATIONS_DEG:
            band = np.abs(elevation - e) < HALF_WIDTH_DEG
            values = radiance[clear & band]
            row[f"l_el{e}"] = round(float(np.median(values)), 3) if values.size > 50 else ""
            row[f"ici_model_el{e}"] = round(float(np.nanmedian(model[sky & band])), 3)
        rows.append(row)

    out = pathlib.Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as fh:
        fh.write(
            "# ARM Infrared Cloud Imager, SGP 2023 (NWB Sensors; doi:10.5439/3001561; ARM data, "
            "no use constraints, cite the DOI).\n"
            "# Derived by scripts/ici_extract_profiles.py: per image, the median calibrated sky "
            "radiance W/(m^2 sr) of the CLEAR pixels\n"
            f"# (cloud_mask == 0) within +-{HALF_WIDTH_DEG} deg of each elevation (l_elNN), the "
            "instrument's own modelled clear sky there (ici_model_elNN),\n"
            "# its cloud fraction, and the surface T and PWV read off the deployment time-lapse "
            "(scripts/ici_met_from_video.py); td_era5_c is ERA5's 2 m dew point that hour (AT.37). "
            "Band: nominal 7.3-14 um, shape not published.\n"
        )
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} images to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
