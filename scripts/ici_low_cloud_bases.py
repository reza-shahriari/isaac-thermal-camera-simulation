#!/usr/bin/env python3
"""XD.6 part 2: give each low-cloud ICI minute its observed cloud base and that base's temperature.

    python scripts/ici_low_cloud_bases.py data/validation/ici_sgp2023_low_cloud.csv \\
        --asos datasets/arm_icii23/asos_PNC_2023.csv datasets/arm_icii23/asos_WDG_2023.csv \\
        --profiles datasets/arm_icii23/profiles_sgp_2023.json

The ICI has no ceilometer, and ERA5's cloud cover by level is model output that can put a
3 km cloud at 1.2 km. The two airport stations nearest the ARM SGP site -- Ponca City (KPNC) and
Enid Woodring (KWDG), each about 37 km away -- report their ceilometer's lowest layer every hour
(Iowa Environmental Mesonet's ASOS archive, no account)::

    https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=PNC&data=tmpc&data=dwpc
      &data=skyc1&data=skyl1&data=skyc2&data=skyl2&data=skyc3&data=skyl3&year1=2023&month1=5
      &day1=15&year2=2023&month2=12&day2=31&tz=Etc/UTC&format=onlycomma&latlon=yes&missing=M

A minute gets ``obs_base_m`` (above the site, the stations' mean) only when both stations
reported a layer within 40 minutes of it and the two agree within 500 m: low cloud is regional,
but not uniform at 37 km. Its temperature, ``t_base_c``, is the hour's air temperature at that
height from the reanalysis profile at the site (Open-Meteo's historical forecast API; pressure
levels 1000-600 hPa with their geopotential heights) -- a temperature analysis, which is far
better constrained than a cloud one. The columns are written back into the CSV in place.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import sys

import numpy as np

#: The site's elevation, m: the reanalysis heights are above sea level, the bases above ground.
SITE_ELEVATION_M = 310.0
#: Two stations must agree this closely for their base to stand for the site's.
AGREE_M = 500.0
#: A report counts for a minute this far either side of it.
WINDOW_S = 2400.0
LEVELS_HPA = (1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600)
FT_M = 0.3048


def _epoch(stamp: str, fmt: str) -> float:
    return dt.datetime.strptime(stamp, fmt).replace(tzinfo=dt.timezone.utc).timestamp()


def read_asos(path: pathlib.Path) -> list[tuple[float, float | None]]:
    """``(time, lowest reported layer's base in m or None)`` per report."""
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            base = None
            for k in (1, 2, 3):
                cover, height = r[f"skyc{k}"].strip(), r[f"skyl{k}"]
                if cover in ("FEW", "SCT", "BKN", "OVC", "VV") and height not in ("M", ""):
                    base = float(height) * FT_M
                    break
            out.append((_epoch(r["valid"], "%Y-%m-%d %H:%M"), base))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="scripts/ici_timelapse_clouds.py's CSV, updated in place")
    parser.add_argument("--asos", nargs=2, required=True, help="two stations' ASOS CSVs")
    parser.add_argument("--profiles", required=True, help="Open-Meteo pressure-level JSON")
    args = parser.parse_args()

    path = pathlib.Path(args.csv)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    header = [line for line in lines if line.startswith("#")]
    rows = list(csv.DictReader(line for line in lines if not line.startswith("#")))
    stations = [read_asos(pathlib.Path(p)) for p in args.asos]
    hourly = json.loads(pathlib.Path(args.profiles).read_text(encoding="utf-8"))["hourly"]
    index = {t: i for i, t in enumerate(hourly["time"])}

    def base_at(t: float) -> tuple[float | None, float | None]:
        bases = []
        for reports in stations:
            near = [r for r in reports if abs(r[0] - t) <= WINDOW_S]
            if not near:
                return None, None
            base = min(near, key=lambda r: abs(r[0] - t))[1]
            if base is None:
                return None, None
            bases.append(base)
        spread = max(bases) - min(bases)
        return (float(np.mean(bases)), spread) if spread <= AGREE_M else (None, spread)

    def temperature_at(stamp: str, height_m: float) -> float | None:
        i = index.get(stamp[:13] + ":00")
        if i is None:
            return None
        z, temp = [], []
        for level in LEVELS_HPA:
            zz = hourly[f"geopotential_height_{level}hPa"][i]
            tt = hourly[f"temperature_{level}hPa"][i]
            if zz is None or tt is None or zz < SITE_ELEVATION_M:
                continue
            z.append(zz - SITE_ELEVATION_M)
            temp.append(tt)
        return float(np.interp(height_m, z, temp)) if len(z) >= 2 else None

    found = 0
    for r in rows:
        base, spread = base_at(_epoch(r["utc"], "%Y-%m-%dT%H:%M:%SZ"))
        t_base = None if base is None else temperature_at(r["utc"], base)
        r["obs_base_m"] = "" if base is None else round(base, 0)
        r["obs_base_spread_m"] = "" if spread is None else round(spread, 0)
        r["t_base_c"] = "" if t_base is None else round(t_base, 2)
        found += base is not None

    note = (
        "# obs_base_m: the KPNC and KWDG ceilometers' lowest layer (mean, both within 40 min and "
        "500 m of each other; scripts/ici_low_cloud_bases.py); t_base_c: the reanalysis air "
        "temperature at that height above the site.\n"
    )
    header = [h for h in header if not h.startswith("# obs_base_m")] + [note]
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.writelines(header)
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{found} of {len(rows)} minutes have an agreed observed base")
    return 0


if __name__ == "__main__":
    sys.exit(main())
