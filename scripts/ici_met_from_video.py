#!/usr/bin/env python3
"""XD.6: surface temperature and PWV for each ICI image, read off the deployment's own time-lapse.

ARM's ICI release (doi:10.5439/3001561) carries per-image radiance and cloud products but not the
surface met and precipitable water vapour the instrument's clear-sky model ran on. Those are
printed in the title of every frame of the time-lapse attached to the release::

    2023-07-17 01:54:00  Surface (˚C): 28.5  PWV (cm): 4.35

This script reads that title: the frames are matplotlib text in one font at one size, so a
glyph-template reader is exact rather than approximate. The templates are learned from seven
frames whose titles were transcribed by eye (:data:`KNOWN`), which between them contain every
character the title uses; the reader then binary-searches the (chronological) video for the frame
nearest each image's own timestamp and records how far apart they are (``dt_s``, at most a minute
here, so the met is that minute's).

    python scripts/ici_met_from_video.py datasets/arm_icii23 \\
        --out datasets/arm_icii23/ici_met_from_video.json

Needs ``ffmpeg`` on PATH, OpenCV, and ``h5py`` for the images' timestamps. A misread digit shows as
a step in the series; the values read here change smoothly minute to minute.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys
from typing import Any

import numpy as np

CROP = (450, 10, 600, 36)  # x, y, w, h of the title in the 1500 x 800 frame
FPS = 15.0

#: (seconds into the video, the title as read by eye). Together they hold every glyph the title
#: uses -- the digit 6 first appears in the fourth -- so the reader never meets an unlearned one.
KNOWN = [
    (5000, "2023-07-17 01:54:00 Surface (˚C): 28.5 PWV (cm): 4.35"),
    (100, "2023-05-20 20:17:00 Surface (˚C): 21.3 PWV (cm): 1.19"),
    (9000, "2023-10-05 12:24:00 Surface (˚C): 15.1 PWV (cm): 2.94"),
    (2000, "2023-06-11 01:58:00 Surface (˚C): 25.0 PWV (cm): 3.09"),
    (3000, "2023-06-21 22:02:00 Surface (˚C): 29.6 PWV (cm): 3.93"),
    (7000, "2023-08-14 13:51:00 Surface (˚C): 20.8 PWV (cm): 3.63"),
    (14000, "2023-11-30 04:14:00 Surface (˚C): 7.3 PWV (cm): 1.57"),
]

PATTERN = re.compile(
    r"(\d{4}-\d\d-\d\d)(\d\d:\d\d:\d\d)Surface\(˚C\):(-?[\d.]+)PWV?\(cm\):([\d.]+)"
)
#: Glyph pairs the renderer draws touching, which come out as one connected component.
LIGATURES = {"rf": "\x01", "WV": "\x02"}


def frame_title(video: pathlib.Path, index: int) -> np.ndarray:
    x, y, w, h = CROP
    raw = subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-ss",
            f"{index / FPS:.4f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            f"crop={w}:{h}:{x}:{y}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w)


def _components(gray: np.ndarray) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    import cv2

    ink = (gray < 140).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    boxes = sorted(tuple(int(v) for v in stats[i][:4]) for i in range(1, n) if stats[i][4] >= 2)
    return [((bx, by, bw, bh), 255 - gray[by : by + bh, bx : bx + bw]) for bx, by, bw, bh in boxes]


def _descriptor(glyph: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    import cv2

    g = cv2.resize(glyph.astype(np.float32), (10, 14), interpolation=cv2.INTER_AREA)
    g = g / (np.linalg.norm(g) + 1e-6)
    _, y, w, h = box
    return np.concatenate([g.ravel(), [w / 10.0, h / 14.0, y / 14.0]])


class TitleReader:
    """Nearest-template glyph classifier, trained on :data:`KNOWN`."""

    def __init__(self, video: pathlib.Path) -> None:
        self.video = video
        self.templates: list[tuple[str, np.ndarray]] = []
        for seconds, text in KNOWN:
            self._learn(frame_title(video, int(seconds * FPS)), text)

    def _learn(self, gray: np.ndarray, text: str) -> None:
        comps = _components(gray)
        for merged in ((), ("WV",)):
            t = text.replace("rf", LIGATURES["rf"])
            for pair in merged:
                t = t.replace(pair, LIGATURES[pair])
            labels: list[str] = []
            for c in (c for c in t if c != " "):
                labels.extend([c, c] if c == ":" else [c])  # a colon is two dots
            if len(labels) == len(comps):
                break
        if len(labels) != len(comps):
            raise ValueError(f"cannot align {len(comps)} glyphs with {text!r}")
        self.templates += [(c, _descriptor(g, b)) for (b, g), c in zip(comps, labels, strict=True)]

    def read(self, index: int) -> tuple[str, float, float] | None:
        chars = []
        for box, glyph in _components(frame_title(self.video, index)):
            d = _descriptor(glyph, box)
            chars.append(min(self.templates, key=lambda t: float(np.linalg.norm(d - t[1])))[0])
        text = "".join(chars).replace("::", ":")
        for pair, code in LIGATURES.items():
            text = text.replace(code, pair)
        m = PATTERN.search(text)
        return (
            None
            if m is None
            else (f"{m.group(1)} {m.group(2)}", float(m.group(3)), float(m.group(4)))
        )


def _epoch(stamp: str) -> float:
    return (
        dt.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="the unpacked ICI request: time_lapse.mp4, data/")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        import h5py
    except ImportError:
        print("the image timestamps are NetCDF-4 and need h5py: pip install h5py", file=sys.stderr)
        return 2
    root = pathlib.Path(args.root)
    video = root / "time_lapse.mp4"
    n_frames = int(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                str(video),
            ],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    )
    reader = TitleReader(video)
    cache: dict[int, tuple[int, tuple[str, float, float]]] = {}

    def at(i: int) -> tuple[int, tuple[str, float, float]]:
        if i not in cache:
            for k in (0, 1, -1, 2, -2, 3):  # a frame mid-transition can blur the title
                j = min(max(i + k, 0), n_frames - 1)
                read = reader.read(j)
                if read is not None:
                    cache[i] = (j, read)
                    break
            else:
                raise RuntimeError(f"no readable title near frame {i}")
        return cache[i]

    out: list[dict[str, Any]] = []
    for path in sorted(root.glob("data/*/ici_*.nc")):
        t = float(h5py.File(path, "r")["img_time"][()])
        lo, hi = 0, n_frames - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if _epoch(at(mid)[1][0]) <= t else (lo, mid)
        j, (stamp, t_c, pwv) = min((at(lo), at(hi)), key=lambda q: abs(_epoch(q[1][0]) - t))
        out.append(
            {
                "file": path.name,
                "img_time": t,
                "video_frame": j,
                "video_time": stamp,
                "dt_s": _epoch(stamp) - t,
                "t_surface_c": t_c,
                "pwv_cm": pwv,
            }
        )
        print(out[-1], flush=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
