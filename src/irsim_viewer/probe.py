"""Read every plane of a frame at one clicked pixel, and say how that pixel was found.

A click lands on the image the page is showing, whose grid is not always every plane's grid. A
G-buffer id plane may be supersampled (ADR 0014: ids are never filtered, so anti-aliasing is done
by supersampling them), and a future writer may store a plane at another resolution. The rule:

* **same shape** -- the pixel is read directly;
* **an integer multiple** (either way) -- the pixel maps to the matching block, and for a
  supersampled plane the sample nearest the pixel's centre is read. That sample is the surface
  under the pixel centre, which is not the same as the pixel's value when the pixel straddles an
  edge; the note says so;
* **anything else** -- the pixel is mapped proportionally and the reading carries a **warning**,
  because a click that reads a neighbouring pixel's temperature looks exactly like a correct one.

Nothing here knows what a plane means. Units come from the source, names from its legends, so a
plane a render starts writing tomorrow shows up in the viewer without an edit.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from irsim_viewer.source import FrameSource, PlaneInfo

__all__ = ["map_pixel", "probe", "KELVIN_TO_CELSIUS"]

#: Offset of the Celsius scale, used only to print a kelvin reading twice. Not a physical constant.
KELVIN_TO_CELSIUS = 273.15


def map_pixel(
    x: int, y: int, shown: tuple[int, int], plane: tuple[int, int]
) -> tuple[int, int, str | None, bool]:
    """Map pixel ``(x, y)`` of an image of shape ``shown`` (rows, cols) onto ``plane``.

    Returns ``(px, py, note, exact)``; ``exact`` is False when the grids do not nest.
    """
    sh, sw = shown
    ph, pw = plane
    if (ph, pw) == (sh, sw):
        return x, y, None, True
    if ph % sh == 0 and pw % sw == 0 and ph // sh == pw // sw:
        k = ph // sh
        return x * k + k // 2, y * k + k // 2, f"{k}x supersampled: sample at pixel centre", True
    if sh % ph == 0 and sw % pw == 0 and sh // ph == sw // pw:
        k = sh // ph
        return x // k, y // k, f"plane is {k}x coarser than the image", True
    px = min(pw - 1, int(x * pw / sw))
    py = min(ph - 1, int(y * ph / sh))
    return px, py, f"plane {ph}x{pw} does not nest in image {sh}x{sw}: mapped proportionally", False


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _is_kelvin(unit: str) -> bool:
    """``"K"``, or ``"K -- explanation"``: the unit is the first word of the sidecar's string."""
    return unit.split(" ", 1)[0] == "K"


def _reading(info: PlaneInfo, value: Any) -> dict[str, Any]:
    arr = np.asarray(value)
    out: dict[str, Any] = {"key": info.key, "unit": info.unit}
    if arr.ndim == 0:
        out["value"] = _scalar(arr[()])
        if info.legend:
            ident = int(arr)
            out["label"] = info.legend.get(ident, "(not in legend)" if ident else "(none)")
        elif _is_kelvin(info.unit) and np.issubdtype(arr.dtype, np.floating):
            out["celsius"] = _scalar(np.float64(arr) - KELVIN_TO_CELSIUS)
    else:
        out["value"] = [_scalar(v) for v in arr.reshape(-1)]
    return out


def probe(source: FrameSource, frame: str, x: int, y: int, shown: str) -> dict[str, Any]:
    """Every plane of ``frame`` at pixel ``(x, y)`` of the plane ``shown`` (the image clicked).

    Labels found in a legend are also looked up in the frame metadata's dictionaries, so a pixel
    labelled ``motor_front_left`` also reports ``node_temperatures_k[motor_front_left]`` -- the
    solver's node temperature beside the pixel's own.
    """
    planes = source.planes(frame)
    if shown not in planes:
        raise KeyError(f"frame {frame!r} has no plane {shown!r}")
    sh, sw = planes[shown].shape[:2]
    if not (0 <= x < sw and 0 <= y < sh):
        raise ValueError(f"pixel ({x}, {y}) is outside the {sw}x{sh} image")

    readings: list[dict[str, Any]] = []
    warnings: list[str] = []
    for key, info in planes.items():
        if len(info.shape) < 2:
            continue
        px, py, note, exact = map_pixel(x, y, (sh, sw), (info.shape[0], info.shape[1]))
        try:
            data = source.read_plane(frame, key)
        except (OSError, ValueError, KeyError) as err:
            readings.append({"key": key, "unit": info.unit, "error": str(err)})
            continue
        if tuple(data.shape[:2]) != info.shape[:2]:
            warnings.append(f"{key}: file is {data.shape[:2]}, sidecar says {info.shape[:2]}")
            px, py, note, exact = map_pixel(x, y, (sh, sw), (data.shape[0], data.shape[1]))
        reading = _reading(info, data[py, px])
        if note:
            reading["note"] = note
        if not exact:
            warnings.append(f"{key}: {note}")
        readings.append(reading)

    metadata = source.metadata(frame)
    joins: list[dict[str, Any]] = []
    for reading in readings:
        label = reading.get("label")
        if not label:
            continue
        for meta_key, table in metadata.items():
            if not (isinstance(table, dict) and label in table):
                continue
            join = {"key": f"{meta_key}[{label}]", "value": _scalar(table[label])}
            if join not in joins:  # part and node often share a name
                joins.append(join)

    return {
        "frame": frame,
        "x": x,
        "y": y,
        "shown": shown,
        "readings": readings,
        "joins": joins,
        "warnings": warnings,
    }
