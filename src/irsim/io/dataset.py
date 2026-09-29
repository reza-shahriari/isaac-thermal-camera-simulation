"""Write one frame's four outputs to disk, in units, with a sidecar that says what they are.

docs/physics-model.md §12.2 ``outputs``; roadmap M10.10a; CLAUDE.md non-negotiable #2 (the disk
rule); ADR 0004 (hashes travel with data).

§12.2 names four outputs and they do not want the same container:

==============  =======================  ==================================================
output          file                     why
==============  =======================  ==================================================
``radiance``    ``.npy`` or ``.exr``     W/m2/sr (or photons/s/m2/sr) -- float32, always
``apparent_t``  ``.npy`` or ``.exr``     kelvin -- float32, always
``dn16``        ``.png`` (uint16)        an ADC code is an integer; nothing is lost
``display8``    ``.png`` (RGBA8)         already 8-bit by construction
==============  =======================  ==================================================

The rule the table encodes: **a plane in physical units never goes to an integer container.**
A 16-bit PNG spanning 200--400 K quantises to 3 mK, which sounds harmless next to a 50 mK NETD
until you remember that NETD is the noise floor the whole sensor model is anchored to -- and it
would look exactly right. ``.npy`` is the default because NumPy reads it back bit-exactly with no
dependency; ``.exr`` is there because no image tool reads ``.npy`` (:mod:`irsim.io.exr`).

**The sidecar is the point of the exercise.** A directory of arrays nobody can trace back to a
configuration is not a dataset, it is a pile of images. Each frame is written with a JSON sidecar
carrying the config and band hashes (ADR 0008), the ISP hash of the display branch, the frame
index, the scene time in seconds and the wall-clock UTC that time corresponds to on the scene's
weather axis, and, for every file, its dtype, shape and **unit**. Two frames that disagree about
any of those are two different cameras, and the hashes are what make that checkable later instead
of arguable.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from irsim.io.exr import write_exr
from irsim.io.png import write_png

__all__ = [
    "FloatFormat",
    "PLANE_UNITS",
    "FrameRecord",
    "FrameWriter",
    "write_float_plane",
    "write_frame",
    "read_float_plane",
]

#: Container for the planes that carry physical units. Both are float32 and lossless.
FloatFormat = Literal["npy", "exr"]

#: What each written plane means. Carried in the sidecar so a consumer never has to guess whether
#: a number is kelvin, watts or a code -- the mistake that makes a dataset quietly useless.
PLANE_UNITS: dict[str, str] = {
    "radiance": "W/m^2/sr (energy form) or photons/s/m^2/sr (photon form) -- see quantity",
    "apparent_t": "K",
    "dn16": "ADC code (uint16, 0..2^bit_depth-1)",
    "display8": "RGBA8 display code (no physical unit; AGC and palette applied)",
    "signal_dn": "DN before quantisation (float32)",
    "flux": "W per pixel (energy form) or photons/s per pixel (photon form)",
    "semantic_id": "semantic class id (uint32)",
    "rgb": (
        "visible-light RGBA8 from the same camera prim, pose and lens -- a registered companion "
        "image with NO infrared content; nothing in the radiometric chain reads it"
    ),
    "instance_id": "renderer instance id (uint32)",
    # AT.29: the cloud the infrared band marched between the camera and each pixel's hit.
    "cloud_transmittance": (
        "fraction of what lies behind the cloud that reaches the camera (1 = no cloud short of "
        "the hit)"
    ),
    "cloud_range_m": "m, where the cloud's emission toward the camera is centred (NaN if none)",
    "cloud_id": "0 = clear, 1 = cloud (transmittance under one half) -- see legend",
    # Scene truth under each pixel centre (irsim.io.truth): what was there, not what the camera saw.
    "temperature_k": (
        "K -- true surface temperature the radiometry was given, at the pixel centre "
        "(sky pixels: apparent sky temperature)"
    ),
    "distance_m": "m -- range from the camera to the surface along the ray (NaN = sky)",
    "material_id": "material index (uint16; names in legends)",
    "part_id": "part index (uint16; names in legends, 0 = sky / no geometry)",
    "node_id": "thermal node index (uint16; names in legends, 0 = sky / no geometry)",
}


@dataclass(frozen=True)
class FrameRecord:
    """What was written for one frame: the files, and the sidecar that describes them."""

    directory: pathlib.Path
    sidecar: pathlib.Path
    files: dict[str, pathlib.Path]
    metadata: dict[str, Any]


def _require_float32_or_better(array: Any, name: str) -> NDArray[np.floating]:
    arr = np.asarray(array)
    if arr.dtype == np.float16:
        raise TypeError(
            f"{name} is float16: at 300 K its spacing is 0.25 K against a 50 mK NETD, so it would "
            "reach disk looking correct with the sensitivity already gone (CLAUDE.md #2)"
        )
    if not np.issubdtype(arr.dtype, np.floating):
        raise TypeError(f"{name} must be a float plane, got {arr.dtype}")
    return arr


def write_float_plane(
    path: str | os.PathLike[str], array: Any, *, fmt: FloatFormat = "npy"
) -> pathlib.Path:
    """Write one plane in physical units as float32, refusing float16 (CLAUDE.md #2)."""
    arr = np.ascontiguousarray(_require_float32_or_better(array, str(path)), dtype=np.float32)
    out = pathlib.Path(path).with_suffix(f".{fmt}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(out, arr, allow_pickle=False)
    elif fmt == "exr":
        write_exr(out, arr)
    else:  # pragma: no cover - Literal is checked by mypy
        raise ValueError(f"unknown float format {fmt!r}")
    return out


def read_float_plane(path: str | os.PathLike[str]) -> NDArray[np.float32]:
    """Read a plane back as float32, whichever container it went to."""
    from irsim.io.exr import read_exr

    p = pathlib.Path(path)
    if p.suffix == ".npy":
        return np.asarray(np.load(p, allow_pickle=False), dtype=np.float32)
    if p.suffix == ".exr":
        return read_exr(p)
    raise ValueError(f"not a float plane container: {p.suffix!r}")


def write_frame(
    directory: str | os.PathLike[str],
    outputs: Any,
    *,
    name: str | None = None,
    frame_index: int = 0,
    t_s: float | None = None,
    start_utc: datetime | None = None,
    config_hash: str | None = None,
    band_hash: str | None = None,
    quantity: str | None = None,
    float_format: FloatFormat = "npy",
    extra_planes: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    legends: dict[str, dict[int, str]] | None = None,
) -> FrameRecord:
    """Write an :class:`~irsim.pipeline.frame.Outputs` and its sidecar into ``directory``.

    ``name`` defaults to ``frame_{frame_index:06d}``, so a sequence sorts lexicographically --
    which is what every downstream tool assumes and what a bare ``frame_9``/``frame_10`` breaks.
    An output that is ``None`` (a flag turned off in ``sensor.outputs``) is simply not written;
    it is never written as zeros, because a zero plane is indistinguishable from a real one.

    ``t_s`` is the scene time on the weather axis and ``start_utc`` the weather series' own start,
    so the sidecar can state the wall-clock time the frame represents. Without them the frame is
    still written and the fields are recorded as null rather than invented.

    ``legends`` names the values of integer planes (``{"part_id": {3: "motor_front_left"}}``);
    it is written to the sidecar under ``legends`` with string keys, as JSON requires.
    """
    out_dir = pathlib.Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = name if name is not None else f"frame_{frame_index:06d}"

    files: dict[str, pathlib.Path] = {}
    planes: dict[str, dict[str, Any]] = {}

    def record(key: str, path: pathlib.Path, arr: NDArray[Any]) -> None:
        files[key] = path
        planes[key] = {
            "file": path.name,
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "unit": PLANE_UNITS.get(key, "unspecified"),
        }

    for key in ("radiance", "apparent_t"):
        value = getattr(outputs, key, None)
        if value is None:
            continue
        plane = np.ascontiguousarray(_require_float32_or_better(value, key), dtype=np.float32)
        record(key, write_float_plane(out_dir / f"{stem}_{key}", plane, fmt=float_format), plane)

    dn16 = getattr(outputs, "dn16", None)
    if dn16 is not None:
        codes: NDArray[Any] = np.asarray(dn16)
        if codes.dtype != np.uint16:
            raise TypeError(f"dn16 must be uint16, got {codes.dtype}")
        path = out_dir / f"{stem}_dn16.png"
        write_png(path, codes)
        record("dn16", path, codes)

    display8 = getattr(outputs, "display8", None)
    if display8 is not None:
        rgba: NDArray[Any] = np.asarray(display8)
        if rgba.dtype != np.uint8:
            raise TypeError(f"display8 must be uint8, got {rgba.dtype}")
        path = out_dir / f"{stem}_display8.png"
        write_png(path, rgba)
        record("display8", path, rgba)

    for key, value in (extra_planes or {}).items():
        other: NDArray[Any] = np.asarray(value)
        if np.issubdtype(other.dtype, np.floating):
            record(
                key, write_float_plane(out_dir / f"{stem}_{key}", other, fmt=float_format), other
            )
        elif other.dtype in (np.uint8, np.uint16):
            path = out_dir / f"{stem}_{key}.png"
            write_png(path, other)
            record(key, path, other)
        else:
            path = out_dir / f"{stem}_{key}.npy"
            np.save(path, other, allow_pickle=False)
            record(key, path, other)

    utc = None
    if start_utc is not None and t_s is not None:
        moment = start_utc + timedelta(seconds=float(t_s))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        utc = moment.isoformat()

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "frame_index": int(frame_index),
        "name": stem,
        "t_s": None if t_s is None else float(t_s),
        "utc": utc,
        "config_hash": config_hash,
        "band_hash": band_hash,
        "isp_hash": getattr(outputs, "isp_hash", None),
        "quantity": quantity,
        "float_format": float_format,
        "planes": planes,
    }
    if legends:
        metadata["legends"] = {
            key: {str(ident): str(name) for ident, name in table.items()}
            for key, table in legends.items()
        }
    metadata.update(extra_metadata or {})

    sidecar = out_dir / f"{stem}.json"
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return FrameRecord(directory=out_dir, sidecar=sidecar, files=files, metadata=metadata)


@dataclass(frozen=True)
class FrameWriter:
    """Bind one run's invariants so a render driver's loop writes a frame in a single line.

    docs/decisions/0068 (an 8-bit frame cannot carry a radiometric claim); roadmap IG.13.

    :func:`write_frame` already does the work, but it takes nine keyword arguments of which
    seven are constant for the whole run -- the sensor's config and band hashes, the radiometric
    quantity, the container, the weather series' start, the per-run metadata. Spelling those out
    at a call site inside a 500-line driver is how three of this project's six drivers ended up
    writing 8-bit PNGs and nothing else: the frame loop was the easy part and the bookkeeping was
    not. Binding them once turns the loop body into ``writer.write(outputs, frame_index=i, ...)``
    and makes "does this driver write planes at all" a one-line question.

    ``stride`` thins the written sequence without thinning the render. A 300-frame time-lapse at
    640x512 is 786 MB of float32 radiance and apparent temperature, and 3.1 GB at the NIR array's
    1280x1024 -- enough that a four-band sweep of two scenes fills a disk. ``stride=4`` writes
    every fourth frame's planes; ``stride=0`` writes none, which is the only honest way to spell
    "this run makes no radiometric claim". The value travels in every sidecar as ``plane_stride``
    so a consumer reading ``frame_000025`` beside no ``frame_000024`` can tell a thinned sequence
    from a failed one.
    """

    directory: pathlib.Path
    config_hash: str | None = None
    band_hash: str | None = None
    quantity: str | None = None
    float_format: FloatFormat = "npy"
    start_utc: datetime | None = None
    stride: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stride < 0:
            raise ValueError(f"stride must be >= 0 (0 disables plane writing), got {self.stride}")
        object.__setattr__(self, "directory", pathlib.Path(self.directory))

    def wants(self, frame_index: int) -> bool:
        """Whether this frame's planes are written, so a caller can skip preparing them."""
        return self.stride > 0 and int(frame_index) % self.stride == 0

    def write(
        self,
        outputs: Any,
        *,
        frame_index: int,
        t_s: float | None = None,
        name: str | None = None,
        extra_planes: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
        legends: dict[str, dict[int, str]] | None = None,
    ) -> FrameRecord | None:
        """Write one frame, or return ``None`` if ``stride`` skips it.

        Per-frame ``extra_metadata`` is layered over the run's own, so a driver can add what
        changes (a throttle, a slant range) without restating what does not.
        """
        if not self.wants(frame_index):
            return None
        merged: dict[str, Any] = {**self.metadata, "plane_stride": int(self.stride)}
        merged.update(extra_metadata or {})
        return write_frame(
            self.directory,
            outputs,
            name=name,
            frame_index=frame_index,
            t_s=t_s,
            start_utc=self.start_utc,
            config_hash=self.config_hash,
            band_hash=self.band_hash,
            quantity=self.quantity,
            float_format=self.float_format,
            extra_planes=extra_planes,
            extra_metadata=merged,
            legends=legends,
        )
