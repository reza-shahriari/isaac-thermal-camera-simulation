"""Where a frame's per-pixel data comes from -- the one layer that knows about files.

The viewer never opens a file itself. It asks a :class:`FrameSource` for a run's frames, their
metadata and their planes, and every plane comes back as a NumPy array indexed ``[row, col]``.
That is the swap point: when a render stops writing ``.npy`` files (a Warp/CUDA path that writes
one packed buffer per frame, say), the change is one new reader or one new source, and the
server, the probe and the page are untouched.

Two extension points, smallest first:

* **A new container for the same sidecar layout** -- register a function with
  :func:`register_plane_reader` for its file suffix. ``.npy``, ``.exr`` and ``.png`` are
  registered here.
* **A new layout altogether** -- write a class satisfying :class:`FrameSource` and register a
  factory with :func:`register_source_factory`. :func:`discover_sources` offers every directory
  to every factory, newest registration first, and the first one that recognises it wins.

The layout supported today is :class:`SidecarDirectorySource`: a directory of per-frame JSON
sidecars written by :func:`irsim.io.dataset.write_frame`, each carrying a ``planes`` manifest
(file, dtype, shape, unit) and, optionally, ``legends`` that turn an integer plane into names
(``part_id`` 3 -> ``motor_front_left``). Everything the sidecar says reaches the viewer verbatim.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "PlaneInfo",
    "FrameSource",
    "SidecarDirectorySource",
    "PlaneReader",
    "register_plane_reader",
    "read_plane_file",
    "register_source_factory",
    "discover_sources",
]


@dataclass(frozen=True)
class PlaneInfo:
    """What a source says about one plane of one frame, before anything is read."""

    key: str
    unit: str
    dtype: str
    shape: tuple[int, ...]
    #: integer value -> name, for id planes (``part_id``, ``material_id``); empty otherwise
    legend: dict[int, str] = field(default_factory=dict)

    @property
    def is_image(self) -> bool:
        """Whether the plane can be shown as it is: 8-bit gray, RGB or RGBA."""
        return self.dtype == "uint8"


@runtime_checkable
class FrameSource(Protocol):
    """One run of frames. Implement this to show a new data layout in the viewer."""

    #: a short name the page lists the run under
    name: str

    def frames(self) -> list[str]:
        """Frame names in playback order."""
        ...

    def metadata(self, frame: str) -> dict[str, Any]:
        """Frame-level facts: scene time, range, node temperatures... JSON-serialisable."""
        ...

    def planes(self, frame: str) -> dict[str, PlaneInfo]:
        """Every per-pixel plane this frame has, by key."""
        ...

    def read_plane(self, frame: str, key: str) -> NDArray[Any]:
        """One plane as an array indexed ``[row, col]`` (``[row, col, channel]`` for colour)."""
        ...


# -- plane readers, by file suffix ---------------------------------------------------------------

PlaneReader = Callable[[pathlib.Path], NDArray[Any]]
_READERS: dict[str, PlaneReader] = {}


def register_plane_reader(suffix: str, reader: PlaneReader) -> None:
    """Teach every sidecar source to read ``*.<suffix>`` files (``suffix`` without the dot)."""
    _READERS[suffix.lower().lstrip(".")] = reader


def _read_npy(path: pathlib.Path) -> NDArray[Any]:
    # Memory-mapped: a probe reads one pixel, so a 640x512 float32 plane costs one page, not 1.3 MB.
    return np.asarray(np.load(path, mmap_mode="r", allow_pickle=False))


def _read_exr(path: pathlib.Path) -> NDArray[Any]:
    from irsim.io.exr import read_exr

    return read_exr(path)


def _read_png(path: pathlib.Path) -> NDArray[Any]:
    from irsim.io.png import read_png

    return read_png(path)


register_plane_reader("npy", _read_npy)
register_plane_reader("exr", _read_exr)
register_plane_reader("png", _read_png)


@lru_cache(maxsize=64)
def _cached_read(path: pathlib.Path, mtime_ns: int) -> NDArray[Any]:
    reader = _READERS.get(path.suffix.lower().lstrip("."))
    if reader is None:
        raise ValueError(
            f"no plane reader for {path.suffix!r} files -- register one with "
            "irsim_viewer.source.register_plane_reader"
        )
    return reader(path)


def read_plane_file(path: pathlib.Path) -> NDArray[Any]:
    """Read a plane file with whichever reader its suffix is registered to, cached by mtime."""
    return _cached_read(path, path.stat().st_mtime_ns)


# -- the sidecar layout written by irsim.io.dataset ----------------------------------------------


def _is_sidecar(doc: Any) -> bool:
    return isinstance(doc, dict) and isinstance(doc.get("planes"), dict)


class SidecarDirectorySource:
    """A directory of ``<frame>.json`` sidecars with their plane files beside them.

    Frames without a sidecar (a clip's in-between frames under ``plane_stride``) are not listed:
    there is no per-pixel data behind them to probe, only a picture.
    """

    def __init__(self, directory: pathlib.Path, *, name: str | None = None) -> None:
        self.directory = pathlib.Path(directory)
        self.name = name if name is not None else self.directory.name
        self._docs: dict[str, dict[str, Any]] = {}
        for path in sorted(self.directory.glob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if _is_sidecar(doc):
                self._docs[path.stem] = doc
        self._order = sorted(self._docs, key=lambda f: (self._docs[f].get("frame_index", 0), f))

    @classmethod
    def recognise(cls, directory: pathlib.Path) -> SidecarDirectorySource | None:
        """The factory :func:`discover_sources` calls: a source if any sidecar is here."""
        source = cls(directory)
        return source if source.frames() else None

    def frames(self) -> list[str]:
        return list(self._order)

    def _doc(self, frame: str) -> dict[str, Any]:
        try:
            return self._docs[frame]
        except KeyError:
            raise KeyError(f"no frame {frame!r} in {self.directory}") from None

    def metadata(self, frame: str) -> dict[str, Any]:
        return {k: v for k, v in self._doc(frame).items() if k not in ("planes", "legends")}

    def planes(self, frame: str) -> dict[str, PlaneInfo]:
        doc = self._doc(frame)
        legends = doc.get("legends") or {}
        return {
            key: PlaneInfo(
                key=key,
                unit=str(entry.get("unit", "unspecified")),
                dtype=str(entry.get("dtype", "")),
                shape=tuple(int(n) for n in entry.get("shape", ())),
                legend={int(k): str(v) for k, v in (legends.get(key) or {}).items()},
            )
            for key, entry in doc["planes"].items()
        }

    def read_plane(self, frame: str, key: str) -> NDArray[Any]:
        entry = self._doc(frame)["planes"].get(key)
        if entry is None:
            raise KeyError(f"frame {frame!r} has no plane {key!r}")
        path = (self.directory / str(entry["file"])).resolve()
        if self.directory.resolve() not in path.parents:
            raise ValueError(f"plane file {entry['file']!r} points outside {self.directory}")
        return read_plane_file(path)


# -- discovery -----------------------------------------------------------------------------------

SourceFactory = Callable[[pathlib.Path], "FrameSource | None"]
_FACTORIES: list[SourceFactory] = []


def register_source_factory(factory: SourceFactory) -> None:
    """Offer every discovered directory to ``factory`` before the ones registered earlier."""
    _FACTORIES.insert(0, factory)


register_source_factory(SidecarDirectorySource.recognise)


def discover_sources(root: pathlib.Path, *, max_depth: int = 3) -> list[FrameSource]:
    """Every run under ``root`` (``root`` itself included), named by its path relative to it."""
    root = pathlib.Path(root)
    found: list[FrameSource] = []
    directories = [root] + sorted(
        p for p in root.rglob("*") if p.is_dir() and len(p.relative_to(root).parts) <= max_depth
    )
    for directory in directories:
        for factory in _FACTORIES:
            source = factory(directory)
            if source is not None:
                rel = directory.relative_to(root).as_posix()
                source.name = root.name if rel == "." else rel
                found.append(source)
                break
    return found
