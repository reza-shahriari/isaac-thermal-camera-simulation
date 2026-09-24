"""Engine-availability probes. Importing this module never imports an engine.

`irsim_isaac` must be importable on a machine with no Isaac Sim and no GPU (the unit gate runs
there), so every engine import in this package happens inside a function, after one of these
probes. `tests/integration/conftest.py` uses them to skip the whole directory cleanly.

Detection is by `importlib.util.find_spec`, which locates a module without executing it. Whether
`isaacsim` resolves outside a running Kit application is exactly the kind of Isaac Sim 6.0 detail
the public docs leave open; roadmap M2.1 verifies it. Set `IRSIM_FORCE_NO_ISAAC=1` (or
`IRSIM_FORCE_NO_WARP=1`) to make the probes answer False, which is how the skip path is tested.

Warp needs one extra step. It ships as the `omni.warp.core` Kit extension, and Kit is what puts
that extension on `sys.path`; ADR 0014 recorded it as "importable only inside a running Kit". It
is in fact an ordinary Python package sitting in the build's extension cache, and
:func:`ensure_warp_on_path` finds it and adds it, so `import warp` — and both its `cpu` and
`cuda:0` devices — work from plain `python.sh` with no Kit boot (ADR 0014 addendum, 2026-09-12).
That is what makes the CPU-vs-GPU equivalence harness runnable in seconds instead of behind a
35 s Kit startup. A pip-installed Warp, if one is ever present, always wins: the cache directory
is appended only when `warp` does not already resolve.
"""

from __future__ import annotations

import ctypes
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "render_device",
    "GPU_ENV_VAR",
    "ensure_openvdb_on_path",
    "ensure_warp_on_path",
    "has_isaac",
    "has_openvdb",
    "has_warp",
    "isaac_root",
    "openvdb_extension_path",
    "require_isaac",
    "require_openvdb",
    "require_warp",
    "simulation_app_config",
    "warp_extension_path",
]

#: The extension cache entry that holds the Warp package (`omni.warp.core-<version>+<platform>`).
WARP_EXTENSION_GLOB = "extscache/omni.warp.core-*"

#: The extension cache entry that holds the OpenVDB and NanoVDB Python bindings.
VOLUME_EXTENSION_GLOB = "extscache/omni.volume-*"

#: Shared libraries `openvdb.cpython-*.so` is linked against, in dependency order. The module
#: carries **no RPATH**, so nothing resolves them unless they are already in the process; the
#: alternative is making every caller export `LD_LIBRARY_PATH` before Python starts, which a
#: library cannot do for itself.
VOLUME_PRELOAD_LIBS = ("libtbb.so.12", "libtbbmalloc.so.2", "libopenvdb.so.12.0")


def _spec_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def has_isaac() -> bool:
    """True when the Isaac Sim Python package (`isaacsim`) is importable."""
    if os.environ.get("IRSIM_FORCE_NO_ISAAC") == "1":
        return False
    return _spec_present("isaacsim")


def isaac_root() -> Path | None:
    """The Isaac Sim build root: `$ISAAC_PATH` if set (python.sh exports it), else the directory
    three levels above the `isaacsim` package (`<root>/python_packages/isaacsim/__init__.py`)."""
    env = os.environ.get("ISAAC_PATH")
    if env:
        root = Path(env)
        return root if root.is_dir() else None
    try:
        spec = importlib.util.find_spec("isaacsim")
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    root = Path(spec.origin).resolve().parents[2]
    return root if root.is_dir() else None


def warp_extension_path() -> Path | None:
    """Directory to put on `sys.path` so that `import warp` resolves, or None if there is none.

    `$IRSIM_WARP_PATH` overrides the search for installs laid out differently. Otherwise the
    build's extension cache is globbed; if it holds several versions the lexicographically last
    is taken, which orders 1.16.0 after 1.9.0 incorrectly but matters only on a build that ships
    two Warps, and the chosen one is verified to contain the package before it is returned.
    """
    override = os.environ.get("IRSIM_WARP_PATH")
    if override:
        candidate = Path(override)
        return candidate if (candidate / "warp" / "__init__.py").is_file() else None
    root = isaac_root()
    if root is None:
        return None
    found = [
        p for p in sorted(root.glob(WARP_EXTENSION_GLOB)) if (p / "warp" / "__init__.py").is_file()
    ]
    return found[-1] if found else None


def ensure_warp_on_path() -> Path | None:
    """Make `import warp` resolve outside Kit; returns the directory added, else None.

    Idempotent, and a no-op when Warp already resolves (so a pip or Kit copy is never shadowed).
    Only `sys.path` is touched: nothing is imported, so this stays safe on a machine with no
    Isaac Sim and no GPU.
    """
    if os.environ.get("IRSIM_FORCE_NO_WARP") == "1" or _spec_present("warp"):
        return None
    ext = warp_extension_path()
    if ext is None:
        return None
    entry = str(ext)
    if entry not in sys.path:
        sys.path.append(entry)
    importlib.invalidate_caches()
    return ext


def has_warp() -> bool:
    """True when NVIDIA Warp is importable (does not check for a CUDA device).

    Calls :func:`ensure_warp_on_path` when Warp does not resolve yet, so the answer reflects the
    Warp this package would actually import rather than whether Kit happens to be running.
    """
    if os.environ.get("IRSIM_FORCE_NO_WARP") == "1":
        return False
    if _spec_present("warp"):
        return True
    ensure_warp_on_path()
    return _spec_present("warp")


def openvdb_extension_path() -> Path | None:
    """Directory to put on `sys.path` so that `import openvdb` resolves, or None if there is none.

    `$IRSIM_OPENVDB_PATH` overrides the search. Otherwise the build's extension cache is globbed
    for `omni.volume`, which is where Isaac Sim keeps OpenVDB and NanoVDB.
    """
    override = os.environ.get("IRSIM_OPENVDB_PATH")
    if override:
        candidate = Path(override)
        return candidate if _has_openvdb_module(candidate) else None
    root = isaac_root()
    if root is None:
        return None
    found = [p for p in sorted(root.glob(VOLUME_EXTENSION_GLOB)) if _has_openvdb_module(p)]
    return found[-1] if found else None


def _has_openvdb_module(directory: Path) -> bool:
    return any(directory.glob("openvdb*.so")) or (directory / "openvdb" / "__init__.py").is_file()


def ensure_openvdb_on_path() -> Path | None:
    """Make `import openvdb` resolve outside Kit; returns the directory added, else None.

    **This is why `irsim_isaac.cloud_volume` no longer hand-builds a NanoVDB grid.** ADR 0127
    recorded that "there is no OpenVDB writer in this environment -- no `pyopenvdb` on any
    interpreter here", and on that premise the deck was voxelised through Warp, whose grid header
    then had to be re-stamped and whose file metadata had to be back-filled by hand, and which
    the renderer refused anyway. The premise is wrong on this build: `omni.volume` ships a full
    OpenVDB 12 binding (file format 224) and a NanoVDB one beside it. They are simply not
    importable until someone does the two things Kit would have done -- preload the shared
    libraries the extension module names but cannot find, and put the directory on `sys.path`.

    Idempotent, and a no-op when `openvdb` already resolves, so a pip copy is never shadowed.
    Preloading opens shared libraries but imports nothing and starts no Kit application, so this
    stays safe on a machine with no GPU.
    """
    if os.environ.get("IRSIM_FORCE_NO_OPENVDB") == "1" or _spec_present("openvdb"):
        return None
    ext = openvdb_extension_path()
    if ext is None:
        return None
    for name in VOLUME_PRELOAD_LIBS:
        library = ext / "bin" / name
        if library.is_file():
            # RTLD_GLOBAL so the extension module's own NEEDED entries resolve against these.
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    entry = str(ext)
    if entry not in sys.path:
        sys.path.append(entry)
    importlib.invalidate_caches()
    return ext


def has_openvdb() -> bool:
    """True when the OpenVDB Python binding is importable (does not check for a GPU)."""
    if os.environ.get("IRSIM_FORCE_NO_OPENVDB") == "1":
        return False
    if _spec_present("openvdb"):
        return True
    ensure_openvdb_on_path()
    return _spec_present("openvdb")


def require_openvdb() -> None:
    if not has_openvdb():
        raise RuntimeError(
            "This code path needs OpenVDB. It ships inside Isaac Sim's `omni.volume` extension: "
            "run through the Isaac Sim interpreter, or point $IRSIM_OPENVDB_PATH at a directory "
            "containing an `openvdb` Python module (docs/decisions/0140)."
        )


def require_isaac() -> None:
    """Raise a clear error at the call site that needs the engine, not deep inside it."""
    if not has_isaac():
        raise RuntimeError(
            "This code path needs Isaac Sim (`isaacsim` is not importable). Run it through the "
            "Isaac Sim interpreter (docs/decisions/0002) or use the engine-free path in `irsim`."
        )


def require_warp() -> None:
    if not has_warp():
        raise RuntimeError(
            "This code path needs NVIDIA Warp. It ships as the `omni.warp.core` Kit extension: "
            "run through the Isaac Sim interpreter, or point $IRSIM_WARP_PATH at a directory "
            "containing the `warp` package (docs/decisions/0014 addendum)."
        )


#: Which GPU the renderer runs on. An index, or ``all`` for Kit's own multi-GPU behaviour.
GPU_ENV_VAR = "IRSIM_GPU"

#: The A6000 on this machine. The other card is the one the owner works on, so a render that
#: spreads onto it takes memory somebody is using.
DEFAULT_GPU = 0


def render_device() -> str:
    """The CUDA device string for work that runs beside the render, e.g. ``"cuda:0"``.

    The same card ``simulation_app_config`` pins the renderer to, read from the same variable, so
    a helper that allocates on the GPU (AT.12 voxelises a cloud deck through Warp) cannot land on
    the card the owner is working on while the renderer sits on the other one. ``all`` gives back
    the bare ``"cuda"``, which is the caller's framework choosing.
    """
    # The same remapping `simulation_app_config` applies, because an index only means the card
    # `nvidia-smi` prints once CUDA is ordered by PCI bus. A caller that reaches this function
    # first would otherwise get the *other* card, which is the one the owner is working on.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    raw = os.environ.get(GPU_ENV_VAR, "").strip()
    if raw.lower() in {"all", "*"}:
        return "cuda"
    try:
        return f"cuda:{DEFAULT_GPU if not raw else int(raw)}"
    except ValueError:
        raise ValueError(f"{GPU_ENV_VAR}={raw!r} is neither a GPU index nor 'all'") from None


def simulation_app_config(**overrides: Any) -> dict[str, Any]:
    """The ``SimulationApp`` config every script boots with, with the render GPU pinned.

    ``$IRSIM_GPU`` selects the card: an index (the default is ``0``), or ``all`` to hand the
    choice back to Kit. It is an environment variable rather than a flag because all eleven
    entry points -- six render drivers, four probes and the material audit -- need the same
    answer, and because it has to be set before ``SimulationApp`` is constructed, which happens
    at import time.

    **``CUDA_VISIBLE_DEVICES`` does not do this job.** Kit picks a *Vulkan* device, so a render
    launched under it still allocates its multi-GPU render graph on every card and dies with
    ``ERROR_OUT_OF_DEVICE_MEMORY`` when another one is busy -- measured 2026-09-16, a car render
    that reached ``app ready`` and then segfaulted while the second GPU held 26 GB of somebody
    else's job. ``active_gpu`` and ``multi_gpu`` are the keys Isaac turns into
    ``--/renderer/activeGpu=`` and ``--/renderer/multiGpu/enabled=``, which do.
    """
    # Make every GPU index in this project mean the one `nvidia-smi` prints. CUDA's own default
    # is FASTEST_FIRST, which on this machine puts the 5090 at index 0 and the A6000 at 1 -- the
    # reverse of nvidia-smi's PCI order, so `CUDA_VISIBLE_DEVICES=0` selects the card it looks
    # like it excludes (measured 2026-09-16: Warp reported `cuda:0` as the 5090 under it). Set
    # before `SimulationApp` is constructed, which is before CUDA initialises; an explicit value
    # in the environment still wins.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    config: dict[str, Any] = {"headless": True}
    raw = os.environ.get(GPU_ENV_VAR, "").strip()
    if raw.lower() not in {"all", "*"}:
        try:
            gpu = DEFAULT_GPU if not raw else int(raw)
        except ValueError:
            raise ValueError(f"{GPU_ENV_VAR}={raw!r} is neither a GPU index nor 'all'") from None
        config["active_gpu"] = gpu
        config["multi_gpu"] = False
    config.update(overrides)
    return config
