"""The cloud deck as real geometry in the stage: NanoVDB on disk, `UsdVol.Volume` in USD
(AT.12, ADR 0127).

The visible companion used to paint cloud into the environment-dome texture (ADR 0073/0076). A
dome light is at infinity, so that cloud had no distance, no thickness and no inside: a camera
could not move relative to it and no ray had a length within it. This module puts the *same*
:class:`~irsim.atmosphere.cloud_deck.CloudDeck` the infrared band marches into the stage as a
participating medium, so the path tracer integrates the field the radiometry integrates.

**Why NanoVDB and not OpenVDB.** There is no OpenVDB writer in this environment -- no `pyopenvdb`
on any interpreter here, and the format is not something to hand-roll -- but Isaac Sim ships
`omni.warp.core`, and `warp.Volume` round-trips a dense NumPy array to a `.nvdb` file. Omniverse
documents `.nvdb` as loadable beside `.vdb`, and `UsdVol.OpenVDBAsset` is the prim either goes on.
That last sentence is the one piece of this module that public documentation does not settle for
Isaac Sim 6.0, and it is **verified by rendering**: if the volume does not load, the visible frame
comes back with no cloud in it, which is loud rather than subtle.

**The grid is named**, because `UsdVol.Volume`'s field relationship binds by name and Warp's
allocator leaves the name empty. The name is written into the NanoVDB grid header before the file
is saved, and the header's checksum is set to NanoVDB's own "not computed" sentinel rather than
left stale -- a stale checksum is exactly the kind of thing a loader is entitled to reject.

**Units.** The grid carries *visible extinction per metre*, which is what
:meth:`CloudDeck.density_at` returns, so the material's density multiplier is 1 and the column
optical depth a renderer integrates is the optical depth the environment preset authored. Nothing
here is in radiometric units for the infrared band and nothing in the infrared path reads it:
this is the visible companion, and ADR 0073's rule that its photometry is a legibility aid rather
than a measurement still holds.

Engine glue by construction: `warp` and `pxr` are imported inside the functions, so importing this
module on a machine with neither still works and `tests/unit/test_layering.py` stays green.
"""

from __future__ import annotations

import ctypes
import pathlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from irsim.atmosphere.cloud_deck import DEFAULT_CELL_M, CloudDeck
from irsim_isaac.env import ensure_warp_on_path

__all__ = [
    "DENSITY_GRID_NAME",
    "NANOVDB_NO_CHECKSUM",
    "CloudVolume",
    "write_nanovdb",
    "author_cloud_volume",
]

#: The NanoVDB grid's name, and the `UsdVol.Volume` field relationship that binds to it. Both
#: sides must agree; they are the same constant so they cannot drift.
DENSITY_GRID_NAME = "density"

#: NanoVDB's sentinel for "this grid carries no checksum". Written after the grid name is patched,
#: because the name lives inside the region a full checksum covers.
NANOVDB_NO_CHECKSUM = 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True)
class CloudVolume:
    """What was written, in the numbers a render log should carry."""

    path: pathlib.Path
    voxel_m: float
    shape: tuple[int, int, int]
    min_world_m: tuple[float, float, float]
    max_density_per_m: float
    file_bytes: int

    @property
    def caption(self) -> str:
        nx, ny, nz = self.shape
        return (
            f"{nx}x{ny}x{nz} voxels at {self.voxel_m:g} m "
            f"({self.file_bytes / 1e6:.1f} MB), peak {self.max_density_per_m:.5f} /m"
        )


class _GridData(ctypes.Structure):
    """The head of a NanoVDB `GridData` header, to the two fields this module edits.

    Only the prefix is declared, because only the prefix is touched and a partial declaration
    cannot accidentally write past what it names. The layout is NanoVDB's and is the same one
    `warp.Volume.save_to_nvdb` reads when it writes the file's own metadata.
    """

    _fields_ = (
        ("magic", ctypes.c_uint64),
        ("checksum", ctypes.c_uint64),
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("gridIndex", ctypes.c_uint32),
        ("gridCount", ctypes.c_uint32),
        ("gridSize", ctypes.c_uint64),
        ("gridName", ctypes.c_char * 256),
    )


def write_nanovdb(
    deck: CloudDeck,
    path: Any,
    *,
    voxel_m: float = DEFAULT_CELL_M,
    device: str = "cuda:0",
    grid_name: str = DENSITY_GRID_NAME,
) -> CloudVolume:
    """Voxelise ``deck`` and write it as a named NanoVDB grid.

    ``warp.Volume.load_from_numpy`` is CUDA-only, so this needs a device; it is the same device
    the render runs on and the allocation is transient. The array itself is built engine-free by
    :meth:`CloudDeck.voxels`, so the expensive and checkable part needs no GPU and has its own
    unit tests.
    """
    # Warp lives in the `omni.warp.core` Kit extension, which is not on `sys.path` until asked
    # for; the same helper every other Warp caller in this package uses (ADR 0014).
    ensure_warp_on_path()
    import warp as wp  # noqa: PLC0415  -- engine glue, deliberately not imported at module scope

    grid, min_world = deck.voxels(voxel_m)
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wp.init()
    volume = wp.Volume.load_from_numpy(
        grid, min_world=min_world, voxel_size=float(voxel_m), bg_value=0.0, device=device
    )
    _name_grid(volume, grid_name)
    volume.save_to_nvdb(str(out))
    return CloudVolume(
        path=out,
        voxel_m=float(voxel_m),
        shape=(int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2])),
        min_world_m=(float(min_world[0]), float(min_world[1]), float(min_world[2])),
        max_density_per_m=float(grid.max()),
        file_bytes=out.stat().st_size,
    )


def _name_grid(volume: Any, name: str) -> None:
    """Write ``name`` into the grid header in place, and clear the now-stale checksum.

    ``warp.Volume.allocate`` leaves the name empty, and an unnamed grid cannot be bound by a
    `UsdVol.Volume` field relationship. The buffer is the volume's own device array read back to
    NumPy, edited and written back, rather than the file patched afterwards, so the file's
    metadata (which `save_to_nvdb` derives from this same header) stays consistent with it.
    """
    encoded = name.encode("utf-8")
    if len(encoded) > 255:
        raise ValueError(f"a NanoVDB grid name must fit 255 bytes, got {len(encoded)}")
    buffer = volume.array().numpy()
    header = _GridData.from_buffer(buffer)
    header.gridName = encoded
    # The name sits inside the range a full NanoVDB checksum covers, so leaving the old value
    # would present a grid that fails its own integrity check. The sentinel says "not computed",
    # which is a state the format defines rather than a value that happens not to match.
    header.checksum = NANOVDB_NO_CHECKSUM
    volume.array().assign(buffer)


def author_cloud_volume(
    stage: Any,
    prim_path: str,
    volume: CloudVolume,
    *,
    grid_name: str = DENSITY_GRID_NAME,
    albedo: float = 0.96,
    density_multiplier: float = 1.0,
) -> Any:
    """Define a `UsdVol.Volume` at ``prim_path`` bound to ``volume``'s grid, and shade it.

    ``albedo`` is the single-scattering albedo of a cloud droplet in the visible, which is very
    nearly 1 -- water barely absorbs at 0.55 um, which is why clouds are white rather than grey
    and why their undersides are dark by shadowing rather than by absorption. 0.96 leaves a little
    absorption so a deep tower does not glow. ESTIMATED; nothing radiometric rests on it, because
    no infrared output reads this prim (ADR 0073).

    The scattering and absorption coefficients are split out of the grid's own extinction, so the
    *sum* is the extinction the deck authored and the optical depth the path tracer integrates is
    the one the environment preset asked for.
    """
    from pxr import Gf, Sdf, UsdShade, UsdVol  # noqa: PLC0415

    prim = UsdVol.Volume.Define(stage, prim_path)
    asset = UsdVol.OpenVDBAsset.Define(stage, f"{prim_path}/{grid_name}")
    asset.CreateFilePathAttr(Sdf.AssetPath(str(volume.path)))
    asset.CreateFieldNameAttr(grid_name)
    prim.CreateFieldRelationship(grid_name, asset.GetPath())

    material = UsdShade.Material.Define(stage, f"{prim_path}/Material")
    shader = UsdShade.Shader.Define(stage, f"{prim_path}/Material/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("OmniVolumeDensity.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniVolumeDensity", "mdl")
    scatter = float(albedo) * float(density_multiplier)
    absorb = (1.0 - float(albedo)) * float(density_multiplier)
    shader.CreateInput("scattering_scale", Sdf.ValueTypeNames.Float).Set(scatter)
    shader.CreateInput("absorption_scale", Sdf.ValueTypeNames.Float).Set(absorb)
    shader.CreateInput("densityMultiplier", Sdf.ValueTypeNames.Float).Set(float(density_multiplier))
    # Neutral, because cloud droplets are far larger than a wavelength and scatter without much
    # spectral preference -- which is itself the visible signature, the sky behind being blue.
    tint = shader.CreateInput("albedo", Sdf.ValueTypeNames.Color3f)
    tint.Set(Gf.Vec3f(1.0, 1.0, 1.0))
    tint.GetAttr().SetColorSpace("raw")
    material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI(prim.GetPrim()).Bind(material)
    return prim.GetPrim()


def deck_bounds_m(deck: CloudDeck) -> tuple[np.ndarray, np.ndarray]:
    """``(min, max)`` corners of the deck in stage metres -- for a caller that wants to check the
    camera is inside the footprint before it spends a render on a cloud it cannot see."""
    half = deck.half_extent_m
    return (
        np.array([-half, deck.base_m, -half], dtype=np.float64),
        np.array([half, deck.top_m, half], dtype=np.float64),
    )
