"""The cloud deck as real geometry in the stage: OpenVDB on disk, a volume box in USD
(AT.12, ADR 0127; AT.13, ADR 0140 and ADR 0144).

The visible companion used to paint cloud into the environment-dome texture (ADR 0073/0076). A
dome light is at infinity, so that cloud had no distance, no thickness and no inside: a camera
could not move relative to it and no ray had a length within it. This module puts the *same*
:class:`~irsim.atmosphere.cloud_deck.CloudDeck` the infrared band marches into the stage as a
participating medium, so the path tracer integrates the field the radiometry integrates.

**OpenVDB is the writer; NanoVDB is the fallback.** ADR 0127 wrote this module on the premise
that "there is no OpenVDB writer in this environment", and everything painful about it followed
from that: the deck was voxelised through `warp.Volume`, whose grid header had to be re-stamped
to a NanoVDB version the renderer accepts, whose file metadata had to be back-filled by hand out
of the tree, and which the renderer refused anyway. **The premise was wrong.** Isaac Sim's
`omni.volume` extension ships a full OpenVDB 12 binding (file format 224) and a NanoVDB one, and
`irsim_isaac.env.ensure_openvdb_on_path` makes them importable outside Kit. :func:`write_openvdb`
therefore writes a real `.vdb` with the library that defines the format -- named grid, fog-volume
class, linear transform, stats metadata -- and none of the hand-patching applies to it.

:func:`write_nanovdb` is kept because a `.nvdb` is still the cheaper thing for Warp itself to read
back and because the two fixes in it are correct, but it is no longer the path a renderer is
handed. ADR 0140 records the switch.

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
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from irsim.atmosphere.cloud_deck import DEFAULT_CELL_M, CloudDeck
from irsim_isaac.env import ensure_openvdb_on_path, ensure_warp_on_path

__all__ = [
    "DENSITY_GRID_NAME",
    "write_openvdb",
    "NANOVDB_NO_CHECKSUM",
    "INDEX_NANOVDB_VERSION",
    "CloudVolume",
    "write_nanovdb",
    "author_cloud_volume",
]

#: The grid's name inside the file. The renderer reads the file's density grid through the
#: material's texture input (ADR 0144); the name is what a stranger's tool lists it as.
DENSITY_GRID_NAME = "density"

#: NanoVDB's sentinel for "this grid carries no checksum". Written after the grid name is patched,
#: because the name lives inside the region a full checksum covers.
NANOVDB_NO_CHECKSUM = 0xFFFFFFFFFFFFFFFF

#: The NanoVDB grid version the Omniverse renderer's IndeX plugin accepts, as (major, minor,
#: patch). **Measured** on this build by authoring a volume and reading the render log:
#:
#:     INDEX main error: VDB subset: The NanoVDB version used on the application side is not
#:     compatible with the version utilized by NVIDIA IndeX.
#:      * Application NanoVDB version: 32.8,
#:      * IndeX NanoVDB version: 32.7.
#:
#: Warp 1.16's allocator stamps 32.8.0 into the grid header, so every volume it writes is refused
#: by this renderer -- not a subtle failure, but a total one: the volume never loads and the frame
#: comes back with no cloud in it.
#:
#: The grid is **re-stamped** to 32.7.0 rather than regenerated, because the two minor versions
#: share a byte-identical `GridData` header and `FloatGrid` tree layout -- the fields Warp itself
#: declares to read the grid back (magic, checksum, version, flags, gridIndex, gridCount,
#: gridSize, gridName, map, worldBBox, voxelSize, gridClass, gridType, blindMetadataOffset,
#: blindMetadataCount, data0..2) are the same in both, and a 32.8 FloatGrid contains nothing a
#: 32.7 reader does not know. That reasoning is not a proof, so it is **verified by rendering**:
#: an accepted volume draws cloud and a misread one does not, and neither is quiet.
#:
#: Revisit when Isaac Sim's bundled Warp and its IndeX plugin agree again; the shim is a
#: four-byte write and removing it is deleting one argument.
INDEX_NANOVDB_VERSION = (32, 7, 0)


def _packed_version(version: tuple[int, int, int]) -> int:
    """NanoVDB packs (major, minor, patch) into one uint32 as 11/11/10 bits."""
    major, minor, patch = version
    return (int(major) << 21) | (int(minor) << 10) | int(patch)


def _unpack_version(packed: int) -> tuple[int, int, int]:
    return (packed >> 21, (packed >> 10) & 0x7FF, packed & 0x3FF)


@dataclass(frozen=True)
class CloudVolume:
    """What was written, in the numbers a render log should carry."""

    path: pathlib.Path
    voxel_m: float
    shape: tuple[int, int, int]
    min_world_m: tuple[float, float, float]
    max_density_per_m: float
    file_bytes: int
    #: ``(written, as generated)`` NanoVDB grid versions. They differ when the renderer's IndeX
    #: plugin is behind the bundled Warp; see :data:`INDEX_NANOVDB_VERSION`.
    nanovdb_version: tuple[tuple[int, int, int], tuple[int, int, int]] = ((0, 0, 0), (0, 0, 0))
    #: Voxels the sparse tree actually stores. Zero when the writer did not report it (NanoVDB).
    active_voxels: int = 0

    @property
    def occupancy(self) -> float:
        """Stored voxels over the dense box -- what makes a sparse format worth using."""
        dense = int(self.shape[0]) * int(self.shape[1]) * int(self.shape[2])
        return float(self.active_voxels) / dense if dense and self.active_voxels else 0.0

    @property
    def caption(self) -> str:
        nx, ny, nz = self.shape
        written, generated = self.nanovdb_version
        if self.path.suffix == ".vdb":
            stamp = f"OpenVDB, {self.active_voxels:,} active ({self.occupancy:.1%})"
        else:
            stamp = "NanoVDB " + ".".join(str(v) for v in written)
            if written != generated:
                stamp += " (Warp wrote " + ".".join(str(v) for v in generated) + ")"
        return (
            f"{nx}x{ny}x{nz} voxels at {self.voxel_m:g} m "
            f"({self.file_bytes / 1e6:.1f} MB), peak {self.max_density_per_m:.5f} /m, {stamp}"
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
    nanovdb_version: tuple[int, int, int] | None = INDEX_NANOVDB_VERSION,
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
    versions = _stamp_grid(volume, grid_name, nanovdb_version)
    volume.save_to_nvdb(str(out))
    _fill_file_metadata(out)
    return CloudVolume(
        path=out,
        voxel_m=float(voxel_m),
        shape=(int(grid.shape[0]), int(grid.shape[1]), int(grid.shape[2])),
        min_world_m=(float(min_world[0]), float(min_world[1]), float(min_world[2])),
        max_density_per_m=float(grid.max()),
        file_bytes=out.stat().st_size,
        nanovdb_version=versions,
    )


def write_openvdb(
    deck: CloudDeck,
    path: Any,
    *,
    voxel_m: float = DEFAULT_CELL_M,
    grid_name: str = DENSITY_GRID_NAME,
    tolerance: float = 1e-4,
) -> CloudVolume:
    """Voxelise ``deck`` and write it as a named OpenVDB fog volume.

    No GPU and no Kit: :meth:`CloudDeck.voxels` is engine-free NumPy and OpenVDB is a CPU library,
    so this runs on the unit gate. That is the point of preferring it -- the old NanoVDB path
    needed a CUDA device to *write a file*, because `warp.Volume.load_from_numpy` is CUDA-only.

    ``tolerance`` is what `copyFromArray` treats as background. It is not cosmetic: a fog volume
    is sparse, and a grid whose empty space is stored as 1e-30 rather than as absence is dense in
    every way that costs memory. At the deck's usual cover the field is about **14 % occupied**,
    so the tolerance is the difference between a megabyte and eight.
    """
    ensure_openvdb_on_path()
    import openvdb  # noqa: PLC0415  -- engine glue, deliberately not imported at module scope

    grid_array, min_world = deck.voxels(voxel_m)
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    grid = openvdb.FloatGrid()
    grid.name = grid_name
    grid.gridClass = openvdb.GridClass.FOG_VOLUME
    # Index space is the array's own (X, Y, Z), so the transform is a scale by the voxel size and
    # a translation putting index (0, 0, 0) at the deck's minimum corner in stage metres. Written
    # as a matrix rather than by composing operations, because the order of `preScale` against
    # `postTranslate` is exactly the kind of thing that silently puts a cloud underground.
    transform = openvdb.createLinearTransform(
        [
            [float(voxel_m), 0.0, 0.0, 0.0],
            [0.0, float(voxel_m), 0.0, 0.0],
            [0.0, 0.0, float(voxel_m), 0.0],
            [float(min_world[0]), float(min_world[1]), float(min_world[2]), 1.0],
        ]
    )
    grid.transform = transform
    grid.copyFromArray(np.ascontiguousarray(grid_array), tolerance=float(tolerance))
    # **The counts the renderer asked for, written by the library that owns them.** IndeX
    # reported "failed to generate VDB subset grid storage" against a Warp grid whose voxel, node
    # and tile counts were zero; `addStatsMetadata` is OpenVDB's own way to put them in the file,
    # so a consumer reading metadata alone sizes its storage correctly.
    grid.addStatsMetadata()
    openvdb.write(str(out), grids=[grid])

    return CloudVolume(
        path=out,
        voxel_m=float(voxel_m),
        shape=(int(grid_array.shape[0]), int(grid_array.shape[1]), int(grid_array.shape[2])),
        min_world_m=(float(min_world[0]), float(min_world[1]), float(min_world[2])),
        max_density_per_m=float(grid_array.max()),
        file_bytes=out.stat().st_size,
        active_voxels=int(grid.activeVoxelCount()),
    )


def _stamp_grid(
    volume: Any, name: str, version: tuple[int, int, int] | None
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Name the grid, optionally re-stamp its version, and clear the now-stale checksum.

    Two edits, both to the grid header, both because of what the consumers need:

    * ``warp.Volume.allocate`` leaves the **name** empty, and an unnamed grid cannot be bound by a
      `UsdVol.Volume` field relationship.
    * The bundled Warp stamps a NanoVDB **version** the renderer's IndeX plugin refuses; see
      :data:`INDEX_NANOVDB_VERSION`. ``None`` leaves it alone, which is what a caller writing a
      file for Warp itself wants.

    The buffer is the volume's own device array read back to NumPy, edited and written back,
    rather than the file patched afterwards, so the file metadata `save_to_nvdb` derives from this
    same header stays consistent with it. Returns ``(written, as generated)``.
    """
    encoded = name.encode("utf-8")
    if len(encoded) > 255:
        raise ValueError(f"a NanoVDB grid name must fit 255 bytes, got {len(encoded)}")
    buffer = volume.array().numpy()
    header = _GridData.from_buffer(buffer)
    header.gridName = encoded
    generated = _unpack_version(int(header.version))
    written = generated if version is None else tuple(int(v) for v in version)
    if version is not None:
        header.version = _packed_version(written)  # type: ignore[arg-type]
    # The name and the version both sit inside the range a full NanoVDB checksum covers, so
    # leaving the old value would present a grid that fails its own integrity check. The sentinel
    # says "not computed", which is a state the format defines rather than a value that happens
    # not to match.
    header.checksum = NANOVDB_NO_CHECKSUM
    volume.array().assign(buffer)
    return written, generated  # type: ignore[return-value]


#: Byte size of NanoVDB's `GridData` header, which `TreeData` follows immediately.
_GRID_DATA_BYTES = 672
#: Byte size of `nanovdb::io::FileMetaData`, which follows the 16-byte file header.
_FILE_META_BYTES = 176


def _fill_file_metadata(path: pathlib.Path) -> dict[str, int]:
    """Copy the tree's own node, tile and voxel counts into the file metadata block.

    ``warp.Volume.save_to_nvdb`` writes a `FileMetaData` with ``voxelCount``, ``nodeCount`` and
    ``tileCount`` left at **zero** -- it only fills the fields it needs to read its own files back.
    Warp does not care; a consumer that sizes its storage from them gets nothing, which is exactly
    what the Omniverse renderer reported:

        IndeX Direct carb-volume importer: unable to create VDB subset
        (failed to generate VDB subset grid storage of size 56657696)

    -- the grid *size* was right and the allocation still produced nothing. The counts are not
    derived or guessed here: they are read straight out of the grid's own `TreeData`, which sits
    immediately after the 672-byte `GridData` header and carries ``mNodeCount[3]``,
    ``mTileCount[3]`` and ``mVoxelCount``. The root node is one, by definition of a tree.

    Returns what was written, so a caller can log it and a test can assert it is not all zeros.
    """
    data = bytearray(path.read_bytes())
    (magic,) = struct.unpack_from("<Q", data, 0)
    if magic not in (0x304244566F6E614E, 0x314244566F6E614E):
        raise ValueError(f"{path} does not start with a NanoVDB file magic")
    (name_size,) = struct.unpack_from("<I", data, 16 + 136)
    grid = 16 + _FILE_META_BYTES + name_size
    tree = grid + _GRID_DATA_BYTES
    node_count = struct.unpack_from("<3I", data, tree + 32)
    tile_count = struct.unpack_from("<3I", data, tree + 44)
    (voxel_count,) = struct.unpack_from("<Q", data, tree + 56)
    struct.pack_into("<Q", data, 16 + 24, voxel_count)  # FileMetaData.voxelCount
    struct.pack_into("<4I", data, 16 + 140, *node_count, 1)  # nodeCount, root included
    struct.pack_into("<3I", data, 16 + 156, *tile_count)
    path.write_bytes(bytes(data))
    return {
        "voxels": int(voxel_count),
        "leaf_nodes": int(node_count[0]),
        "lower_nodes": int(node_count[1]),
        "upper_nodes": int(node_count[2]),
    }


def author_cloud_volume(
    stage: Any,
    prim_path: str,
    volume: CloudVolume,
    *,
    grid_name: str = DENSITY_GRID_NAME,
    albedo: float = 0.96,
    density_multiplier: float = 1.0,
    phase_bias: float = 0.85,
    with_material: bool = True,
) -> Any:
    """Put ``volume`` in the stage the way this build's path tracer actually renders it.

    **The recipe, measured on 2026-09-26 (ADR 0144), is not the USD-textbook one.** A
    `UsdVol.Volume` with an `OpenVDBAsset` field never rendered here in any variant. What renders
    is the recipe NVIDIA documents for USD Composer: a **mesh box** whose vertices sit at the
    grid's world bounds and which carries **no transform**, marked ``primvars:isVolume``, bound to
    `OmniVolumeDensity` with the ``.vdb`` handed to it as ``volume_density_texture``. Each of
    those four clauses was found by its failure:

    * the box's local space must *be* world space. A unit `UsdGeom.Cube` scaled to the same
      bounds magnifies the grid by the scale factor and puts the camera inside cloud everywhere
      -- the whole frame, backdrop included, darkened forty-fold in linear radiance;
    * without ``isVolume`` the prim is a glass block with a homogeneous interior, writes depth,
      and shows the sun as a specular highlight on its face;
    * the material's inputs are ``volume_density_texture``, ``volume_albedo``,
      ``volume_density_scale`` and ``directional_bias`` (read off the MDL source in the Kit
      kernel). Inputs the MDL does not declare are dropped silently, and the first version of
      this function set four of those;
    * the renderer's "Non-uniform Volumes" frame (``/rtx/pathtracing/ptvol/enabled``) has to be
      on and the bounce limits raised, or a thick scattering slab terminates black. Those are
      render settings, not stage content: see ``VOLUME_SETTINGS`` in
      ``scripts/probe_cloud_volume.py`` and the driver that renders with it.

    ``albedo`` is the single-scattering albedo of a cloud droplet in the visible, very nearly 1
    -- water barely absorbs at 0.55 um, which is why clouds are white and their undersides are
    dark by shadowing rather than by absorption. 0.96 leaves a little absorption so a deep tower
    does not glow. ESTIMATED; nothing radiometric rests on it, because no infrared output reads
    this prim (ADR 0073). ``phase_bias`` is the material's Henyey-Greenstein-like forward bias;
    0.85 is the visible-band cloud value.

    The grid carries visible extinction per metre and the stage is in metres, so a
    ``density_multiplier`` of 1 makes the optical depth the path tracer integrates the one the
    deck authored. A volume writes **no depth**: ``distance_to_camera`` behind a cloud is the
    backdrop's, so occlusion in a band computed from AOVs still has to come from this project's
    own march (AT.14).
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade  # noqa: PLC0415

    low = tuple(float(v) for v in volume.min_world_m)
    high = tuple(float(low[i] + volume.shape[i] * volume.voxel_m) for i in range(3))
    x0, y0, z0 = low
    x1, y1, z1 = high

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]  # fmt: skip
    mesh.CreatePointsAttr([Gf.Vec3f(*c) for c in corners])
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr(
        [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
    )
    # Authored, not computed: a prim with no bounding box is culled before anything asks what is
    # inside it, and a volume's contents live in a file the scene delegate has not opened yet.
    mesh.CreateExtentAttr([Gf.Vec3f(*low), Gf.Vec3f(*high)])
    # **No transform.** The density texture is sampled in the prim's local frame, and the grid's
    # own transform is in world metres; the two coincide only when local is world.
    UsdGeom.Xformable(mesh.GetPrim()).ClearXformOpOrder()
    UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar("isVolume", Sdf.ValueTypeNames.Bool).Set(True)
    prim = mesh.GetPrim()
    if not with_material:
        # Unshaded, so a probe can tell "the grid did not load" from "the grid loaded and the
        # material did not": those two failures look identical in a frame and are different bugs.
        return prim

    material = UsdShade.Material.Define(stage, f"{prim_path}/Material")
    shader = UsdShade.Shader.Define(stage, f"{prim_path}/Material/Shader")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("OmniVolumeDensity.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniVolumeDensity", "mdl")
    # The density texture *is* the grid. ``grid_name`` is kept in the signature for the callers
    # that name it; the material reads the file's first density grid, which is the one written.
    del grid_name
    shader.CreateInput("volume_density_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath(str(volume.path))
    )
    shader.CreateInput("volume_density_scale", Sdf.ValueTypeNames.Float).Set(
        float(density_multiplier)
    )
    # The MDL splits the coefficients itself: scattering = albedo, absorption = 1 - albedo, both
    # times the density. A grey albedo, because cloud droplets are far larger than a wavelength
    # and scatter without spectral preference -- the sky behind being blue is the visible sign.
    tint = shader.CreateInput("volume_albedo", Sdf.ValueTypeNames.Color3f)
    tint.Set(Gf.Vec3f(float(albedo), float(albedo), float(albedo)))
    tint.GetAttr().SetColorSpace("raw")
    shader.CreateInput("directional_bias", Sdf.ValueTypeNames.Float).Set(float(phase_bias))
    # Kit binds an MDL material through all three of its outputs; a stage authored by Composer
    # carries `mdl:surface`, `mdl:displacement` and `mdl:volume` on the same shader.
    for output in (
        material.CreateSurfaceOutput("mdl"),
        material.CreateDisplacementOutput("mdl"),
        material.CreateVolumeOutput("mdl"),
    ):
        output.ConnectToSource(shader.ConnectableAPI(), "out")
    UsdShade.MaterialBindingAPI(prim).Bind(material)
    return prim


def deck_bounds_m(deck: CloudDeck) -> tuple[np.ndarray, np.ndarray]:
    """``(min, max)`` corners of the deck in stage metres -- for a caller that wants to check the
    camera is inside the footprint before it spends a render on a cloud it cannot see."""
    half = deck.half_extent_m
    return (
        np.array([-half, deck.base_m, -half], dtype=np.float64),
        np.array([half, deck.top_m, half], dtype=np.float64),
    )
