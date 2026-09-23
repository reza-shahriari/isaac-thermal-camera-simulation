"""File I/O for the engine-free core: PNG for the look, float32 .npy / EXR for the physics.

The rule these modules exist to enforce is CLAUDE.md non-negotiable #2 at the disk boundary: a
plane in physical units never reaches an integer or half-float container, however good the
resulting picture looks (roadmap M10.10a).
"""

from irsim.io.assets import (
    AssetMesh,
    AssetMeshes,
    load_asset_meshes,
)
from irsim.io.dataset import (
    PLANE_UNITS,
    FrameRecord,
    FrameWriter,
    read_float_plane,
    write_float_plane,
    write_frame,
)
from irsim.io.exr import exr_bytes, read_exr, write_exr
from irsim.io.png import png_bytes, write_png

__all__ = [
    "AssetMesh",
    "AssetMeshes",
    "load_asset_meshes",
    "write_png",
    "png_bytes",
    "write_exr",
    "read_exr",
    "exr_bytes",
    "write_frame",
    "write_float_plane",
    "read_float_plane",
    "FrameRecord",
    "FrameWriter",
    "PLANE_UNITS",
]
