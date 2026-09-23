"""Engine-free mesh archives: a prepared asset's real triangles, without importing ``pxr``.

`src/irsim/` may not import an engine (CLAUDE.md #1), and on this build `pxr` is a Kit extension
that is not importable outside a running Kit application (ADR 0014). That is why
:class:`~irsim.config.scene.MeshSpec` generated its geometry rather than reading it, and why
`config/scene.py` recorded ingesting a real asset's triangles as an unowned follow-on.

This module closes it from the other side. ``scripts/prep_asset.py`` runs inside **Blender**,
which ships a complete OpenUSD, and writes one ``.npz`` holding every prim's world-space triangles
in metres. Nothing here knows what produced it: it is two arrays and a manifest, so a scene stays
loadable and solvable on a machine with no renderer, no Isaac Sim and no GPU (ADR 0128, ADR 0132).

**Why the archive is a *thermal* mesh and not the render mesh.** They do not have to be the same
surface, and should not be. ADR 0110's closest-point query locates a pixel on whatever mesh the
field is built on, so the renderer can keep the asset's full tessellation while the solver carries
a coarser one sized to the physics — a Phantom 4's 0.2938 m² wants cells of millimetres, not the
0.4 mm triangles a product-visualisation asset happens to ship. The bridge's distance tolerance is
what bounds how far the two may diverge.

**Area is the quantity the archive preserves.** It sets both the radiated power and the convective
load, so the prep tool decimates only by *planar dissolve* — merging coplanar faces without moving
a vertex — and refuses to write an archive whose area moved. Measured on the Phantom 4: collapse
decimation at ratio 0.05 removed **37 %** of the area (31,068 of its shells are small enough to
vanish entirely), where planar dissolve at 3° removed 38 % of the triangles for **+0.06 %** area.

docs/physics-model.md §6.1 (a facet is what the energy balance balances); ADR 0110, ADR 0128.
"""

from __future__ import annotations

import json
import os
import pathlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = ["AssetMesh", "AssetMeshes", "load_asset_meshes"]

#: Archives live beside the rest of a prepared asset, which is generated and not in git.
ASSETS_DIR = pathlib.Path(__file__).resolve().parents[3] / "data" / "assets"


@dataclass(frozen=True)
class AssetMesh:
    """One prim's triangles, in world-frame metres."""

    name: str
    material_name: str | None
    vertices_m: NDArray[np.float64]
    faces: NDArray[np.intp]
    #: Area as the prep tool measured it *after* decimation. Kept rather than recomputed so a
    #: mismatch between the archive and this loader's arithmetic is detectable, not invisible.
    area_m2: float
    area_before_m2: float

    def __post_init__(self) -> None:
        if self.vertices_m.ndim != 2 or self.vertices_m.shape[1] != 3:
            raise ValueError(f"{self.name}: vertices must be (n, 3), got {self.vertices_m.shape}")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError(f"{self.name}: faces must be (m, 3) triangles, got {self.faces.shape}")
        if len(self.faces) and (
            self.faces.min() < 0 or self.faces.max() >= self.vertices_m.shape[0]
        ):
            raise ValueError(f"{self.name}: a face indexes a vertex that does not exist")

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    def computed_area_m2(self) -> float:
        """Area from the arrays themselves — the cross-check against :attr:`area_m2`."""
        if not len(self.faces):
            return 0.0
        v = self.vertices_m
        a, b, c = v[self.faces[:, 0]], v[self.faces[:, 1]], v[self.faces[:, 2]]
        return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=-1).sum())


@dataclass(frozen=True)
class AssetMeshes(Mapping[str, AssetMesh]):
    """Every prim of one prepared asset, keyed by prim name."""

    name: str
    meshes: dict[str, AssetMesh]

    def __getitem__(self, key: str) -> AssetMesh:
        try:
            return self.meshes[key]
        except KeyError:
            raise KeyError(
                f"{self.name}: no mesh named {key!r}; have {sorted(self.meshes)}"
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.meshes)

    def __len__(self) -> int:
        return len(self.meshes)

    @property
    def area_m2(self) -> float:
        return float(sum(m.area_m2 for m in self.meshes.values()))

    @property
    def n_faces(self) -> int:
        return int(sum(m.n_faces for m in self.meshes.values()))

    def by_material(self, material_name: str) -> tuple[AssetMesh, ...]:
        """Every prim carrying one source material name, case-insensitively.

        An imported asset is grouped by material rather than by function — the Phantom 4's white
        plastic is one prim spanning 987 disconnected shells — so "every prim of this material" is
        the granularity a scene actually has to bind to.
        """
        target = material_name.lower()
        return tuple(
            m
            for _, m in sorted(self.meshes.items())
            if m.material_name is not None and m.material_name.lower() == target
        )


def load_asset_meshes(path: str | os.PathLike[str], *, check_area: float = 1e-6) -> AssetMeshes:
    """Read an ``.npz`` written by ``scripts/prep_asset.py --emit-mesh``.

    A bare name (``"phantom4"``) resolves to ``data/assets/<name>/<name>.meshes.npz``, so a scene
    can name an asset without knowing the layout.

    ``check_area`` is the relative tolerance between each prim's recorded area and the area of the
    arrays as loaded. It is not a formality: the recorded value is what the prep tool's own gate
    was applied to, so a silent disagreement here would mean the gate passed on geometry that is
    not the geometry being solved.
    """
    p = pathlib.Path(path)
    if not p.suffix:
        p = ASSETS_DIR / p.name / f"{p.name}.meshes.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"no mesh archive at {p}. Generate it with:\n"
            f"    python scripts/prep_asset.py --asset {p.stem.split('.')[0]} --emit-mesh"
        )
    with np.load(p, allow_pickle=False) as data:
        manifest = json.loads(str(data["manifest"]))
        meshes: dict[str, AssetMesh] = {}
        for entry in manifest:
            i = int(entry["index"])
            mesh = AssetMesh(
                name=str(entry["name"]),
                material_name=(
                    None if entry.get("material_name") is None else str(entry["material_name"])
                ),
                vertices_m=np.asarray(data[f"v{i}"], dtype=np.float64),
                faces=np.asarray(data[f"f{i}"], dtype=np.intp),
                area_m2=float(entry["area_m2"]),
                area_before_m2=float(entry["area_before_m2"]),
            )
            if mesh.n_faces and mesh.area_m2 > 0.0:
                got = mesh.computed_area_m2()
                if abs(got / mesh.area_m2 - 1.0) > check_area:
                    raise ValueError(
                        f"{p}: {mesh.name} records {mesh.area_m2:.8f} m2 but its arrays give "
                        f"{got:.8f} m2 — the archive and its manifest disagree"
                    )
            if mesh.name in meshes:
                raise ValueError(f"{p}: two prims named {mesh.name!r}")
            meshes[mesh.name] = mesh
    return AssetMeshes(name=p.stem.split(".")[0], meshes=meshes)
