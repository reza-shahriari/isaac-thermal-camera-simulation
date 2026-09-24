"""A mesh whose faces carry three materials, audited from a committed USD fixture (`AI.4`).

``tests/unit/data/subset_building.usda`` is the first asset in this repository with
``materialBind`` subsets, and ``subset_building.prims.json`` is the record dump Kit produced
from it (``scripts/audit_materials.py --stage ... --dump-prims ...``). The stage walk itself
needs Kit and is covered by ``tests/integration/test_subset_walk_isaac.py``; what is engine-free
-- and what actually decides the physics -- is what the resolver then makes of those records.

The failure this guards against is the quiet one named in ADR 0128. Blender writes slot 0 as a
mesh-level binding *beside* the subsets, as a documented Hydra workaround, so a walk that reads
the mesh binding maps every face of a multi-material mesh to whichever material was first in the
artist's stack. Nothing errors; a whole concrete wall simply becomes glass, and the image looks
entirely reasonable.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from irsim.materials.library import MaterialLibrary
from irsim.materials.mapping import MaterialResolver, PrimRecord, load_mapping_rules
from irsim.materials.table import MaterialTable
from irsim.radiometry.lut import BandLUT
from irsim.radiometry.spectral_response import load_spectral_response

DUMP = pathlib.Path(__file__).parent / "data" / "subset_building.prims.json"
FIXTURE_USD = pathlib.Path(__file__).parent / "data" / "subset_building.usda"

#: The facade is one mesh. These are its three face groups, in the file's own order.
FACADE = {
    "/World/Building/Facade/glazing": "glass_windshield",
    "/World/Building/Facade/precast": "concrete",
    "/World/Building/Facade/cladding": "aircraft_aluminium_painted",
}

#: A surface at midday against a clear winter sky: the contrast that makes emissivity legible.
T_SURFACE_K = 300.0
T_SKY_K = 250.0

#: FLIR Boson 640 LWIR, the sensor every aerial scene in this project uses.
NETD_K = 0.050


@pytest.fixture(scope="module")
def records() -> list[PrimRecord]:
    return [PrimRecord.from_dict(d) for d in json.loads(DUMP.read_text(encoding="utf-8"))]


@pytest.fixture(scope="module")
def resolver() -> MaterialResolver:
    library = MaterialLibrary.load()
    names = MaterialTable.from_library(library, "lwir").names
    return MaterialResolver(load_mapping_rules(known_materials=library.names), names)


@pytest.fixture(scope="module")
def lut() -> BandLUT:
    repo = pathlib.Path(__file__).resolve().parents[2]
    return BandLUT.build(
        load_spectral_response(repo / "data" / "spectra" / "responses" / "boson_vox.csv")
    )


def _apparent_k(lut: BandLUT, emissivity: float) -> float:
    """What the camera reads off a surface at ``T_SURFACE_K`` reflecting a ``T_SKY_K`` sky.

    docs/physics-model.md §4.1: the reflected term is what makes a wrong emissivity a wrong
    *temperature* rather than a wrong number nobody looks at.
    """
    lb = float(lut.lookup(np.float32(T_SURFACE_K)))
    sky = float(lut.lookup(np.float32(T_SKY_K)))
    mixed = emissivity * lb + (1.0 - emissivity) * sky
    return float(lut.apparent_temperature(np.float32(mixed)))


def test_the_committed_dump_is_the_committed_fixture() -> None:
    """Both halves are in git, and the dump names the subsets the USD file authors.

    A dump regenerated from a different stage would pass every test below while describing
    something nobody can read, which is the failure mode a committed fixture exists to prevent.
    """
    assert FIXTURE_USD.exists()
    text = FIXTURE_USD.read_text(encoding="utf-8")
    paths = [d["path"] for d in json.loads(DUMP.read_text(encoding="utf-8"))]
    for path in paths:
        leaf = path.rsplit("/", 1)[-1]
        assert f'"{leaf}"' in text, f"{leaf} is in the dump but not in the USD file"
    assert text.count('familyName = "materialBind"') == 7


def test_one_mesh_resolves_to_three_different_materials(
    records: list[PrimRecord], resolver: MaterialResolver
) -> None:
    """The facade's three face groups map to three library materials, not to one.

    This is the branch that had never run on a USD file before `AI.4`: every asset committed
    here until now had one material per mesh.
    """
    resolved = {r.path: r.material for r in resolver.resolve_all(records)}
    got = {path: resolved[path] for path in FACADE}
    assert got == FACADE
    assert len(set(got.values())) == 3


def test_reading_the_mesh_binding_instead_makes_the_whole_wall_glass(
    resolver: MaterialResolver, lut: BandLUT
) -> None:
    """The negative control, and the cost of it in kelvin.

    `/World/Building/Facade` binds `Facade_Glass_Clear` at the mesh level as well -- Blender's
    slot 0. A walk that reads that binding produces one record for the mesh, and two thirds of
    the wall is then made of the wrong substance. Measured through the band LUT rather than
    asserted: the error is tens of NETD, so it is not a rounding question.
    """
    slot_zero = PrimRecord(
        path="/World/Building/Facade",
        material_name="Facade_Glass_Clear",
        semantic_class=None,
        override=None,
    )
    (resolved,) = resolver.resolve_all([slot_zero])
    assert resolved.material == "glass_windshield"

    library = MaterialLibrary.load()
    as_glass = _apparent_k(
        lut, float(library["glass_windshield"].band_properties("lwir").emissivity)
    )
    as_concrete = _apparent_k(lut, float(library["concrete"].band_properties("lwir").emissivity))
    as_metal = _apparent_k(
        lut, float(library["aircraft_aluminium_painted"].band_properties("lwir").emissivity)
    )
    assert abs(as_concrete - as_glass) > 20.0 * NETD_K
    assert abs(as_metal - as_glass) > 5.0 * NETD_K


def test_the_invisible_scaffold_contributes_no_surfaces(records: list[PrimRecord]) -> None:
    """Visibility is decided before subsets are read.

    `/World/Building/Scaffold` is invisible and carries two subsets. A walk that read subsets
    first would report two surfaces that are not in the picture, which inflates the coverage
    fraction ADR 0047 gates on with geometry nobody can see.
    """
    assert not [r for r in records if "Scaffold" in r.path]


def test_a_single_material_mesh_still_comes_through_whole(records: list[PrimRecord]) -> None:
    """The plinth has no subsets and must stay one record, at its own path.

    The render path looks a material up by prim path, so a mesh that grew a subset-shaped path
    it does not have would resolve to nothing at all.
    """
    plinth = [r for r in records if r.path.endswith("Plinth")]
    assert len(plinth) == 1
    assert plinth[0].material_name == "Plinth_Concrete_Base"
