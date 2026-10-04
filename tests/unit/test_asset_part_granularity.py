"""Selectors may judge whole source objects rather than connected shells (AI.16).

The DJI Inspire 3's artist modelled one object per physical piece -- a blade, a landing foot, a
motor can -- but built most of them from several shells. Judged shell by shell, a geometric cut
split 41 of its 345 objects across parts: 3 % of every blade landed in its rotor head and two
camera objects went 55/45 to the fuselage. The fixtures here are that situation in miniature:
a blade of three shells, one of which sits right at the hub.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from irsim.io.asset_parts import (
    Component,
    PartsConfig,
    PartSelector,
    PartSpec,
    assign_parts,
    object_unit,
)

REPO = pathlib.Path(__file__).resolve().parents[2]


def _shell(index, source, centre, area, material="default", half=0.005):
    lo = tuple(c - half for c in centre)
    hi = tuple(c + half for c in centre)
    return Component(
        index=index,
        faces=10,
        area_m2=area,
        centroid=tuple(centre),
        lo=lo,
        hi=hi,
        material_name=material,
        source=source,
    )


def rotor_station():
    """A motor can at the hub, and one blade object of three shells reaching out along x.

    The blade's root shell sits 8 mm from the axis -- inside the motor's 12 mm circle -- exactly
    as the Inspire 3's blade roots do.
    """
    return [
        _shell(0, "can", (0.0, 0.0, 0.0), 0.004),
        _shell(1, "blade", (0.008, 0.0, 0.01), 0.0004),  # root, inside the motor circle
        _shell(2, "blade", (0.05, 0.0, 0.01), 0.005),
        _shell(3, "blade", (0.09, 0.0, 0.01), 0.004),
    ]


def station_config(granularity):
    return PartsConfig(
        granularity=granularity,
        parts=[
            PartSpec(name="motor", select=PartSelector(near_xy=(0.0, 0.0), within_m=0.012)),
            PartSpec(name="propeller", select=PartSelector(near_xy=(0.0, 0.0), within_m=0.2)),
        ],
    )


def test_shell_by_shell_a_cut_splits_one_physical_piece():
    """The failure the field exists for, kept as a control: it must still happen at the default."""
    claimed = station_config("component").claim_all(rotor_station())
    assert claimed[1].name == "motor" and claimed[2].name == "propeller"


def test_judged_whole_no_object_is_split():
    claimed = station_config("object").claim_all(rotor_station())
    assert {claimed[i].name for i in (1, 2, 3)} == {"propeller"}, "the blade is one part"
    assert claimed[0].name == "motor"
    _, report = assign_parts(rotor_station(), station_config("object"))
    assert report.area_by_part["propeller"] == pytest.approx(0.0094)
    assert report.area_by_part["motor"] == pytest.approx(0.004)


def test_an_object_is_judged_at_its_area_weighted_centre():
    """Area-weighted, not a vertex mean: the archive's dissolve moves vertices but keeps area."""
    unit = object_unit(rotor_station()[1:])
    expected_x = (0.008 * 0.0004 + 0.05 * 0.005 + 0.09 * 0.004) / 0.0094
    assert unit.centroid[0] == pytest.approx(expected_x)
    assert unit.area_m2 == pytest.approx(0.0094) and unit.faces == 30
    assert unit.lo[0] == pytest.approx(0.003) and unit.hi[0] == pytest.approx(0.095)
    assert unit.source == "blade"


def test_the_dominant_material_names_the_object():
    members = [_shell(0, "o", (0, 0, 0), 0.001, "glass"), _shell(1, "o", (0, 0, 0), 0.003, "paint")]
    assert object_unit(members).material_name == "paint"


def test_an_object_is_excluded_whole_or_not_at_all():
    """A case whose lid shell rises above the cut must not leave a lid behind in the aircraft."""
    case = [_shell(0, "case", (0.0, 0.0, 0.05), 0.9), _shell(1, "case", (0.0, 0.0, 0.40), 0.05)]
    config = PartsConfig(
        granularity="object",
        parts=[PartSpec(name="airframe", select=PartSelector())],
        exclude=[PartSelector(z_max_m=0.25)],
    )
    assert config.excluded_indices(case) == {0, 1}
    assert all(v is None for v in config.claim_all(case).values())


def test_object_granularity_refuses_components_without_their_object():
    bare = [
        Component(index=0, faces=1, area_m2=1e-4, centroid=(0, 0, 0), lo=(0, 0, 0), hi=(0, 0, 0))
    ]
    with pytest.raises(ValueError, match="re-emit the components"):
        station_config("object").claim_all(bare)


def test_the_archive_split_keeps_each_prim_whole():
    """In the thermal archive an object is a prim; its shells must travel together there too."""
    from irsim.io.asset_parts import split_by_part
    from irsim.io.assets import AssetMesh, AssetMeshes

    def tri(x0):
        v = np.array([[x0, 0, 0], [x0 + 0.004, 0, 0], [x0, 0.004, 0]], dtype=np.float64)
        return v, np.array([[0, 1, 2]], dtype=np.intp)

    v1, f1 = tri(0.006)  # a blade root shell, inside the motor's circle
    v2, f2 = tri(0.08)  # the rest of the blade
    blade = AssetMesh(
        name="blade",
        material_name="default",
        vertices_m=np.concatenate([v1, v2]),
        faces=np.concatenate([f1, f2 + 3]),
        area_m2=1.6e-5,
        area_before_m2=1.6e-5,
    )
    meshes = AssetMeshes(name="toy", meshes={"blade": blade})
    by_shell, _ = split_by_part(meshes, station_config("component"))
    whole, _ = split_by_part(meshes, station_config("object"))
    assert set(by_shell) == {"motor", "propeller"}, "control: shell by shell the blade splits"
    assert set(whole) == {"propeller"} and whole["propeller"].n_faces == 2


@pytest.fixture(scope="module")
def prep():
    spec = importlib.util.spec_from_file_location("prep_asset", REPO / "scripts" / "prep_asset.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_blender_split_is_told_to_keep_objects_whole(prep):
    stats = [
        {
            "index": c.index,
            "faces": c.faces,
            "area_m2": c.area_m2,
            "centroid": list(c.centroid),
            "lo": list(c.lo),
            "hi": list(c.hi),
            "material_name": c.material_name,
            "object": c.source,
        }
        for c in rotor_station()
    ]
    plan = prep.part_assignment(stats, station_config("object"))
    assert plan["part_of_component"] == {
        "0": "motor",
        "1": "propeller",
        "2": "propeller",
        "3": "propeller",
    }
