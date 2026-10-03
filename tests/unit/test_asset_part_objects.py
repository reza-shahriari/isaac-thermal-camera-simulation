"""A part may be selected by the name of the object it is (the Blender add-on's export).

The add-on exports one USD prim per Blender object, each one a part a person separated and named:
`battery`, `shell_lower`, `arm_front_left`. Its contacts and hidden parts (AI.11) name those
parts, so the asset config needs parts that are *those objects* -- not a geometric cut that has
to be re-tuned whenever the model is edited, and not a material, which several parts may share.
``PartSelector.objects`` matches ``Component.source``, which names the object in the render pass
and the archive prim in the thermal pass (the same names).
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.io.asset_parts import Component, PartsConfig, PartSelector, PartSpec, assign_parts


def _shell(index, source, centre, area, material="white_plastic"):
    lo = tuple(c - 0.005 for c in centre)
    hi = tuple(c + 0.005 for c in centre)
    return Component(
        index=index,
        faces=12,
        area_m2=area,
        centroid=tuple(centre),
        lo=lo,
        hi=hi,
        material_name=material,
        source=source,
    )


def quad():
    """Two shells of one shell object, a battery inside it, and an arm -- all one plastic."""
    return [
        _shell(0, "shell_lower", (0.0, 0.0, 0.00), 0.02),
        _shell(1, "shell_lower", (0.0, 0.0, 0.02), 0.01),
        _shell(2, "Battery", (0.0, 0.0, 0.01), 0.005),
        _shell(3, "arm_front_left", (0.12, 0.12, 0.0), 0.004),
    ]


def by_object(granularity, names=("shell_lower", "battery", "arm_front_left")):
    return PartsConfig(
        granularity=granularity,
        parts=[PartSpec(name=n, select=PartSelector(objects=[n])) for n in names],
    )


@pytest.mark.parametrize("granularity", ["component", "object"])
def test_each_object_becomes_its_own_part(granularity):
    """One plastic everywhere and the battery inside the shell: neither material nor geometry could
    tell them apart, and the object names do, at either granularity."""
    claimed = by_object(granularity).claim_all(quad())
    assert [claimed[i].name for i in range(4)] == [
        "shell_lower",
        "shell_lower",
        "battery",
        "arm_front_left",
    ]
    _, report = assign_parts(quad(), by_object(granularity))
    assert report.area_by_part["shell_lower"] == pytest.approx(0.03)
    assert report.area_by_part["battery"] == pytest.approx(0.005)


def test_object_names_match_whatever_their_case():
    """`Battery` in the source and `battery` in the config are one object."""
    assert PartSelector(objects=["BATTERY"]).accepts(quad()[2], (0.0, 0.0, 0.0))


def test_an_object_the_config_does_not_name_is_left_unclaimed():
    claimed = by_object("object", names=("battery",)).claim_all(quad())
    assert claimed[2].name == "battery"
    assert claimed[0] is None and claimed[3] is None


def test_a_shell_with_no_source_object_is_never_claimed_by_name():
    bare = Component(index=0, faces=1, area_m2=1e-4, centroid=(0, 0, 0), lo=(0, 0, 0), hi=(0, 0, 0))
    assert not PartSelector(objects=["battery"]).accepts(bare, (0.0, 0.0, 0.0))
    assert PartSelector().accepts(bare, (0.0, 0.0, 0.0)), "an empty selector still matches all"


def test_the_archive_split_puts_each_prim_in_its_named_part():
    """The thermal pass: each archive prim lands whole in the part of the same name."""
    from irsim.io.asset_parts import split_by_part
    from irsim.io.assets import AssetMesh, AssetMeshes

    def mesh(name, x0):
        v = np.array([[x0, 0, 0], [x0 + 0.01, 0, 0], [x0, 0.01, 0]], dtype=np.float64)
        return AssetMesh(
            name=name,
            material_name="white_plastic",
            vertices_m=v,
            faces=np.array([[0, 1, 2]], dtype=np.intp),
            area_m2=5e-5,
            area_before_m2=5e-5,
        )

    meshes = AssetMeshes(
        name="toy",
        meshes={"battery": mesh("battery", 0.0), "shell_lower": mesh("shell_lower", 0.0)},
    )
    parts, report = split_by_part(meshes, by_object("object", names=("shell_lower", "battery")))
    assert set(parts) == {"shell_lower", "battery"}
    assert report.coverage == pytest.approx(1.0)
