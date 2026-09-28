"""A STEP assembly tessellates to metres, keeps its part names, and obeys its facet angle (AI.10).

Skipped unless OpenCascade (`cadquery-ocp`, the optional ``cad`` extra) is importable: it is not a
default dependency, so the default gate cannot run this. The claims are the three things the
`low2high` skill relies on the tessellator for — no hand-typed scale, no hand-measured part
selectors, and a facet angle that the requested tolerance actually bounds. ADR 0155.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

pytest.importorskip("OCP")

from irsim.io.mesh_facets import edge_dihedrals, glb_meshes  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tess():
    spec = importlib.util.spec_from_file_location(
        "tessellate_step", REPO / "scripts" / "tessellate_step.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def step_file(tmp_path_factory) -> pathlib.Path:
    """A two-part assembly in millimetres: a 50 mm diameter shaft and a bracket, both named."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
    from OCP.Interface import Interface_Static
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    doc = TDocStd_Document(TCollection_ExtendedString("XmlXCAF"))
    XCAFApp_Application.GetApplication_s().InitDocument(doc)
    tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    shaft = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 25.0, 100.0)
    bracket = BRepPrimAPI_MakeBox(gp_Pnt(40, 0, 0), 30.0, 20.0, 10.0)
    for shape, name in [(shaft.Shape(), "motor_shaft"), (bracket.Shape(), "mount_bracket")]:
        TDataStd_Name.Set_s(tool.AddShape(shape, False), TCollection_ExtendedString(name))
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.Transfer(doc, STEPControl_AsIs)
    path = tmp_path_factory.mktemp("cad") / "assembly.step"
    writer.Write(str(path))
    return path


def test_part_names_survive_and_millimetres_become_metres(tess, step_file, tmp_path) -> None:
    summary = tess.tessellate_step(step_file, tmp_path / "a.glb", angle_deg=5.0)
    assert {"motor_shaft", "mount_bracket"} <= set(summary["parts"])
    lo, hi = summary["bounds_m"]
    # The assembly spans x from -25 mm (shaft) to 70 mm (bracket) and z over the 100 mm shaft.
    assert hi[0] - lo[0] == pytest.approx(0.095, abs=1e-4)
    assert hi[2] - lo[2] == pytest.approx(0.100, abs=1e-4)


@pytest.mark.parametrize("angle", [20.0, 5.0])
def test_the_requested_angle_bounds_the_facets_on_the_shaft(tess, step_file, tmp_path, angle):
    out = tmp_path / f"a{angle:g}.glb"
    # A generous chord so the angle, not the chord, is the binding tolerance.
    tess.tessellate_step(step_file, out, angle_deg=angle, chord_mm=5.0)
    curved = []
    for _name, verts, faces in glb_meshes(out):
        d = edge_dihedrals(verts, faces)
        curved.append(d.angle_deg[(d.angle_deg > 0.01) & (d.angle_deg < 60.0)])
    worst = max(float(a.max()) for a in curved if a.size)
    assert worst <= angle + 1e-6


def test_a_finer_angle_costs_more_triangles(tess, step_file, tmp_path) -> None:
    coarse = tess.tessellate_step(step_file, tmp_path / "c.glb", angle_deg=20.0, chord_mm=5.0)
    fine = tess.tessellate_step(step_file, tmp_path / "f.glb", angle_deg=5.0, chord_mm=5.0)
    assert fine["triangles"] > 2 * coarse["triangles"]
