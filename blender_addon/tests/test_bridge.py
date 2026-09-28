"""The bridge: the add-on's only way to read the library or write into the repository.

Every write here goes to a temporary directory (``--material-dir`` / ``--assets-dir``); the
committed library is only read. What these tests pin is that the bridge refuses exactly what the
project's loaders refuse -- so a material created in Blender is one the committed library walk
(``tests/unit/test_materials_library.py``) will accept, and nothing it rejects reaches disk.
"""

import json
import os
import pathlib
import shutil
import sys

import pytest
from irsim_thermal import bridge, bridge_client

REPO = pathlib.Path(__file__).resolve().parents[2]
LIBRARY = REPO / "configs" / "materials"
BANDS = ("nir", "swir", "mwir", "lwir")

#: The interpreter the add-on would call. Not ``sys.executable``: under Isaac Sim's ``python.sh``
#: that is the bare kit binary, which only works inside the environment ``python.sh`` sets up.
ISAAC = pathlib.Path("~/IsaacSim/_build/linux-x86_64/release/python.sh").expanduser()
PROJECT_PYTHON = os.environ.get("IRSIM_PYTHON") or (
    str(ISAAC) if ISAAC.is_file() else sys.executable
)


def call(tmp_path: pathlib.Path, command: str, payload=None, *extra: str) -> dict:
    out = tmp_path / f"{command}.json"
    argv = [command, "--repo", str(REPO), "--out", str(out), *extra]
    if payload is not None:
        inp = tmp_path / f"{command}.in.json"
        inp.write_text(json.dumps(payload), encoding="utf-8")
        argv += ["--in", str(inp)]
    assert bridge.main(argv) == 0
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture
def small_library(tmp_path: pathlib.Path) -> pathlib.Path:
    lib = tmp_path / "materials"
    lib.mkdir()
    for name in ("carbon_fibre", "abs_plastic_white", "bare_aluminium"):
        shutil.copy(LIBRARY / f"{name}.yaml", lib / f"{name}.yaml")
    return lib


def draft(**overrides) -> dict:
    material = {
        "name": "nylon_black",
        "source": "estimated",
        "surface_treatment": "as_manufactured",
        "reference": "estimated from similar moulded polymers",
        "description": "Black moulded nylon (PA6) housing",
        "thermal": {
            "density_kg_m3": 1140.0,
            "specific_heat_j_kgk": 1700.0,
            "conductivity_w_mk": 0.25,
            "thickness_m": 0.002,
            "solar_absorptivity": 0.92,
        },
        "optical": {
            "emissivity_per_band": {"nir": 0.90, "swir": 0.88, "mwir": 0.90, "lwir": 0.92},
            "transmittance_per_band": {"nir": 0.0, "swir": 0.0, "mwir": 0.0, "lwir": 0.0},
            "roughness_per_band": {"nir": 0.2, "swir": 0.2, "mwir": 0.1, "lwir": 0.1},
            "angular_model": {"type": "empirical", "a": 0.25, "p": 5.0},
        },
    }
    material.update(overrides)
    return {"material": material}


def test_the_library_closes_in_every_band_as_the_bridge_reports_it(tmp_path) -> None:
    result = call(tmp_path, "library")
    assert result["ok"], result
    assert tuple(result["bands"])[:4] == BANDS
    names = {m["name"] for m in result["materials"]}
    assert {"carbon_fibre", "abs_plastic_white", "bare_aluminium", "car_paint_black"} <= names
    for m in result["materials"]:
        for band in BANDS:
            b = m["bands"][band]
            assert "error" not in b, (m["name"], band, b)
            total = b["emissivity"] + b["reflectance"] + b["transmittance"]
            assert total == pytest.approx(1.0, abs=1e-6), (m["name"], band)
    alu = next(m for m in result["materials"] if m["name"] == "bare_aluminium")
    assert alu["bands"]["lwir"]["emissivity"] == pytest.approx(0.09)
    assert result["mirror_emissivity"] == pytest.approx(0.2)


def test_a_spectral_material_is_shown_band_integrated(tmp_path) -> None:
    # car_paint_black authors a spectral file; the add-on must show what the solver uses.
    result = call(tmp_path, "library")
    paint = next(m for m in result["materials"] if m["name"] == "car_paint_black")
    assert paint["optical"]["spectral"] is True
    assert paint["bands"]["lwir"]["emissivity"] == pytest.approx(0.90, abs=1e-6)


def test_a_valid_material_is_written_and_the_library_still_loads(tmp_path, small_library) -> None:
    result = call(tmp_path, "write-material", draft(), "--material-dir", str(small_library))
    assert result["ok"], result
    written = small_library / "nylon_black.yaml"
    assert written.exists()
    text = written.read_text(encoding="utf-8")
    assert bridge.MARKER in text.splitlines()[1]
    assert "reflectance_per_band" not in text  # derived, never authored (CLAUDE.md #4)
    lwir = result["bands"]["lwir"]
    assert lwir["reflectance"] == pytest.approx(1.0 - 0.92)

    from irsim.materials.library import MaterialLibrary

    lib = MaterialLibrary.load(small_library)
    assert "nylon_black" in lib
    assert lib["nylon_black"].band_properties("mwir").emissivity == pytest.approx(0.90)


def test_emissivity_plus_transmittance_over_one_is_refused_and_nothing_is_written(
    tmp_path, small_library
) -> None:
    bad = draft()
    bad["material"]["optical"]["transmittance_per_band"]["swir"] = 0.2  # 0.88 + 0.2 > 1
    result = call(tmp_path, "write-material", bad, "--material-dir", str(small_library))
    assert not result["ok"]
    assert result["kind"] == "invalid"
    assert "closure" in result["error"].lower()
    assert not (small_library / "nylon_black.yaml").exists()


def test_authoring_reflectance_as_well_is_refused(tmp_path, small_library) -> None:
    bad = draft()
    bad["material"]["optical"]["reflectance_per_band"] = {b: 0.1 for b in BANDS}
    result = call(tmp_path, "check-material", bad, "--material-dir", str(small_library))
    assert not result["ok"]
    assert "exactly one optical property" in result["error"]


def test_a_material_that_leaves_a_band_out_is_refused(tmp_path, small_library) -> None:
    bad = draft()
    del bad["material"]["optical"]["emissivity_per_band"]["swir"]
    result = call(tmp_path, "check-material", bad, "--material-dir", str(small_library))
    assert not result["ok"]
    assert "swir" in result["error"]


def test_an_existing_name_is_refused(tmp_path, small_library) -> None:
    result = call(
        tmp_path,
        "write-material",
        draft(name="carbon_fibre"),
        "--material-dir",
        str(small_library),
    )
    assert not result["ok"] and result["kind"] == "exists"


def test_a_name_the_schema_would_refuse_is_refused(tmp_path, small_library) -> None:
    result = call(
        tmp_path, "check-material", draft(name="Nylon Black"), "--material-dir", str(small_library)
    )
    assert not result["ok"] and result["kind"] == "invalid"


def test_a_mirror_like_material_is_accepted_with_a_warning(tmp_path, small_library) -> None:
    shiny = draft(name="chrome_trim", surface_treatment="polished")
    shiny["material"]["optical"]["emissivity_per_band"] = {b: 0.05 for b in BANDS}
    result = call(tmp_path, "check-material", shiny, "--material-dir", str(small_library))
    assert result["ok"]
    assert any("reflected surroundings" in w for w in result["warnings"])


def asset_payload(**overrides) -> dict:
    payload = {
        "name": "test_quad",
        "source_file": "3d_models/test_quad/test_quad.usdc",
        "scale_to_metres": 1.0,
        "materials": {"white_plastic": "abs_plastic_white", "Metal_Matte": "carbon_fibre"},
        "parts": ["arm_left", "shell"],
        "notes": ["Coverage at export: 100.0 %"],
    }
    payload.update(overrides)
    return payload


def test_an_asset_config_is_written_and_reloads_through_the_project_loader(tmp_path) -> None:
    assets = tmp_path / "assets"
    result = call(tmp_path, "write-asset", asset_payload(), "--assets-dir", str(assets))
    assert result["ok"], result
    from irsim.materials.mapping import load_asset_mapping

    mapping = load_asset_mapping(assets / "test_quad.yaml")
    assert mapping.lookup("WHITE_PLASTIC") == "abs_plastic_white"  # case-insensitive, as audited
    assert mapping.scale_to_metres == 1.0
    assert "arm_left" in (assets / "test_quad.yaml").read_text(encoding="utf-8")


def test_an_asset_mapping_to_a_material_nobody_has_is_refused(tmp_path) -> None:
    payload = asset_payload(materials={"white_plastic": "unobtainium"})
    result = call(tmp_path, "write-asset", payload, "--assets-dir", str(tmp_path / "a"))
    assert not result["ok"] and "unobtainium" in result["error"]


def test_keys_differing_only_by_case_are_refused(tmp_path) -> None:
    payload = asset_payload(materials={"Metal": "carbon_fibre", "metal": "carbon_fibre"})
    result = call(tmp_path, "write-asset", payload, "--assets-dir", str(tmp_path / "a"))
    assert not result["ok"] and result["kind"] == "invalid"


def test_a_hand_written_asset_config_is_never_overwritten(tmp_path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    precious = "# DJI Phantom 4 Pro: why every line of this map is what it is\nschema_version: 1\n"
    (assets / "test_quad.yaml").write_text(precious, encoding="utf-8")
    result = call(
        tmp_path, "write-asset", asset_payload(overwrite=True), "--assets-dir", str(assets)
    )
    assert not result["ok"] and result["kind"] == "handwritten"
    assert (assets / "test_quad.yaml").read_text(encoding="utf-8") == precious


def test_an_earlier_export_is_replaced_only_when_asked(tmp_path) -> None:
    assets = tmp_path / "assets"
    assert call(tmp_path, "write-asset", asset_payload(), "--assets-dir", str(assets))["ok"]
    again = call(tmp_path, "write-asset", asset_payload(), "--assets-dir", str(assets))
    assert not again["ok"] and again["kind"] == "exists"
    replaced = call(
        tmp_path,
        "write-asset",
        asset_payload(overwrite=True, materials={"white_plastic": "carbon_fibre"}),
        "--assets-dir",
        str(assets),
    )
    assert replaced["ok"] and replaced["materials"] == {"white_plastic": "carbon_fibre"}


def test_the_client_runs_the_bridge_in_a_clean_interpreter(tmp_path, monkeypatch) -> None:
    # What Blender does: a separate process, with Blender's own Python variables stripped.
    monkeypatch.setenv("PYTHONHOME", "/nonexistent/blender/python")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/snap/blender/1/lib:/usr/lib/x86_64-linux-gnu")
    env = bridge_client.child_env(str(REPO))
    assert "PYTHONHOME" not in env
    assert env["LD_LIBRARY_PATH"] == "/usr/lib/x86_64-linux-gnu"
    assert env["PYTHONPATH"] == str(REPO / "src")
    result = bridge_client.run_bridge(PROJECT_PYTHON, str(REPO), "library")
    assert len(result["materials"]) >= 20


def test_the_client_reports_a_refusal_with_its_kind(tmp_path, small_library) -> None:
    with pytest.raises(bridge_client.BridgeError) as err:
        bridge_client.run_bridge(
            PROJECT_PYTHON,
            str(REPO),
            "write-material",
            draft(name="carbon_fibre"),
            extra_args=("--material-dir", str(small_library)),
        )
    assert err.value.kind == "exists"


def test_the_client_explains_a_missing_setup() -> None:
    with pytest.raises(bridge_client.BridgeError, match="repository is not set"):
        bridge_client.run_bridge(sys.executable, "", "library")
    with pytest.raises(bridge_client.BridgeError, match="does not exist"):
        bridge_client.run_bridge("/no/such/python", str(REPO), "library")
