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


def test_asset_configs_are_listed_with_who_wrote_them(tmp_path) -> None:
    assets = tmp_path / "assets"
    assert call(tmp_path, "write-asset", asset_payload(), "--assets-dir", str(assets))["ok"]
    (assets / "handmade.yaml").write_text("# hand-written\nschema_version: 1\n", encoding="utf-8")
    listed = call(tmp_path, "list-assets", None, "--assets-dir", str(assets))
    assert listed["assets"] == [
        {"name": "handmade", "by_addon": False},
        {"name": "test_quad", "by_addon": True},
    ]


def test_a_hand_written_map_is_read_through_the_project_loader(tmp_path) -> None:
    # The committed Phantom 4 map: every target checked against the library on the way out.
    result = call(tmp_path, "read-asset", {"name": "phantom4"})
    assert result["ok"], result
    assert result["by_addon"] is False
    assert result["scale_to_metres"] == pytest.approx(0.01)
    assert result["materials"]["Metal_Matte"] == "aircraft_aluminium_painted"
    assert result["materials"]["chrome_shiny"] == "bare_aluminium"


def test_reading_an_asset_that_does_not_exist_is_refused(tmp_path) -> None:
    for name in ("no_such_asset", "../materials/carbon_fibre"):
        result = call(tmp_path, "read-asset", {"name": name})
        assert not result["ok"] and result["kind"] == "missing", name


# --- contacts, facing pairs and hidden parts -------------------------------------------------


def test_the_library_read_carries_the_joint_table(tmp_path) -> None:
    result = call(tmp_path, "library")
    joints = {j["name"]: j for j in result["joints"]}
    assert joints["dry_default"]["h_c_w_m2_k"] == 1000.0
    assert joints["dry_default"]["status"] == "ESTIMATED"
    assert joints["bolted_ferrous_new"]["status"] == "MEASURED"


def structure(**overrides) -> dict:
    payload = {
        "name": "test_car",
        "part_areas": {"body": 20.0, "floor_pan": 4.0, "bonnet": 1.5},
        "contacts": [
            {
                "parts": ["engine", "floor_pan"],
                "joint": "dry_default",
                "area_m2": 0.05,
                "reviewed": True,
                "found_by": "finder",
            },
        ],
        "facing": [
            {
                "parts": ["bonnet", "engine"],
                "area_m2": [0.6, 0.4],
                "gap_m": 0.08,
                "reviewed": False,
                "found_by": "finder",
            },
        ],
        "hidden_parts": [
            {
                "name": "engine",
                "kind": "piston_engine",
                "material": "rusted_steel",
                "inside": "body",
                "area_m2": 1.9,
                "box": {
                    "centre_m": [1.2, 0.0, 0.5],
                    "size_m": [0.6, 0.5, 0.5],
                    "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "mass_kg": 120.0,
                "heat_w": {"idle": 4000.0, "rated": 60000.0},
                "values": {
                    "source": "estimated",
                    "reference": "a 2 L petrol engine, 30 % efficient",
                },
            },
        ],
    }
    payload.update(overrides)
    return payload


def write_structure(tmp_path, payload) -> dict:
    return call(tmp_path, "write-structure", payload, "--models-dir", str(tmp_path / "models"))


def test_a_structure_is_written_beside_the_model_marked_as_the_add_ons(tmp_path) -> None:
    result = write_structure(tmp_path, structure())
    assert result["ok"], result
    assert (result["contacts"], result["facing"], result["hidden_parts"]) == (1, 1, 1)
    path = tmp_path / "models" / "test_car" / "test_car.structure.yaml"
    text = path.read_text(encoding="utf-8")
    assert bridge.MARKER in text.splitlines()[0]  # so a later export may replace it
    import yaml

    doc = yaml.safe_load(text)
    assert doc["hidden_parts"][0]["mass_kg"] == 120.0
    assert "area_m2" not in doc["hidden_parts"][0]  # used for the check, not a property
    assert doc["contacts"][0]["joint"] == "dry_default"


@pytest.mark.parametrize(
    ("change", "says"),
    [
        (lambda p: p["contacts"][0].update(parts=["engine", "gearbox"]), "no part named"),
        (lambda p: p["contacts"][0].update(joint="welded_magic"), "not in joints.yaml"),
        (lambda p: p["contacts"][0].update(area_m2=5.0), "larger than the smaller part"),
        (lambda p: p["contacts"][0].update(area_m2=0.0), "no contact area"),
        (lambda p: p["hidden_parts"][0].update(mass_kg=0.0), "has no mass"),
        (lambda p: p["hidden_parts"][0].update(material="unobtainium"), "not a library"),
        (lambda p: p["hidden_parts"][0]["heat_w"].update(rated=10.0), "less heat at full load"),
        (lambda p: p["facing"][0].update(parts=["bonnet", "bonnet"]), "to itself"),
    ],
)
def test_what_the_asset_format_will_refuse_is_refused_and_nothing_is_written(
    tmp_path, change, says
) -> None:
    """The refusals roadmap row AI.11 lists for its loader, plus the add-on's own two."""
    payload = structure()
    change(payload)
    result = write_structure(tmp_path, payload)
    assert not result["ok"] and result["kind"] == "invalid"
    assert says in result["error"]
    assert not (tmp_path / "models").exists()


def test_a_structure_is_replaced_only_when_asked_and_never_one_written_by_hand(tmp_path) -> None:
    assert write_structure(tmp_path, structure())["ok"]
    again = write_structure(tmp_path, structure())
    assert not again["ok"] and again["kind"] == "exists"
    assert write_structure(tmp_path, structure(overwrite=True))["ok"]
    path = tmp_path / "models" / "test_car" / "test_car.structure.yaml"
    path.write_text("# measured on the real car\ncontacts: []\n", encoding="utf-8")
    refused = write_structure(tmp_path, structure(overwrite=True))
    assert not refused["ok"] and refused["kind"] == "handwritten"
    assert path.read_text(encoding="utf-8").startswith("# measured on the real car")
