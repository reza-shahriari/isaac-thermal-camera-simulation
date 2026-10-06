"""Every contact an asset authors joins two parts that touch on the solver's cells (AI.17).

A scene refuses a contact whose parts have no cell pair within its gap (`proximity_contactor`,
TC.16). `scripts/check_asset_contacts.py` runs that test against the part-split archive a scene
reads. This module runs it over every asset that authors contacts. The archives are generated, not
in git, so each case skips without one. The control proves the check can fail: a propeller
"bolted" to the opposite corner's landing foot is refused.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
WITH_CONTACTS = ("phantom4_parts", "dji_inspire_3", "dji_matrice_300_rtk")


def _archive(name: str) -> pathlib.Path:
    """The part-split archive a scene reads: ``<name>_parts`` when prep wrote one, else its own."""
    split = REPO / "data" / "assets" / f"{name}_parts" / f"{name}_parts.meshes.npz"
    return split if split.exists() else REPO / "data" / "assets" / name / f"{name}.meshes.npz"


@pytest.fixture(scope="module")
def check():  # type: ignore[no-untyped-def]
    path = REPO / "scripts" / "check_asset_contacts.py"
    spec = importlib.util.spec_from_file_location("check_asset_contacts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", WITH_CONTACTS)
def test_every_authored_contact_touches(check, name: str) -> None:  # type: ignore[no-untyped-def]
    if not _archive(name).exists():
        pytest.skip(f"the {name} archive is generated, not in git")
    from irsim.materials.mapping import load_asset_mapping

    parts = load_asset_mapping(name).parts
    assert parts is not None and parts.contacts, f"{name} is listed as authoring contacts"
    assert check.main([name]) == 0


def test_a_contact_between_parts_that_do_not_touch_is_refused(check, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    if not _archive("dji_inspire_3").exists():
        pytest.skip("the dji_inspire_3 archive is generated, not in git")
    import irsim.materials.mapping as mapping
    from irsim.io.asset_parts import ContactSpec

    real = mapping.load_asset_mapping

    def with_a_bad_contact(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        asset = real(name, *args, **kwargs)
        far = ContactSpec(
            a="propeller_front_left", b="landing_foot_rear_right", joint="dry_default", area_m2=1e-4
        )
        parts = asset.parts.model_copy(update={"contacts": [*asset.parts.contacts, far]})
        return asset.model_copy(update={"parts": parts})

    monkeypatch.setattr(mapping, "load_asset_mapping", with_a_bad_contact)
    assert check.main(["dji_inspire_3"]) == 1
