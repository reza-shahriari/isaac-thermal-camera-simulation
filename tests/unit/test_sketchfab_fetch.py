"""The licence gate fails towards quarantine, and provenance survives a round trip (AI.8).

The gate is the one piece of this pipeline whose failure no image would ever reveal: a
CC-BY-NC model committed to a public library *renders identically* to a CC-BY one. So these
tests pin the legal semantics — exactly three licence slugs may ever be shareable, everything
unknown quarantines — rather than exercising code paths. ADR 0150.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from irsim.io.sketchfab import (
    KNOWN_LICENSE_SLUGS,
    SHAREABLE_LICENSE_SLUGS,
    asset_name_from_title,
    attribution_markdown,
    gate_license,
    load_provenance,
    model_uid_from_url,
    plan_fetch,
    provenance_from_local,
    provenance_from_model,
    write_provenance,
)

UID = "f1722ab650ad4d8dbe6fc4bf44e33d38"


def model_response(**overrides: Any) -> dict[str, Any]:
    """A Data API response with the fields the pipeline reads."""
    base: dict[str, Any] = {
        "uid": UID,
        "name": "Chunky Knight",
        "isDownloadable": True,
        "viewerUrl": f"https://sketchfab.com/3d-models/chunky-knight-{UID}",
        "user": {"username": "alwayshasbean", "profileUrl": "https://sketchfab.com/alwayshasbean"},
        "license": {
            "slug": "by",
            "label": "CC Attribution",
            "url": "https://creativecommons.org/licenses/by/4.0/",
        },
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------ the UID from a shared link


@pytest.mark.parametrize(
    "url",
    [
        f"https://sketchfab.com/3d-models/chunky-knight-{UID}",
        f"https://sketchfab.com/3d-models/chunky-knight-{UID}/",
        f"https://sketchfab.com/3d-models/chunky-knight-{UID}?utm_source=shared",
        f"https://sketchfab.com/models/{UID}/embed",
        f"HTTPS://SKETCHFAB.COM/3d-models/Chunky-Knight-{UID.upper()}",
        UID,
    ],
)
def test_uid_is_found_in_every_link_shape(url: str) -> None:
    assert model_uid_from_url(url) == UID


def test_uid_is_the_last_hex_run_not_the_first() -> None:
    # A slug can contain a 32-hex run of its own; the UID is the trailing one.
    decoy = "deadbeefdeadbeefdeadbeefdeadbeef"
    assert model_uid_from_url(f"https://sketchfab.com/3d-models/{decoy}-thing-{UID}") == UID


@pytest.mark.parametrize(
    "url",
    [
        "https://sketchfab.com/alwayshasbean",  # a user page has no model uid
        "https://sketchfab.com/3d-models/",
        "f1722ab650ad4d8dbe6fc4bf44e33d3",  # 31 chars: one digit short
    ],
)
def test_a_link_without_a_uid_is_refused(url: str) -> None:
    with pytest.raises(ValueError, match="no Sketchfab model UID"):
        model_uid_from_url(url)


# ------------------------------------------------------------------------------ the licence gate


@pytest.mark.parametrize("slug", sorted(SHAREABLE_LICENSE_SLUGS))
def test_the_three_permissive_licences_are_shareable(slug: str) -> None:
    assert gate_license({"slug": slug, "label": slug}).shareable is True


@pytest.mark.parametrize("slug", sorted(KNOWN_LICENSE_SLUGS - SHAREABLE_LICENSE_SLUGS))
def test_every_other_known_licence_quarantines_with_a_reason(slug: str) -> None:
    decision = gate_license({"slug": slug, "label": slug})
    assert decision.shareable is False
    # The reason must say why, not just that: it is what the operator reads before overriding.
    assert len(decision.reason) > 20


def test_the_shareable_set_is_exactly_cc0_by_and_by_sa() -> None:
    # The legal core of the gate. Widening this set is an ADR-level decision, not a refactor:
    # ND forbids the derivative we make, NC forbids the project's use, Standard forbids
    # stand-alone redistribution. If this test surprises you, read ADR 0150 before touching it.
    assert {"cc0", "by", "by-sa"} == SHAREABLE_LICENSE_SLUGS


@pytest.mark.parametrize(
    "license_info",
    [
        {},  # nothing at all
        {"slug": "brand-new-licence-2027"},  # a slug this code has never seen
        {"label": "Some Future Licence"},  # a label with no slug and no mapping
    ],
)
def test_an_unknown_licence_is_never_shareable(license_info: dict[str, Any]) -> None:
    decision = gate_license(license_info)
    assert decision.shareable is False
    assert "quarantin" in decision.reason  # fails towards quarantine, and says so


def test_a_label_without_a_slug_still_gates_correctly_both_ways() -> None:
    assert gate_license({"label": "CC0 Public Domain"}).shareable is True
    assert gate_license({"label": "CC Attribution-NoDerivs"}).shareable is False


# ------------------------------------------------------------------------------------ the plan


def test_a_shareable_model_plans_into_the_library_tree() -> None:
    plan = plan_fetch(model_response())
    assert plan.problems == ()
    assert plan.asset_name == "chunky_knight"
    assert plan.dest_subdir == "chunky_knight"


def test_an_nd_model_plans_into_quarantine() -> None:
    plan = plan_fetch(model_response(license={"slug": "by-nd", "label": "CC Attribution-NoDerivs"}))
    assert plan.decision.shareable is False
    assert plan.dest_subdir == "quarantine/chunky_knight"


def test_a_view_only_model_is_a_named_problem_not_a_crash() -> None:
    plan = plan_fetch(model_response(isDownloadable=False))
    assert any("downloadable" in p for p in plan.problems)


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Chunky Knight", "chunky_knight"),
        ("DJI Phantom 4 Pro (scan!)", "dji_phantom_4_pro_scan"),
        ("4K Drone", "m_4k_drone"),  # a USD prim path cannot start with a digit
        ("###", f"m_{UID[:8]}"),  # an all-symbol title falls back to the uid
    ],
)
def test_asset_names_obey_the_usd_identifier_grammar(title: str, expected: str) -> None:
    name = asset_name_from_title(title, UID)
    assert name == expected
    assert name[0].isalpha() or name[0] == "_"
    assert all(c.isalnum() or c == "_" for c in name)


# ------------------------------------------------------------------- provenance and attribution


def test_provenance_round_trips_through_yaml(tmp_path: Path) -> None:
    model = model_response()
    record = provenance_from_model(
        model,
        asset_name="chunky_knight",
        decision=gate_license(model["license"]),
        downloaded_at="2026-09-26T12:00:00+00:00",
        archive_sha256="ab" * 32,
    )
    path = tmp_path / "chunky_knight.provenance.yaml"
    write_provenance(record, path)
    assert load_provenance(path) == record


def test_attribution_carries_the_four_required_elements() -> None:
    model = model_response()
    record = provenance_from_model(
        model, asset_name="chunky_knight", decision=gate_license(model["license"])
    )
    text = attribution_markdown(record)
    assert "alwayshasbean" in text  # creator, linked
    assert record.model_url in text  # the model page
    assert "creativecommons.org/licenses/by" in text  # the licence terms
    assert "Changes:" in text  # CC-BY requires indicating modifications


# --------------------------------------------------- the real-world name, and a file handed over


def test_the_real_name_names_the_asset_over_the_page_title() -> None:
    # A page titled "free low poly drone" is a DJI Mavic 3 once the owner says so; the asset key
    # and every search for its published specs follow the real name.
    plan = plan_fetch(model_response(name="free low poly drone"), real_name="DJI Mavic 3")
    assert plan.asset_name == "dji_mavic_3"
    assert (
        plan_fetch(model_response(name="free low poly drone")).asset_name == "free_low_poly_drone"
    )


def test_the_real_name_travels_in_provenance_and_attribution(tmp_path: Path) -> None:
    model = model_response()
    record = provenance_from_model(
        model,
        asset_name="dji_mavic_3",
        decision=gate_license(model["license"]),
        real_name="DJI Mavic 3",
    )
    assert record.real_name == "DJI Mavic 3"
    assert "Depicts: DJI Mavic 3" in attribution_markdown(record)
    path = tmp_path / "p.yaml"
    write_provenance(record, path)
    assert load_provenance(path).real_name == "DJI Mavic 3"


def test_a_local_file_with_no_stated_licence_is_quarantined() -> None:
    # No special case for local files: "somebody gave me this file" is the unknown-licence case.
    record = provenance_from_local(
        asset_name="dji_mavic_3",
        file_name="mavic.fbx",
        decision=gate_license({}),
        real_name="DJI Mavic 3",
    )
    assert record.source == "local"
    assert record.shareable is False
    assert record.source_format == "fbx"
    assert record.title == "DJI Mavic 3"


def test_a_local_file_with_a_stated_permissive_licence_is_shareable(tmp_path: Path) -> None:
    record = provenance_from_local(
        asset_name="dji_mavic_3",
        file_name="mavic.glb",
        decision=gate_license({"slug": "by"}),
        author="some_artist",
        archive_sha256="cd" * 32,
    )
    assert record.shareable is True
    text = attribution_markdown(record)
    assert "Supplied as a file" in text and "some_artist" in text
    path = tmp_path / "p.yaml"
    write_provenance(record, path)
    assert load_provenance(path) == record


def test_records_written_before_real_name_existed_still_load(tmp_path: Path) -> None:
    # Additive schema change: an AI.8-era record has no `source` or `real_name` key.
    path = tmp_path / "old.provenance.yaml"
    path.write_text(
        "provenance:\n  schema_version: 1\n  asset_name: x\n  uid: " + UID + "\n  title: X\n"
        "  author: a\n  model_url: https://sketchfab.com/3d-models/x\n"
        "  license: {slug: by, label: CC Attribution}\n  shareable: true\n  gate_reason: ok\n"
    )
    record = load_provenance(path)
    assert record.source == "sketchfab"
    assert record.real_name == ""
