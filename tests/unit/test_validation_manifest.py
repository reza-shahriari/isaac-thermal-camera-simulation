"""The validation-data index (ME.1): the fields that gate a measurement, and the refusals.

This index is the reason a statistic from public video can be trusted at all, so the tests are
about the things that would let an untrustworthy one through: a set with no recorded signal path,
an analyser run on data that cannot support it, a licence silently assumed to be permissive, and a
README that has drifted from the fields the code actually reads.

docs/physics-model.md §15 T4; ADR 0003.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml
from pydantic import ValidationError

from irsim.validation.signal_path import SIGNAL_PATHS
from irsim_eval.fetch import Action, download, plan, sha256_of, target_dir, verify
from irsim_eval.manifest import (
    MANIFEST_SCHEMA_VERSION,
    UNSTATED,
    Dataset,
    Manifest,
    load_manifest,
    manifest_path,
    readme_path,
    render_readme,
)


@pytest.fixture(scope="module")
def manifest() -> Manifest:
    return load_manifest()


def decisions(actions: list[Action]) -> dict[str, str]:
    return {a.name: a.decision for a in actions}


# --- the fields that gate a measurement ---------------------------------------------------------


def test_every_set_records_a_licence_and_a_signal_path(manifest: Manifest) -> None:
    """The ME.1 criterion. A set that cannot say what its frames are cannot be measured."""
    for name, dataset in manifest.datasets.items():
        assert dataset.licence.strip(), name
        assert dataset.signal_path in SIGNAL_PATHS, name
        assert dataset.signal_path_note.strip(), name
        assert dataset.analysers, name
        assert dataset.why.strip(), name


def test_the_index_names_exactly_one_primary_set(manifest: Manifest) -> None:
    """Two primaries would invite a comparison that averages sets with different signal paths."""
    name, primary = manifest.primary
    assert name == "halmstad_drone_detection"
    assert primary.licence_known
    with pytest.raises(ValueError, match="exactly one primary"):
        Manifest.model_validate(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "checked_utc": "2026-09-13",
                "datasets": {
                    k: {**v.model_dump(exclude_none=True), "role": "primary"}
                    for k, v in list(manifest.datasets.items())[:2]
                },
            }
        )


def test_only_the_y16_set_may_carry_a_noise_decomposition(manifest: Manifest) -> None:
    """A 3-D noise decomposition needs the sensor's own stream, not an ISP's output.

    This is the index earning its place: the same flat-sky patch is a laboratory measurement on a
    Y16-derived clip and a measurement of somebody's unknown AGC on a display-output one. Only the
    Halmstad set states a signal path that supports it.
    """
    assert manifest.usable_for("noise_3d") == ["halmstad_drone_detection"]
    assert manifest.usable_for("temporal_psd") == ["halmstad_drone_detection"]


def test_an_agc_signature_may_not_be_measured_on_the_set_that_has_no_agc(
    manifest: Manifest,
) -> None:
    """The Halmstad clips never went through an AGC, so its histogram shape is the recorder's."""
    halmstad = manifest.datasets["halmstad_drone_detection"]
    assert not halmstad.may_run("agc_signature")
    assert "agc_signature" in halmstad.excluded_analysers
    # Two sets now, both display output through an undocumented ISP. `anti_uav_600` joined in
    # `XD.1`; it is the same kind of picture as `anti_uav_410` and twice the size of it, so an
    # AGC signature measured on one can be checked against the other rather than taken alone.
    assert manifest.usable_for("agc_signature") == ["anti_uav_410", "anti_uav_600"]


def test_an_exclusion_beats_an_inclusion(manifest: Manifest) -> None:
    """Belt and braces: a name in both lists must be refused, not allowed."""
    both = manifest.datasets["halmstad_drone_detection"].model_copy(
        update={"analysers": ["noise_3d", "agc_signature"], "excluded_analysers": ["agc_signature"]}
    )
    assert both.may_run("noise_3d")
    assert not both.may_run("agc_signature")


def test_the_range_labelled_set_is_the_only_one_that_can_do_size_versus_range(
    manifest: Manifest,
) -> None:
    """Without a range label, size-versus-range is unmeasurable however many frames there are."""
    assert manifest.usable_for("size_vs_range") == ["lrddv3"]


def test_single_frame_sets_are_priors_and_nothing_temporal(manifest: Manifest) -> None:
    """One frame cannot show an FFC freeze, a temporal PSD or a smear, whatever else it shows."""
    for name in ("irstd_1k", "nuaa_sirst"):
        dataset = manifest.datasets[name]
        assert dataset.role == "prior"
        for analyser in ("ffc_freeze", "temporal_psd", "noise_3d", "size_vs_range"):
            assert not dataset.may_run(analyser), f"{name} must not claim {analyser}"


# --- licences -----------------------------------------------------------------------------------


def test_unstated_is_recorded_rather_than_assumed(manifest: Manifest) -> None:
    """Five of the seven publishers stated no terms, and the index says so in those words.

    `lrddv3` left this set in `XD.1`: its publisher *did* state terms, on the dataset page rather
    than in the paper, and the index had recorded the absence of a licence it had not gone and
    read. `anti_uav_600` joined it for the opposite reason -- its repository states a licence
    loudly, and that licence is the code's.
    """
    unstated = {k for k, v in manifest.datasets.items() if not v.licence_known}
    assert unstated == {
        "anti_uav_410",
        "anti_uav_600",
        "cst_anti_uav",
        "irstd_1k",
        "nuaa_sirst",
    }
    assert all(manifest.datasets[k].licence == UNSTATED for k in unstated)


def test_a_set_with_no_stated_licence_is_refused_by_default(manifest: Manifest) -> None:
    """Not an error -- a decision handed back to a person, with the reason attached."""
    actions = plan(manifest, ["anti_uav_410"])
    assert decisions(actions) == {"anti_uav_410": "refused"}
    assert "no licence" in actions[0].reason
    assert actions[0].needs_a_person


def test_the_refusal_can_be_overridden_deliberately(manifest: Manifest) -> None:
    """Once a person has read the terms, the flag says so and the set proceeds as manual."""
    actions = plan(manifest, ["anti_uav_410"], accept_unstated_licence=True)
    assert decisions(actions) == {"anti_uav_410": "manual"}


def test_the_licence_gate_is_checked_before_the_access_mode(manifest: Manifest) -> None:
    """An unreleased, unlicensed set reports the licence: that decides whether to want it."""
    assert decisions(plan(manifest, ["cst_anti_uav"])) == {"cst_anti_uav": "refused"}
    assert decisions(plan(manifest, ["cst_anti_uav"], accept_unstated_licence=True)) == {
        "cst_anti_uav": "unreleased"
    }


def test_the_primary_set_is_fetchable_and_says_how_large_it_is(manifest: Manifest) -> None:
    """The one set with a licence *and* a direct URL, with its size in the plan.

    It was indexed `manual` on 2026-09-13 from the repository README, which points at the DOI
    landing page rather than at the file. CHECKED 2026-09-15 at the DOI: Zenodo serves the archive
    over https with no interstitial, so the landing page needs a human and the file does not, and
    the index now says so. The size is in the reason because these sets run to tens of gigabytes
    and "download" is not a decision anyone should make blind.
    """
    actions = plan(manifest, ["halmstad_drone_detection"])
    assert decisions(actions) == {"halmstad_drone_detection": "download"}
    assert actions[0].url is not None and actions[0].url.startswith("https://zenodo.org/")
    assert "0.31 GB" in actions[0].reason


def test_a_drive_link_is_still_a_human_job(manifest: Manifest) -> None:
    """The policy the set above used to carry: no direct URL means no download, ever.

    A Google Drive interstitial or a university access form is a human's job, and a script that
    pretended otherwise would fail in a way that looks like a network error.
    """
    actions = plan(manifest, ["anti_uav_410"], accept_unstated_licence=True)
    assert decisions(actions) == {"anti_uav_410": "manual"}
    assert "by hand" in actions[0].reason


def test_an_unknown_set_name_is_an_error_not_a_silent_skip(manifest: Manifest) -> None:
    with pytest.raises(KeyError, match="not in the index"):
        plan(manifest, ["there_is_no_such_dataset"])


# --- fetching and hashing -----------------------------------------------------------------------


def test_a_direct_url_is_planned_as_a_download(manifest: Manifest) -> None:
    """The primary set is the one that has one; this pins the path on a synthetic entry too, so
    it keeps working if the index ever loses its only direct URL again."""
    direct = manifest.datasets["halmstad_drone_detection"].model_copy(
        update={"access": "direct", "download_url": "https://example.invalid/x.zip"}
    )
    synthetic = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        checked_utc="2026-09-13",
        datasets={"synthetic": direct},
    )
    actions = plan(synthetic)
    assert decisions(actions) == {"synthetic": "download"}
    assert actions[0].url == "https://example.invalid/x.zip"


def test_download_refuses_an_unencrypted_scheme(tmp_path: pathlib.Path) -> None:
    """These archives are huge; silently downgrading the transport is not ours to decide."""
    with pytest.raises(ValueError, match="non-https"):
        download("http://example.invalid/x.zip", tmp_path / "x.zip")
    with pytest.raises(ValueError, match="non-https"):
        download("ftp://example.invalid/x.zip", tmp_path / "x.zip")


def test_download_streams_a_file_url_byte_for_byte(tmp_path: pathlib.Path) -> None:
    """The transfer path itself, checked without a network: file:// is a real urllib scheme."""
    source = tmp_path / "source.bin"
    payload = bytes(range(256)) * 5000  # larger than one chunk
    source.write_bytes(payload)
    out = download(source.as_uri(), tmp_path / "out" / "copy.bin")
    assert out.read_bytes() == payload
    assert sha256_of(out) == sha256_of(source)


def test_verify_matches_a_recorded_hash_and_reports_a_mismatch(
    manifest: Manifest, tmp_path: pathlib.Path
) -> None:
    """The point of recording a hash: 'measured on these exact bytes' stays checkable."""
    path = tmp_path / "clip.bin"
    path.write_bytes(b"thermal video, honestly")
    digest = sha256_of(path)

    indexed = manifest.datasets["halmstad_drone_detection"]
    unrecorded = indexed.model_copy(update={"sha256": None})
    ok, seen = verify(unrecorded, path)
    assert ok and seen == digest, "nothing recorded yet, so nothing to contradict"

    recorded = indexed.model_copy(update={"sha256": digest})
    assert verify(recorded, path) == (True, digest)

    path.write_bytes(b"thermal video, edited")
    matched, changed = verify(recorded, path)
    assert not matched and changed != digest


def test_the_measured_set_has_its_hash_recorded(manifest: Manifest) -> None:
    """ADR 0004's rule, applied to somebody else's data: the reference statistics were measured on
    specific bytes, so the index has to say which bytes, or 'regenerates deterministically' is a
    claim nobody can check a year later."""
    indexed = manifest.datasets["halmstad_drone_detection"]
    assert indexed.sha256 is not None and len(indexed.sha256) == 64
    assert indexed.download_bytes == 311348350


def test_a_malformed_hash_is_rejected_by_the_schema() -> None:
    """A truncated or non-hex digest in the index is worse than none: it looks authoritative."""
    base = load_manifest().datasets["halmstad_drone_detection"].model_dump(exclude_none=True)
    for bad in ("abc", "z" * 64, "AB" * 31):
        with pytest.raises(ValueError, match="64 hex"):
            Dataset.model_validate({**base, "sha256": bad})
    good = "a" * 64
    assert Dataset.model_validate({**base, "sha256": good.upper()}).sha256 == good


def test_datasets_land_outside_the_repository_tree_but_under_data(manifest: Manifest) -> None:
    """`data/validation/<set>/`, which .gitignore excludes -- the frames never enter git."""
    destination = target_dir("halmstad_drone_detection")
    assert destination.parent == manifest_path().parent
    assert destination.name == "halmstad_drone_detection"
    ignore = (manifest_path().parents[2] / ".gitignore").read_text()
    assert "data/validation/*/" in ignore


# --- the generated index ------------------------------------------------------------------------


def test_the_readme_is_generated_and_current(manifest: Manifest) -> None:
    """A prose index that disagrees with the machine-readable one is worse than having neither.

    Regenerate with `python scripts/fetch_validation_data.py --render-readme`.
    """
    assert readme_path().read_text(encoding="utf-8") == render_readme(manifest)


def test_the_readme_states_every_licence_and_signal_path(manifest: Manifest) -> None:
    """What a reader must not have to dig for: the terms, and what the frames actually are."""
    text = readme_path().read_text(encoding="utf-8")
    assert "GENERATED FILE" in text
    for name, dataset in manifest.datasets.items():
        assert f"`{name}`" in text
        assert dataset.licence in text
        assert f"`{dataset.signal_path}`" in text
        assert dataset.signal_path_note.strip().split("\n")[0][:40] in text.replace("\n", " ")


def test_the_manifest_file_is_the_one_the_package_ships(manifest: Manifest) -> None:
    """The loader must find the committed index, not a copy that drifted."""
    raw = yaml.safe_load(manifest_path().read_text(encoding="utf-8"))
    assert set(raw["datasets"]) == set(manifest.datasets)
    assert raw["schema_version"] == manifest.schema_version


# --- XD.1: the four fields, and which of them is load-bearing -----------------------------------


def test_the_range_set_carries_the_datas_licence_and_not_the_papers(manifest: Manifest) -> None:
    """`XD.1` asked for CC BY 4.0 here. That is the **paper's** arXiv licence, not the data's.

    The dataset page states its own terms -- "fully free to use for commercial or R&D purposes
    under CDLA-v2", linking CDLA-Permissive-2.0 -- and this field gates `plan`, so writing the
    paper's badge into it would have opened the gate on terms the frames do not carry. That is
    precisely the "silently wrong permission" the row was written to prevent, arrived at from the
    other direction.
    """
    lrddv3 = manifest.datasets["lrddv3"]
    assert lrddv3.licence == "CDLA-Permissive-2.0"
    assert lrddv3.licence_known, "a stated licence must open the gate"
    assert decisions(plan(manifest, ["lrddv3"])) == {"lrddv3": "manual"}

    note = (lrddv3.licence_note or "").lower()
    assert "cc by 4.0" in note and "arxiv" in note, "the note must name the licence it is not"
    assert "redistribute" in note and "export" in note, "and the two conditions on top of it"


def test_a_code_licence_is_not_a_data_licence(manifest: Manifest) -> None:
    """Anti-UAV600's repository says MIT, and means its toolkit.

    The set therefore stays `unstated` and stays refused. This is the same substitution as the
    `lrddv3` case and the opposite outcome, which is why both are asserted: the question is never
    "is a licence written down nearby" but "is this a grant over these frames".
    """
    dataset = manifest.datasets["anti_uav_600"]
    assert dataset.licence == UNSTATED
    assert decisions(plan(manifest, ["anti_uav_600"])) == {"anti_uav_600": "refused"}
    note = (dataset.licence_note or "").lower()
    assert "mit" in note and "project" in note, "the note must say what the MIT licence covers"


def test_the_largest_infrared_set_is_indexed(manifest: Manifest) -> None:
    """It was missing entirely, and it is the biggest one: 600 sequences, over 723k frames."""
    dataset = manifest.datasets["anti_uav_600"]
    assert dataset.clips == 600
    assert dataset.frames is not None and dataset.frames >= 723_000
    assert "infrared only" in dataset.signal_path_note.lower()
    biggest = max((d for d in manifest.datasets.values() if d.frames), key=lambda d: d.frames or 0)
    assert biggest is dataset, "nothing indexed should be larger"


def test_the_stored_rate_and_the_camera_rate_are_different_numbers(manifest: Manifest) -> None:
    """LRDDv3's camera records IR at 30 fps and the set stores 5. Both are true; only one is usable.

    They live in different fields because an analyser must fit against the rate of the file it is
    reading. A one-pole fit run at the camera's 30 against frames stored at 5 returns a time
    constant wrong by six, and looks entirely plausible.
    """
    dataset = manifest.datasets["lrddv3"]
    assert dataset.frame_rate_hz == 5.0
    assert dataset.sensor_frame_rate_hz == 30.0
    assert dataset.frame_rate_hz != dataset.sensor_frame_rate_hz


def test_an_unsourced_frame_rate_is_not_written_into_the_index(manifest: Manifest) -> None:
    """The one field `XD.1` asked for that is deliberately still null.

    640x512 at 25 Hz describes the Anti-UAV **RGBT** parent set; no primary source states it for
    these 410 IR sequences, and the entry's own note says they vary. `decode.probe_clip` reads
    both from the file rather than from here, so a number written in cannot reach a temporal fit
    and cannot earn its way in on usefulness either -- it would only be an unchecked claim in a
    file whose whole job is to separate checked from unchecked.
    """
    dataset = manifest.datasets["anti_uav_410"]
    assert dataset.resolution is None
    assert dataset.frame_rate_hz is None


# --- the signal path is a value, and the index is checked against it (XD.2) ----------------------


def test_a_set_may_not_claim_a_measurement_its_own_path_cannot_carry(manifest: Manifest) -> None:
    """The index refuses to load, rather than the analyser refusing weeks later.

    By the time a run-time refusal fires, somebody has downloaded a set on the strength of a
    permission this file granted and written the claim into a plan. This is the cheap end of the
    same check.
    """
    display = manifest.datasets["anti_uav_410"]
    with pytest.raises(ValidationError, match="rescales every frame on its own content"):
        Dataset.model_validate(
            {
                **display.model_dump(exclude_none=True),
                "analysers": [*display.analysers, "noise_3d"],
                "excluded_analysers": [],
            }
        )

    recorder = manifest.datasets["halmstad_drone_detection"]
    with pytest.raises(ValidationError, match="no DDE in it to measure"):
        Dataset.model_validate(
            {**recorder.model_dump(exclude_none=True), "analysers": ["dde_overshoot"]}
        )


def test_a_measurement_name_that_does_not_exist_is_refused(manifest: Manifest) -> None:
    """A typo in a permission list grants a permission to nothing, or to the wrong thing."""
    recorder = manifest.datasets["halmstad_drone_detection"]
    with pytest.raises(ValidationError, match="unknown measurement"):
        Dataset.model_validate({**recorder.model_dump(exclude_none=True), "analysers": ["noise3d"]})
    with pytest.raises(ValidationError, match="unknown measurement"):
        Dataset.model_validate(
            {**recorder.model_dump(exclude_none=True), "excluded_analysers": ["agc_signatures"]}
        )


def test_nothing_indexed_is_radiometric_yet_and_the_index_says_so(manifest: Manifest) -> None:
    """The state of the shelf, asserted rather than assumed, because it is about to change.

    Every set indexed today is 8-bit and went through a recorder, an ISP, or something nobody
    wrote down. That is exactly why `noise_3d_kelvin` currently has nowhere to run, and it is what
    XD.3, XD.4 and XD.10 are for. When one of them lands this test changes, deliberately -- which
    is the point of pinning it.
    """
    assert manifest.with_signal_path("radiometric") == []
    assert manifest.usable_for("noise_3d_kelvin") == []
    assert {d.signal_path for d in manifest.datasets.values()} == {
        "recorder",
        "display",
        "unknown",
    }


def test_the_paths_recorded_match_what_each_set_says_it_is(manifest: Manifest) -> None:
    """One word per set, and the prose behind it has to agree -- the two are written separately."""
    assert manifest.datasets["halmstad_drone_detection"].signal_path == "recorder"
    assert manifest.with_signal_path("display") == ["anti_uav_410", "anti_uav_600", "cst_anti_uav"]
    assert manifest.with_signal_path("unknown") == ["irstd_1k", "lrddv3", "nuaa_sirst"]
    for name, dataset in manifest.datasets.items():
        if dataset.signal_path == "unknown":
            note = dataset.signal_path_note.lower()
            assert "unknown" in note or "unverified" in note, name


def test_asking_about_a_measurement_that_does_not_exist_is_an_error(manifest: Manifest) -> None:
    """`usable_for` returning [] for a typo would read as "no set supports it", which is a lie."""
    with pytest.raises(KeyError, match="unknown measurement"):
        manifest.usable_for("noise_3d_kelvins")
    with pytest.raises(ValueError, match="unknown signal path"):
        manifest.with_signal_path("y16")  # type: ignore[arg-type]
