"""Sketchfab arrival logic: the model UID, the licence gate, the provenance record.

A shared link is the front door of the asset pipeline (roadmap ``AI.8``, ADR 0150). Two facts
about that door decide this module's shape:

* **The Download API serves a converted glTF archive — never the source FBX.** "Models are not
  currently available in their source formats such as FBX and OBJ through the API." A
  well-formatted FBX is therefore something ``prep_asset.py`` *exports* at the end of the
  pipeline, not something this step downloads.
* **A cleaned asset is a derivative, and the licence decides whether it may leave the machine.**
  The pipeline rescales, regroups and renames the model, so CC-BY-ND ("unchanged and in whole")
  and the Sketchfab Standard licence (no redistribution as a stand-alone file) both forbid
  committing or publishing the result, even though the download itself succeeds. The gate is
  therefore decided **before** anything is fetched, and it fails towards quarantine: a licence
  this module does not recognise is never shareable.

Everything here is engine-free and network-free: functions take the URL string or the JSON
``dict`` the Sketchfab Data API returned, and ``scripts/fetch_sketchfab.py`` owns the HTTP. That
keeps the gate and the provenance schema testable in the plain unit suite, which matters because
a wrong gate is a legal defect that no rendered image would ever reveal.

The provenance record travels with the asset for the same reason the per-asset material map
carries ``ESTIMATED`` flags: a library entry must answer "where is this from, who made it, and
what are we allowed to do with it" without anyone re-opening the model page. The Download API
guidelines require the attribution to follow the asset wherever it is used;
:func:`attribution_markdown` renders exactly that text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

__all__ = [
    "SHAREABLE_LICENSE_SLUGS",
    "KNOWN_LICENSE_SLUGS",
    "GateDecision",
    "FetchPlan",
    "LicenseRecord",
    "Provenance",
    "asset_name_from_title",
    "attribution_markdown",
    "gate_license",
    "load_provenance",
    "model_uid_from_url",
    "plan_fetch",
    "provenance_from_local",
    "provenance_from_model",
    "write_provenance",
]

#: A Sketchfab model UID: 32 hex characters, the trailing run of every model page URL
#: (``…/3d-models/<slug>-<uid>``) and the key of every Data API call.
_UID_RE = re.compile(r"[0-9a-f]{32}")

#: Licences under which a *modified* model may be redistributed, given attribution. This is the
#: whole shareable set on purpose: ND forbids the derivative, NC forbids this project's use, and
#: the Sketchfab Standard/Editorial licences forbid stand-alone redistribution outright.
SHAREABLE_LICENSE_SLUGS: frozenset[str] = frozenset({"cc0", "by", "by-sa"})

#: Every licence slug the Data API is known to return. A slug outside this set is not an error —
#: it is a licence added after this file was written, and it quarantines like ND does.
KNOWN_LICENSE_SLUGS: frozenset[str] = frozenset(
    {"cc0", "by", "by-sa", "by-nd", "by-nc", "by-nc-sa", "by-nc-nd", "st", "ed"}
)

_QUARANTINE_REASONS: dict[str, str] = {
    "by-nd": (
        "CC-BY-ND allows redistribution only 'unchanged and in whole'; a rescaled, regrouped, "
        "renamed asset is a derivative, so the cleaned copy may not leave this machine"
    ),
    "by-nc": "CC-BY-NC forbids commercial use, which this project cannot promise",
    "by-nc-sa": "CC-BY-NC-SA forbids commercial use, which this project cannot promise",
    "by-nc-nd": "CC-BY-NC-ND forbids both commercial use and derivatives",
    "st": (
        "the Sketchfab Standard licence forbids making the model available 'as a stand-alone "
        "file', so it may never be committed or published, only used locally"
    ),
    "ed": "the Sketchfab Editorial licence forbids redistribution and non-editorial use",
}

#: Fallback for a response that carries a ``label`` but no ``slug``. Keys are lower-cased labels
#: as the site prints them; anything not matched stays unknown and therefore quarantined.
_LABEL_TO_SLUG: dict[str, str] = {
    "cc0 public domain": "cc0",
    "cc attribution": "by",
    "cc attribution-sharealike": "by-sa",
    "cc attribution-noderivs": "by-nd",
    "cc attribution-noncommercial": "by-nc",
    "cc attribution-noncommercial-sharealike": "by-nc-sa",
    "cc attribution-noncommercial-noderivs": "by-nc-nd",
    "standard": "st",
    "editorial": "ed",
}


def model_uid_from_url(url: str) -> str:
    """The 32-hex model UID from a model page URL, an API URL, or a bare UID.

    The slug part of a page URL is free text chosen by the author, so the UID is found by
    pattern, not by splitting on ``-``: the *last* 32-hex run in the URL path is the UID
    (``…/chunky-knight-f1722ab650ad4d8dbe6fc4bf44e33d38``). Query string and fragment are
    ignored. Raises ``ValueError`` when no UID is present, e.g. for a collection or user URL.
    """
    bare = url.strip().lower()
    if _UID_RE.fullmatch(bare):
        return bare
    path = bare.split("?", 1)[0].split("#", 1)[0].rstrip("/")
    found: list[str] = _UID_RE.findall(path)
    if not found:
        raise ValueError(
            f"no Sketchfab model UID in {url!r}: expected a model page URL ending in the "
            "32-character id, or the bare id itself"
        )
    return found[-1]


@dataclass(frozen=True)
class GateDecision:
    """The licence gate's verdict on one model, with the sentence that justifies it."""

    shareable: bool
    slug: str
    label: str
    reason: str


def gate_license(license_info: Mapping[str, Any]) -> GateDecision:
    """Decide whether a cleaned derivative of this model may be committed and published.

    ``license_info`` is the ``license`` object of a Data API model response. The decision keys
    on the slug, falls back to the label, and **fails towards quarantine**: a missing or
    unrecognised licence is never shareable, because the failure mode of guessing generous is a
    published file this project has no right to publish.
    """
    slug = str(license_info.get("slug", "") or "").lower()
    label = str(license_info.get("label", "") or "").strip()
    if not slug:
        slug = _LABEL_TO_SLUG.get(label.lower(), "")
    if not label:
        label = slug or "unknown"
    if slug in SHAREABLE_LICENSE_SLUGS:
        return GateDecision(
            shareable=True,
            slug=slug,
            label=label,
            reason=f"{label}: a modified copy may be redistributed with attribution",
        )
    reason = _QUARANTINE_REASONS.get(
        slug,
        f"licence {label or slug or 'unknown'!r} is not one this gate recognises; "
        "quarantined until a human reads its terms",
    )
    return GateDecision(shareable=False, slug=slug or "unknown", label=label, reason=reason)


def asset_name_from_title(title: str, uid: str) -> str:
    """A library-safe asset name from the model title: ``^[a-z_][a-z0-9_]*$``, at most 48 chars.

    The name becomes a ``configs/assets/`` key, a directory under the models root and a USD
    prim-path component, so it obeys the strictest of those three grammars (USD identifiers).
    An empty or all-symbol title falls back to ``m_<first 8 of the uid>``.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:48].rstrip("_")
    if not slug:
        return f"m_{uid[:8]}"
    if slug[0].isdigit():
        slug = f"m_{slug}"
    return slug


@dataclass(frozen=True)
class FetchPlan:
    """What one model's download would do, decided entirely from Data API metadata.

    ``dest_subdir`` is relative to the models root (``3d_models/``): shareable models land in
    ``<asset_name>/``, everything else in ``quarantine/<asset_name>/`` — a directory whose name
    states the obligation, inside a tree that is gitignored either way.
    """

    uid: str
    asset_name: str
    downloadable: bool
    decision: GateDecision
    dest_subdir: str
    problems: tuple[str, ...]


def plan_fetch(model: Mapping[str, Any], real_name: str = "") -> FetchPlan:
    """Turn a Data API model response into a :class:`FetchPlan`, before any download happens.

    The asset is named after ``real_name`` when the owner gave one, since that is what the object
    is; the page title ("free low poly drone") is the fallback. Problems that make the fetch
    pointless are collected rather than raised, so a driver can print all of them at once: a
    model that is not downloadable (view-only), or one with no glTF among its archives when the
    response says so up front.
    """
    uid = str(model.get("uid", ""))
    title = str(model.get("name", "") or "")
    name = asset_name_from_title(real_name.strip() or title, uid)
    decision = gate_license(model.get("license", {}) or {})
    downloadable = bool(model.get("isDownloadable", False))
    problems: list[str] = []
    if not uid:
        problems.append("response carries no model uid")
    if not downloadable:
        problems.append(
            "the author did not mark this model downloadable; the API will refuse it — "
            "pick another model, do not scrape the site"
        )
    subdir = name if decision.shareable else f"quarantine/{name}"
    return FetchPlan(
        uid=uid,
        asset_name=name,
        downloadable=downloadable,
        decision=decision,
        dest_subdir=subdir,
        problems=tuple(problems),
    )


class LicenseRecord(BaseModel):
    """The licence as the Data API stated it at download time."""

    slug: str
    label: str
    url: str = ""


#: What the pipeline does to every asset — recorded because CC attribution must indicate changes.
MODIFICATIONS_NOTE = (
    "Rescaled to metres, regrouped into functional parts, renamed, and re-exported to USD/FBX; "
    "materials remapped onto an infrared material library. A modified derivative, not the "
    "original."
)


class Provenance(BaseModel):
    """Where an asset came from and what may be done with it (``configs/assets/*.provenance.yaml``).

    One record per ingested asset, written at arrival time by ``scripts/fetch_sketchfab.py`` (a
    shared link, ``source: sketchfab``) or ``scripts/register_local_asset.py`` (a file the owner
    handed over, ``source: local``). For a shareable asset it is committed beside the material
    map; for a quarantined one it stays with the quarantined files. ``downloaded_at`` and
    ``archive_sha256`` pin *which* upload this was — a Sketchfab author can re-upload under the
    same UID.

    ``real_name`` is what the object *is* in the world ("DJI Mavic 3", "Cessna 172"), as the
    owner stated it. It outranks the page title — often "low poly drone" or a free-text joke — as
    the search key for published dimensions and part materials, and it is recorded so the next
    reader of the material map knows which product its ``ESTIMATED`` lines were researched
    against. Empty when nobody said.
    """

    schema_version: int = 1
    asset_name: str
    source: str = "sketchfab"
    real_name: str = ""
    uid: str = ""
    title: str
    author: str
    author_url: str = ""
    model_url: str = ""
    license: LicenseRecord
    shareable: bool
    gate_reason: str
    source_format: str = "gltf"
    archive_sha256: str = ""
    downloaded_at: str = ""
    modifications: str = MODIFICATIONS_NOTE


def provenance_from_model(
    model: Mapping[str, Any],
    *,
    asset_name: str,
    decision: GateDecision,
    downloaded_at: str = "",
    archive_sha256: str = "",
    source_format: str = "gltf",
    real_name: str = "",
) -> Provenance:
    """Build the :class:`Provenance` record from the Data API response and the gate's verdict."""
    uid = str(model.get("uid", ""))
    user_raw = model.get("user", {}) or {}
    user: Mapping[str, Any] = user_raw if isinstance(user_raw, Mapping) else {}
    license_raw = model.get("license", {}) or {}
    license_info: Mapping[str, Any] = license_raw if isinstance(license_raw, Mapping) else {}
    return Provenance(
        asset_name=asset_name,
        source="sketchfab",
        real_name=real_name,
        uid=uid,
        title=str(model.get("name", "") or ""),
        author=str(user.get("username", "") or "unknown"),
        author_url=str(user.get("profileUrl", "") or ""),
        model_url=str(model.get("viewerUrl", "") or f"https://sketchfab.com/3d-models/{uid}"),
        license=LicenseRecord(
            slug=decision.slug,
            label=decision.label,
            url=str(license_info.get("url", "") or ""),
        ),
        shareable=decision.shareable,
        gate_reason=decision.reason,
        source_format=source_format,
        archive_sha256=archive_sha256,
        downloaded_at=downloaded_at,
    )


def provenance_from_local(
    *,
    asset_name: str,
    file_name: str,
    decision: GateDecision,
    real_name: str = "",
    author: str = "",
    origin_url: str = "",
    archive_sha256: str = "",
    registered_at: str = "",
) -> Provenance:
    """The :class:`Provenance` record for a file the owner handed over rather than a link.

    A local file carries no licence metadata, so ``decision`` comes from :func:`gate_license`
    on whatever licence the owner *stated* — and with none stated that is the unknown-licence
    branch, which quarantines. The gate does not get a special case for local files: "somebody
    gave me this file" is exactly the situation in which nobody knows the terms.
    """
    return Provenance(
        asset_name=asset_name,
        source="local",
        real_name=real_name,
        title=real_name or file_name,
        author=author or "unknown",
        model_url=origin_url,
        license=LicenseRecord(slug=decision.slug, label=decision.label),
        shareable=decision.shareable,
        gate_reason=decision.reason,
        source_format=file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "unknown",
        archive_sha256=archive_sha256,
        downloaded_at=registered_at,
    )


def attribution_markdown(provenance: Provenance) -> str:
    """The attribution text the Download API guidelines require to follow the asset.

    Creator username with a link to their profile, model title with a link to the model,
    licence name with a link to its terms, and an indication of changes — the four things
    CC-BY obliges and CC0 deserves anyway.
    """
    author = (
        f"[{provenance.author}]({provenance.author_url})"
        if provenance.author_url
        else provenance.author
    )
    license_text = (
        f"[{provenance.license.label}]({provenance.license.url})"
        if provenance.license.url
        else provenance.license.label
    )
    where = f" ({provenance.model_url})" if provenance.model_url else ""
    via = "Provided via Sketchfab" if provenance.source == "sketchfab" else "Supplied as a file"
    lines = [
        "# Attribution",
        "",
        f'"{provenance.title}"{where} by {author}, licensed under {license_text}.',
        "",
        f"{via}. Received {provenance.downloaded_at or 'date not recorded'}; "
        f"archive sha256 {provenance.archive_sha256 or 'not recorded'}.",
        "",
        f"Changes: {provenance.modifications}",
        "",
    ]
    if provenance.real_name and provenance.real_name != provenance.title:
        lines[3:3] = [f"Depicts: {provenance.real_name}.", ""]
    return "\n".join(lines)


def write_provenance(provenance: Provenance, path: Path) -> None:
    """Write the record as YAML, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"provenance": provenance.model_dump()}
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


def load_provenance(path: Path) -> Provenance:
    """Read a record written by :func:`write_provenance`."""
    payload = yaml.safe_load(path.read_text())
    return Provenance.model_validate(payload["provenance"])
