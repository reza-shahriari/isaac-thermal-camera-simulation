#!/usr/bin/env python3
"""Fetch a Sketchfab model from a shared link, through the licence gate, into `3d_models/`.

    python scripts/fetch_sketchfab.py https://sketchfab.com/3d-models/<slug>-<uid>
    python scripts/fetch_sketchfab.py <uid> --dry-run
    python scripts/fetch_sketchfab.py <link> --real-name "DJI Mavic 3"

The front door of asset ingestion (roadmap ``AI.8``, ADR 0150). What it does, in order:

1. **Parse the UID** from whatever form of link was shared (page URL, API URL, bare id).
2. **Read the metadata** from the Data API — title, author, licence, downloadability. No token
   is needed for this, so ``--dry-run`` answers "may we use this model?" with no account at all.
3. **Gate the licence** (`irsim.io.sketchfab.gate_license`) *before* downloading. CC0 / CC-BY /
   CC-BY-SA are shareable; everything else — ND, NC, Sketchfab Standard, Editorial, anything
   unrecognised — still downloads but into ``3d_models/quarantine/<name>/``, which must never be
   committed, uploaded or published. The verdict and its reason are printed and recorded.
4. **Download the glTF archive.** The Download API serves glTF/GLB/USDZ only — never the source
   FBX — through S3 links that expire in ~5 minutes, so the fetch happens immediately and only
   the file is kept, never the URL. Requires ``SKETCHFAB_API_TOKEN`` (Settings → Password & API).
   429 responses are retried with backoff; 401/403 get actionable messages, not tracebacks.
5. **Record provenance.** ``ATTRIBUTION.md`` lands beside the files (the Download API guidelines
   require attribution to follow the asset); the full record goes to
   ``configs/assets/<name>.provenance.yaml`` for a shareable model — that file is committed —
   or stays inside the quarantine directory otherwise.

Everything decidable without the network lives in ``irsim.io.sketchfab`` and is unit-tested;
this script owns only argv, HTTP and the filesystem. The whole tree under ``3d_models/`` is
gitignored either way: what gets committed is the provenance YAML and, later, the material map —
the binaries travel by the owner's cloud drive, not by git.

Next step after a successful fetch: the `ingest-asset` skill — author `configs/assets/<name>.yaml`
and run `scripts/prep_asset.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from irsim.io.sketchfab import (  # noqa: E402
    FetchPlan,
    attribution_markdown,
    model_uid_from_url,
    plan_fetch,
    provenance_from_model,
    write_provenance,
)

DATA_API = "https://api.sketchfab.com/v3/models/{uid}"
DOWNLOAD_API = "https://api.sketchfab.com/v3/models/{uid}/download"
USER_AGENT = "irsim-asset-ingest/1.0 (+https://github.com/; scripts/fetch_sketchfab.py)"

#: Retries for a 429, honouring Retry-After when present. One model per invocation is the
#: intended cadence; this is not, and must not become, a bulk scraper.
MAX_ATTEMPTS = 3


def _api_json(url: str, token: str | None, timeout: float) -> dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"
    last_error = "no attempt made"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
                if not isinstance(payload, dict):
                    raise SystemExit(f"unexpected non-object response from {url}")
                return payload
        except urllib.error.HTTPError as error:
            if error.code == 401:
                raise SystemExit(
                    "401 from the Sketchfab API: the token was refused. Set "
                    "SKETCHFAB_API_TOKEN from sketchfab.com Settings → Password & API."
                ) from error
            if error.code == 403:
                raise SystemExit(
                    "403 from the Sketchfab API: this model is not downloadable by this "
                    "account (view-only, or store content that moved to Fab). Pick another."
                ) from error
            if error.code == 404:
                raise SystemExit(f"404: no model with this uid ({url}).") from error
            if error.code == 429 and attempt < MAX_ATTEMPTS:
                wait = float(error.headers.get("Retry-After", "0") or 0) or 5.0 * attempt
                print(f"429 rate-limited; waiting {wait:.0f} s (attempt {attempt})")
                time.sleep(wait)
                last_error = "429 Too Many Requests"
                continue
            raise SystemExit(f"HTTP {error.code} from {url}: {error.reason}") from error
    raise SystemExit(f"gave up after {MAX_ATTEMPTS} attempts: {last_error}")


def _download(url: str, dest: pathlib.Path, timeout: float) -> str:
    """Stream the archive to ``dest`` immediately (the link expires) and return its sha256."""
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=timeout) as response, dest.open("wb") as sink:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            sink.write(chunk)
    return digest.hexdigest()


def _extract_zip(archive: pathlib.Path, target: pathlib.Path) -> list[str]:
    """Extract, refusing members that would land outside ``target`` (zip path traversal)."""
    names: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            resolved = (target / member.filename).resolve()
            if not resolved.is_relative_to(target.resolve()):
                raise SystemExit(f"archive member escapes the target directory: {member.filename}")
        bundle.extractall(target)
        names = [m.filename for m in bundle.infolist() if not m.is_dir()]
    return names


def _print_plan(plan: FetchPlan, model: dict[str, Any]) -> None:
    verdict = "SHAREABLE — goes into the library" if plan.decision.shareable else "QUARANTINE"
    faces = model.get("faceCount", "?")
    print(f"model     : {model.get('name', '?')}  ({plan.uid})")
    print(f"author    : {(model.get('user') or {}).get('username', '?')}")
    print(f"faces     : {faces}")
    print(f"licence   : {plan.decision.label}")
    print(f"gate      : {verdict}")
    print(f"            {plan.decision.reason}")
    print(f"asset name: {plan.asset_name}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url", help="model page URL, API URL, or bare 32-hex uid")
    ap.add_argument(
        "--real-name",
        default="",
        help="what the object is in the world, e.g. 'DJI Mavic 3': names the asset and is the "
        "search key for its published dimensions and part materials",
    )
    ap.add_argument(
        "--dest",
        type=pathlib.Path,
        default=REPO_ROOT / "3d_models",
        help="models root (default 3d_models/, which is gitignored)",
    )
    ap.add_argument(
        "--token-env",
        default="SKETCHFAB_API_TOKEN",
        help="environment variable holding the personal API token",
    )
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="metadata and gate verdict only; needs no token and downloads nothing",
    )
    args = ap.parse_args(argv)

    uid = model_uid_from_url(args.url)
    token = os.environ.get(args.token_env)
    model = _api_json(DATA_API.format(uid=uid), token, args.timeout)
    plan = plan_fetch(model, real_name=args.real_name)
    _print_plan(plan, model)

    for problem in plan.problems:
        print(f"problem   : {problem}")
    if plan.problems:
        return 2
    if args.dry_run:
        return 0
    if not token:
        print(
            f"no token: set {args.token_env} (sketchfab.com Settings → Password & API). "
            "Metadata worked without it; the download will not."
        )
        return 2

    download = _api_json(DOWNLOAD_API.format(uid=uid), token, args.timeout)
    gltf = download.get("gltf")
    if not isinstance(gltf, dict) or "url" not in gltf:
        offered = ", ".join(sorted(download)) or "nothing"
        print(f"no glTF archive offered for this model (offered: {offered}); cannot proceed.")
        return 2

    asset_dir = args.dest / plan.dest_subdir
    archive = asset_dir / f"{plan.asset_name}.gltf.zip"
    sha256 = _download(str(gltf["url"]), archive, args.timeout)
    extracted = _extract_zip(archive, asset_dir / "gltf")
    scene_files = [n for n in extracted if n.endswith((".gltf", ".glb"))]

    provenance = provenance_from_model(
        model,
        asset_name=plan.asset_name,
        decision=plan.decision,
        downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        archive_sha256=sha256,
        real_name=args.real_name.strip(),
    )
    (asset_dir / "ATTRIBUTION.md").write_text(attribution_markdown(provenance))
    if plan.decision.shareable:
        record = REPO_ROOT / "configs" / "assets" / f"{plan.asset_name}.provenance.yaml"
    else:
        record = asset_dir / "provenance.yaml"
        (asset_dir / "QUARANTINED.md").write_text(
            "# Quarantined\n\n"
            f"{plan.decision.reason}.\n\n"
            "Local use only: never commit, upload, publish or bundle anything in this "
            "directory. The provenance record stays here rather than in configs/assets/ so "
            "that nothing shareable ever points at it.\n"
        )
    write_provenance(provenance, record)

    print(f"archive   : {archive}  (sha256 {sha256[:12]}…, {len(extracted)} files)")
    print(f"scene     : {', '.join(scene_files) or 'NO .gltf/.glb IN ARCHIVE — inspect by hand'}")
    print(f"provenance: {record}")
    print(f"next      : author configs/assets/{plan.asset_name}.yaml and run scripts/prep_asset.py")
    print("            (the ingest-asset skill is the checklist)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
