#!/usr/bin/env python3
"""Register a 3D file the owner handed over — the second front door beside a Sketchfab link.

    python scripts/register_local_asset.py ~/Downloads/mavic.fbx --real-name "DJI Mavic 3"
    python scripts/register_local_asset.py drone.zip --real-name "DJI Mavic 3" \
        --license by --author "some_artist" --origin-url https://example.com/model

What it does (roadmap ``AI.9``, ADR 0150):

1. **Names the asset** from ``--real-name`` — what the object is in the world — falling back to
   the file name. The real name is also the search key the `ingest-asset` skill uses for the
   object's published dimensions and part materials.
2. **Gates the licence you state.** A file carries no licence metadata, so ``--license`` takes
   the same slugs a Sketchfab page does (``cc0``, ``by``, ``by-sa``, ``by-nd``, ``by-nc``, …).
   Your own work is ``cc0`` or ``by``. **With no ``--license`` the file is quarantined** — usable
   on this machine, never committed or published — because "somebody gave me this file" is the
   case where nobody knows the terms. The gate is `irsim.io.sketchfab.gate_license`, unchanged.
3. **Copies the file** into ``3d_models/<name>/source/`` (or ``3d_models/quarantine/<name>/``),
   extracting a ``.zip``, and pins its sha256.
4. **Records provenance** exactly as a fetch does: ``ATTRIBUTION.md`` beside the files, and
   ``configs/assets/<name>.provenance.yaml`` (committed) for a shareable asset.

Accepted formats are what ``prep_asset.py`` imports: .fbx .obj .gltf .glb .usd .usda .usdc, or a
.zip holding one of them. A .blend is refused: open it in Blender and export glTF first, so every
asset enters the pipeline through an importer that preserves material names.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import sys
import zipfile
from collections.abc import Sequence
from datetime import datetime, timezone

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from irsim.io.sketchfab import (  # noqa: E402
    asset_name_from_title,
    attribution_markdown,
    gate_license,
    provenance_from_local,
    write_provenance,
)

MODEL_SUFFIXES = {".fbx", ".obj", ".gltf", ".glb", ".usd", ".usda", ".usdc"}


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_zip(archive: pathlib.Path, target: pathlib.Path) -> list[pathlib.Path]:
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if not (target / member.filename).resolve().is_relative_to(target.resolve()):
                raise SystemExit(f"archive member escapes the target directory: {member.filename}")
        bundle.extractall(target)
    return [p for p in target.rglob("*") if p.suffix.lower() in MODEL_SUFFIXES]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", type=pathlib.Path, help="the model file, or a .zip holding it")
    ap.add_argument("--real-name", default="", help="what the object is, e.g. 'DJI Mavic 3'")
    ap.add_argument(
        "--license", default="", help="licence slug you know applies (none = quarantine)"
    )
    ap.add_argument("--author", default="", help="who made the model, for attribution")
    ap.add_argument("--origin-url", default="", help="where it came from, if anywhere")
    ap.add_argument("--dest", type=pathlib.Path, default=REPO_ROOT / "3d_models")
    args = ap.parse_args(argv)

    src = args.file.expanduser().resolve()
    if not src.is_file():
        ap.error(f"no such file: {src}")
    suffix = src.suffix.lower()
    if suffix == ".blend":
        ap.error("a .blend is not accepted: open it in Blender and export glTF (.glb) first")
    if suffix not in MODEL_SUFFIXES | {".zip"}:
        ap.error(f"unsupported {suffix!r}; accepted: {sorted(MODEL_SUFFIXES | {'.zip'})}")

    sha256 = _sha256(src)
    name = asset_name_from_title(args.real_name.strip() or src.stem, sha256)
    decision = gate_license({"slug": args.license} if args.license else {})
    asset_dir = args.dest / (name if decision.shareable else f"quarantine/{name}")
    source_dir = asset_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    if suffix == ".zip":
        models = _extract_zip(src, source_dir)
    else:
        shutil.copy2(src, source_dir / src.name)
        models = [source_dir / src.name]

    provenance = provenance_from_local(
        asset_name=name,
        file_name=src.name,
        decision=decision,
        real_name=args.real_name.strip(),
        author=args.author,
        origin_url=args.origin_url,
        archive_sha256=sha256,
        registered_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    (asset_dir / "ATTRIBUTION.md").write_text(attribution_markdown(provenance))
    if decision.shareable:
        record = REPO_ROOT / "configs" / "assets" / f"{name}.provenance.yaml"
    else:
        record = asset_dir / "provenance.yaml"
        (asset_dir / "QUARANTINED.md").write_text(
            f"# Quarantined\n\n{decision.reason}.\n\n"
            "Local use only: never commit, upload, publish or bundle anything in this directory. "
            "Re-register with --license <slug> once the terms are known.\n"
        )
    write_provenance(provenance, record)

    print(f"asset name: {name}")
    print(f"real name : {args.real_name.strip() or '(not given — the skill will ask or infer)'}")
    print(f"gate      : {'SHAREABLE' if decision.shareable else 'QUARANTINE'} — {decision.reason}")
    print(f"models    : {', '.join(str(m) for m in models) or 'NONE FOUND'}")
    print(f"provenance: {record}")
    return 0 if models else 2


if __name__ == "__main__":
    raise SystemExit(main())
