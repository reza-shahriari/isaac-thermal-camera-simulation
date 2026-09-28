"""The irsim side of the Blender add-on. Runs under the **project** interpreter, never in Blender.

Blender's Python has neither pydantic nor PyYAML, and the add-on must not carry a second copy of
the material schema, of Kirchhoff closure (CLAUDE.md #4) or of the asset-map rules: a copy is a
rule that drifts from the one the solver enforces. So every question the add-on asks about the
library, and every file it writes into the repository, goes through this script, which imports
``irsim`` from the repository it is pointed at and uses the project's own loaders:

* ``library``        -- every material, with ε/ρ/τ per band as ``Material.band_properties``
                        derives them (so a spectral material shows its band-integrated values).
* ``check-material`` -- validate a draft exactly as the loader will, and return its derived bands.
* ``write-material`` -- the same, then write ``configs/materials/<name>.yaml`` and reload the
                        whole library; a file that breaks the library is removed again.
* ``list-assets``    -- the asset configs there are, and which ones this add-on wrote.
* ``read-asset``     -- one asset config's map, loaded and checked by the project loader.
* ``write-asset``    -- validate and write ``configs/assets/<name>.yaml``
                        (``irsim.materials.mapping.AssetConfig``), then reload it with every target
                        checked against the library.

Protocol, so a banner on stdout can never be mistaken for a result::

    <python> bridge.py <command> --repo <root> --out <result.json> [--in <payload.json>]

The result is a JSON object with ``"ok": true`` or ``"ok": false`` plus ``"error"``. Exit status is
0 whenever a result was written, including a refusal; non-zero means the bridge itself failed.

This file lives inside the add-on so the two always ship together, but nothing here imports
``bpy`` and nothing in the add-on imports this module: Blender only runs it as a subprocess.
"""

import argparse
import datetime
import json
import pathlib
import re
import sys
import tempfile
import traceback
from typing import Any

#: The add-on's display order. The library may declare more; these are the ones the UI shows.
BAND_ORDER = ("nir", "swir", "mwir", "lwir")

#: First line of every file this script writes. `write-asset` only overwrites a file that carries
#: it, so a hand-written asset config -- whose comments are the only record of how its map was
#: decided -- can never be replaced by an export.
MARKER = "irsim Blender add-on"


class RefusalError(Exception):
    """A request the bridge declines, with a message meant for the person in Blender."""

    def __init__(self, message: str, kind: str = "refused") -> None:
        super().__init__(message)
        self.kind = kind


def _import_irsim(repo: pathlib.Path) -> None:
    """Import ``irsim`` from *this* repository, not from wherever the interpreter installed one.

    ``irsim.materials.library.MATERIAL_DIR`` is computed from the package's own location, so the
    package and the configs must come from the same checkout or a new material would be checked
    against one library and written into another.
    """
    src = repo / "src"
    if not (src / "irsim").is_dir():
        raise RefusalError(f"{repo} does not look like the irsim repository (no src/irsim)")
    sys.path.insert(0, str(src))


def _pydantic_message(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return str(exc)
    lines = []
    for err in errors():
        where = ".".join(str(part) for part in err.get("loc", ()))
        lines.append(f"{where}: {err.get('msg', '')}" if where else str(err.get("msg", "")))
    return "\n".join(lines) or str(exc)


def _bands_of(material: Any, band_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for band in band_ids:
        try:
            p = material.band_properties(band)
        except (KeyError, ValueError) as exc:
            out[band] = {"error": str(exc)}
            continue
        out[band] = {
            "emissivity": p.emissivity,
            "reflectance": p.reflectance,
            "transmittance": p.transmittance,
            "authored": p.authored,
        }
    return out


def _band_ids() -> tuple[str, ...]:
    from irsim.config.bands import BAND_IDS

    ordered = [b for b in BAND_ORDER if b in BAND_IDS]
    return (*ordered, *(b for b in BAND_IDS if b not in ordered))


def _material_record(material: Any, repo: pathlib.Path, band_ids: tuple[str, ...]) -> dict:
    spec = material.spec
    thermal = spec.thermal
    optical = spec.optical
    angular = optical.angular_model.model_dump(mode="json")
    try:
        rel = str(material.path.relative_to(repo))
    except ValueError:
        rel = str(material.path)
    return {
        "name": spec.name,
        "description": spec.description,
        "reference": spec.reference,
        "source": spec.source,
        "surface_treatment": spec.surface_treatment,
        "file": rel,
        "thermal": {
            "density_kg_m3": thermal.density_kg_m3,
            "specific_heat_j_kgk": thermal.specific_heat_j_kgk,
            "conductivity_w_mk": thermal.conductivity_w_mk,
            "thickness_m": thermal.thickness_m,
            "solar_absorptivity": thermal.solar_absorptivity,
            "heat_capacity_j_m2_k": thermal.heat_capacity_j_m2_k,
        },
        "optical": {
            "authored": optical.authored,
            "spectral": optical.spectral_file is not None,
            "angular": angular,
            "roughness": dict(optical.roughness_per_band or {}),
        },
        "bands": _bands_of(material, band_ids),
    }


def _material_dir(args: argparse.Namespace, repo: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(args.material_dir) if args.material_dir else repo / "configs" / "materials"


def cmd_library(args: argparse.Namespace, repo: pathlib.Path, payload: Any) -> dict:
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MIRROR_EMISSIVITY

    material_dir = _material_dir(args, repo)
    library = MaterialLibrary.load(material_dir)
    band_ids = _band_ids()
    return {
        "ok": True,
        "bands": list(band_ids),
        "mirror_emissivity": MIRROR_EMISSIVITY,
        "library_hash": library.content_hash(),
        "material_dir": str(material_dir),
        "materials": [_material_record(library[n], repo, band_ids) for n in sorted(library)],
    }


def _validated_material(
    draft: dict, material_dir: pathlib.Path, repo: pathlib.Path
) -> tuple[dict, Any, list[str]]:
    """``(document, loaded Material, warnings)`` for a draft, or a :class:`RefusalError`.

    The draft is validated by ``MaterialConfig`` and then loaded from a scratch file by
    ``load_material``, then asked for every band -- the same three gates a library file passes,
    so a material this accepts is one the committed library walk
    (``tests/unit/test_materials_library.py``) will accept too.
    """
    from pydantic import ValidationError

    from irsim.config.materials import MATERIAL_SCHEMA_VERSION, MaterialConfig
    from irsim.materials.library import MaterialLibrary, load_material
    from irsim.materials.mapping import MIRROR_EMISSIVITY

    if not isinstance(draft, dict) or "material" not in draft:
        raise RefusalError("the payload needs a 'material' block")
    doc = {"schema_version": MATERIAL_SCHEMA_VERSION, "material": draft["material"]}
    try:
        spec = MaterialConfig.model_validate(doc).material
    except ValidationError as exc:
        raise RefusalError(_pydantic_message(exc), kind="invalid") from exc

    existing = MaterialLibrary.load(material_dir)
    if spec.name in existing:
        raise RefusalError(
            f"a material named {spec.name!r} already exists ({existing[spec.name].path.name}). "
            "Choose another name, or edit that file by hand if it is the same material.",
            kind="exists",
        )

    band_ids = _band_ids()
    with tempfile.TemporaryDirectory(prefix="irsim_material_") as tmp:
        path = pathlib.Path(tmp) / f"{spec.name}.yaml"
        path.write_text(_material_yaml(doc, spec.source), encoding="utf-8")
        try:
            material = load_material(path)
        except (ValueError, FileNotFoundError) as exc:
            raise RefusalError(str(exc), kind="invalid") from exc
        bands = _bands_of(material, band_ids)
    failed = {b: v["error"] for b, v in bands.items() if "error" in v}
    if failed:
        raise RefusalError(
            "every standard band must close (ε + ρ + τ = 1): "
            + "; ".join(f"{b}: {e}" for b, e in failed.items()),
            kind="invalid",
        )

    warnings = []
    lwir = bands.get("lwir", {}).get("emissivity")
    if lwir is not None and lwir < MIRROR_EMISSIVITY:
        warnings.append(
            f"LWIR emissivity {lwir:.2f} is below {MIRROR_EMISSIVITY}: a surface like this shows "
            "mostly reflected surroundings, not its own temperature. Right for polished metal; "
            "wrong for anything painted, anodised or oxidised."
        )
    if spec.source != "measured" and not spec.reference.strip():
        warnings.append(
            "No reference given. Say where the numbers came from, even if it is 'estimated from "
            "similar materials', so the next person can tell a guess from a measurement."
        )
    return doc, material, warnings


def _material_yaml(doc: dict, source: str) -> str:
    import yaml

    today = datetime.date.today().isoformat()
    description = str(doc["material"].get("description", "")).strip()
    header = [f"# {description}" if description else f"# {doc['material']['name']}"]
    header.append(f"# Created with the {MARKER} (blender_addon/irsim_thermal) on {today}.")
    if source == "estimated":
        header.append("# source: estimated -- values entered by hand; replace them with a citation")
        header.append("# or a measurement when one exists.")
    header.append(
        "# Schema: irsim.config.materials. Emissivity and transmittance are authored per band; "
        "reflectance"
    )
    header.append("# is derived as 1 - emissivity - transmittance (CLAUDE.md #4).")
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return "\n".join(header) + "\n" + body


def cmd_check_material(args: argparse.Namespace, repo: pathlib.Path, payload: Any) -> dict:
    _, material, warnings = _validated_material(payload, _material_dir(args, repo), repo)
    return {
        "ok": True,
        "name": material.name,
        "bands": _bands_of(material, _band_ids()),
        "warnings": warnings,
    }


def cmd_write_material(args: argparse.Namespace, repo: pathlib.Path, payload: Any) -> dict:
    from irsim.materials.library import MaterialLibrary

    material_dir = _material_dir(args, repo)
    doc, material, warnings = _validated_material(payload, material_dir, repo)
    target = material_dir / f"{material.name}.yaml"
    if target.exists():
        raise RefusalError(f"{target} already exists", kind="exists")
    target.write_text(_material_yaml(doc, material.spec.source), encoding="utf-8")
    try:
        library = MaterialLibrary.load(material_dir)
        written = library[material.name]
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise RefusalError(
            f"the library no longer loads with this file, so it was removed: {exc}"
        ) from exc
    return {
        "ok": True,
        "name": material.name,
        "file": str(target),
        "bands": _bands_of(written, _band_ids()),
        "warnings": warnings,
    }


def _asset_yaml(doc: dict, parts: list[str], notes: list[str]) -> str:
    import yaml

    asset = doc["asset"]
    today = datetime.date.today().isoformat()
    header = [
        f"# {asset['name']}: exported from Blender by the {MARKER} (blender_addon/irsim_thermal)",
        f"# on {today}. Schema: irsim.materials.mapping.AssetConfig.",
        "#",
        "# `materials:` maps each Blender material, by the name it carries in the exported USD, to",
        "# a library material in configs/materials/. The names were made USD-safe before export,",
        "# so they read the same in the .blend, in the USD and here.",
    ]
    if parts:
        header.append("#")
        header.append(f"# Parts (one USD prim per Blender object, {len(parts)}):")
        line = "#  "
        for name in parts:
            if len(line) + len(name) + 2 > 100:
                header.append(line.rstrip(","))
                line = "#  "
            line += f" {name},"
        header.append(line.rstrip(","))
    for note in notes:
        header.append(f"# {note}")
    header.append(
        "# Re-exporting from the add-on replaces this file; hand edits belong in a copy under a "
        "new name."
    )
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return "\n".join(header) + "\n" + body


def _assets_dir(args: argparse.Namespace, repo: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(args.assets_dir) if args.assets_dir else repo / "configs" / "assets"


def _by_addon(path: pathlib.Path) -> bool:
    first = path.read_text(encoding="utf-8").splitlines()[:1]
    return bool(first) and MARKER in first[0]


def cmd_list_assets(args: argparse.Namespace, repo: pathlib.Path, payload: Any) -> dict:
    assets_dir = _assets_dir(args, repo)
    paths = sorted(assets_dir.glob("*.yaml")) if assets_dir.is_dir() else []
    return {
        "ok": True,
        "assets": [{"name": p.stem, "by_addon": _by_addon(p)} for p in paths],
    }


def cmd_read_asset(args: argparse.Namespace, repo: pathlib.Path, payload: Any) -> dict:
    """An asset config's map, loaded by the project loader with every target checked.

    This is how an asset is reopened: a hand-written map (the Phantom 4's, whose comments record
    why each line is what it is) is applied to the Blender materials of the same names, so the
    add-on starts from the decisions already made instead of from nothing.
    """
    from pydantic import ValidationError

    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import load_asset_mapping

    name = str((payload or {}).get("name", ""))
    path = _assets_dir(args, repo) / f"{name}.yaml"
    if not re.fullmatch(r"[A-Za-z0-9_]+", name) or not path.is_file():
        raise RefusalError(f"there is no asset config named {name!r}", kind="missing")
    library = MaterialLibrary.load(_material_dir(args, repo))
    try:
        asset = load_asset_mapping(path, known_materials=library.names)
    except ValidationError as exc:
        raise RefusalError(_pydantic_message(exc), kind="invalid") from exc
    except ValueError as exc:
        raise RefusalError(str(exc), kind="invalid") from exc
    return {
        "ok": True,
        "name": asset.name,
        "file": str(path),
        "source_file": asset.source_file,
        "scale_to_metres": asset.scale_to_metres,
        "materials": dict(asset.materials),
        "by_addon": _by_addon(path),
    }


def cmd_write_asset(args: argparse.Namespace, repo: pathlib.Path, payload: Any) -> dict:
    from pydantic import ValidationError

    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import ASSET_SCHEMA_VERSION, AssetConfig, load_asset_mapping

    name = str(payload.get("name", ""))
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise RefusalError(
            f"asset name {name!r}: use lower-case letters, digits and underscores, as scene "
            "configs refer to it by that name",
            kind="invalid",
        )
    asset = {
        "name": name,
        "source_file": payload.get("source_file"),
        "scale_to_metres": float(payload.get("scale_to_metres", 1.0)),
        "materials": dict(payload.get("materials", {})),
    }
    doc = {"schema_version": ASSET_SCHEMA_VERSION, "asset": asset}
    try:
        AssetConfig.model_validate(doc)
    except ValidationError as exc:
        raise RefusalError(_pydantic_message(exc), kind="invalid") from exc
    library = MaterialLibrary.load(_material_dir(args, repo))
    unknown = sorted(set(asset["materials"].values()) - set(library.names))
    if unknown:
        raise RefusalError(f"these are not library materials: {unknown}", kind="invalid")

    assets_dir = _assets_dir(args, repo)
    target = assets_dir / f"{name}.yaml"
    if target.exists():
        if not _by_addon(target):
            raise RefusalError(
                f"{target.name} exists and was written by hand; its comments are the record of "
                "how its map was decided, so an export will not replace it. Export under another "
                "name.",
                kind="handwritten",
            )
        if not payload.get("overwrite"):
            raise RefusalError(f"{target.name} exists (from an earlier export)", kind="exists")
    assets_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(
        _asset_yaml(doc, list(payload.get("parts", [])), list(payload.get("notes", []))),
        encoding="utf-8",
    )
    reloaded = load_asset_mapping(target, known_materials=library.names)
    return {"ok": True, "file": str(target), "materials": dict(reloaded.materials)}


COMMANDS = {
    "library": cmd_library,
    "check-material": cmd_check_material,
    "write-material": cmd_write_material,
    "list-assets": cmd_list_assets,
    "read-asset": cmd_read_asset,
    "write-asset": cmd_write_asset,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--repo", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--in", dest="payload", type=pathlib.Path, default=None)
    ap.add_argument("--material-dir", default=None, help="default: <repo>/configs/materials")
    ap.add_argument("--assets-dir", default=None, help="default: <repo>/configs/assets")
    args = ap.parse_args(argv)

    repo = args.repo.expanduser().resolve()
    payload = None
    if args.payload is not None:
        payload = json.loads(args.payload.read_text(encoding="utf-8"))
    try:
        _import_irsim(repo)
        result = COMMANDS[args.command](args, repo, payload)
    except RefusalError as exc:
        result = {"ok": False, "kind": exc.kind, "error": str(exc)}
    except Exception as exc:
        result = {
            "ok": False,
            "kind": "crash",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
