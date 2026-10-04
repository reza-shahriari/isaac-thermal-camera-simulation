"""The irsim side of the Blender add-on. Runs under the **project** interpreter, never in Blender.

Blender's Python has neither pydantic nor PyYAML, and the add-on must not carry a second copy of
the material schema, of Kirchhoff closure (CLAUDE.md #4) or of the asset-map rules: a copy is a
rule that drifts from the one the solver enforces. So every question the add-on asks about the
library, and every file it writes into the repository, goes through this script, which imports
``irsim`` from the repository it is pointed at and uses the project's own loaders:

* ``library``        -- every material, with ε/ρ/τ per band as ``Material.band_properties``
                        derives them (so a spectral material shows its band-integrated values),
                        the joint table (``configs/thermal/joints.yaml``) a contact names, and the
                        component library (``configs/components/``) a hidden part may name.
* ``check-material`` -- validate a draft exactly as the loader will, and return its derived bands.
* ``check-curve``    -- read a spectral CSV the person picked through the library's own loader:
                        its span, range of values, which nominal bands it covers, a plot sample.
* ``plot-material``  -- draw one material's authored quantity across the bands as a PNG the
                        panel shows: the curve where it has data, its per-band or grey fill
                        dashed where it does not, the nominal bands shaded.
* ``write-material`` -- the same, then write ``configs/materials/<name>.yaml`` and reload the
                        whole library; a file that breaks the library is removed again.
* ``list-assets``    -- the asset configs there are, and which ones this add-on wrote.
* ``read-asset``     -- one asset config's map, loaded and checked by the project loader, with
                        its parts block's contacts and hidden parts when it has one.
* ``write-asset``    -- validate and write ``configs/assets/<name>.yaml``
                        (``irsim.materials.mapping.AssetConfig``), then reload it with every target
                        checked against the library. With a ``structure``, the config also gets a
                        ``parts:`` block: each exported object a named part (``objects:``, AI.18),
                        and the contacts and hidden parts the solver reads (AI.11, AI.12).
* ``write-structure`` -- the add-on's own record beside the USD,
                        ``3d_models/<name>/<name>.structure.yaml``: the facing pairs, each hidden
                        part's idle heat and reference, and which contacts nobody has reviewed --
                        what the asset config has no field for. With ``check_only`` it is the
                        pre-export check of the contacts and hidden parts.

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
import textwrap
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
            # ADR 0175: the share of the band the curve supplied; the rest is per-band or grey.
            "curve_fraction": p.curve_fraction,
        }
    return out


#: Points in the plot sample of a curve. Log-spaced, because a curve may run 0.35-14 µm and the
#: short-wave detail would otherwise be a few points.
CURVE_SAMPLE_POINTS = 240


def _curve_sample(curve: Any) -> dict | None:
    """``{wavelength_um, value}`` for drawing a material's curve; ``None`` in its gaps."""
    import numpy as np

    if curve is None:
        return None
    lo, hi = curve.support_um
    lam = np.geomspace(lo, hi, CURVE_SAMPLE_POINTS)
    values = curve.values(lam)
    return {
        "wavelength_um": [round(float(x), 5) for x in lam],
        "value": [None if np.isnan(v) else round(float(v), 6) for v in values],
    }


def _forms_of(material: Any) -> dict:
    """Which of ADR 0175's three forms a material authors, and what each covers."""
    optical = material.spec.optical
    segments = []
    if material.curve is not None:
        for seg, flip, spec_seg in zip(
            material.curve.segments,
            material.curve.complement,
            optical.spectral_segments,
            strict=True,
        ):
            lo, hi = seg.support_um
            segments.append(
                {
                    "file": spec_seg.file,
                    "quantity": spec_seg.quantity,
                    "complement": bool(flip),
                    "lo_um": lo,
                    "hi_um": hi,
                }
            )
    table = optical.band_table
    return {
        "curve": segments,
        "per_band": dict(table) if table else None,
        "grey": optical.grey_value,
        "summary": _forms_summary(segments, table, optical.grey_value),
    }


def _forms_summary(segments: list[dict], table: Any, grey: float | None) -> str:
    """One line for the panel, e.g. 'curve 0.35-2.5 µm (1 - R), 8-14 µm; per band MWIR, LWIR'."""
    parts = []
    if segments:
        spans = ", ".join(
            f"{s['lo_um']:.3g}-{s['hi_um']:.3g} µm" + (" (1 - R)" if s["complement"] else "")
            for s in segments
        )
        parts.append(f"curve {spans}")
    if table:
        parts.append("per band " + ", ".join(b.upper() for b in table))
    if grey is not None:
        parts.append(f"grey {grey:.3g}")
    return "; ".join(parts)


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
            "forms": _forms_of(material),
            "curve_sample": _curve_sample(material.curve),
            "angular": angular,
            "roughness": dict(optical.roughness_per_band or {}),
        },
        "bands": _bands_of(material, band_ids),
    }


def _material_dir(args: argparse.Namespace, repo: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(args.material_dir) if args.material_dir else repo / "configs" / "materials"


def _data_dir(args: argparse.Namespace, repo: pathlib.Path) -> pathlib.Path:
    """Where curve files resolve and where a picked curve is copied.

    ``--data-dir``, else ``$IRSIM_DATA_DIR``, else ``<repo>/data`` -- the loader's own order, but
    rooted at the repository the bridge was pointed at rather than wherever the ``irsim``
    package happens to live, so a curve is read back from exactly where it was copied.
    """
    import os

    raw = args.data_dir or os.environ.get("IRSIM_DATA_DIR") or str(repo / "data")
    return pathlib.Path(raw).expanduser().resolve()


_SPECTRAL_KEYS = ("spectral_emissivity", "spectral_reflectance")


def _segments(optical: dict) -> list[tuple[str, int | None, dict | str]]:
    """``(key, index or None, entry)`` for every curve file a draft's optical block names."""
    out: list[tuple[str, int | None, dict | str]] = []
    for key in _SPECTRAL_KEYS:
        value = optical.get(key)
        if isinstance(value, str):
            out.append((key, None, value))
        elif isinstance(value, list):
            out.extend((key, i, entry) for i, entry in enumerate(value))
    return out


def _entry_file(entry: dict | str) -> str:
    return entry if isinstance(entry, str) else str(entry.get("file", ""))


def _set_entry_file(optical: dict, key: str, index: int | None, new: str) -> None:
    if index is None:
        optical[key] = new
        return
    entry = optical[key][index]
    optical[key][index] = new if isinstance(entry, str) else {**entry, "file": new}


def _resolved(raw: str, data_dir: pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(raw).expanduser()
    return (p if p.is_absolute() else data_dir / p).resolve()


def _order_segments(optical: dict) -> None:
    """Sort a list of curve segments by where each file starts, as the loader requires."""
    from irsim.materials.spectra import load_property_spectrum

    for key in _SPECTRAL_KEYS:
        value = optical.get(key)
        if not isinstance(value, list) or len(value) < 2:
            continue
        try:
            starts = [load_property_spectrum(_entry_file(e)).support_um[0] for e in value]
        except (ValueError, FileNotFoundError) as exc:
            raise RefusalError(str(exc), kind="invalid") from exc
        optical[key] = [e for _, e in sorted(zip(starts, value, strict=True), key=lambda t: t[0])]


def _prune_covered_bands(optical: dict, data_dir: pathlib.Path) -> list[str]:
    """Drop per-band values for registry bands the draft's curve wholly covers (ADR 0175).

    The form lets a person type all four band values and pick a curve; the library refuses a band
    value the curve covers as a second authoring. Rather than make the person work out which
    bands those are, the bridge reads the curve and removes them -- and says which it removed.
    """
    from irsim.config.bands import NOMINAL_RANGES_UM
    from irsim.materials.spectra import SpectralCurve, load_property_spectrum

    segments = _segments(optical)
    if not segments:
        return []
    own = "emissivity" if "spectral_emissivity" in optical else "reflectance"
    files, flips = [], []
    for _, _, entry in segments:
        try:
            files.append(load_property_spectrum(_resolved(_entry_file(entry), data_dir)))
        except (ValueError, FileNotFoundError) as exc:
            raise RefusalError(str(exc), kind="invalid") from exc
        quantity = entry if isinstance(entry, str) else entry.get("quantity", own)
        flips.append(isinstance(entry, dict) and quantity != own)
    order = sorted(range(len(files)), key=lambda i: files[i].support_um[0])
    try:
        curve = SpectralCurve(tuple(files[i] for i in order), tuple(flips[i] for i in order))
    except ValueError as exc:
        raise RefusalError(str(exc), kind="invalid") from exc
    removed = []
    for key in ("emissivity_per_band", "reflectance_per_band"):
        table = optical.get(key)
        if not isinstance(table, dict):
            continue
        for band in list(table):
            rng = NOMINAL_RANGES_UM.get(band)  # type: ignore[call-overload]
            if rng is not None and curve.covers_interval(*rng):
                del table[band]
                removed.append(band)
        if not table:
            del optical[key]
    return removed


def cmd_library(args: argparse.Namespace, repo: pathlib.Path, payload: Any) -> dict:
    from irsim.materials.library import MaterialLibrary
    from irsim.materials.mapping import MIRROR_EMISSIVITY

    material_dir = _material_dir(args, repo)
    library = MaterialLibrary.load(material_dir, _data_dir(args, repo))
    band_ids = _band_ids()
    return {
        "ok": True,
        "bands": list(band_ids),
        "mirror_emissivity": MIRROR_EMISSIVITY,
        "library_hash": library.content_hash(),
        "material_dir": str(material_dir),
        "materials": [_material_record(library[n], repo, band_ids) for n in sorted(library)],
        "joints": _joint_records(),
        "components": _component_records(repo),
    }


def _components(repo: pathlib.Path) -> Any:
    """The repository's component library (AI.12), or an empty mapping when it has none."""
    from irsim.config.components import load_component_library

    folder = repo / "configs" / "components"
    return load_component_library(folder) if folder.is_dir() else {}


def _component_records(repo: pathlib.Path) -> list[dict]:
    return [
        {
            "name": c.name,
            "kind": c.kind,
            "description": c.description,
            "mass_kg": c.mass_kg,
            "specific_heat_j_kgk": c.specific_heat_j_kgk,
            "dissipation_idle_w": c.dissipation_idle_w,
            "dissipation_rated_w": c.dissipation_rated_w,
            "material": c.material or "",
            "status": c.status,
            "source": " ".join(c.source.split()),
        }
        for c in sorted(_components(repo).values(), key=lambda c: c.name)
    ]


def _joint_records() -> list[dict]:
    from irsim.config.joints import load_joint_table

    table = load_joint_table()
    return [
        {
            "name": name,
            "h_c_w_m2_k": spec.h_c_w_m2_k,
            "status": spec.status,
            "source": " ".join(spec.source.split()),
        }
        for name, spec in sorted(table.joints.items())
    ]


def _validated_material(
    draft: dict,
    material_dir: pathlib.Path,
    repo: pathlib.Path,
    data_dir: pathlib.Path | None = None,
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
    data_dir = data_dir if data_dir is not None else repo / "data"
    optical = draft["material"].get("optical") or {}
    # A picked curve is validated where it lies: every file is given to the loader by absolute
    # path, so a CSV outside data/ is read as it is and copied in only on write.
    for key, index, entry in _segments(optical):
        _set_entry_file(optical, key, index, str(_resolved(_entry_file(entry), data_dir)))
    _order_segments(optical)
    pruned = _prune_covered_bands(optical, data_dir) if draft.get("prune_covered") else []
    doc = {"schema_version": MATERIAL_SCHEMA_VERSION, "material": draft["material"]}
    try:
        spec = MaterialConfig.model_validate(doc).material
    except ValidationError as exc:
        raise RefusalError(_pydantic_message(exc), kind="invalid") from exc

    existing = MaterialLibrary.load(material_dir, data_dir)
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
            material = load_material(path, data_dir)
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
    if pruned:
        warnings.append(
            "The curve covers " + ", ".join(b.upper() for b in pruned) + " wholly, so the typed "
            "value" + ("s for those bands were" if len(pruned) > 1 else " for it was") + " left "
            "out: the curve is the material there (ADR 0175)."
        )
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
        "# Schema: irsim.config.materials. Emissivity is authored -- as a curve, per band and/or"
    )
    header.append(
        "# one grey value (ADR 0175) -- with transmittance per band; reflectance is derived as"
    )
    header.append("# 1 - emissivity - transmittance (CLAUDE.md #4).")
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, allow_unicode=True)
    return "\n".join(header) + "\n" + body


def cmd_check_material(args: argparse.Namespace, repo: pathlib.Path, payload: Any) -> dict:
    _, material, warnings = _validated_material(
        payload, _material_dir(args, repo), repo, _data_dir(args, repo)
    )
    return {
        "ok": True,
        "name": material.name,
        "bands": _bands_of(material, _band_ids()),
        "warnings": warnings,
    }


def _copy_curves_in(doc: dict, data_dir: pathlib.Path, name: str, reference: str) -> list[str]:
    """Copy every curve file outside ``data_dir`` to ``spectra/materials/`` and point at it.

    The library resolves curve paths against the data root and a file elsewhere would not travel
    with the repository, so a CSV the person picked is copied in, with a header that says where it
    came from and that the material's ``reference`` is its citation. A target that already exists
    is refused rather than overwritten. Returns the files written.
    """
    import shutil

    optical = doc["material"]["optical"]
    segments = _segments(optical)
    written = []
    for n, (key, index, entry) in enumerate(segments):
        src = pathlib.Path(_entry_file(entry))
        inside = src.resolve().is_relative_to(data_dir)
        if inside:
            rel = src.resolve().relative_to(data_dir)
        else:
            suffix = "" if len(segments) == 1 else f"_{n + 1}"
            rel = pathlib.Path("spectra") / "materials" / f"{name}{suffix}.csv"
            target = data_dir / rel
            if target.exists():
                raise RefusalError(f"{target} already exists", kind="exists")
            target.parent.mkdir(parents=True, exist_ok=True)
            today = datetime.date.today().isoformat()
            head = (
                f"# Copied by the {MARKER} on {today} from {src.name} for material {name}.\n"
                f"# source: {reference.strip() or 'see the material file'} (the material's "
                "reference; the copy is otherwise unmodified).\n"
            )
            with target.open("w", encoding="utf-8") as out:
                out.write(head)
                with src.open(encoding="utf-8") as fh:
                    shutil.copyfileobj(fh, out)
            written.append(str(target))
        _set_entry_file(optical, key, index, rel.as_posix())
    return written


def cmd_check_curve(args: argparse.Namespace, repo: pathlib.Path, payload: Any) -> dict:
    """Read one picked CSV through ``load_property_spectrum`` and describe it for the form."""
    import numpy as np

    from irsim.config.bands import NOMINAL_RANGES_UM
    from irsim.materials.spectra import SpectralCurve, load_property_spectrum

    if not isinstance(payload, dict) or not payload.get("file"):
        raise RefusalError("the payload needs a 'file'")
    path = _resolved(str(payload["file"]), _data_dir(args, repo))
    try:
        spectrum = load_property_spectrum(path)
    except (ValueError, FileNotFoundError) as exc:
        raise RefusalError(str(exc), kind="invalid") from exc
    curve = SpectralCurve((spectrum,), (False,))
    lo, hi = spectrum.support_um
    head = path.read_text(encoding="utf-8", errors="replace")[:4000].lower()
    return {
        "ok": True,
        "file": str(path),
        "lo_um": lo,
        "hi_um": hi,
        "rows": int(spectrum.wavelength_um.size),
        "min": float(np.min(spectrum.values)),
        "max": float(np.max(spectrum.values)),
        "covers": {
            b: curve.covers_interval(*NOMINAL_RANGES_UM[b])  # type: ignore[index]
            for b in _band_ids()
            if b in NOMINAL_RANGES_UM
        },
        "has_source": "# source:" in head,
        "sample": _curve_sample(curve),
    }


#: The panel shows the plot as an icon; this is its pixel size.
PLOT_SIZE_PX = (640, 300)


def cmd_plot_material(args: argparse.Namespace, repo: pathlib.Path, payload: Any) -> dict:
    """Write ``payload['out']`` (PNG): the material's ε(λ) or ρ(λ) as the library resolves it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    from irsim.config.bands import NOMINAL_RANGES_UM
    from irsim.materials.library import MaterialLibrary

    if not isinstance(payload, dict) or not payload.get("name") or not payload.get("out"):
        raise RefusalError("the payload needs 'name' and 'out'")
    library = MaterialLibrary.load(_material_dir(args, repo), _data_dir(args, repo))
    if payload["name"] not in library:
        raise RefusalError(f"no material named {payload['name']!r}")
    material = library[payload["name"]]
    optical = material.spec.optical
    table = optical.band_table or {}
    grey = optical.grey_value

    w, h = PLOT_SIZE_PX
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
    shades = ("#e8eef7", "#f4ece2")
    for i, band in enumerate(_band_ids()):
        rng = NOMINAL_RANGES_UM.get(band)  # type: ignore[call-overload]
        if rng is None:
            continue
        ax.axvspan(*rng, color=shades[i % 2], zorder=0)
        ax.text(sum(rng) / 2, 1.02, band.upper(), ha="center", va="bottom", fontsize=8)
        fill = table.get(band, grey)
        covered = material.curve is not None and material.curve.covers_interval(*rng)
        if fill is not None and not covered:
            ax.hlines(fill, *rng, colors="#b0592b", linestyles="dashed", linewidth=1.5, zorder=2)
    if material.curve is not None:
        lo, hi = material.curve.support_um
        lam = np.geomspace(lo, hi, 1200)
        ax.plot(lam, material.curve.values(lam), color="#1f4e8c", linewidth=1.4, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 16.0)
    ax.set_ylim(0.0, 1.08)
    ticks = [0.4, 0.75, 1, 1.7, 3, 5, 7.5, 10, 13.5]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks], fontsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("wavelength (µm)", fontsize=8)
    ax.set_ylabel(optical.authored, fontsize=8)
    ax.set_title(material.name, fontsize=9, loc="left")
    fig.tight_layout()
    out = pathlib.Path(payload["out"]).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return {"ok": True, "file": str(out), "summary": _forms_of(material)["summary"]}


def cmd_write_material(args: argparse.Namespace, repo: pathlib.Path, payload: Any) -> dict:
    from irsim.materials.library import MaterialLibrary

    material_dir = _material_dir(args, repo)
    data_dir = _data_dir(args, repo)
    doc, material, warnings = _validated_material(payload, material_dir, repo, data_dir)
    target = material_dir / f"{material.name}.yaml"
    if target.exists():
        raise RefusalError(f"{target} already exists", kind="exists")
    curves = _copy_curves_in(doc, data_dir, material.name, material.spec.reference)
    target.write_text(_material_yaml(doc, material.spec.source), encoding="utf-8")
    try:
        library = MaterialLibrary.load(material_dir, data_dir)
        written = library[material.name]
    except Exception as exc:
        target.unlink(missing_ok=True)
        for path in curves:
            pathlib.Path(path).unlink(missing_ok=True)
        raise RefusalError(
            f"the library no longer loads with this file, so it was removed: {exc}"
        ) from exc
    return {
        "ok": True,
        "name": material.name,
        "file": str(target),
        "curves": curves,
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
        header.extend(f"# {line}" for line in textwrap.wrap(note, 96))
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
    library = MaterialLibrary.load(_material_dir(args, repo), _data_dir(args, repo))
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
        "parts": [s.name for s in (asset.parts.parts if asset.parts else [])],
        "contacts": [
            {"a": c.a, "b": c.b, "joint": c.joint, "area_m2": c.area_m2}
            for c in (asset.parts.contacts if asset.parts else [])
        ],
        "hidden_parts": [h.model_dump() for h in (asset.parts.hidden_parts if asset.parts else [])],
    }


def _parts_block(structure: dict) -> dict:
    """The asset config's ``parts:`` block (``irsim.io.asset_parts.PartsConfig``) for an export.

    Every exported object is a part of its own name, selected by that name (AI.18) and judged
    whole (AI.16); its contacts and hidden parts (AI.11) name those parts. A hidden part names its
    component (AI.12) when it has one, and carries only the numbers the person set, which then
    override the component's.
    """
    hidden = []
    for h in structure.get("hidden_parts", []):
        record: dict[str, Any] = {
            "name": h["name"],
            "centre_m": [float(f"{x:.6g}") for x in h["aabb"]["centre_m"]],
            "size_m": [float(f"{x:.6g}") for x in h["aabb"]["size_m"]],
        }
        for key in ("component", "mass_kg", "specific_heat_j_kgk", "dissipation_w", "material"):
            value = h.get(key)
            if value not in (None, "", 0, 0.0):
                record[key] = value
        record["status"] = h.get("status", "ESTIMATED")
        hidden.append(record)
    return {
        "centre": [float(f"{x:.6g}") for x in structure.get("centre", (0.0, 0.0, 0.0))],
        "granularity": "object",
        "parts": [
            {"name": n, "select": {"objects": [n]}} for n in sorted(structure.get("part_areas", {}))
        ],
        "contacts": [
            {
                "a": c["parts"][0],
                "b": c["parts"][1],
                "joint": c["joint"],
                "area_m2": float(f"{c['area_m2']:.6g}"),
            }
            for c in structure.get("contacts", [])
        ],
        "hidden_parts": hidden,
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
    structure = payload.get("structure")
    if structure:
        problems = _structure_problems(repo, args, structure)
        if problems:
            raise RefusalError("\n".join(problems), kind="invalid")
        asset["parts"] = _parts_block(structure)
    doc = {"schema_version": ASSET_SCHEMA_VERSION, "asset": asset}
    try:
        AssetConfig.model_validate(doc)
    except ValidationError as exc:
        raise RefusalError(_pydantic_message(exc), kind="invalid") from exc
    library = MaterialLibrary.load(_material_dir(args, repo), _data_dir(args, repo))
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
    notes = list(payload.get("notes", []))
    if structure:
        unreviewed = [
            " / ".join(c["parts"]) for c in structure.get("contacts", []) if not c.get("reviewed")
        ]
        notes.append(
            "`parts:` -- every exported object is a part of its own name (selected by that name, "
            "judged whole); `contacts:` and `hidden_parts:` are what the thermal solve reads "
            "(AI.11). A hidden part's box is the axis-aligned box around it as placed in Blender; "
            "the numbers it carries override its component's."
        )
        if unreviewed:
            notes.append(
                f"Contacts the finder proposed and nobody reviewed: {', '.join(unreviewed)}."
            )
    previous = target.read_text(encoding="utf-8") if target.exists() else None
    target.write_text(_asset_yaml(doc, list(payload.get("parts", [])), notes), encoding="utf-8")
    try:
        reloaded = load_asset_mapping(target, known_materials=library.names)
    except (ValidationError, ValueError) as exc:  # the project loader has the last word
        if previous is None:
            target.unlink()
        else:
            target.write_text(previous, encoding="utf-8")
        message = _pydantic_message(exc) if isinstance(exc, ValidationError) else str(exc)
        raise RefusalError(message, kind="invalid") from exc
    out = {"ok": True, "file": str(target), "materials": dict(reloaded.materials)}
    if reloaded.parts is not None:
        out["parts"] = len(reloaded.parts.parts)
        out["contacts"] = len(reloaded.parts.contacts)
        out["hidden_parts"] = len(reloaded.parts.hidden_parts)
    return out


def _structure_yaml(name: str, doc: dict) -> str:
    import yaml

    today = datetime.date.today().isoformat()
    header = [
        f"# {name}: contacts, facing pairs and hidden parts, from the {MARKER}",
        f"# (blender_addon/irsim_thermal) on {today}.",
        "#",
        "# contacts:     two parts that touch; `joint` names configs/thermal/joints.yaml,",
        "#               `area_m2` is the area where they touch.",
        "# facing:       two parts that look straight at each other across `gap_m`; `area_m2`",
        "#               is, for each, the area whose outward normal reaches the other. This is",
        "#               not a view factor (roadmap TC.9).",
        "# hidden_parts: parts the model does not have (an engine inside a car shell), each a box",
        "#               in the model's frame, in metres, with its mass and heat output.",
        "# `reviewed: false` marks what the finder proposed and nobody has looked at yet.",
        "#",
        "# The contacts and hidden parts irsim solves with are in configs/assets/<name>.yaml",
        "# (its `parts:` block, AI.11). This file is the add-on's fuller record of the same",
        "# export -- the facing pairs, each hidden part's idle heat and reference, what nobody",
        "# has reviewed -- and nothing in irsim reads it. Edit in Blender and export again.",
    ]
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, allow_unicode=True)
    return "\n".join(header) + "\n" + body


def _structure_problems(repo: pathlib.Path, args: argparse.Namespace, payload: dict) -> list[str]:
    """Contacts and hidden parts checked the way AI.11's loader checks them, before anything is
    written: a contact naming a part that is not there, a joint not in the table, a contact area
    larger than either part, a hidden part with neither a mass nor a component (AI.12) to take one
    from, a component or material the libraries do not have, or less heat at full load than idle.
    """
    from irsim.config.joints import load_joint_table
    from irsim.materials.library import MaterialLibrary

    part_areas = {str(k): float(v) for k, v in payload.get("part_areas", {}).items()}
    hidden = list(payload.get("hidden_parts", []))
    names = set(part_areas) | {str(h.get("name", "")) for h in hidden}
    joints = load_joint_table().joints
    library = MaterialLibrary.load(_material_dir(args, repo), _data_dir(args, repo))
    components = _components(repo)
    problems: list[str] = []

    for h in hidden:
        what = f"hidden part {h.get('name')!r}"
        component = str(h.get("component") or "")
        if component and component not in components:
            problems.append(f"{what}: {component!r} is not in the component library")
        if not component and not float(h.get("mass_kg", 0.0)) > 0.0:
            problems.append(f"{what} has no mass, and no component to take one from")
        material = h.get("material") or (
            components[component].material if component in components else None
        )
        if material not in library.names:
            problems.append(f"{what}: {material!r} is not a library material")
        heat = h.get("heat_w", {})
        if float(heat.get("rated", 0.0)) < float(heat.get("idle", 0.0)):
            problems.append(f"{what} makes less heat at full load than at idle")
        if min(h.get("box", {}).get("size_m", [0.0])) <= 0.0:
            problems.append(f"{what} has no size")
        part_areas.setdefault(str(h.get("name", "")), float(h.get("area_m2", 0.0)))
    for kind in ("contacts", "facing"):
        for c in payload.get(kind, []):
            a, b = (str(x) for x in c.get("parts", ("", "")))
            what = f"{kind[:-1] if kind == 'contacts' else 'facing pair'} {a} / {b}"
            missing = [x for x in (a, b) if x not in names]
            if missing:
                problems.append(f"{what}: no part named {missing}")
                continue
            if a == b:
                problems.append(f"{what}: a part cannot be connected to itself")
            if kind == "contacts":
                if c.get("joint") not in joints:
                    problems.append(f"{what}: joint {c.get('joint')!r} is not in joints.yaml")
                area = float(c.get("area_m2", 0.0))
                smaller = min(part_areas.get(a, 0.0), part_areas.get(b, 0.0))
                if not area > 0.0:
                    problems.append(f"{what}: no contact area")
                elif area > smaller * 1.0001:
                    problems.append(
                        f"{what}: a contact of {area:.4g} m2 is larger than the smaller part "
                        f"({smaller:.4g} m2)"
                    )
    return problems


def cmd_write_structure(args: argparse.Namespace, repo: pathlib.Path, payload: Any) -> dict:
    """The add-on's own record of the structure, after the same check the asset config gets.

    With ``check_only`` it stops after the check: the export asks first, so a refusal arrives
    before the USD or the asset config is written. Otherwise it writes what the asset config has
    no field for -- facing pairs, idle heat, references, which contacts nobody has reviewed --
    beside the USD. The contacts and hidden parts the solver reads are in the asset config.
    """
    name = str(payload.get("name", ""))
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise RefusalError(f"asset name {name!r} is not valid", kind="invalid")
    hidden = list(payload.get("hidden_parts", []))
    problems = _structure_problems(repo, args, payload)
    if problems:
        raise RefusalError("\n".join(problems), kind="invalid")
    if payload.get("check_only"):
        return {"ok": True, "checked": True}

    models = pathlib.Path(args.models_dir) if args.models_dir else repo / "3d_models"
    target = models / name / f"{name}.structure.yaml"
    if target.exists():
        if not _by_addon(target):
            raise RefusalError(f"{target.name} was not written by the add-on", kind="handwritten")
        if not payload.get("overwrite"):
            raise RefusalError(f"{target.name} exists (from an earlier export)", kind="exists")
    doc = {
        "asset": name,
        "contacts": list(payload.get("contacts", [])),
        "facing": list(payload.get("facing", [])),
        "hidden_parts": [{k: v for k, v in h.items() if k != "area_m2"} for h in hidden],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_structure_yaml(name, doc), encoding="utf-8")
    return {
        "ok": True,
        "file": str(target),
        "contacts": len(doc["contacts"]),
        "facing": len(doc["facing"]),
        "hidden_parts": len(doc["hidden_parts"]),
    }


COMMANDS = {
    "library": cmd_library,
    "check-curve": cmd_check_curve,
    "plot-material": cmd_plot_material,
    "check-material": cmd_check_material,
    "write-material": cmd_write_material,
    "list-assets": cmd_list_assets,
    "read-asset": cmd_read_asset,
    "write-asset": cmd_write_asset,
    "write-structure": cmd_write_structure,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--repo", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--in", dest="payload", type=pathlib.Path, default=None)
    ap.add_argument("--material-dir", default=None, help="default: <repo>/configs/materials")
    ap.add_argument("--assets-dir", default=None, help="default: <repo>/configs/assets")
    ap.add_argument("--models-dir", default=None, help="default: <repo>/3d_models")
    ap.add_argument("--data-dir", default=None, help="default: <repo>/data")
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
