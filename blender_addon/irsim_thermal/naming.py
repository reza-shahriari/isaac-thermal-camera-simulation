"""Names that survive the trip from Blender to USD to the irsim asset map.

Pure Python, no ``bpy``: importable and testable outside Blender.

Three places read a name the add-on writes, and each has its own rules:

* **USD** turns every character outside ``[A-Za-z0-9_]`` into ``_`` when Blender exports a prim
  (``Tf.MakeValidIdentifier``), so ``white plastic.001`` in Blender is ``white_plastic_001`` in the
  stage that ``scripts/prep_asset.py`` audits.
* **The asset map** (``irsim.materials.mapping.AssetMapping``) matches a source material name
  exactly but case-insensitively, and refuses two keys that differ only by case.
* **Blender** caps an ID name at 63 bytes and answers a collision by appending ``.001``.

The export therefore renames, once and visibly, every material and part whose name would not come
out of that trip unchanged. The map is then written against names that are identical in the
``.blend``, in the USD and in the YAML, which is the only arrangement a person can check by eye.
"""

import re
from collections.abc import Iterable

__all__ = [
    "MAX_NAME_BYTES",
    "SPLIT_SEPARATOR",
    "is_default_name",
    "is_safe_identifier",
    "safe_identifier",
    "split_name",
    "unique_safe_names",
]

#: Blender's ID name limit (``MAX_ID_NAME - 2``) in bytes, not characters.
MAX_NAME_BYTES = 63

#: Between a material's original name and the thermal material a per-part copy of it carries.
SPLIT_SEPARATOR = "__"

_UNSAFE = re.compile(r"[^A-Za-z0-9_]")
_DEFAULT = re.compile(
    r"^(Cube|Sphere|UVSphere|Icosphere|Cylinder|Cone|Plane|Circle|Torus|Grid|Monkey|Suzanne|"
    r"Mesh|Object|Material|Untitled|Default|mesh|object)(\.\d+|_\d+)?$"
)


def _truncate_utf8(text: str, limit: int) -> str:
    """The longest prefix of ``text`` whose UTF-8 encoding fits in ``limit`` bytes."""
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text
    return encoded[:limit].decode("utf-8", errors="ignore")


def safe_identifier(name: str) -> str:
    """``name`` as USD will write it: ASCII letters, digits and ``_``, not starting with a digit."""
    out = _UNSAFE.sub("_", name)
    if not out:
        out = "_"
    if out[0].isdigit():
        out = "_" + out
    return _truncate_utf8(out, MAX_NAME_BYTES)


def is_safe_identifier(name: str) -> bool:
    return safe_identifier(name) == name


def unique_safe_names(names: Iterable[str], taken: Iterable[str] = ()) -> dict[str, str]:
    """``{old: new}`` for every name that must change to reach the asset map unchanged.

    A name changes when it is not a safe identifier, or when its safe form collides -- case-
    insensitively, which is how the asset map compares -- with a name already claimed. ``taken``
    holds names that stay as they are (other Blender materials that are not being exported), so a
    rename never lands on one of them and triggers Blender's ``.001``. Names are visited in sorted
    order, so the same scene always yields the same renames.
    """
    claimed = {t.lower() for t in taken}
    renames: dict[str, str] = {}
    for name in sorted(set(names)):
        candidate = safe_identifier(name)
        stem = candidate
        n = 2
        while candidate.lower() in claimed:
            suffix = f"_{n}"
            candidate = _truncate_utf8(stem, MAX_NAME_BYTES - len(suffix)) + suffix
            n += 1
        claimed.add(candidate.lower())
        if candidate != name:
            renames[name] = candidate
    return renames


def split_name(origin: str, thermal: str) -> str:
    """The name of a per-part copy of material ``origin`` that carries thermal material ``thermal``.

    ``white_plastic`` shared by a shell and a propeller, with the propeller assigned
    ``carbon_fibre``, gives the propeller ``white_plastic__carbon_fibre``: the same look, a
    different infrared material, and a name that says both. The thermal suffix is kept whole and
    the origin is shortened instead when the result would pass Blender's 63-byte limit.
    """
    suffix = f"{SPLIT_SEPARATOR}{thermal}"
    return _truncate_utf8(origin, MAX_NAME_BYTES - len(suffix.encode("utf-8"))) + suffix


def is_default_name(name: str) -> bool:
    """True for names Blender or an exporter made up (``Cube.003``, ``Mesh``).

    A scene config addresses a part by its prim name (``prim: battery``), so a part still called
    ``Cylinder.012`` at export is a part nobody will be able to find again.
    """
    return bool(_DEFAULT.match(name))
