"""Finding a material quickly: search, favourites, recently used, and a filter by band (B13).

Pure Python, so it is tested without Blender (``tests/test_pure.py``). An *item* is anything with
the attributes of a library entry as the bridge reports it (``properties.IrsimLibraryItem``):
``name``, ``description``, ``surface_treatment``, ``spectral``, ``error`` and ``eps_<band>``.

- **Search** matches every typed word, in any order, against the start of the words of the name
  (underscores read as spaces), the description or the surface treatment: ``polished al`` finds
  ``aluminium_polished``, and ``painted`` does not find "unpainted".
- **Favourites** and **recently used** are lists of names kept in the add-on's preferences, so
  they follow the person from file to file. Favourites come first in every list.
- **Band filter**: show the emissivity of one band, keep only materials whose emissivity there is
  within a range, sort by it. "Mirror-like in LWIR" is ``band="lwir", eps_max=0.2``.
"""

from __future__ import annotations

import re

BANDS = ("nir", "swir", "mwir", "lwir")
#: How many recently used materials are remembered.
RECENT = 6
_WORD = re.compile(r"[^0-9a-z]+")


def words(text: str) -> list[str]:
    """Lower-case words of ``text``; underscores, hyphens and punctuation separate them."""
    return [w for w in _WORD.split(text.lower()) if w]


def matches(item, query: str) -> bool:
    """True if every word of ``query`` begins a word of the item's name, description or surface.

    Words match from their start, so ``painted`` does not find "unpainted"; the name run together
    is also searched, so ``carbonfibre`` finds ``carbon_fibre``.
    """
    wanted = words(query)
    if not wanted:
        return True
    hay = words(f"{item.name} {item.description} {getattr(item, 'surface_treatment', '')}")
    joined = "".join(words(item.name))
    return all(joined.startswith(w) or any(h.startswith(w) for h in hay) for w in wanted)


def emissivity(item, band: str) -> float:
    return float(getattr(item, f"eps_{band}"))


def shown(
    items,
    *,
    query: str = "",
    band: str = "lwir",
    eps_min: float = 0.0,
    eps_max: float = 1.0,
    favourites: list[str] | tuple[str, ...] = (),
    only_favourites: bool = False,
    only_curves: bool = False,
) -> list[bool]:
    """For each item, whether the picker shows it."""
    fav = set(favourites)
    out = []
    for item in items:
        keep = matches(item, query)
        if keep and only_favourites:
            keep = item.name in fav
        if keep and only_curves:
            keep = bool(item.spectral)
        if keep and (eps_min > 0.0 or eps_max < 1.0):
            # a material whose band could not be evaluated has no emissivity to compare
            keep = not item.error and eps_min <= emissivity(item, band) <= eps_max
        out.append(keep)
    return out


def order(
    items, *, sort: str = "NAME", band: str = "lwir", favourites: list[str] | tuple[str, ...] = ()
) -> list[int]:
    """The display order: favourites first, then by name, or by emissivity in ``band``.

    Returns the item indices in the order they are shown (``UIList.filter_items`` wants the
    inverse, :func:`positions`).
    """
    fav = set(favourites)

    def key(i: int):
        item = items[i]
        eps = emissivity(item, band) if not item.error else -1.0
        by = {"EPS_HIGH": -eps, "EPS_LOW": eps if eps >= 0 else 2.0}.get(sort, 0.0)
        return (item.name not in fav, by, item.name)

    return sorted(range(len(items)), key=key)


def positions(sequence: list[int]) -> list[int]:
    """The inverse of :func:`order`: for each item index, the row it is drawn in."""
    out = [0] * len(sequence)
    for row, i in enumerate(sequence):
        out[i] = row
    return out


def parse_names(text: str) -> list[str]:
    """A stored list of names (comma-separated), without blanks or repeats."""
    seen: list[str] = []
    for name in (n.strip() for n in text.split(",")):
        if name and name not in seen:
            seen.append(name)
    return seen


def join_names(names: list[str]) -> str:
    return ",".join(names)


def toggle(names: list[str], name: str) -> list[str]:
    """``names`` with ``name`` added at the end, or removed if it was there."""
    return [n for n in names if n != name] if name in names else [*names, name]


def remember(recent: list[str], name: str, keep: int = RECENT) -> list[str]:
    """``recent`` with ``name`` moved to the front, at most ``keep`` long."""
    return [name, *(n for n in recent if n != name)][:keep]


def label(item, band: str) -> str:
    """One line for the search menu: the name and its emissivity in ``band``."""
    if item.error:
        return f"{item.name}   ({band.upper()} not available)"
    return f"{item.name}   ε {emissivity(item, band):.2f} {band.upper()}"
