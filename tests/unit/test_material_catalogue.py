"""The material library's page shows the numbers the simulator uses (roadmap XD.13).

`docs/materials.md` lists every material with its band emissivities and says where each came
from. Its tables are written by `scripts/material_catalogue.py`; this fails when a material is
added or changed and the page is not rewritten, so the page cannot drift from the library.
"""

from __future__ import annotations

import pathlib
import sys

from irsim.materials.library import MaterialLibrary

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from material_catalogue import PAGE, render  # noqa: E402


def test_the_page_is_what_the_library_says() -> None:
    current = PAGE.read_text(encoding="utf-8")
    assert render(current, MaterialLibrary.load()) == current, (
        "docs/materials.md is out of date: run scripts/material_catalogue.py"
    )


def test_every_material_has_a_row() -> None:
    text = PAGE.read_text(encoding="utf-8")
    library = MaterialLibrary.load()
    missing = [n for n in library.names if f"| `{n}` |" not in text]
    assert not missing


def test_the_page_states_the_counts_it_claims() -> None:
    """The prose quotes the library's size; a new material must update the sentence too."""
    text = PAGE.read_text(encoding="utf-8")
    assert f"The library has {len(MaterialLibrary.load())} materials." in text
