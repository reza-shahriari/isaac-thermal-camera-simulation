"""RP.1 — the status table (TECHNICAL_REPORT.md) is a table, and stays one.

The table lived in README.md until 2026-09-28, when the README became a short public front page
and the engineering record moved to TECHNICAL_REPORT.md.

The status table is the first thing a new reader looks at, and four of its fifteen rows had lost
their **State** cell to pasted changelog prose: the state token had been pushed to the end of a
1,473-character paragraph and the tier column held it instead. A markdown table with a 1,473
character cell still renders, which is why nobody saw it -- it renders as a table whose second
column is an essay.

This is the third clobber of the same file by parallel sessions writing it whole (see the
`docs: restore ...` commits), so the repair needs something executable behind it rather than a
convention. What this pins is **shape, not content**: any row may say anything in its Notes, but
the State cell holds a state and the Tier cell holds a tier.

docs/roadmap.md RP.1
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
README = REPO / "TECHNICAL_REPORT.md"

#: The four tokens the table is allowed to say, plus an optional parenthetical fidelity level.
STATE = re.compile(r"^(🟢 done|🟡 partial|⬜ not started|🔴 blocked)(\s*\([^)]*\))?$")
#: A §15 validation tier, a range of them, a comma-separated list, and an optional qualifier --
#: `eval` is "T4 infra" and `validation` is "T2, T4 infra", which are statements about what the
#: package supports rather than what it has been validated to, and both are deliberate.
TIER = re.compile(r"^T[1-5]([–-]T[1-5])?(,\s*T[1-5]([–-]T[1-5])?)*(\s+[a-z]+)?$")

#: Every package the layout section promises. A row that disappears is as bad as a malformed one.
REQUIRED = {
    "radiometry",
    "materials",
    "thermal",
    "atmosphere",
    "optics",
    "detector",
    "noise",
    "isp",
    "config",
    "pipeline",
    "validation",
    "io",
    "eval",
    "irsim_isaac",
}


def _status_rows() -> list[tuple[int, list[str]]]:
    """(line number, cells) for each row of the component status table."""
    rows = []
    for number, line in enumerate(README.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) == 4:
            rows.append((number, cells))
    return rows


def test_the_status_table_was_found_at_all() -> None:
    """Guards the guard: a renamed heading would make every assertion below vacuous."""
    rows = _status_rows()
    assert len(rows) >= len(REQUIRED), (
        f"only {len(rows)} status rows found; the table moved or broke"
    )


def test_every_promised_package_has_a_row() -> None:
    named = {cells[0].strip("`") for _, cells in _status_rows()}
    missing = REQUIRED - named
    assert not missing, f"packages with no status row: {sorted(missing)}"


def test_the_state_cell_holds_a_state_and_nothing_else() -> None:
    """The defect RP.1 repaired: changelog prose pasted where the state token belongs.

    Reported with the cell length, because the failure mode is not a typo -- it is a paragraph,
    and the number is what makes that obvious in the failure output.
    """
    broken = [
        (number, cells[0], len(cells[1]), cells[1][:60])
        for number, cells in _status_rows()
        if not STATE.match(cells[1])
    ]
    assert not broken, "\n".join(
        f"  TECHNICAL_REPORT.md:{n} {name}: State cell is {length} chars, starts {text!r}"
        for n, name, length, text in broken
    )


def test_the_tier_cell_holds_a_tier() -> None:
    broken = [
        (number, cells[0], cells[2][:40])
        for number, cells in _status_rows()
        if not TIER.match(cells[2])
    ]
    assert not broken, "\n".join(
        f"  TECHNICAL_REPORT.md:{n} {name}: Tier cell is {text!r}" for n, name, text in broken
    )


def test_the_notes_cell_is_where_the_prose_lives() -> None:
    """The positive half: prose is not banned, it is *placed*. Notes carries the detail."""
    rows = _status_rows()
    assert rows, "no status rows"
    longest_state = max(len(cells[1]) for _, cells in rows)
    longest_notes = max(len(cells[3]) for _, cells in rows)
    assert longest_state < 40, f"a State cell is {longest_state} chars; prose belongs in Notes"
    assert longest_notes > longest_state, "Notes should be the long column, not State"
