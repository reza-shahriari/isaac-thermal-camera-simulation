"""RP.7 — `docs/spec-issues.md` says per row what has been applied, and the summary matches.

The file's header has always promised that "the **status** column below records what has been
applied". There was no status column. What there was instead was one prose sentence listing ten
M0-era resolutions and ending "Everything else is open" — which had been wrong for months, because
fifty-six of the sixty rows had shipped, thirty-seven of them with an ADR written.

A sentence somebody has to remember to edit is the failure mode. These tests make the claim per row
and mechanical: an ADR named in a status cell must exist, a row may not be `open` while the ADR its
own resolution names is already written, and the summary paragraph's counts are recomputed from the
table rather than trusted.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SPEC_ISSUES = REPO / "docs" / "spec-issues.md"
DECISIONS = REPO / "docs" / "decisions"

#: The two main tables. The short table above them repeats three ids in a statusless three-column
#: form, so everything here reads from after this marker.
MAIN = "**Physics (fix the spec):**"

ROW = re.compile(r"^\|\s*([ST]\d+)\s*\|(.*)\|\s*([^|]*?)\s*\|\s*$")

#: Written out in the summary paragraph, so a change to the table has to move the words too.
WORDS = {
    4: "four",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    16: "sixteen",
    19: "nineteen",
    20: "twenty",
    21: "twenty-one",
    22: "twenty-two",
    23: "twenty-three",
    37: "thirty-seven",
    39: "thirty-nine",
    38: "thirty-eight",
    56: "fifty-six",
    57: "fifty-seven",
    58: "fifty-eight",
    59: "fifty-nine",
    60: "sixty",
    61: "sixty-one",
    62: "sixty-two",
    69: "sixty-nine",
    70: "seventy",
}


def _rows() -> dict[str, tuple[str, str]]:
    """``{id: (resolution cell, status cell)}`` for the sixty rows of the two main tables."""
    main = SPEC_ISSUES.read_text().split(MAIN, 1)[1]
    out: dict[str, tuple[str, str]] = {}
    for line in main.splitlines():
        match = ROW.match(line)
        if match:
            cells = match.group(2).split("|")
            out[match.group(1)] = (cells[-1] if cells else "", match.group(3))
    return out


def _existing_adrs() -> set[str]:
    return {p.name[:4] for p in DECISIONS.glob("[0-9][0-9][0-9][0-9]-*.md")}


def _adrs_in(text: str) -> list[str]:
    numbers: list[str] = []
    for match in re.finditer(r"ADRs?\s+(\d{4}(?:\s*(?:,|/|,?\s+and)\s*\d{4})*)", text):
        numbers.extend(re.findall(r"\d{4}", match.group(1)))
    return numbers


# --- every row carries a status -------------------------------------------------------------------


def test_the_file_still_holds_seventy_issues() -> None:
    """If this moves, the summary paragraph's counts move with it and the tests below say so.

    Sixty until 2026-09-20; `S41`-`S49` were opened by roadmap revision 6 for the owner's
    per-point, part-to-part, water and fire requirements, and `S50` by `AT.12` for the cloud
    field's spectral-slope convention.
    """
    assert len(_rows()) == 70


def test_every_row_has_a_non_empty_status() -> None:
    blank = [ident for ident, (_, status) in _rows().items() if not status.strip()]
    assert not blank, f"rows with no status: {blank}"


def test_every_status_is_one_of_the_three_forms() -> None:
    """`ADR NNNN`, `code — <step>`, or `open`. A free-text status is one nobody can audit."""
    bad = [
        f"{ident}: {status!r}"
        for ident, (_, status) in _rows().items()
        if not re.match(r"^(ADR \d{4}|code —|\*\*open\*\*|open\b)", status.strip())
    ]
    assert not bad, "statuses that fit none of the three forms: " + "; ".join(bad)


# --- the mechanical half --------------------------------------------------------------------------


def test_every_adr_named_by_a_status_exists() -> None:
    existing = _existing_adrs()
    dangling = [
        f"{ident} → ADR {number}"
        for ident, (_, status) in _rows().items()
        for number in _adrs_in(status)
        if number not in existing
    ]
    assert not dangling, "statuses cite ADRs that do not exist: " + "; ".join(dangling)


def test_no_row_is_open_while_its_own_resolution_names_a_written_adr() -> None:
    """The exact way the old prose went stale: the decision shipped and the ledger did not move."""
    existing = _existing_adrs()
    stale = []
    for ident, (resolution, status) in _rows().items():
        if not status.strip().lstrip("*").startswith("open"):
            continue
        written = [n for n in _adrs_in(resolution) if n in existing]
        if written:
            stale.append(f"{ident} is open but ADR {', '.join(written)} is written")
    assert not stale, "; ".join(stale)


# --- the summary paragraph ------------------------------------------------------------------------


def test_the_summary_counts_are_the_table_s_own() -> None:
    """The paragraph is spelled out in words, so it cannot be left behind by a status edit."""
    rows = _rows()
    counts = {
        "adr": sum(1 for _, s in rows.values() if s.strip().startswith("ADR")),
        "code": sum(1 for _, s in rows.values() if s.strip().startswith("code")),
        "open": sum(1 for _, s in rows.values() if s.strip().lstrip("*").startswith("open")),
    }
    assert sum(counts.values()) == len(rows), f"{counts} does not account for all {len(rows)} rows"

    text = SPEC_ISSUES.read_text()
    for kind, n in counts.items():
        assert n in WORDS, f"no spelled-out word for {kind} = {n}; add it to WORDS and to the file"
        assert f"**{WORDS[n]}**" in text.lower() or WORDS[n] in text.lower(), (
            f"the summary paragraph does not say {WORDS[n]!r} for the {n} {kind} rows"
        )
    shipped = counts["adr"] + counts["code"]
    assert WORDS[shipped] in text.lower(), (
        f"the paragraph should say {WORDS[shipped]!r} of the sixty rows had shipped"
    )


# --- S13, which this step reopened ----------------------------------------------------------------


def test_s13_is_open_and_says_why() -> None:
    resolution, status = _rows()["S13"]
    assert status.lstrip("*").startswith("open")
    row = next(
        line
        for line in SPEC_ISSUES.read_text().splitlines()
        if line.startswith("| S13 |") and "Reopened" in line
    )
    # The reason has to be the proxy, not "the number disagrees" -- the number is fine for
    # soda-lime; it is the table that cannot speak for it.
    for phrase in ("fused silica", "soda-lime", "0.109"):
        assert phrase in row, f"S13's row never mentions {phrase!r}"


def test_the_glass_material_and_the_issue_agree() -> None:
    """Two files, one claim: the YAML defers to S13 and S13 must still be open to receive it."""
    glass = (REPO / "configs" / "materials" / "glass_windshield.yaml").read_text()
    assert 'transmittance_derivation: "authored: S13"' in glass
    assert _rows()["S13"][1].lstrip("*").startswith("open")


# --- the parser itself ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [("ADR 0042", ["0042"]), ("ADR 0051, 0064", ["0051", "0064"]), ("code — M6.5", [])],
)
def test_the_status_parser_reads_compound_citations(text: str, expected: list[str]) -> None:
    assert _adrs_in(text) == expected
