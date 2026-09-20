"""Revision 6 — the roadmap's phase-plan table says what the step rows say.

The table under *Phase plan* is the one place a reader sees the whole order at a glance, and
revision 5's had drifted from its own rows in seven cells and omitted five steps: `IG.12` listed
under B while its row said X, `XD.*` listed under X while `XD.3` was B and `XD.10` was C, and
`AT.5`, `AT.10`, `IG.16`, `GT.7` and `SC.14` in no cell at all. The queue is computed from the rows,
so the table was describing an order the tooling did not use.

The table is now generated from the rows (the revision-6 script that wrote it is not in the tree;
this test is what keeps the next hand edit honest). Three claims: every step id appears in exactly
one phase cell, that cell's phase letter is the row's phase, and nothing in a cell is not a step.

docs/roadmap.md ("Phase plan"); scripts/next_step.py (`load`, `PHASE_RANK`).
"""

from __future__ import annotations

import importlib.util
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
ROADMAP = REPO / "docs" / "roadmap.md"

PHASE_ROW = re.compile(r"^\| \*\*([0-9A-Z]) — [^|]*\*\* \| (.*?) \| (.*) \|\s*$")
ID = re.compile(r"`([A-Z]{2})\.(\d+)`")
RANGE = re.compile(r"`([A-Z]{2})\.(\d+)`–`([A-Z]{2})\.(\d+)`")


@pytest.fixture(scope="module")
def ns():
    spec = importlib.util.spec_from_file_location("next_step", REPO / "scripts" / "next_step.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _expand(cell: str) -> list[str]:
    """`AT.1`–`AT.4`, `AT.6` → [AT.1, AT.2, AT.3, AT.4, AT.6]."""
    ids: list[str] = []
    rest = cell
    for m in RANGE.finditer(cell):
        lane_a, a, lane_b, b = m.groups()
        assert lane_a == lane_b, f"a range must stay in one lane: {m.group(0)}"
        ids.extend(f"{lane_a}.{n}" for n in range(int(a), int(b) + 1))
        rest = rest.replace(m.group(0), "", 1)
    ids.extend(f"{lane}.{n}" for lane, n in ID.findall(rest))
    return ids


def _table() -> dict[str, list[str]]:
    """``{phase letter: [ids in its contents cell]}``."""
    out: dict[str, list[str]] = {}
    for line in ROADMAP.read_text(encoding="utf-8").splitlines():
        m = PHASE_ROW.match(line)
        if m:
            out[m.group(1)] = _expand(m.group(2))
    return out


def test_the_table_was_found_and_covers_every_phase(ns) -> None:
    """Guards the guard: a reformatted table would make the assertions below vacuous."""
    table = _table()
    assert set(table) == set(ns.PHASE_RANK), f"phase table lists {sorted(table)}"


def test_every_step_is_in_exactly_one_phase_cell(ns) -> None:
    steps = ns.load(ROADMAP)
    seen: dict[str, list[str]] = {}
    for phase, ids in _table().items():
        for sid in ids:
            seen.setdefault(sid, []).append(phase)
    missing = sorted(set(steps) - set(seen))
    twice = sorted(sid for sid, phases in seen.items() if len(phases) > 1)
    assert not missing, f"steps in no phase cell: {missing}"
    assert not twice, f"steps in more than one phase cell: {twice}"


def test_the_cell_a_step_sits_in_is_its_own_phase(ns) -> None:
    steps = ns.load(ROADMAP)
    wrong = [
        f"{sid}: row says {steps[sid].phase}, table says {phase}"
        for phase, ids in _table().items()
        for sid in ids
        if sid in steps and steps[sid].phase != phase
    ]
    assert not wrong, "\n".join(wrong)


def test_nothing_in_a_cell_is_not_a_step(ns) -> None:
    steps = ns.load(ROADMAP)
    ghosts = sorted(sid for ids in _table().values() for sid in ids if sid not in steps)
    assert not ghosts, f"phase table names ids with no row: {ghosts}"


def test_a_range_expands_inclusively() -> None:
    """The one piece of parsing that could silently drop a step."""
    assert _expand("`AT.1`–`AT.4`, `AT.6`") == ["AT.1", "AT.2", "AT.3", "AT.4", "AT.6"]
    assert _expand("`PT.20`") == ["PT.20"]
