#!/usr/bin/env python3
"""The pages of the site that are measured from the repository rather than written by hand.

A project site that is maintained separately from the project starts lying within a week. So the
module map, the configuration catalogue, the ADR index and the headline numbers on the front page
are all derived here, at build time, from the tree itself: module docstrings, the physics-model
section each file cites, the YAML headers, the roadmap's own step tables (via
`scripts/next_step.py`, the same parser `make check` runs). Nothing on those pages can disagree
with the repository, because nothing on them is stored.
"""

from __future__ import annotations

import ast
import collections
import html
import pathlib
import re
import subprocess
from dataclasses import dataclass

import next_step
import site_markdown
from site_layout import GITHUB, Page

#: `# docs/physics-model.md §3.2` -- the citation CLAUDE.md requires of every physics docstring.
CITATION = re.compile(r"§\s?([0-9]+(?:\.[0-9]+)*)")

PACKAGES = {
    "irsim": "the engine-free physics core — pure Python and NumPy, no engine import anywhere",
    "irsim_isaac": "the Isaac Sim glue — the only place an engine import is allowed",
    "irsim_eval": "evaluation and imaging: dataset readers, detection metrics, sim-to-real scoring",
}


@dataclass
class Adr:
    """One architecture decision record, as the index needs it."""

    number: str
    title: str
    status: str
    date: str
    source: str
    url: str


@dataclass
class Module:
    """One Python module in `src/`."""

    package: str
    dotted: str
    path: str
    summary: str
    lines: int
    sections: list[str]


def commit() -> str:
    """The commit the site was built from, short form, `unknown` outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def adr_meta(path: pathlib.Path, url: str) -> Adr:
    """Number, title, status and date, read from an ADR's own header."""
    text = path.read_text(encoding="utf-8")
    heading = re.search(r"^#\s*(?:ADR\s*)?(\d{4})\s*[—-]\s*(.+)$", text, re.MULTILINE)
    number = heading.group(1) if heading else path.name[:4]
    title = heading.group(2).strip() if heading else path.stem
    status = re.search(r"^\*\*Status:\*\*\s*(.+)$", text, re.MULTILINE)
    date = re.search(r"^\*\*Date:\*\*\s*(.+)$", text, re.MULTILINE)
    return Adr(
        number=number,
        title=title,
        status=status.group(1).strip() if status else "—",
        date=date.group(1).strip() if date else "—",
        source=f"docs/decisions/{path.name}",
        url=url,
    )


def modules(repo: pathlib.Path) -> list[Module]:
    """Every module under `src/`, with its docstring summary and the spec sections it cites."""
    found: list[Module] = []
    for package in PACKAGES:
        root = repo / "src" / package
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            try:
                doc = ast.get_docstring(ast.parse(text)) or ""
            except SyntaxError:  # pragma: no cover -- a broken file is not the site's problem
                doc = ""
            summary = doc.strip().split("\n\n")[0].replace("\n", " ").strip()
            rel = path.relative_to(repo).as_posix()
            dotted = path.relative_to(repo / "src").with_suffix("").as_posix().replace("/", ".")
            found.append(
                Module(
                    package=package,
                    dotted=dotted.removesuffix(".__init__"),
                    path=rel,
                    summary=summary,
                    lines=text.count("\n") + 1,
                    sections=sorted(set(CITATION.findall(text)), key=_section_key),
                )
            )
    return found


def _section_key(section: str) -> tuple[int, ...]:
    return tuple(int(part) for part in section.split("."))


def _inline(text: str) -> str:
    """A docstring summary, rendered as Markdown so its `code spans` read as code."""
    rendered = site_markdown.render(text).html
    return rendered[3:-4] if rendered.startswith("<p>") and rendered.endswith("</p>") else rendered


def _blob(path: str, label: str | None = None) -> str:
    return f'<a href="{GITHUB}/blob/main/{path}" target="_blank" rel="noopener">{label or path}</a>'


def code_map(repo: pathlib.Path) -> Page:
    """A page listing every module in `src/`, and what each test file is guarding."""
    mods = modules(repo)
    body: list[str] = [
        "<p>Every module under <code>src/</code>, its own one-line docstring, and the sections of "
        "the physics specification it cites. The layering rule that keeps the first column "
        "engine-free is not a convention here — <code>tests/unit/test_layering.py</code> fails the "
        "build if <code>irsim</code> imports an engine module.</p>"
    ]
    for package, blurb in PACKAGES.items():
        rows = [m for m in mods if m.package == package]
        if not rows:
            continue
        total = sum(m.lines for m in rows)
        body.append(f'<h2 id="{package}">{package}</h2>')
        body.append(f"<p>{html.escape(blurb)}. {len(rows)} modules, {total:,} lines.</p>")
        body.append("<div class='table-wrap'><table><thead><tr><th>module</th><th>what it is</th>")
        body.append("<th>spec §</th><th>lines</th></tr></thead><tbody>")
        for m in rows:
            sections = ", ".join(f"§{s}" for s in m.sections[:6]) or "—"
            body.append(
                f"<tr><td>{_blob(m.path, m.dotted)}</td>"
                f"<td>{_inline(m.summary) or '—'}</td>"
                f"<td class='nowrap'>{sections}</td><td>{m.lines:,}</td></tr>"
            )
        body.append("</tbody></table></div>")

    body.append('<h2 id="tests">tests</h2>')
    body.append(
        "<p>The gate is <code>make check</code>: ruff, mypy <code>--strict</code>, and the unit "
        "and "
        "golden suites, none of which need a GPU or Isaac Sim. Integration tests are marked "
        "<code>@pytest.mark.isaac</code> and skipped by default.</p>"
    )
    counts = test_counts(repo)
    body.append("<div class='table-wrap'><table><thead><tr><th>suite</th><th>files</th>")
    body.append("<th>test functions</th><th>what it covers</th></tr></thead><tbody>")
    for suite, (files, tests, note) in counts.items():
        body.append(
            f"<tr><td><code>{suite}</code></td><td>{files}</td><td>{tests}</td>"
            f"<td>{html.escape(note)}</td></tr>"
        )
    body.append("</tbody></table></div>")
    return Page(
        url="code/",
        title="Code map",
        subtitle=(
            "What is in <code>src/</code>, module by module, measured from the tree at build time."
        ),
        body="\n".join(body),
        nav_section="Reference",
        wide=True,
        search_text=" ".join(f"{m.dotted} {m.summary}" for m in mods),
    )


def test_counts(repo: pathlib.Path) -> dict[str, tuple[int, int, str]]:
    """Files and `def test_` counts per suite, with a one-line description of each."""
    notes = {
        "tests/unit": "fast, engine-free; the developer loop and the commit gate",
        "tests/golden": "regression fixtures: reference arrays with stated tolerances",
        "tests/integration": "needs a running Isaac Sim; marked `isaac` and skipped by default",
    }
    out: dict[str, tuple[int, int, str]] = {}
    for suite, note in notes.items():
        root = repo / suite
        if not root.is_dir():
            continue
        files = sorted(root.rglob("test_*.py"))
        tests = sum(
            len(re.findall(r"^\s*def test_", f.read_text(encoding="utf-8"), re.M)) for f in files
        )
        out[suite] = (len(files), tests, note)
    return out


def _yaml_headline(path: pathlib.Path) -> str:
    """The first comment line of a config file, which is where these files say what they are."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
        if line.strip():
            break
    return "—"


CONFIG_GROUPS = {
    "sensors": "One camera each: band, optics, detector, noise and ISP (physics-model §12.2).",
    "scenes": "A scene is a config, not code: geometry, weather, mission and what the camera does.",
    "materials": "Spectral material library. ε + ρ + τ = 1 holds in every band, to 1e-6.",
    "atmospheres": "Atmosphere presets — visibility, humidity, aerosol regime.",
    "environments": "Illumination and weather regimes a scene can name.",
    "thermal": "Thermal network and solver settings.",
    "assets": "Per-asset material mapping for an imported mesh.",
}


def config_catalogue(repo: pathlib.Path) -> Page:
    """Every YAML under `configs/`, grouped, with the headline comment each file carries."""
    body = [
        "<p>Everything the simulator renders is named by one of these files. Bands are data rather "
        "than code: adding a camera is a new file here, not an edit to the radiance kernel.</p>"
    ]
    for group, blurb in CONFIG_GROUPS.items():
        root = repo / "configs" / group
        if not root.is_dir():
            continue
        files = sorted(root.glob("*.yaml")) + sorted(root.glob("*.yml"))
        body.append(f'<h2 id="{group}">configs/{group}</h2>')
        body.append(f"<p>{html.escape(blurb)} {len(files)} file(s).</p>")
        body.append("<div class='table-wrap'><table><thead><tr><th>file</th><th>what it is</th>")
        body.append("</tr></thead><tbody>")
        for path in files:
            rel = path.relative_to(repo).as_posix()
            body.append(
                f"<tr><td class='nowrap'>{_blob(rel, path.name)}</td>"
                f"<td>{html.escape(_yaml_headline(path))}</td></tr>"
            )
        body.append("</tbody></table></div>")

    data_root = repo / "data"
    if data_root.is_dir():
        body.append('<h2 id="data">data/</h2>')
        body.append(
            "<p>Spectral response curves, optical constants, weather files and generated lookup "
            "tables. <code>$IRSIM_DATA_DIR</code> overrides the root. Generated artefacts (the "
            "band "
            "LUTs) are not in git — <code>make luts</code> rebuilds them from the configs and the "
            "curves, which is ADR 0012.</p>"
        )
        counts: collections.Counter[str] = collections.Counter()
        for path in data_root.rglob("*"):
            if path.is_file():
                counts[path.suffix or "(none)"] += 1
        body.append("<div class='table-wrap'><table><thead><tr><th>kind</th><th>files</th></tr>")
        body.append("</thead><tbody>")
        for suffix, count in counts.most_common(12):
            body.append(f"<tr><td><code>{html.escape(suffix)}</code></td><td>{count}</td></tr>")
        body.append("</tbody></table></div>")

    return Page(
        url="configs/",
        title="Configuration catalogue",
        subtitle=(
            "Cameras, scenes, materials and atmospheres — the data the simulator is driven by."
        ),
        body="\n".join(body),
        nav_section="Reference",
        wide=True,
    )


@dataclass
class Plan:
    """The roadmap, counted: what is shipped, what is open, and what comes next."""

    total: int
    done: int
    by_phase: list[tuple[str, int, int]]
    by_lane: list[tuple[str, str, int, int]]
    queue: list[tuple[str, str, str]]


def plan(repo: pathlib.Path) -> Plan | None:
    """Parse `docs/roadmap.md` with the project's own parser, or `None` if it cannot be read."""
    try:
        steps = next_step.load(repo / "docs" / "roadmap.md")
        order = next_step.order(steps)
    except (OSError, SystemExit):  # pragma: no cover -- a roadmap edit in flight, not a site bug
        return None
    phases: dict[str, list[next_step.Step]] = collections.defaultdict(list)
    lanes: dict[str, list[next_step.Step]] = collections.defaultdict(list)
    for step in steps.values():
        phases[step.phase].append(step)
        lanes[step.lane].append(step)
    by_phase = [
        (phase, sum(s.done for s in group), len(group))
        for phase, group in sorted(
            phases.items(), key=lambda kv: next_step.PHASE_RANK.get(kv[0], 9)
        )
    ]
    by_lane = [
        (lane, group[0].lane_name, sum(s.done for s in group), len(group))
        for lane, group in sorted(lanes.items())
    ]
    queue = [
        (sid, steps[sid].lane_name.split("(")[0].strip(), steps[sid].what) for sid in order[:8]
    ]
    return Plan(
        total=len(steps),
        done=sum(s.done for s in steps.values()),
        by_phase=by_phase,
        by_lane=by_lane,
        queue=queue,
    )
