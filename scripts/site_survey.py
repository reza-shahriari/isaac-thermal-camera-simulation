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
from site_layout import GITHUB, Page, relative

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


def _first_paragraph(doc: str) -> str:
    """A docstring's first paragraph on one line — what a summary column can hold."""
    return doc.strip().split("\n\n")[0].replace("\n", " ").strip()


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


# ---------------------------------------------------------------------------------------------
# the test suite, presented


#: Roadmap step ids (`PT.9`, `M12.2`), ADR numbers and physics-model sections, wherever a test
#: names one. These are the threads a reader follows from a test back to the thing it guards.
REFERENCE = re.compile(
    r"\bADR\s?(\d{4})\b|\b([A-Z]{2}|M)\d{1,2}(?:\.\d{1,2}[a-z]?)?\b|§\s?([\d.]+)"
)

SUITES = {
    "tests/unit": (
        "Unit",
        "Fast, engine-free, and the half of the commit gate that runs on every change. No GPU, no "
        "Isaac Sim, no network.",
    ),
    "tests/golden": (
        "Golden",
        "Regression fixtures: a reference array plus a JSON sidecar carrying the config hash, and "
        "a stated tolerance in physical units (ADR 0004). Regenerated deliberately, never to "
        "silence a failure.",
    ),
    "tests/integration": (
        "Integration",
        "Needs the engine. Marked `isaac` (a running Isaac Sim) or `gpu` (Warp and a CUDA device) "
        "and skipped by default, so the gate never depends on hardware.",
    ),
}


@dataclass
class TestFn:
    """One test function."""

    name: str
    line: int
    summary: str
    marks: list[str]

    @property
    def sentence(self) -> str:
        """The name as the sentence it was written to be."""
        return self.name.removeprefix("test_").replace("_", " ")


@dataclass
class TestModule:
    """One test file: what it guards, and every test in it."""

    suite: str
    path: str
    url: str
    title: str
    doc: str
    summary: str
    marks: list[str]
    tests: list[TestFn]
    refs: list[str]


def _marks(decorators: list[ast.expr]) -> list[str]:
    found = []
    for node in decorators:
        target = node.func if isinstance(node, ast.Call) else node
        name = ast.unparse(target)
        if name.startswith("pytest.mark."):
            found.append(name.removeprefix("pytest.mark."))
    return found


def _module_marks(tree: ast.Module) -> list[str]:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            text = ast.unparse(node.value)
            return [part.split("pytest.mark.")[-1] for part in text.split(",") if "mark." in part]
    return []


def _references(text: str) -> list[str]:
    """The roadmap steps, ADRs and spec sections a test file names, in the order they appear."""
    seen: list[str] = []
    for match in REFERENCE.finditer(text):
        if match.group(1):
            token = f"ADR {match.group(1)}"
        elif match.group(3):
            token = f"§{match.group(3)}"
        else:
            token = match.group(0)
        if token not in seen:
            seen.append(token)
    return seen


def test_modules(repo: pathlib.Path) -> list[TestModule]:
    """Every test file, its docstring, and every test function inside it."""
    found: list[TestModule] = []
    for suite in SUITES:
        root = repo / suite
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("test_*.py")):
            text = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(text)
            except SyntaxError:  # pragma: no cover -- a broken file is not the site's problem
                continue
            doc = ast.get_docstring(tree) or ""
            tests = [
                TestFn(
                    name=node.name,
                    line=node.lineno,
                    summary=_first_paragraph(ast.get_docstring(node) or ""),
                    marks=_marks(node.decorator_list),
                )
                for node in tree.body
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                and node.name.startswith("test_")
            ]
            rel = path.relative_to(repo).as_posix()
            found.append(
                TestModule(
                    suite=suite,
                    path=rel,
                    url=f"{rel.removesuffix('.py')}/",
                    title=path.stem.removeprefix("test_").replace("_", " "),
                    doc=doc,
                    summary=doc.strip().split("\n\n")[0].replace("\n", " "),
                    marks=_module_marks(tree),
                    tests=tests,
                    refs=_references(doc),
                )
            )
    return found


def test_index(modules: list[TestModule]) -> Page:
    """One page listing every test file in the repository, grouped by suite."""
    total = sum(len(m.tests) for m in modules)
    body: list[str] = [
        "<p>Nobody has intuition for what a correct thermal image looks like — a frame that is "
        "wrong by 20 K looks exactly as convincing as one that is right. That is why this project "
        f"carries {total:,} tests across {len(modules)} files, and why a test here is expected "
        "to <em>fail if the physics were wrong</em> rather than merely to call the function. Each "
        "file below links to its own page, where every test is listed with what it asserts.</p>",
        '<p class="filterline"><input id="table-filter" type="search" '
        f'placeholder="Filter {len(modules)} test files…" aria-label="Filter test files"></p>',
    ]
    for suite, (label, blurb) in SUITES.items():
        rows = [m for m in modules if m.suite == suite]
        if not rows:
            continue
        count = sum(len(m.tests) for m in rows)
        body.append(f'<h2 id="{label.lower()}">{label} — <code>{suite}</code></h2>')
        body.append(f"<p>{_inline(blurb)} {len(rows)} files, {count:,} tests.</p>")
        body.append(
            "<div class='table-wrap'><table class=\"filterable suite\"><thead><tr><th>file</th>"
            "<th>what it guards</th><th>cites</th><th>tests</th></tr></thead><tbody>"
        )
        for module in sorted(rows, key=lambda m: m.path):
            marks = "".join(f'<span class="tag">{m}</span>' for m in module.marks)
            refs = ", ".join(module.refs[:4]) or "—"
            name = pathlib.PurePosixPath(module.path).name
            body.append(
                f'<tr><td><a href="{relative("tests/", module.url)}">{name}</a>{marks}</td>'
                f"<td>{_inline(module.summary) or '—'}</td>"
                f"<td>{html.escape(refs)}</td><td>{len(module.tests)}</td></tr>"
            )
        body.append("</tbody></table></div>")
    return Page(
        url="tests/",
        title="The test suite",
        subtitle=(
            f"{total:,} tests across {len(modules)} files — every one listed, with what it asserts."
        ),
        body="\n".join(body),
        nav_section="Evidence",
        wide=True,
        search_text=" ".join(f"{m.title} {m.summary}" for m in modules),
    )


def test_page(module: TestModule) -> Page:
    """One test file's own page: its docstring, then every test it contains."""
    body: list[str] = []
    # The first paragraph is already the page's subtitle; repeating it under the heading reads
    # like a stutter, so the body starts at the second.
    rest = module.doc.strip().split("\n\n", 1)
    if len(rest) > 1 and rest[1].strip():
        body.append(site_markdown.render(rest[1]).html)
    marks = ", ".join(module.marks)
    note = f" The whole file is marked <code>{html.escape(marks)}</code>." if marks else ""
    body.append(
        f"<p class='made-by'>{len(module.tests)} test(s) in {_blob(module.path)}.{note}</p>"
    )
    body.append(
        "<div class='table-wrap'><table><thead><tr><th>test</th><th>what it asserts</th>"
        "</tr></thead><tbody>"
    )
    for test in module.tests:
        marks = "".join(f'<span class="tag">{m.split("(")[0]}</span>' for m in test.marks)
        link = f"{GITHUB}/blob/main/{module.path}#L{test.line}"
        body.append(
            f'<tr><td><a href="{link}" target="_blank" rel="noopener">{html.escape(test.sentence)}'
            f"</a>{marks}<br><code>{html.escape(test.name)}</code></td>"
            f"<td>{_inline(test.summary) or '—'}</td></tr>"
        )
    body.append("</tbody></table></div>")
    return Page(
        url=module.url,
        title=module.title[:1].upper() + module.title[1:],
        subtitle=module.summary,
        body="\n".join(body),
        nav_section="Evidence",
        source=module.path,
        wide=True,
        search_text=" ".join(f"{t.sentence} {t.summary}" for t in module.tests),
    )


def _dedent_usage(block: str) -> str:
    """Line up a usage block whose first line the docstring parser already un-indented.

    `ast.get_docstring` dedents against the lines *after* the first, so a block written as a list
    of commands comes back with its first command flush and the rest indented. Only lines that are
    themselves commands are moved; a continuation line aligned under an argument stays where the
    author put it.
    """
    lines = block.splitlines()
    rest = [line for line in lines[1:] if line.strip()]
    if not rest or not all(line.strip().startswith(("python", "$")) for line in rest):
        return block
    indent = min(len(line) - len(line.lstrip()) for line in rest)
    return "\n".join([lines[0].strip(), *(line[indent:] for line in lines[1:])])


def script_catalogue(repo: pathlib.Path) -> Page:
    """Every command in `scripts/`, with the first paragraph of its own docstring."""
    rows: list[tuple[str, str, str]] = []
    for path in sorted((repo / "scripts").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        try:
            doc = ast.get_docstring(ast.parse(text)) or ""
        except SyntaxError:  # pragma: no cover
            doc = ""
        blocks = [b for b in doc.strip().split("\n\n") if b.strip()]
        summary = blocks[0].replace("\n", " ") if blocks else ""
        usage = ""
        for block in blocks[:3]:
            if block.lstrip().startswith(("python ", "    python", "$ python")):
                usage = _dedent_usage(block)
                break
        rows.append((path.relative_to(repo).as_posix(), summary, usage))
    for path in sorted((repo / "scripts").glob("*.sh")):
        head = [
            line.lstrip("# ").rstrip()
            for line in path.read_text(encoding="utf-8").splitlines()[1:6]
            if line.startswith("#")
        ]
        rows.append((path.relative_to(repo).as_posix(), " ".join(head), ""))

    body = [
        "<p>Everything this repository can be asked to do from a shell. The render drivers need "
        "Isaac Sim and a GPU; everything else is engine-free and runs on any CPython with the dev "
        "extras installed.</p>",
        "<div class='table-wrap'><table><thead><tr><th>command</th><th>what it does</th>"
        "</tr></thead><tbody>",
    ]
    for rel, summary, usage in rows:
        name = pathlib.PurePosixPath(rel).name
        block = f"<pre><code>{html.escape(usage.strip())}</code></pre>" if usage else ""
        body.append(
            f'<tr><td class="nowrap">{_blob(rel, name)}</td>'
            f"<td>{_inline(summary) or '—'}{block}</td></tr>"
        )
    body.append("</tbody></table></div>")
    return Page(
        url="scripts/",
        title="Commands",
        subtitle="Every script in the repository, with the first line of its own docstring.",
        body="\n".join(body),
        nav_section="Reference",
        wide=True,
        search_text=" ".join(f"{rel} {summary}" for rel, summary, _ in rows),
    )
