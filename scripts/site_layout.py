#!/usr/bin/env python3
"""The shell every page of the project site is poured into: nav, table of contents, styling.

Two constraints shape it.

**Every URL is relative.** GitHub project pages are served from a subdirectory
(`<user>.github.io/<repo>/`), so an absolute `/physics/` would 404 there while working perfectly in
a local preview -- the classic way a Pages site breaks only once published. Links are therefore
computed from the current page to the target with `posixpath.relpath`, which is correct at both
roots, and `tests/unit/test_site_build.py` asserts no built page contains a root-absolute one.

**No build step in the browser.** No framework, no bundler, one stylesheet and one small script.
The only external asset is KaTeX, loaded from a CDN to typeset the maths in
`docs/physics-model.md`; if it fails to load the equations degrade to their LaTeX source rather
than to nothing.
"""

from __future__ import annotations

import html
import pathlib
import posixpath
import subprocess
from dataclasses import dataclass, field

from site_markdown import Heading


#: The repository the "edit on GitHub" links point at. Derived from the checkout's own remote, so
#: a rename (this repository has had one) does not leave the published links pointing at the old
#: name and relying on GitHub's redirect.
def _origin_url() -> str:
    try:
        url = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
            cwd=pathlib.Path(__file__).resolve().parents[1],
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover -- not a checkout
        return DEFAULT_GITHUB
    if url.startswith("git@"):
        url = "https://" + url.removeprefix("git@").replace(":", "/", 1)
    return url.removesuffix(".git") or DEFAULT_GITHUB


DEFAULT_GITHUB = "https://github.com/reza-shahriari/isaac-thermal-camera-simulation"
GITHUB = _origin_url()


@dataclass
class Page:
    """One built page. `url` is the site-root-relative directory, e.g. `physics/`."""

    url: str
    title: str
    body: str
    nav_section: str = ""
    subtitle: str = ""
    headings: list[Heading] = field(default_factory=list)
    source: str | None = None
    wide: bool = False
    search_text: str = ""

    @property
    def out_path(self) -> str:
        return posixpath.join(self.url, "index.html") if self.url else "index.html"


@dataclass
class NavItem:
    label: str
    url: str
    note: str = ""


@dataclass
class NavGroup:
    label: str
    items: list[NavItem]


def relative(from_url: str, to_url: str) -> str:
    """A link from one page directory to another, as a relative URL.

    `from_url` and `to_url` are site-root-relative directories (`""` is the home page).
    """
    if to_url.startswith(("http://", "https://", "#", "mailto:")):
        return to_url
    fragment = ""
    if "#" in to_url:
        to_url, fragment = to_url.split("#", 1)
        fragment = "#" + fragment
    here = posixpath.join("/", from_url, "x")  # a fake file, so relpath works on directories
    there = posixpath.join("/", to_url)
    rel = posixpath.relpath(there, posixpath.dirname(here))
    if not to_url or to_url.endswith("/"):
        rel = rel.rstrip("/") + "/"
    return (rel if rel != "./" else "./") + fragment


def _document_title(page: Page) -> str:
    """The browser tab's title.

    The front page is already called irsim and does not need to say so twice.
    """
    if page.title.lower() == "irsim":
        return "irsim — physically-based multi-band IR camera simulator"
    return f"{html.escape(page.title)} — irsim"


def _toc(page: Page) -> str:
    """The in-page contents list, for documents long enough to need one."""
    entries = [h for h in page.headings if 2 <= h.level <= 3]
    if len(entries) < 4:
        return ""
    items = []
    for h in entries:
        cls = "toc-2" if h.level == 2 else "toc-3"
        items.append(f'<li class="{cls}"><a href="#{h.slug}">{html.escape(h.text)}</a></li>')
    body = "\n".join(items)
    return f'<nav class="toc" aria-label="On this page"><h2>On this page</h2><ul>{body}</ul></nav>'


def _nav(groups: list[NavGroup], page: Page) -> str:
    out = ['<nav class="sidebar" aria-label="Sections">']
    for group in groups:
        out.append(f"<h2>{html.escape(group.label)}</h2><ul>")
        for item in group.items:
            here = ' class="here"' if item.url == page.url else ""
            href = relative(page.url, item.url)
            note = f'<span class="note">{html.escape(item.note)}</span>' if item.note else ""
            out.append(f'<li{here}><a href="{href}">{html.escape(item.label)}</a>{note}</li>')
        out.append("</ul>")
    out.append("</nav>")
    return "\n".join(out)


def render_page(page: Page, groups: list[NavGroup], *, built: str, commit: str) -> str:
    """The finished HTML for one page."""
    rel = lambda url: relative(page.url, url)  # noqa: E731 -- a local alias, used a dozen times
    source = ""
    if page.source:
        source = (
            f'<a class="source" href="{GITHUB}/blob/main/{page.source}" '
            f'target="_blank" rel="noopener">{html.escape(page.source)} on GitHub</a>'
        )
    subtitle = f'<p class="subtitle">{page.subtitle}</p>' if page.subtitle else ""
    toc = _toc(page)
    main_class = "main wide" if page.wide else "main"
    if toc:
        main_class += " has-toc"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_document_title(page)}</title>
<meta name="description" content="{html.escape(page.subtitle[:180])}">
<link rel="stylesheet" href="{rel("assets/")}style.css">
<link rel="icon" href="{rel("assets/")}icon.svg" type="image/svg+xml">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
      integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV"
      crossorigin="anonymous">
</head>
<body>
<a class="skip" href="#content">Skip to content</a>
<header class="top">
  <a class="brand" href="{rel("")}"><span class="mark"></span>irsim</a>
  <form class="search" role="search" onsubmit="return false">
    <input id="q" type="search" placeholder="Search the project…" autocomplete="off"
           aria-label="Search" data-index="{rel("search.json")}">
    <div id="results" class="results" hidden></div>
  </form>
  <a class="ghlink" href="{GITHUB}" target="_blank" rel="noopener">GitHub</a>
</header>
<div class="shell">
{_nav(groups, page)}
<main id="content" class="{main_class}">
<div class="page-head">
  <h1>{html.escape(page.title)}</h1>
  {subtitle}
  {source}
</div>
{toc}
<article class="prose">
{page.body}
</article>
<footer class="foot">
  <p>Built {built} from <code>{commit}</code>. The physics specification is
  <a href="{rel("physics/")}">docs/physics-model.md</a>; every figure on this site was generated by
  the code in the repository.</p>
</footer>
</main>
</div>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"
        integrity="sha384-XjKyOOlGwcjNTAIQHIpgOno0Hl1YQqzUOEleOLALmuqehneUG+vnGctmUb0ZY0l8"
        crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
        integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05"
        crossorigin="anonymous"></script>
<script defer src="{rel("assets/")}site.js"></script>
</body>
</html>
"""
