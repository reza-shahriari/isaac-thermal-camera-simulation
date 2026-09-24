"""The project site builds, and every link in it goes somewhere.

The site is 150-odd pages generated from documents that cross-reference each other heavily, and
the failure mode is not a crash: it is a page that builds fine and links to `/physics/`, which
works in a local preview and 404s once GitHub Pages serves the site from `/<repo>/`. So the tests
here are about the *links*, not about the prose:

* every internal href resolves to a file this build actually wrote;
* no href is root-absolute, because the published site lives under a subdirectory;
* the search index names pages that exist.

The build runs with `media=False`: encoding the gallery needs ffmpeg and a tree of renders that is
not in git, and neither belongs in the commit gate. The gallery's own behaviour when those renders
are absent -- list them rather than drop the section -- is asserted instead.
"""

from __future__ import annotations

import json
import pathlib
import posixpath
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

import build_site  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
HREF = re.compile(r'(?:href|src)="([^"]+)"')

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    out = tmp_path_factory.mktemp("site")
    build_site.build(out, REPO / "outputs", media=False)
    return out


def test_it_builds_a_page_for_every_document(site: pathlib.Path) -> None:
    pages = list(site.rglob("index.html"))
    adrs = list((REPO / "docs" / "decisions").glob("*.md"))
    assert len(pages) >= len(adrs) + 10
    assert (site / "index.html").is_file()
    assert (site / "physics" / "index.html").is_file()
    assert (site / "gallery" / "index.html").is_file()
    # GitHub Pages skips directories beginning with an underscore unless this file is present.
    assert (site / ".nojekyll").is_file()


def test_no_link_is_root_absolute(site: pathlib.Path) -> None:
    """Works locally, 404s under the `/<repo>/` prefix a project page is served from."""
    for page in site.rglob("*.html"):
        for href in HREF.findall(page.read_text(encoding="utf-8")):
            assert not href.startswith("/"), f"{page.relative_to(site)} links to {href}"


def test_every_internal_link_resolves(site: pathlib.Path) -> None:
    missing: list[str] = []
    for page in site.rglob("*.html"):
        here = page.parent
        for href in HREF.findall(page.read_text(encoding="utf-8")):
            if href.startswith(("http://", "https://", "mailto:", "#", "data:")):
                continue
            target = (here / posixpath.normpath(href.split("#")[0])).resolve()
            if href.endswith("/") or target.is_dir():
                target = target / "index.html"
            if not target.exists():
                missing.append(f"{page.relative_to(site)} -> {href}")
    assert not missing, "dead internal links: " + ", ".join(sorted(missing)[:10])


def test_the_search_index_names_pages_that_exist(site: pathlib.Path) -> None:
    index = json.loads((site / "search.json").read_text(encoding="utf-8"))
    assert len(index) >= 100
    urls = {entry["u"] for entry in index}
    assert "physics/" in urls and "gallery/" in urls and "" in urls
    for entry in index:
        assert (site / entry["u"] / "index.html").is_file()
    physics = next(entry for entry in index if entry["u"] == "physics/")
    assert any("Planck" in heading["t"] for heading in physics["h"])


def test_the_gallery_reports_renders_it_could_not_find(tmp_path: pathlib.Path) -> None:
    """With no renders to hand the section still publishes, naming what is missing."""
    out = tmp_path / "site"
    result = build_site.build(out, tmp_path / "no-renders-here", media=False)
    assert result["missing"] > 0
    page = (out / "gallery" / "index.html").read_text(encoding="utf-8")
    assert "not in this checkout" in page.lower()
    assert "quad_pointwise_vs_objectwise.mp4" in page


def test_the_front_page_counts_the_repository(tmp_path: pathlib.Path) -> None:
    """Every headline figure is measured at build time, so it cannot go stale in a commit."""
    out = tmp_path / "site"
    result = build_site.build(out, REPO / "outputs", media=False)
    home = (out / "index.html").read_text(encoding="utf-8")
    stats = result["stats"]
    assert stats["adrs"] == len(list((REPO / "docs" / "decisions").glob("*.md")))
    assert stats["modules"] > 100 and stats["tests"] > 1000
    for key in ("modules", "tests", "adrs", "configs"):
        assert f"{stats[key]:,}" in home, f"{key} is not on the front page"


def test_a_citation_in_the_specification_links_to_the_code(site: pathlib.Path) -> None:
    """A repository path that is not a rendered page becomes a link to the file on GitHub."""
    physics = (site / "physics" / "index.html").read_text(encoding="utf-8")
    assert "github.com/reza-shahriari/TCIsaacSim/blob/main/" in physics
