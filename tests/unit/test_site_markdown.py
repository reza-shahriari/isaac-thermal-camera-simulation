"""The site's Markdown renderer, against the constructs this repository's documents actually use.

Two kinds of test here. The first are unit tests of constructs where a generic renderer is known
to get IR documentation wrong -- `$\\tau_{\\text{opt}}$` mangled into emphasis by the underscores
inside it, a code span emphasised because it contains an asterisk, a section anchor swallowed by
the `<a name=...>` line above it.

The second is the corpus test: every Markdown file in the repository is rendered and the output is
checked for the symptoms of a construct the renderer does not know -- a leftover extraction
sentinel, or link syntax that reached the page as literal text. That is what stops the site from
quietly publishing `[see §3.2](docs/physics-model.md)` as prose.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from site_markdown import render, slugify  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_heading_slugs_match_the_anchors_the_documents_link_to() -> None:
    # The documents cross-link by GitHub's slug; if ours differs, every such link 404s in-page.
    assert slugify("3. Radiometry core: Planck, band integration, inversion") == (
        "3-radiometry-core-planck-band-integration-inversion"
    )
    assert slugify("**L2** — the target") == "l2--the-target"
    assert slugify("Kirchhoff's law") == "kirchhoffs-law"


def test_repeated_headings_get_distinct_ids() -> None:
    doc = render("## Context\n\ntext\n\n## Context\n\nmore\n")
    assert [h.slug for h in doc.headings] == ["context", "context-1"]


def test_inline_maths_survives_intact() -> None:
    # The failure this pins: `\tau_{\text{opt}}` has two underscores, which a naive emphasis pass
    # turns into <em>, and KaTeX then receives something that is no longer the equation.
    doc = render(r"the factor $\tau_{\text{opt}}/(4F^2+1)$ closes the units")
    assert r"$\tau_{\text{opt}}/(4F^2+1)$" in doc.html
    assert "<em>" not in doc.html


def test_display_maths_is_passed_through_as_a_block() -> None:
    doc = render("before\n\n$$\nL_{\\text{sensor}} = \\varepsilon B(T_s)\n$$\n\nafter\n")
    assert '<div class="math-block">' in doc.html
    assert "L_{\\text{sensor}}" in doc.html


def test_a_lone_dollar_is_not_treated_as_maths() -> None:
    doc = render("costs $5 and $10, not an equation")
    assert "$5 and $10" in doc.html


def test_code_spans_are_not_emphasised_and_are_escaped() -> None:
    doc = render("use `a*b*c` and `x < y` here")
    assert "<code>a*b*c</code>" in doc.html
    assert "<code>x &lt; y</code>" in doc.html


def test_a_link_whose_label_is_a_code_span_renders_both() -> None:
    # Regression: the label's code span is extracted first, so the finished <a> holds a sentinel
    # that a single substitution pass leaves in the page as an invisible control character.
    doc = render("see [`docs/physics-model.md`](../physics/) for the equations")
    assert '<a href="../physics/"><code>docs/physics-model.md</code></a>' in doc.html
    assert "\x00" not in doc.html


def test_links_are_resolved_through_the_callback() -> None:
    doc = render("[the spec](docs/physics-model.md#3-radiometry-core)", lambda href: "/PHYS" + href)
    assert 'href="/PHYSdocs/physics-model.md#3-radiometry-core"' in doc.html


def test_tables_carry_alignment_and_render_their_cells() -> None:
    doc = render("| a | b |\n|:--|--:|\n| `x` | **y** |\n")
    assert "<table>" in doc.html
    assert 'style="text-align:right"' in doc.html
    assert "<code>x</code>" in doc.html
    assert "<strong>y</strong>" in doc.html


def test_an_html_anchor_above_a_heading_does_not_swallow_it() -> None:
    # docs/physics-model.md writes `<a name="7-atmosphere"></a>` on the line directly above the
    # heading, with no blank line. Consuming to the next blank line eats the heading with it.
    doc = render('<a name="7-atmosphere"></a>\n## 7. Atmosphere\n\nbody\n')
    assert '<a id="7-atmosphere"></a>' in doc.html  # name= is dead in HTML5; id= is the anchor
    assert "<h2" in doc.html
    assert [h.text for h in doc.headings] == ["7. Atmosphere"]


def test_nested_lists_nest() -> None:
    doc = render("- one\n  - inner\n- two\n")
    assert doc.html.count("<ul>") == 2
    assert "inner" in doc.html


def test_fenced_code_keeps_its_language_and_escapes_its_body() -> None:
    doc = render("```python\nif a < b:\n    pass\n```\n")
    assert '<code class="language-python">' in doc.html
    assert "a &lt; b" in doc.html


def corpus() -> list[Path]:
    paths = [REPO / "README.md", REPO / "CLAUDE.md", REPO / "CHANGELOG.md"]
    paths += sorted(REPO.joinpath("docs").rglob("*.md"))
    return [p for p in paths if p.is_file()]


def test_every_document_in_the_repository_renders_without_leftovers() -> None:
    """No sentinel escapes, and no link or image syntax reaches the page as literal text."""
    literal_link = re.compile(r"\[[^\]\n]{1,80}\]\((?!\s)[^)\n]{1,120}\)")
    for path in corpus():
        doc = render(path.read_text(encoding="utf-8"))
        assert "\x00" not in doc.html, f"unsubstituted sentinel in {path}"
        # Link syntax survives legitimately inside code blocks; strip those before looking.
        prose = re.sub(r"<pre><code.*?</code></pre>", "", doc.html, flags=re.S)
        prose = re.sub(r"<code>.*?</code>", "", prose, flags=re.S)
        prose = re.sub(r'<div class="math-block">.*?</div>', "", prose, flags=re.S)
        leftover = literal_link.search(prose)
        assert leftover is None, f"unrendered link in {path}: {leftover.group(0)!r}"


def test_the_specification_renders_its_structure() -> None:
    """The physics model is the document the site exists to publish; assert it arrives whole."""
    doc = render((REPO / "docs" / "physics-model.md").read_text(encoding="utf-8"))
    assert doc.title is not None and "Infrared Camera Model" in doc.title
    assert len([h for h in doc.headings if h.level == 2]) >= 17  # §1..§17
    assert doc.html.count("<table>") >= 15
    assert doc.html.count('<div class="math-block">') >= 20
