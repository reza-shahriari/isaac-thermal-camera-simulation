#!/usr/bin/env python3
"""Markdown to HTML, with no dependency outside the standard library.

The project site (`scripts/build_site.py`) renders the documents that already exist in this repo --
`docs/physics-model.md`, the 130-odd ADRs, the roadmap, the validation reports -- rather than
keeping a second copy of them in HTML. That needs a Markdown renderer, and the two obvious ways to
get one are both worse than writing it:

* a pip dependency (`markdown-it-py`, `mistune`) is not installed in the project interpreter
  (Isaac Sim's bundled Python, ADR 0002) and would make `make site` fail on a fresh checkout for a
  reason that has nothing to do with the simulator;
* GitHub's own Jekyll build would render the Markdown but cannot reach `outputs/`, which is
  gitignored and where every frame and clip on the site comes from.

So this is a deliberately small renderer covering exactly what these documents use, which is
CommonMark plus GitHub tables, checked against the real corpus by
`tests/unit/test_site_markdown.py`:

    headings (with GitHub-compatible anchor slugs)   fenced code, with a language class
    paragraphs, hard breaks                          GitHub pipe tables, with alignment
    bullet and ordered lists, nested by indent       block quotes
    `code`, **bold**, *italic*, [links](), images    thematic breaks
    raw HTML blocks and inline tags, passed through  $inline$ and $$display$$ maths, passed through

Maths is *not* rendered here. `$...$` and `$$...$$` spans are passed through untouched (their
contents escaped for HTML, which is what KaTeX's auto-render reads) and typeset in the browser.
The point of extracting them first is that `\tau_{\text{opt}}` must not be mangled into emphasis
by the underscores it contains, which is the failure mode of rendering maths-bearing Markdown with
a generic renderer.

Anything the corpus does not use -- reference-style links, footnotes, setext headings, HTML
comments spanning blocks -- is not supported, and `tests/unit/test_site_markdown.py` asserts the
corpus stays inside the supported set rather than letting an unsupported construct render as
literal text on the site.
"""

from __future__ import annotations

import html
import re
from collections.abc import Callable
from dataclasses import dataclass, field

#: The sentinel that guards an extracted span (code, maths, a finished link) from every later
#: regex. \x00 cannot occur in a Markdown source file, so there is nothing to escape.
_MARK = "\x00{}\x00"
_MARK_RE = re.compile("\x00(\\d+)\x00")


@dataclass
class Heading:
    """One heading, for the page's table of contents."""

    level: int
    slug: str
    text: str


@dataclass
class Document:
    """A rendered document: its HTML body and the headings found in it."""

    html: str
    headings: list[Heading] = field(default_factory=list)
    title: str | None = None


def slugify(text: str) -> str:
    """GitHub's anchor slug: lowercase, punctuation dropped, spaces to hyphens.

    Matching GitHub matters because the documents link to each other by anchors that were written
    against GitHub's rendering -- `docs/physics-model.md#3-radiometry-core` has to keep working.
    """
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[`*_]", "", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    # One hyphen per space, not one per run: GitHub turns "**L2** — the target" into
    # "l2--the-target" because the em dash is dropped and both spaces survive as hyphens.
    return re.sub(r"\s", "-", text).strip("-")


class _Inline:
    """Inline rendering, with every span that must not be touched extracted first."""

    def __init__(self, resolve: Callable[[str], str] | None = None) -> None:
        self._resolve = resolve or (lambda href: href)
        self._parts: list[str] = []

    def _stash(self, rendered: str) -> str:
        self._parts.append(rendered)
        return _MARK.format(len(self._parts) - 1)

    def _unstash(self, text: str) -> str:
        """Put the extracted spans back, repeatedly: a link's label may itself hold one.

        `[`docs/physics-model.md`](../physics/)` is the common case -- the label is a code span,
        so the finished `<a>` contains a placeholder of its own and one pass leaves it in the page.
        """
        for _ in range(4):
            if not _MARK_RE.search(text):
                break
            text = _MARK_RE.sub(lambda m: self._parts[int(m.group(1))], text)
        return text

    def render(self, text: str) -> str:
        self._parts = []
        text = self._extract_code(text)
        text = self._extract_math(text)
        text = html.escape(text, quote=False)
        text = self._extract_media(text)
        text = self._emphasis(text)
        text = self._autolink(text)
        text = text.replace("  \n", "<br>\n")
        return self._unstash(text)

    def _extract_code(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            body = html.escape(match.group(2).strip(), quote=False)
            return self._stash(f"<code>{body}</code>")

        return re.sub(r"(`+)(.+?)\1", repl, text, flags=re.DOTALL)

    def _extract_math(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            return self._stash(html.escape(match.group(0), quote=False))

        # `$$..$$` first so the greedy single-dollar pattern cannot split one in half. A lone `$`
        # (a price, a shell prompt) is left alone: both patterns need a closing delimiter on the
        # same line and no space just inside it, which is the usual heuristic.
        text = re.sub(r"\$\$(?:[^$]|\$(?!\$))+\$\$", repl, text)
        return re.sub(r"(?<![\w$])\$(?!\s)((?:[^$\n]|\\\$)+?)(?<!\s)\$(?![\w$])", repl, text)

    def _extract_media(self, text: str) -> str:
        """Images, then links. Both become finished HTML so `_autolink` cannot reach inside them."""

        def image(match: re.Match[str]) -> str:
            alt, href = match.group(1), self._resolve(match.group(2))
            return self._stash(
                f'<img src="{html.escape(href)}" alt="{html.escape(alt)}" loading="lazy">'
            )

        def link(match: re.Match[str]) -> str:
            label, href = match.group(1), self._resolve(match.group(2))
            inner = self._emphasis(label)
            extra = (
                ' target="_blank" rel="noopener"'
                if href.startswith(("http://", "https://"))
                else ""
            )
            return self._stash(f'<a href="{html.escape(href)}"{extra}>{inner}</a>')

        text = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+&quot;[^&]*&quot;)?\)", image, text)
        return re.sub(r"\[([^\]]+)\]\(([^)\s]*)(?:\s+&quot;[^&]*&quot;)?\)", link, text)

    def _emphasis(self, text: str) -> str:
        text = re.sub(r"\*\*(?=\S)(.+?)(?<=\S)\*\*", r"<strong>\1</strong>", text)
        text = re.sub(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])", r"<em>\1</em>", text)
        text = re.sub(r"(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])", r"<em>\1</em>", text)
        return re.sub(r"(?<!\w)~~(?=\S)(.+?)(?<=\S)~~", r"<del>\1</del>", text)

    def _autolink(self, text: str) -> str:
        def repl(match: re.Match[str]) -> str:
            url = match.group(0).rstrip(".,;:")
            tail = match.group(0)[len(url) :]
            link = f'<a href="{url}" target="_blank" rel="noopener">{url}</a>'
            return self._stash(link) + tail

        return re.sub(r"(?<![\"'=>])\bhttps?://[^\s<>)\]]+", repl, text)


class _Renderer:
    """Block-level rendering. One pass over the lines, recursing for quotes and list items."""

    def __init__(self, resolve: Callable[[str], str] | None = None) -> None:
        self.inline = _Inline(resolve)
        self.headings: list[Heading] = []
        self._seen_slugs: dict[str, int] = {}

    # -- helpers ---------------------------------------------------------------------------

    def _slug(self, text: str) -> str:
        """A slug, made unique the way GitHub does it: `-1`, `-2`, ... on a repeat."""
        base = slugify(text)
        count = self._seen_slugs.get(base, 0)
        self._seen_slugs[base] = count + 1
        return base if count == 0 else f"{base}-{count}"

    @staticmethod
    def _indent(line: str) -> int:
        return len(line) - len(line.lstrip(" "))

    # -- the block loop --------------------------------------------------------------------

    def render(self, text: str) -> str:
        lines = text.replace("\r\n", "\n").replace("\t", "    ").split("\n")
        return self._blocks(lines)

    def _blocks(self, lines: list[str]) -> str:
        out: list[str] = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            for handler in (
                self._fence,
                self._math_block,
                self._heading,
                self._rule,
                self._table,
                self._quote,
                self._list,
                self._raw_html,
            ):
                taken = handler(lines, i)
                if taken is not None:
                    chunk, i = taken
                    out.append(chunk)
                    break
            else:
                chunk, i = self._paragraph(lines, i)
                out.append(chunk)
        return "\n".join(out)

    # -- block handlers --------------------------------------------------------------------

    def _fence(self, lines: list[str], i: int) -> tuple[str, int] | None:
        match = re.match(r"^\s*(`{3,}|~{3,})\s*([\w+-]*)\s*$", lines[i])
        if not match:
            return None
        marker, lang = match.group(1)[0] * 3, match.group(2)
        body: list[str] = []
        i += 1
        while i < len(lines) and not lines[i].strip().startswith(marker):
            body.append(lines[i])
            i += 1
        i += 1  # the closing fence
        code = html.escape("\n".join(body), quote=False)
        cls = f' class="language-{lang}"' if lang else ""
        return f"<pre><code{cls}>{code}</code></pre>", i

    def _math_block(self, lines: list[str], i: int) -> tuple[str, int] | None:
        if lines[i].strip() != "$$":
            return None
        body = [lines[i]]
        i += 1
        while i < len(lines):
            body.append(lines[i])
            if lines[i].strip() == "$$":
                i += 1
                break
            i += 1
        escaped = html.escape("\n".join(body), quote=False)
        return f'<div class="math-block">{escaped}</div>', i

    def _heading(self, lines: list[str], i: int) -> tuple[str, int] | None:
        match = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", lines[i])
        if not match:
            return None
        level, raw = len(match.group(1)), match.group(2)
        slug = self._slug(raw)
        self.headings.append(Heading(level, slug, re.sub(r"[`*_]", "", raw)))
        body = self.inline.render(raw)
        anchor = f'<a class="anchor" href="#{slug}" aria-label="permalink">#</a>'
        return f'<h{level} id="{slug}">{body}{anchor}</h{level}>', i + 1

    def _rule(self, lines: list[str], i: int) -> tuple[str, int] | None:
        if re.match(r"^\s*([-*_])(\s*\1){2,}\s*$", lines[i]):
            return "<hr>", i + 1
        return None

    def _table(self, lines: list[str], i: int) -> tuple[str, int] | None:
        if i + 1 >= len(lines) or "|" not in lines[i]:
            return None
        if not re.match(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", lines[i + 1]) or "-" not in lines[i + 1]:
            return None

        def cells(row: str) -> list[str]:
            row = row.strip()
            row = row[1:] if row.startswith("|") else row
            row = row[:-1] if row.endswith("|") else row
            return [c.strip() for c in re.split(r"(?<!\\)\|", row)]

        aligns = []
        for spec in cells(lines[i + 1]):
            left, right = spec.startswith(":"), spec.endswith(":")
            aligns.append(
                "center" if left and right else "right" if right else "left" if left else ""
            )
        header = cells(lines[i])
        rows: list[list[str]] = []
        j = i + 2
        while j < len(lines) and "|" in lines[j] and lines[j].strip():
            rows.append(cells(lines[j]))
            j += 1

        def cell(tag: str, text: str, col: int) -> str:
            align = aligns[col] if col < len(aligns) else ""
            style = f' style="text-align:{align}"' if align else ""
            return f"<{tag}{style}>{self.inline.render(text.replace(chr(92) + '|', '|'))}</{tag}>"

        head = "".join(cell("th", c, n) for n, c in enumerate(header))
        body = "\n".join(
            "<tr>" + "".join(cell("td", c, n) for n, c in enumerate(row)) + "</tr>" for row in rows
        )
        return (
            f"<div class='table-wrap'><table>\n<thead><tr>{head}</tr></thead>\n"
            f"<tbody>\n{body}\n</tbody></table></div>",
            j,
        )

    def _quote(self, lines: list[str], i: int) -> tuple[str, int] | None:
        if not lines[i].lstrip().startswith(">"):
            return None
        body: list[str] = []
        while i < len(lines) and (lines[i].lstrip().startswith(">") or lines[i].strip()):
            stripped = lines[i].lstrip()
            body.append(re.sub(r"^>\s?", "", stripped) if stripped.startswith(">") else stripped)
            i += 1
        return f"<blockquote>\n{self._blocks(body)}\n</blockquote>", i

    def _list(self, lines: list[str], i: int) -> tuple[str, int] | None:
        match = re.match(r"^(\s*)([-*+]|\d+[.)])\s+", lines[i])
        if not match:
            return None
        indent = len(match.group(1))
        ordered = match.group(2)[0].isdigit()
        items: list[list[str]] = []
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                # A blank line ends the list unless the next line continues an item.
                if (
                    i + 1 < len(lines)
                    and self._indent(lines[i + 1]) > indent
                    and lines[i + 1].strip()
                ):
                    items[-1].append("")
                    i += 1
                    continue
                break
            item = re.match(rf"^ {{{indent}}}([-*+]|\d+[.)])\s+(.*)$", line)
            if item:
                items.append([item.group(2)])
            elif self._indent(line) > indent and items:
                items[-1].append(line[indent + 2 :] if len(line) > indent + 2 else line.strip())
            else:
                break
            i += 1
        rendered = []
        for item in items:
            if len(item) == 1:
                rendered.append(f"<li>{self.inline.render(item[0])}</li>")
            else:
                inner = self._blocks([item[0], *item[1:]])
                # A one-paragraph item reads better without the <p>; keep it for anything richer.
                if inner.startswith("<p>") and inner.count("<p>") == 1 and inner.endswith("</p>"):
                    inner = inner[3:-4]
                rendered.append(f"<li>{inner}</li>")
        tag = "ol" if ordered else "ul"
        body = "\n".join(rendered)
        return f"<{tag}>\n{body}\n</{tag}>", i

    def _raw_html(self, lines: list[str], i: int) -> tuple[str, int] | None:
        """A raw HTML block, passed through.

        It ends at a blank line *or* at the first line that starts another Markdown block, because
        `docs/physics-model.md` writes its section anchors as `<a name="3-radiometry-core"></a>`
        immediately above the `##` heading they belong to, with no blank line between them.
        `name=` is rewritten to `id=` on the way through: HTML5 dropped the `name` anchor and only
        the id form is guaranteed to be a scroll target.
        """
        if not re.match(r"^\s*<(/?[a-zA-Z][\w-]*|!--)", lines[i]):
            return None
        body = [lines[i]]
        i += 1
        while i < len(lines) and lines[i].strip():
            if re.match(r"^\s*(#{1,6}\s|```|~~~|>|([-*+]|\d+[.)])\s)", lines[i]):
                break
            body.append(lines[i])
            i += 1
        chunk = "\n".join(body)
        return re.sub(r"(<a\s+)name=", r"\1id=", chunk), i

    def _paragraph(self, lines: list[str], i: int) -> tuple[str, int]:
        body: list[str] = []
        while i < len(lines) and lines[i].strip():
            if re.match(r"^(#{1,6})\s|^\s*(```|~~~)|^\s*>", lines[i]) and body:
                break
            if re.match(r"^\s*([-*+]|\d+[.)])\s+", lines[i]) and body:
                break
            if lines[i].strip() == "$$" and body:
                break
            body.append(lines[i].strip())
            i += 1
        return f"<p>{self.inline.render(' '.join(body))}</p>", i


def render(text: str, resolve: Callable[[str], str] | None = None) -> Document:
    """Render `text`, resolving every link through `resolve` (repo path -> site URL)."""
    renderer = _Renderer(resolve)
    body = renderer.render(text)
    title = next((h.text for h in renderer.headings if h.level == 1), None)
    return Document(html=body, headings=renderer.headings, title=title)
