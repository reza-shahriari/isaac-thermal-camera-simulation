# ADR 0139 — The project site is generated from the repository, and carries web-sized renders

**Status:** Accepted
**Date:** 2026-09-24

## Context

The owner asked for a GitHub Pages site showing the whole project: the physics specification, the
plan, the decision log, the validation reports, the code and — the part that cannot be read out of
the repository — what the simulator actually produces.

Three constraints shaped the answer.

**The documents already exist and are the source of truth.** `docs/physics-model.md` is 114 kB of
specification with LaTeX in it, the roadmap is generated in part by `scripts/next_step.py`, the
status table lives in `README.md` and there are 137 ADRs. A site that *copies* any of that starts
lying the first time somebody edits the original, and this repository has several sessions editing
it at once.

**The renders are not in git and cannot be.** `outputs/` is 8.3 GB. The clips the existing gallery
referenced come to 609 MB on their own — the drone NIR clip is 97 MB at 163 Mbit/s, which is what
you want for analysis and is absurd for a web page. GitHub Pages publishes about 1 GB with a 100 MB
per-file limit, and the `.npy` planes beside those clips are of no use to a reader at all.

**The project interpreter is Isaac Sim's bundled Python** (ADR 0002), which has no Markdown library
and no static-site generator, and `make check` must keep running on a plain CPython 3.10 with only
`.[dev]` installed.

## Options considered

1. **GitHub's own Jekyll build, from `docs/`.** No local tooling at all, and it renders Markdown
   well. But the build runs on GitHub, where `outputs/` does not exist, so the site could carry no
   imagery — and imagery is the half of the request the repository cannot otherwise answer. It also
   renders only what is in the branch, so the module map and the configuration catalogue would have
   to be written by hand and maintained.
2. **MkDocs Material (or similar) plus a Pages action.** Excellent output; adds a pip dependency
   that the project interpreter does not have, a `mkdocs.yml` that duplicates the document tree, and
   the same problem with `outputs/`.
3. **A generator in this repository, publishing a built directory.** Costs a Markdown renderer —
   about 350 lines, since the corpus is CommonMark plus GitHub tables plus LaTeX — and buys the
   ability to measure the repository at build time and to re-encode the renders on the machine that
   has them.

## Decision

Option 3. `scripts/build_site.py` builds a static site into the gitignored `_site/` from three
sources: the documents (rendered, never copied), the tree itself (the module map, the configuration
catalogue and every headline number are measured during the build), and the renders named by
`site/gallery.yaml`.

Specifics worth recording:

* **The gallery is a committed manifest over uncommitted media.** `site/gallery.yaml` names the
  frames and clips worth showing and says what each one demonstrates; the files themselves come from
  `$IRSIM_OUTPUTS` (default `outputs/`). An entry the checkout cannot supply is listed on the page
  rather than dropped, so a reader can tell the difference between "not modelled" and "not rendered
  here". No `.npy` plane is ever published.
* **Everything is re-encoded**, at the source's own resolution up to 1280 px, H.264 CRF 23,
  `+faststart`, no audio; stills to WebP. Measured on this checkout: 436.8 MB of source renders
  become 38.5 MB, and the whole site is 46 MB against the ~1 GB Pages publishes. The first version
  of this capped at 720 px and CRF 34, which is right for the four reflective-band clips whose
  content is mostly sensor grain and wrong for everything else: it halved each panel of a
  1280 × 512 side-by-side comparison and then blurred what was left. Those four clips now carry
  `max_width: 720` and `crf: 34` as per-item overrides — the aggressive setting belongs on the
  items that need it, not on the default. The grain survives either way, which matters: it is the
  model's output, not a compression artefact.
* **Maths is passed through, not rendered.** `$…$` and `$$…$$` spans are extracted before any
  inline pass so that `\tau_{\text{opt}}` cannot be mangled into emphasis by its own underscores,
  and KaTeX typesets them in the browser.
* **Every URL is relative.** Project pages are served from `/<repo>/`, where a root-absolute link
  404s while working perfectly in a local preview. A test asserts no built page contains one.
* **Everything presentable is published, and the docstring is the publication.** The owner's rule
  is that anything a reader outside the conversation would be better off seeing belongs on the site,
  and only debug output does not. So the test suite is published *in full* — a page per test file,
  every test listed with what it asserts — rather than counted, because in this project a frame that
  is wrong by 20 K looks exactly as convincing as one that is right, and the assertions are the only
  evidence there is. The same reasoning puts `scripts/` and the project's own skills on the site. The
  cost is that a missing docstring is now a blank cell on a public page rather than a private
  annoyance, which is the intended pressure; `tests/unit/test_site_build.py` holds the floor by
  failing if any test file has no module docstring.
* **Publication rewrites a `gh-pages` branch** (`scripts/publish_site.sh`), one commit, no history:
  the site is an artefact, and keeping every encoded clip of every build would add tens of megabytes
  per publish to a repository whose main branch deliberately carries none of it.

## Consequences

Easy: the site cannot contradict the repository, because it stores nothing. Adding a document to
`docs/` puts it on the site; shipping a module changes the module map; ticking a roadmap step moves
the progress bars. A contributor with no renders can still build the whole site.

Hard: the Markdown renderer is ours to maintain. It covers the constructs this corpus uses and
refuses to pretend otherwise — `tests/unit/test_site_markdown.py` renders every Markdown file in
the repository and fails if link syntax reaches a page as literal text, which is how an unsupported
construct announces itself. Reference-style links, footnotes and setext headings are not supported.

The site also publishes only what the machine that built it had rendered. A publish from a checkout
with a stale `outputs/` shows stale clips, and nothing in the build can detect that.

Not done here: no CI job builds or publishes the site. It would have to build without `outputs/`,
which is the one thing the Jekyll option was rejected for.

## Revisit when

A second person needs to publish, or the site needs to update on every push — at that point the
document half could be built in CI and the media half published from a machine that has the
renders, and the two merged; or when the corpus starts wanting a Markdown construct the renderer
does not have, and adding it costs more than a dependency would.
