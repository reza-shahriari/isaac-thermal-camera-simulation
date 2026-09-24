# ADR 0141 — The front page is a front door, and its renders are named in the gallery manifest

**Status:** Accepted
**Date:** 2026-09-24

## Context

ADR 0139 built the site and got the depth right: 457 pages, the specification, 139 decision
records, every test with what it asserts, the module and configuration catalogues measured from
the tree at build time. What it did not get right was the first screen. The front page opened with
three paragraphs of prose, a counter row and one clip, while the eighteen sections of the gallery
— the same drone through four bands, a vessel departing, a car warming the metal around its engine
— sat one click away and were the only things on the site a reader can judge without taking the
project's word for anything.

The owner asked whether the site should stay technical or be made beautiful. It is a false choice:
the depth is the differentiator and nothing about it needs to go. What was missing was a lobby in
front of it.

Two constraints from ADR 0139 still hold and rule out the obvious answer of adopting a template.
The generator *measures the repository at build time*, which no off-the-shelf theme does; and the
renders are not in git, so the page has to be built on the machine that has `outputs/`. A theme
swap would trade both for nicer chrome.

## Decision

Four changes to the generator, no new dependency, no framework, one stylesheet and one script as
before.

**The hero is full bleed and carries the title.** `Page.hero` replaces the standard title block for
a page that sets it, and the front page sets it: the render runs edge to edge with `<h1>` over it
behind a two-axis scrim. The scrim is not decoration. These frames are grayscale, their brightness
is the sensor model's business, and several of them carry a telemetry readout burnt into the top
left — exactly where a title goes.

**Three demonstration blocks come before the prose.** The band strip (one scene, four bands, four
up), the point-wise headline clip, and a wipe. Show first, explain second: the signal chain, the
counters and the plan now follow them rather than preceding them.

**The gallery opens with a thumbnail of every section** and no longer floats a contents list beside
it. Eighteen titles in a list is not how anyone finds the section they were sent for; eighteen of
the sections' own first frames is.

**Which renders appear is declared in `site/gallery.yaml`, not in the page builder.** The manifest
already owned the hero for this reason. It now owns the band strip, the headline clip and the wipe
pair too, and `GalleryBuilder.front()` resolves them against media the sections have already paid
to encode, so the front page adds no bytes of its own. Every one of them must also be named by a
section, which a test asserts: a front page showing something the site explains nowhere is how a
demo reel starts to diverge from the work.

## What the wipe may be pointed at, and why it is not pointed at the headline clip

The obvious subject for a before/after wipe is `pointwise_flight/quad_pointwise_vs_objectwise.mp4`,
since the point-wise claim is the project's headline. It cannot be. That file is a single 926 × 460
composite of two 463-wide panels, and **the panels carry different burnt-in annotations** — the
left owns the run header, and each owns its own footer readout. Wiping between them puts
`OBJECT-WISE — whole aircraft 26.2 C` over an image that is mostly point-wise. That is not a nicer
way to make the argument, it is a wrong one, and the script that produced the composite is no
longer in `scripts/`, so splitting it into two bare panels is not a small change.

So the wipe has a rule, and `tests/unit/test_site_build.py` enforces it: **both halves must be
stills, and both must come from the same gallery section.** Stills, because two clips under a wipe
drift out of sync and end up comparing different instants as well as different physics. One
section, because that is what makes them one scene under one change. Today it is the flat-field
pair, which satisfies it exactly.

The headline clip is instead shown whole, full width, with the two solvers named in HTML above
their own halves — and without `controls`, because Chrome draws its control bar precisely over the
readout that is the clip's evidence.

## Consequences

* The front page costs nothing extra to serve. Every asset on it was already encoded for a gallery
  section, and the band strip is poster frames with hover-to-play: the SWIR clip alone is 5.7 MB
  re-encoded, and four autoplaying clips would be about 7 MB spent before a reader has scrolled.
* A checkout with no renders drops each block rather than publishing an empty frame, so the
  commit gate — which builds with `media=False` — still exercises the page.
* Without JavaScript the wipe sits at half and reads as a static split comparison, which is what
  it is. Everything else on the page is CSS.
* Changing what the project opens with is now a manifest edit, not a code change.
