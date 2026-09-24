---
name: present-on-the-site
description: How work in this repository reaches the public project site (the GitHub Pages site built by `make site`) — what is published automatically, what has to be added by hand to `site/gallery.yaml`, the rules for captions and media encoding, and the line between something worth presenting and debug output. Use this skill whenever a step produces something a reader outside the conversation would want to see — a render, a clip, a validation report, a new test file, a new module, a new scene or sensor config — and whenever the user says "put this on the site", "update the site", "publish", or asks why something is or is not on it. Run through it as part of finishing a step, before the commit.
---

# Presenting work on the project site

The site (`make site`, ADR 0139) is not a brochure kept beside the repository. It is the repository,
rendered: 440-odd pages built from the documents, the tree and the renders, published to the
`gh-pages` branch and served at `https://reza-shahriari.github.io/isaac-thermal-camera-simulation/`.

The rule the owner set is short: **everything that can be presented should be on the site.** A test,
a module, a decision, a report, a scene config, a clip. Not debug output — a probe dump, a scratch
print, a one-off timing, a half-rendered frame kept to look at once.

If you are not sure which side of that line something falls on, ask: *would somebody who was not in
this conversation be better off seeing it?* If yes, it goes on the site.

---

## What is already automatic

These need nothing from you beyond doing the work well, because the build measures the tree:

| You added | It appears as | Because |
|---|---|---|
| A module under `src/` | a row on **Code map**, with its docstring and the spec sections it cites | `site_survey.modules` |
| A test file | its **own page** under **The test suite**, every test listed with what it asserts | `site_survey.test_modules` |
| An ADR | a row in the **Decision log** and its own page | `docs/decisions/*.md` is a collection |
| A document in `docs/` | a rendered page under Physics, Plan or Evidence | the same |
| A config in `configs/` | a row in the **Configuration catalogue**, with its headline comment | `site_survey.config_catalogue` |
| A script in `scripts/` | a row on **Commands**, with its usage block | `site_survey.script_catalogue` |
| A CHANGELOG or README edit | the Changelog and Status pages | rendered, not copied |

So: **write the docstring**. A module, a test file or a script with no docstring publishes an em dash
where its explanation should be, and that is now visible to everyone rather than only to the next
reader of the file. The first paragraph is what the site shows in a table; the rest becomes the body
of its page.

A test's own docstring is the "what it asserts" column. `test_the_bonnet_is_a_gradient` with no
docstring tells a reader the name and nothing else; one sentence saying *6.3 K across one prim after
thirty minutes, which is 127 NETD of structure* is the evidence the project claims to have.

---

## What you have to add by hand: renders

`outputs/` is gitignored and 8 GB. The site carries a curated, re-encoded subset named by
**`site/gallery.yaml`**, which is committed and reviewable. If your step produced frames or a clip
worth showing, add a section:

```yaml
  - id: exhaust-plume              # the anchor; keep it stable, people link to it
    title: The plume as a chord per pixel
    lead: |
      Markdown. Say what the reader is looking at and what it demonstrates, with the numbers
      that make it a result rather than a picture.
    media:
      - video: car_exhaust/plume_agc.mp4     # relative to outputs/
        width: full                          # full | half | third
        caption: the camera's own AGC
      - image: pointwise/exhaust_line/under_car_whitehot.png
        width: half
        caption: 425 °C at the downpipe falling to 128 °C through the silencer
      - frames: vessel_departure/frames/agc_%05d.png   # a still sequence, assembled into a clip
        fps: 8
    note: |
      Optional: the caveat, the known limit, the thing a reader would otherwise mistake.
    source: scripts/render_car_ignition.py   # what produced it
```

Then `make site` re-encodes it. Defaults are H.264 CRF 34 capped at 720 px and WebP for stills;
override per item with `crf:`, `width:` or `quality:` only if you can say why.

### The rules for that entry

1. **Every number in a caption is measured from the run on disk** — the one the manifest points at,
   not the one the CHANGELOG describes. Read `outputs/<run>/summary.json`, or read the gauge off the
   frames themselves. A previous run's numbers beside this run's pixels is the kind of error nobody
   catches by looking.
2. **No `.npy`, ever.** The radiance and apparent-temperature planes are for the code. Publish the
   frames and the clips.
3. **Say what it demonstrates, not what it is.** "A quadrotor in LWIR" is a caption; "the deck moves
   16.6 K over the mission while the object-wise aircraft moves 0.5 K" is the reason the clip exists.
4. **A missing render is listed, not hidden.** If the manifest names something this checkout lacks,
   the build says so on the page and in its output. Leave the entry in: it tells a reader the
   scene exists and which script regenerates it.
5. **Check the licence before publishing somebody else's imagery** (ADR 0041, XD.1). A licence noted
   in a dataset's README is not a grant over its frames.

---

## Finishing a step

Add this to the `ship-step` checklist, between the CHANGELOG entry and the commit:

```bash
make site PYTHON=…      # or `make site-preview` to look at it
```

and read its output, which is three things:

* **the page count and the media budget** — `442 pages, 55 media files (20.7 MB re-encoded from
  436.8 MB), 28.0 MB total`. The whole site must stay well under the ~1 GB GitHub Pages publishes;
  if one item has blown up, cap it with `crf:`/`width:` rather than dropping the section.
* **`N manifest entries were not in outputs/`** — expected on a machine that has not run the
  renders, a mistake if it names the thing you just made.
* **dead links** — `dead link in docs/decisions/0120-…: 0036-rk2-….md`. These are real broken
  references in the documents; fix them where they are, do not silence the report.

`tests/unit/test_site_build.py` runs in the gate and fails on a dead internal link or a
root-absolute URL, so a broken site is a red `make check`, not a surprise after publishing.

Publishing itself is manual and stays that way (ADR 0139): `make site-publish` rewrites the
`gh-pages` branch locally, one commit, no history, and prints the push command. **Do not push it
unless the user asks.**

---

## What not to publish

* Raw arrays, golden fixtures, weather CSVs, prim dumps — the code's input, not a reader's.
* Probe output kept to answer one question (`outputs/isaac_probe/`), unless the answer itself is
  worth a section, in which case write it up as an ADR and let the ADR be the thing on the site.
* A frame you have not looked at. Everything in the gallery is something you have opened and can
  describe; an unopened render on a public page is a claim you have not checked.
* Anything whose numbers you cannot source.

---

## Do not hand-edit the built site

`_site/` is generated and gitignored, and so is every file in it. If a page is wrong, the fix is in
`scripts/build_site.py`, `scripts/site_survey.py`, `scripts/site_gallery.py`, `site/gallery.yaml` or
`site/assets/`, with a test in `tests/unit/test_site_build.py` or
`tests/unit/test_site_markdown.py` if the failure could recur. The Markdown renderer is ours
(`scripts/site_markdown.py`): if a document uses a construct it does not support, the corpus test
fails rather than the site quietly publishing the raw syntax.
