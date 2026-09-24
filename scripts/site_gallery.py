#!/usr/bin/env python3
"""The demo gallery: the frames and clips the simulator has actually produced, made web-sized.

`outputs/` is 8 GB and is not in git — it is what the render scripts write, at full resolution and
at bitrates that make sense for something you will analyse, not something you will stream. The
drone NIR clip alone is 97 MB at 163 Mbit/s, and GitHub Pages has a per-file limit of 100 MB and a
site budget of about 1 GB. So the gallery is not a copy of `outputs/`: it is a *manifest*
(`site/gallery.yaml`) naming the frames and clips worth showing, each re-encoded here into
something a browser will open on a phone.

The manifest is committed and the media are not, which is the point: a reviewer can see what the
site claims to show, and `make site` reproduces the media from whatever renders the machine has to
hand. Anything the manifest names but the checkout does not have is reported and skipped — the
section still renders, with a line saying which script would generate it.

Encoding defaults (overridable per item): H.264 at CRF 34, capped at 720 px wide, `+faststart` so
playback starts before the file has arrived, and no audio track, because none of these have one.
That is a 97 MB clip at 1.1 MB with the sensor noise — which is real output, not compression
artefact — still visible. Stills become WebP, which takes a 1.1 MB grayscale contact sheet to 40 kB.
"""

from __future__ import annotations

import html
import json
import pathlib
import shutil
import subprocess
from dataclasses import dataclass, field

import site_markdown
import yaml
from site_layout import GITHUB, Page

#: Encoding defaults. `crf` is x264's quality knob (higher is smaller); `width` is a cap, never an
#: upscale, so a 640 px Boson frame is recompressed at its own size rather than blown up.
VIDEO_WIDTH = 720
VIDEO_CRF = 34
IMAGE_WIDTH = 1400
IMAGE_QUALITY = 86


@dataclass
class Asset:
    """One encoded piece of media, as the page needs it."""

    url: str
    width: int
    height: int
    source_bytes: int
    bytes: int
    poster: str | None = None
    duration: float | None = None


@dataclass
class Missing:
    """A manifest entry this checkout cannot show."""

    src: str
    section: str


@dataclass
class GalleryResult:
    page: Page
    assets: list[Asset] = field(default_factory=list)
    missing: list[Missing] = field(default_factory=list)

    @property
    def total_bytes(self) -> int:
        return sum(a.bytes + (0 if a.poster is None else 0) for a in self.assets)


def _probe(path: pathlib.Path) -> tuple[int, int, float | None]:
    """Width, height and duration of a media file, via ffprobe."""
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height:format=duration",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0, 0, None
    data = json.loads(out or "{}")
    stream = (data.get("streams") or [{}])[0]
    duration = data.get("format", {}).get("duration")
    return int(stream.get("width", 0)), int(stream.get("height", 0)), float(duration or 0) or None


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


class Cache:
    """What each encoded file was built from: its source's mtime and the encoder settings.

    Freshness cannot be mtime alone. Changing a clip's frame rate or its CRF in `site/gallery.yaml`
    leaves the source untouched, so an mtime test would keep serving the old encode and the
    manifest would quietly stop describing the site. The settings are part of the key.
    """

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        try:
            self.entries: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.entries = {}

    def fresh(self, key: str, src_mtime: float, signature: str, dest: pathlib.Path) -> bool:
        return dest.exists() and self.entries.get(key) == f"{src_mtime:.0f}|{signature}"

    def record(self, key: str, src_mtime: float, signature: str) -> None:
        self.entries[key] = f"{src_mtime:.0f}|{signature}"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=0, sort_keys=True), encoding="utf-8")


def encode_video(
    src: pathlib.Path, out_root: pathlib.Path, rel: str, *, width: int, crf: int, cache: Cache
) -> Asset:
    """Re-encode one clip for the web, with a poster frame so it has something to show unplayed."""
    dest = out_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    signature = f"video:{width}:{crf}:{IMAGE_QUALITY}"
    if not cache.fresh(rel, src.stat().st_mtime, signature, dest):
        _run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(src),
                "-vf",
                f"scale='min({width},iw)':-2:flags=lanczos",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(dest),
            ]
        )
    poster = dest.with_suffix(".webp")
    if not cache.fresh(rel + ":poster", src.stat().st_mtime, signature, poster):
        _run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(dest),
                "-frames:v",
                "1",
                "-c:v",
                "libwebp",
                "-q:v",
                str(IMAGE_QUALITY),
                str(poster),
            ]
        )
    cache.record(rel, src.stat().st_mtime, signature)
    cache.record(rel + ":poster", src.stat().st_mtime, signature)
    w, h, duration = _probe(dest)
    return Asset(
        url=rel,
        width=w,
        height=h,
        source_bytes=src.stat().st_size,
        bytes=dest.stat().st_size,
        poster=poster.relative_to(out_root).as_posix(),
        duration=duration,
    )


def encode_frames(
    pattern: pathlib.Path,
    out_root: pathlib.Path,
    rel: str,
    *,
    fps: int,
    width: int,
    crf: int,
    cache: Cache,
) -> Asset:
    """Assemble a numbered still sequence into a clip.

    Several demos wrote frames and never an mp4. A folder of 96 PNGs is not something anyone
    watches, so the gallery builds the clip here rather than asking for the render to be repeated
    on a GPU (`scripts/render_*.py` writes both when it is run again).
    """
    dest = out_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    first = sorted(pattern.parent.glob(pattern.name.replace("%05d", "*").replace("%04d", "*")))
    newest = max((f.stat().st_mtime for f in first), default=0.0)
    signature = f"frames:{fps}:{width}:{crf}:{len(first)}"
    if not cache.fresh(rel, newest, signature, dest):
        _run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(pattern),
                "-vf",
                f"scale='min({width},iw)':-2:flags=lanczos",
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dest),
            ]
        )
    poster = dest.with_suffix(".webp")
    if not cache.fresh(rel + ":poster", newest, signature, poster):
        _run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(dest),
                "-frames:v",
                "1",
                "-c:v",
                "libwebp",
                "-q:v",
                str(IMAGE_QUALITY),
                str(poster),
            ]
        )
    cache.record(rel, newest, signature)
    cache.record(rel + ":poster", newest, signature)
    w, h, duration = _probe(dest)
    return Asset(
        url=rel,
        width=w,
        height=h,
        source_bytes=sum(f.stat().st_size for f in first),
        bytes=dest.stat().st_size,
        poster=poster.relative_to(out_root).as_posix(),
        duration=duration,
    )


def encode_image(
    src: pathlib.Path, out_root: pathlib.Path, rel: str, *, width: int, quality: int, cache: Cache
) -> Asset:
    """Re-encode one still as WebP, unless the original is already smaller."""
    dest = (out_root / rel).with_suffix(".webp")
    dest.parent.mkdir(parents=True, exist_ok=True)
    signature = f"image:{width}:{quality}"
    if not cache.fresh(rel, src.stat().st_mtime, signature, dest):
        _run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(src),
                "-vf",
                f"scale='min({width},iw)':-2:flags=lanczos",
                "-c:v",
                "libwebp",
                "-q:v",
                str(quality),
                str(dest),
            ]
        )
    cache.record(rel, src.stat().st_mtime, signature)
    if dest.stat().st_size > src.stat().st_size:
        shutil.copy2(src, out_root / rel)
        dest = out_root / rel
    w, h, _ = _probe(dest)
    return Asset(
        url=dest.relative_to(out_root).as_posix(),
        width=w,
        height=h,
        source_bytes=src.stat().st_size,
        bytes=dest.stat().st_size,
    )


class GalleryBuilder:
    """Turns `site/gallery.yaml` plus a tree of renders into one page and its media."""

    def __init__(
        self,
        manifest: pathlib.Path,
        outputs: pathlib.Path,
        media_root: pathlib.Path,
        *,
        media_prefix: str,
        encode: bool = True,
    ) -> None:
        self.spec = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        self.outputs = outputs
        self.media_root = media_root
        self.media_prefix = media_prefix
        self.encode = encode
        self.cache = Cache(media_root / ".encode.json")
        self.assets: list[Asset] = []
        self.missing: list[Missing] = []

    # -- media ------------------------------------------------------------------------------

    def _asset(self, src_rel: str, section: str, *, video: bool, **options: int) -> Asset | None:
        src = self.outputs / src_rel
        if not src.is_file():
            self.missing.append(Missing(src_rel, section))
            return None
        if not self.encode:  # a docs-only build: keep the page, drop the megabytes
            return None
        if video:
            asset = encode_video(
                src,
                self.media_root,
                src_rel,
                width=int(options.get("width", VIDEO_WIDTH)),
                crf=int(options.get("crf", VIDEO_CRF)),
                cache=self.cache,
            )
        else:
            asset = encode_image(
                src,
                self.media_root,
                src_rel,
                width=int(options.get("width", IMAGE_WIDTH)),
                quality=int(options.get("quality", IMAGE_QUALITY)),
                cache=self.cache,
            )
        self.assets.append(asset)
        return asset

    def _frames(self, pattern: str, section: str, fps: int, **options: int) -> Asset | None:
        """A clip assembled from a still sequence named by a printf pattern."""
        src = self.outputs / pattern
        if not any(src.parent.glob(src.name.replace("%05d", "*").replace("%04d", "*"))):
            self.missing.append(Missing(pattern, section))
            return None
        if not self.encode:
            return None
        rel = f"{pathlib.PurePosixPath(pattern).parent}.mp4"
        asset = encode_frames(
            src,
            self.media_root,
            rel,
            fps=fps,
            width=int(options.get("width", VIDEO_WIDTH)),
            crf=int(options.get("crf", VIDEO_CRF)),
            cache=self.cache,
        )
        self.assets.append(asset)
        return asset

    # -- rendering --------------------------------------------------------------------------

    def _md(self, text: str) -> str:
        return site_markdown.render(text).html

    def _figure(self, item: dict[str, object], section: str) -> str:
        caption = str(item.get("caption", ""))
        span = str(item.get("width", "half"))
        options = {k: v for k, v in item.items() if k in ("crf", "quality") and isinstance(v, int)}
        if "frames" in item:
            pattern = str(item["frames"])
            asset = self._frames(pattern, section, int(item.get("fps", 24)), **options)  # type: ignore[arg-type]
            if asset is None:
                return self._placeholder(pattern, caption, span)
            poster = f' poster="{self.media_prefix}{asset.poster}"' if asset.poster else ""
            size = f' width="{asset.width}" height="{asset.height}"' if asset.width else ""
            media = (
                f'<video src="{self.media_prefix}{asset.url}"{poster}{size} controls loop muted '
                f'playsinline preload="none"></video>'
            )
        elif "video" in item:
            rel = str(item["video"])
            asset = self._asset(rel, section, video=True, **options)  # type: ignore[arg-type]
            if asset is None:
                return self._placeholder(rel, caption, span)
            poster = f' poster="{self.media_prefix}{asset.poster}"' if asset.poster else ""
            size = f' width="{asset.width}" height="{asset.height}"' if asset.width else ""
            media = (
                f'<video src="{self.media_prefix}{asset.url}"{poster}{size} controls loop muted '
                f'playsinline preload="none"></video>'
            )
        else:
            rel = str(item["image"])
            asset = self._asset(rel, section, video=False, **options)  # type: ignore[arg-type]
            if asset is None:
                return self._placeholder(rel, caption, span)
            size = f' width="{asset.width}" height="{asset.height}"' if asset.width else ""
            media = (
                f'<a href="{self.media_prefix}{asset.url}">'
                f'<img src="{self.media_prefix}{asset.url}"{size} loading="lazy" '
                f'alt="{html.escape(caption)}">'
                f"</a>"
            )
        label = (
            f'<span class="label">{html.escape(str(item["label"]))}</span>'
            if "label" in item
            else ""
        )
        body = (
            f"<figcaption>{label}{self._inline(caption)}</figcaption>" if caption or label else ""
        )
        return f'<figure class="shot {span}">{media}{body}</figure>'

    def _placeholder(self, rel: str, caption: str, span: str) -> str:
        return (
            f'<figure class="shot {span} absent"><div class="absent-box">not in this checkout'
            f"<code>outputs/{html.escape(rel)}</code></div>"
            f"<figcaption>{self._inline(caption)}</figcaption></figure>"
        )

    def _inline(self, text: str) -> str:
        rendered = self._md(text)
        return (
            rendered[3:-4] if rendered.startswith("<p>") and rendered.endswith("</p>") else rendered
        )

    def build(self, url: str) -> GalleryResult:
        body: list[str] = [self._md(str(self.spec.get("intro", "")))]
        headings: list[site_markdown.Heading] = []
        search: list[str] = []
        for section in self.spec.get("sections", []):
            title = str(section["title"])
            slug = site_markdown.slugify(str(section.get("id", title)))
            headings.append(site_markdown.Heading(2, slug, title))
            search.append(f"{title} {section.get('lead', '')}")
            body.append(f'<section class="demo"><h2 id="{slug}">{html.escape(title)}</h2>')
            if section.get("lead"):
                body.append(self._md(str(section["lead"])))
            media = section.get("media") or []
            if media:
                body.append('<div class="shots">')
                body.extend(self._figure(item, title) for item in media)
                body.append("</div>")
            if section.get("note"):
                body.append(f'<div class="note">{self._md(str(section["note"]))}</div>')
            if section.get("source"):
                sources = section["source"]
                sources = sources if isinstance(sources, list) else [sources]
                links = " · ".join(
                    f'<a href="{GITHUB}/blob/main/{s}" target="_blank" rel="noopener">'
                    f"<code>{s}</code></a>"
                    for s in sources
                )
                body.append(f'<p class="made-by">Made by {links}</p>')
            body.append("</section>")

        if self.missing:
            names = "".join(
                f"<li><code>outputs/{html.escape(m.src)}</code></li>" for m in self.missing
            )
            body.append(
                '<section class="demo"><h2 id="not-in-this-checkout">Not in this checkout</h2>'
                "<p>The manifest names these and this build could not find them. They are render "
                "outputs, not repository files: run the script named in the section to regenerate "
                f"them.</p><ul class='absent-list'>{names}</ul></section>"
            )
            headings.append(
                site_markdown.Heading(2, "not-in-this-checkout", "Not in this checkout")
            )

        page = Page(
            url=url,
            title=str(self.spec.get("title", "Gallery")),
            subtitle=str(self.spec.get("subtitle", "")),
            body="\n".join(body),
            nav_section="Demos",
            headings=headings,
            wide=True,
            search_text=" ".join(search),
        )
        if self.encode:
            self.cache.save()
        return GalleryResult(page=page, assets=self.assets, missing=self.missing)
