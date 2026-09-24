"""The gallery's media cache, which decides what gets re-encoded and what gets republished.

`make site` re-encodes 55 clips and stills from `outputs/`, so it caches. The cache is the part
worth testing: every stale answer it gives is published, and a published clip that does not match
the manifest is worse than a missing one, because nothing about the page says it is wrong.

The failure this pins actually happened. A build that changed the encoder defaults and added
per-item overrides in one go left a 29 MB, 1280-pixel clip on disk under a cache entry claiming
720 pixels; every later build trusted the entry, skipped the re-encode, and republished the file
the manifest no longer described.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))

from site_gallery import Cache  # noqa: E402


def _cache(tmp_path: pathlib.Path) -> tuple[Cache, pathlib.Path, pathlib.Path]:
    src = tmp_path / "source.mp4"
    src.write_bytes(b"x" * 64)
    dest = tmp_path / "out" / "encoded.mp4"
    dest.parent.mkdir()
    dest.write_bytes(b"y" * 32)
    return Cache(tmp_path / ".encode.json"), src, dest


def test_a_recorded_encode_is_fresh(tmp_path: pathlib.Path) -> None:
    cache, src, dest = _cache(tmp_path)
    cache.record("clip", src.stat().st_mtime, "video:720:34:86", dest)
    assert cache.fresh("clip", src.stat().st_mtime, "video:720:34:86", dest)


def test_changed_settings_are_not_fresh(tmp_path: pathlib.Path) -> None:
    """A CRF or a resolution edited in the manifest leaves the source untouched."""
    cache, src, dest = _cache(tmp_path)
    cache.record("clip", src.stat().st_mtime, "video:720:34:86", dest)
    assert not cache.fresh("clip", src.stat().st_mtime, "video:1280:23:86", dest)
    assert not cache.fresh("clip", src.stat().st_mtime, "video:720:20:86", dest)


def test_a_newer_source_is_not_fresh(tmp_path: pathlib.Path) -> None:
    cache, src, dest = _cache(tmp_path)
    cache.record("clip", src.stat().st_mtime, "video:720:34:86", dest)
    assert not cache.fresh("clip", src.stat().st_mtime + 60, "video:720:34:86", dest)


def test_an_encoded_file_that_changed_underneath_is_not_fresh(tmp_path: pathlib.Path) -> None:
    """The one that bit: the entry was right and the file on disk was not."""
    cache, src, dest = _cache(tmp_path)
    cache.record("clip", src.stat().st_mtime, "video:720:34:86", dest)
    dest.write_bytes(b"z" * 4096)  # a different encode, same settings in the entry
    assert not cache.fresh("clip", src.stat().st_mtime, "video:720:34:86", dest)


def test_a_missing_file_is_not_fresh(tmp_path: pathlib.Path) -> None:
    cache, src, dest = _cache(tmp_path)
    cache.record("clip", src.stat().st_mtime, "video:720:34:86", dest)
    dest.unlink()
    assert not cache.fresh("clip", src.stat().st_mtime, "video:720:34:86", dest)


def test_the_cache_survives_a_round_trip_and_a_corrupt_file(tmp_path: pathlib.Path) -> None:
    cache, src, dest = _cache(tmp_path)
    cache.record("clip", src.stat().st_mtime, "video:720:34:86", dest)
    cache.save()
    assert Cache(cache.path).fresh("clip", src.stat().st_mtime, "video:720:34:86", dest)
    cache.path.write_text("{ not json", encoding="utf-8")
    assert not Cache(cache.path).fresh("clip", src.stat().st_mtime, "video:720:34:86", dest)
