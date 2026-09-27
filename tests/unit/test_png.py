"""The stdlib PNG codec (M5.7): valid signature and chunks, a decodable payload, an exact read-back.

The reader exists so a tool can read a frame's ``dn16`` ADC code back at a pixel. An off-by-one in
a row filter would return a plausible code for the wrong pixel, so the filters are checked against
an encoder written from the PNG specification's own definitions, not against the writer (which
only ever uses filter 0).
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from irsim.io.png import png_bytes, read_png, write_png


def test_png_structure_and_payload_round_trip() -> None:
    img = np.arange(4 * 3 * 4, dtype=np.uint8).reshape(3, 4, 4)
    data = png_bytes(img)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    w, h, depth, ctype = struct.unpack(">IIBB", data[16:26])
    assert (w, h, depth, ctype) == (4, 3, 8, 6)
    idat_len = struct.unpack(">I", data[33:37])[0]
    assert data[37:41] == b"IDAT"
    raw = zlib.decompress(data[41 : 41 + idat_len])
    rows = np.frombuffer(raw, np.uint8).reshape(3, 1 + 4 * 4)
    assert np.all(rows[:, 0] == 0) and np.array_equal(rows[:, 1:].reshape(3, 4, 4), img)
    assert data[-12:-8] == struct.pack(">I", 0) and data[-8:-4] == b"IEND"
    gray = png_bytes(np.zeros((2, 2), np.uint8))
    assert struct.unpack(">B", gray[25:26])[0] == 0
    with pytest.raises(TypeError):
        png_bytes(np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError):
        png_bytes(np.zeros((2, 2, 2), np.uint8))


@pytest.mark.parametrize(
    "image",
    [
        np.arange(5 * 7, dtype=np.uint8).reshape(5, 7),
        np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3),
        np.arange(5 * 7 * 4, dtype=np.uint8).reshape(5, 7, 4),
        (np.arange(5 * 7, dtype=np.uint32) * 1871 % 65536).astype(np.uint16).reshape(5, 7),
    ],
    ids=["gray8", "rgb8", "rgba8", "gray16"],
)
def test_what_the_writer_writes_the_reader_reads_back_exactly(tmp_path, image) -> None:
    path = tmp_path / "x.png"
    write_png(path, image)
    back = read_png(path)
    assert back.dtype == image.dtype and np.array_equal(back, image)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def _filtered_png(img: np.ndarray, kinds: list[int]) -> bytes:
    """RGB8 PNG with row ``y`` filtered by ``kinds[y]``, straight from PNG spec §9.2."""
    h, w, c = img.shape
    rows = img.reshape(h, w * c).astype(int)
    raw = b""
    for y in range(h):
        prev = rows[y - 1] if y else np.zeros(w * c, int)
        out = []
        for i in range(w * c):
            a = rows[y, i - c] if i >= c else 0
            b = prev[i]
            cc = prev[i - c] if i >= c else 0
            pred = [0, a, b, (a + b) // 2, _paeth(a, b, cc)][kinds[y]]
            out.append((rows[y, i] - pred) % 256)
        raw += bytes([kinds[y]]) + bytes(out)

    def chunk(tag: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_every_row_filter_decodes_to_the_original_pixels(tmp_path) -> None:
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(10, 6, 3), dtype=np.uint8)
    path = tmp_path / "filtered.png"
    path.write_bytes(_filtered_png(img, [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]))
    assert np.array_equal(read_png(path), img)


def test_a_file_that_is_not_a_png_is_refused(tmp_path) -> None:
    path = tmp_path / "x.png"
    path.write_bytes(b"not a png at all")
    with pytest.raises(ValueError, match="not a PNG"):
        read_png(path)
