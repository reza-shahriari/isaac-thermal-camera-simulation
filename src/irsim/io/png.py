"""Minimal stdlib PNG writer and reader (8-bit gray, RGB, RGBA; 16-bit gray).

The physics core may not import PIL / imageio / cv2 (layering guard, CLAUDE.md #1), yet a
human needs to look at the first image. PNG is zlib + a CRC, so this is thirty lines of stdlib.
Display images are uint8; a **uint16** grayscale PNG is also supported for the DN16 output, which
is an integer ADC code and therefore loses nothing by being stored as an integer (§12.2). Anything
in physical units -- radiance, apparent temperature -- goes to .npy or float32 EXR, never to a PNG:
a 16-bit PNG over a 200-400 K range quantises to 3 mK and looks fine doing it (CLAUDE.md #2).

:func:`read_png` is the inverse, for tools that must read a frame's ``dn16`` code back at a pixel
(the frame viewer, :mod:`irsim_viewer`) without an image library. It decodes any non-interlaced
8- or 16-bit gray, gray+alpha, RGB or RGBA PNG, all five row filters included, so it also reads a
PNG this module did not write.
"""

from __future__ import annotations

import os
import struct
import zlib

import numpy as np
from numpy.typing import NDArray

__all__ = ["write_png", "png_bytes", "read_png"]

_COLOR_TYPES = {1: 0, 3: 2, 4: 6}  # channels -> PNG colour type (gray, RGB, RGBA)
_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}  # PNG colour type -> channels (gray, RGB, gray+alpha, RGBA)


def _chunk(tag: bytes, payload: bytes) -> bytes:
    body = tag + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def png_bytes(image: NDArray[np.unsignedinteger]) -> bytes:
    """Encode a uint8 (H, W), (H, W, 3) or (H, W, 4) array -- or a uint16 (H, W) -- as PNG bytes.

    uint16 is single-channel only and is written big-endian, as the PNG specification requires.
    It exists for the DN16 output alone: an ADC code is an integer and survives the trip exactly.
    """
    arr = np.asarray(image)
    if arr.dtype == np.uint16:
        if arr.ndim != 2:
            raise ValueError(f"uint16 PNG is single-channel; got shape {arr.shape}")
        h, w = arr.shape
        be = np.ascontiguousarray(arr, dtype=">u2")
        raw = b"".join(b"\x00" + be[row].tobytes() for row in range(h))
        ihdr = struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw, 9))
            + _chunk(b"IEND", b"")
        )
    if arr.dtype != np.uint8:
        raise TypeError(
            f"PNG output is uint8 display data or uint16 DN, got {arr.dtype}: anything in "
            "physical units goes to .npy or float32 EXR (CLAUDE.md #2)"
        )
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[2] not in _COLOR_TYPES:
        raise ValueError("image must be (H, W), (H, W, 3) or (H, W, 4)")
    h, w, c = arr.shape
    raw = b"".join(b"\x00" + arr[row].tobytes() for row in range(h))  # filter type 0 per row
    ihdr = struct.pack(">IIBBBBB", w, h, 8, _COLOR_TYPES[c], 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(raw, 9))
        + _chunk(b"IEND", b"")
    )


def write_png(path: str | os.PathLike[str], image: NDArray[np.unsignedinteger]) -> None:
    with open(path, "wb") as fh:
        fh.write(png_bytes(image))


def _unfilter(raw: bytes, h: int, stride: int, bpp: int) -> NDArray[np.uint8]:
    """Undo the per-row PNG filters (PNG spec §9) and return the (h, stride) byte image."""
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(h, stride + 1)
    out = np.zeros((h, stride), dtype=np.uint8)
    prev: NDArray[np.int32] = np.zeros(stride, dtype=np.int32)
    for y in range(h):
        kind = int(rows[y, 0])
        line = rows[y, 1:].astype(np.int32)
        if kind == 0:
            cur = line
        elif kind == 2:  # Up
            cur = (line + prev) & 0xFF
        elif kind == 1:  # Sub: a running sum per byte lane, one lane per byte of a pixel
            lanes = np.zeros(stride + (-stride) % bpp, dtype=np.int32)
            lanes[:stride] = line
            cur = (np.cumsum(lanes.reshape(-1, bpp), axis=0) & 0xFF).reshape(-1)[:stride]
        elif kind in (3, 4):  # Average, Paeth: each byte depends on the one decoded before it
            cur = np.zeros(stride, dtype=np.int32)
            for i in range(stride):
                a = int(cur[i - bpp]) if i >= bpp else 0
                b = int(prev[i])
                if kind == 3:
                    pred = (a + b) // 2
                else:
                    c = int(prev[i - bpp]) if i >= bpp else 0
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                cur[i] = (int(line[i]) + pred) & 0xFF
        else:
            raise ValueError(f"unknown PNG filter type {kind} on row {y}")
        out[y] = cur
        prev = cur
    return out


def read_png(path: str | os.PathLike[str]) -> NDArray[np.unsignedinteger]:
    """Decode a PNG to uint8 or uint16, shaped (H, W) for gray and (H, W, C) otherwise.

    Palette and interlaced files are refused rather than half-decoded: nothing this project
    writes uses either, and a wrong pixel value is worse than an error in a debugging tool.
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    pos, idat, header = 8, bytearray(), None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
    if header is None:
        raise ValueError(f"{path} has no IHDR chunk")
    w, h, depth, colour, _, _, interlace = header
    if colour not in _CHANNELS or depth not in (8, 16) or interlace != 0:
        raise ValueError(
            f"{path}: only non-interlaced 8/16-bit gray, RGB and alpha PNGs are read "
            f"(colour type {colour}, depth {depth}, interlace {interlace})"
        )
    channels = _CHANNELS[colour]
    bpp = channels * depth // 8
    pixels = _unfilter(zlib.decompress(bytes(idat)), h, w * bpp, bpp)
    if depth == 16:
        image: NDArray[np.unsignedinteger] = (
            pixels.view(">u2").astype(np.uint16).reshape(h, w, channels)
        )
    else:
        image = pixels.reshape(h, w, channels)
    return image[:, :, 0] if channels == 1 else image
