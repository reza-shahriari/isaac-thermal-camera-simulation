"""Turning a rendered sequence into something a person can watch (ADR 0074).

Lives in ``irsim_eval`` and not in ``irsim`` because it needs the imaging stack -- PIL for the
burnt-in readout, ffmpeg for the container -- which CLAUDE.md keeps out of the physics core. It is
a presentation layer and nothing in the radiometric chain reads it back.

The readout matters more than it looks. A thermal video after automatic gain control is a picture
of *contrast*, not of temperature: plateau equalisation will happily render a 27 C motor and a
70 C motor as the same bright blob if nothing else in the frame moved. Burning the numbers the
physics actually produced into the corner is what stops the viewer reading the AGC's opinion as a
measurement.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "annotate",
    "target_span_k",
    "overlay_readout",
    "temperature_bar",
    "palette_scale",
    "encode_mp4",
    "ffmpeg_available",
]

_MARGIN = 10


def target_span_k(
    samples: Iterable[Mapping[str, float]],
    *,
    below: float = 0.25,
    above: float = 0.10,
    minimum_spread_k: float = 5.0,
) -> tuple[float, float]:
    """A display span that spends its range on the **target**, not on the scene.

    Eight bits is 256 levels and a sky-target scene spans hundreds of kelvin, so the choice of
    what to spend them on is the whole of the picture. Spanning ambient +/- 50 K -- the obvious
    rule -- gives half the palette to sky-to-ambient, which is one flat region and one flat
    region, and leaves every part of the target squeezed into the top half. Measured on the
    quadrotor at full throttle: arms at code 128 and motors at 227, real contrast but compressed
    into a third of the range.

    So the span is taken from the target's own nodes over the whole sequence instead. The floor
    sits a quarter of the node spread *below* the coldest node rather than on it, for two reasons:
    a node at the very bottom of the span is black and invisible, and at the start of a flight
    every node is at ambient, so a floor on the coldest node would make the aircraft disappear
    exactly when the viewer is looking for it. The ceiling gets a tenth of headroom so the hottest
    node is bright rather than clipped.

    The sky falls below the floor and clips to black. That is deliberate: the sky is the thing
    there is least to see in, and giving it any of the range costs the target all of the
    difference. Pass ``--span-c`` when the scene context matters more than the target.

    **None of this is a measurement.** The mapping from temperature to brightness is a display
    choice; the burnt-in gauge and the JSON sidecar carry the temperatures.
    """
    values = [float(v) for sample in samples for v in sample.values()]
    if not values:
        raise ValueError("need at least one node temperature to span")
    coldest, hottest = min(values), max(values)
    spread = max(hottest - coldest, float(minimum_spread_k))
    return (coldest - below * spread, hottest + above * spread)


def _font(size: int) -> Any:
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if pathlib.Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:  # pragma: no cover - a broken font file, not a code path
                continue
    return ImageFont.load_default()


def annotate(
    frame: NDArray[np.uint8],
    lines: Sequence[str],
    *,
    corner: str = "tl",
    size: int = 15,
) -> NDArray[np.uint8]:
    """Burn ``lines`` into a copy of an RGB or RGBA frame, on a dark plate so it stays readable.

    A plate rather than an outline because the background here is a thermal image under AGC: it
    can be any brightness anywhere, and white text on a white sky is not a readout.
    """
    from PIL import Image, ImageDraw

    arr = np.asarray(frame, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"frame must be (H, W, 3) or (H, W, 4), got {arr.shape}")
    image = Image.fromarray(arr[..., :3], mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(size)
    step = size + 5
    widths = [int(draw.textlength(line, font=font)) for line in lines] or [0]
    box_w, box_h = max(widths) + 2 * _MARGIN, step * len(lines) + 2 * _MARGIN - 5
    x = _MARGIN if corner.endswith("l") else image.width - box_w - _MARGIN
    y = _MARGIN if corner.startswith("t") else image.height - box_h - _MARGIN
    draw.rectangle([x, y, x + box_w, y + box_h], fill=(0, 0, 0, 150))
    for index, line in enumerate(lines):
        draw.text((x + _MARGIN, y + _MARGIN + index * step), line, font=font, fill=(255, 255, 255))
    out = np.asarray(image, dtype=np.uint8)
    if arr.shape[2] == 4:
        out = np.dstack([out, arr[..., 3]])
    return np.ascontiguousarray(out)


def temperature_bar(
    frame: NDArray[np.uint8],
    values_k: Mapping[str, float],
    span_k: tuple[float, float],
    *,
    width_px: int = 150,
    size: int = 13,
) -> NDArray[np.uint8]:
    """A small horizontal gauge per node, so a *change* is visible without reading the digits.

    The span is fixed by the caller and never auto-scaled: a gauge that rescales itself shows the
    same picture for a 2 K swing and a 40 K one, which is the failure the gauge exists to prevent.
    """
    from PIL import Image, ImageDraw

    arr = np.asarray(frame, dtype=np.uint8)
    image = Image.fromarray(arr[..., :3], mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(size)
    lo, hi = float(span_k[0]), float(span_k[1])
    if not hi > lo:
        raise ValueError("span_k must be increasing")
    step = size + 9
    box_h = step * len(values_k) + 2 * _MARGIN - 6
    label_w = max((int(draw.textlength(f"{n} ", font=font)) for n in values_k), default=0)
    box_w = label_w + width_px + 62 + 2 * _MARGIN
    x, y = _MARGIN, image.height - box_h - _MARGIN
    draw.rectangle([x, y, x + box_w, y + box_h], fill=(0, 0, 0, 150))
    for index, (name, value) in enumerate(values_k.items()):
        row = y + _MARGIN - 3 + index * step
        draw.text((x + _MARGIN, row), name, font=font, fill=(220, 220, 220))
        bx = x + _MARGIN + label_w
        draw.rectangle([bx, row + 3, bx + width_px, row + size - 2], outline=(120, 120, 120))
        filled = int(width_px * min(max((float(value) - lo) / (hi - lo), 0.0), 1.0))
        hot = int(255 * min(max((float(value) - lo) / (hi - lo), 0.0), 1.0))
        draw.rectangle([bx, row + 3, bx + filled, row + size - 2], fill=(255, 255 - hot, 60, 255))
        draw.text(
            (bx + width_px + 8, row),
            f"{float(value) - 273.15:5.1f}C",
            font=font,
            fill=(255, 255, 255),
        )
    out = np.asarray(image, dtype=np.uint8)
    if arr.shape[2] == 4:
        out = np.dstack([out, arr[..., 3]])
    return np.ascontiguousarray(out)


def palette_scale(
    frame: NDArray[np.uint8],
    palette: NDArray[np.uint8],
    span_k: tuple[float, float],
    *,
    ticks: int = 5,
    width_px: int = 26,
    size: int = 13,
    caption: str = "apparent T",
) -> NDArray[np.uint8]:
    """The colour bar: the **actual palette**, from the span's floor to its ceiling, with ticks.

    A thermal image is a picture of a number that has been through a mapping, and the mapping is
    a choice. `temperature_bar` says what each part of the target *is* in Celsius; this says how
    any temperature became the grey (or the colour) on screen, which is the other half and the
    one a viewer cannot reconstruct from the picture. Without it, "that part is brighter" is an
    observation about the display and not about the scene.

    It is drawn from ``palette`` itself -- the same lookup table the display branch indexed to
    make the frame -- rather than from a gradient that resembles it. A bar that only resembled
    the mapping would be a second display path, free to drift from the first, which is the exact
    failure this project keeps a single `quantise_display` to avoid.

    **Only ever drawn beside a fixed span.** Under either AGC mode the mapping is rebuilt from
    each frame's own histogram, so a bar drawn once would be wrong for every frame but one; the
    caller is expected to leave it off there, and that difference between the two videos is
    itself the lesson.
    """
    from PIL import Image, ImageDraw

    arr = np.asarray(frame, dtype=np.uint8)
    lut = np.asarray(palette, dtype=np.uint8)
    if lut.ndim != 2 or lut.shape[0] < 2:
        raise ValueError(f"palette must be a (levels, channels) table, got {lut.shape}")
    lo, hi = float(span_k[0]), float(span_k[1])
    if not hi > lo:
        raise ValueError("span_k must be increasing")

    levels = lut.shape[0]
    bar_h = max(60, int(0.55 * arr.shape[0]))
    label_w = 52
    box_w = width_px + label_w + 2 * _MARGIN
    box_h = bar_h + 2 * _MARGIN + size + 4
    x = arr.shape[1] - box_w - _MARGIN
    y = (arr.shape[0] - box_h) // 2

    image = Image.fromarray(arr[..., :3], mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rectangle([x, y, x + box_w, y + box_h], fill=(0, 0, 0, 150))
    font = _font(size)
    draw.text((x + _MARGIN, y + _MARGIN - 3), caption, font=font, fill=(220, 220, 220))

    # The ramp: row 0 is the top of the bar and the top of the span, so hot is up -- the
    # convention every thermal core ships with.
    top = y + _MARGIN + size + 4
    codes = np.rint(np.linspace(levels - 1, 0, bar_h)).astype(np.int64)
    ramp = np.repeat(lut[codes][:, None, :3], width_px, axis=1)
    image.paste(
        Image.fromarray(np.ascontiguousarray(ramp, dtype=np.uint8), "RGB"), (x + _MARGIN, top)
    )
    draw.rectangle(
        [x + _MARGIN, top, x + _MARGIN + width_px, top + bar_h - 1], outline=(140, 140, 140)
    )
    for index in range(max(2, ticks)):
        f = index / (max(2, ticks) - 1)
        row = int(round(top + f * (bar_h - 1)))
        kelvin = hi - f * (hi - lo)
        draw.line(
            [x + _MARGIN + width_px, row, x + _MARGIN + width_px + 5, row], fill=(200, 200, 200)
        )
        draw.text(
            (x + _MARGIN + width_px + 8, row - size // 2),
            f"{kelvin - 273.15:5.1f}C",
            font=font,
            fill=(255, 255, 255),
        )
    out = np.asarray(image, dtype=np.uint8)
    if arr.shape[2] == 4:
        out = np.dstack([out, arr[..., 3]])
    return np.ascontiguousarray(out)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def encode_mp4(
    frame_glob: str,
    out_path: str | pathlib.Path,
    *,
    fps: float = 30.0,
    crf: int = 17,
    extra: Iterable[str] = (),
) -> pathlib.Path:
    """H.264 in MP4 from a numbered PNG sequence. Raises if ffmpeg is not installed.

    ``yuv420p`` and the even-dimension filter are not decoration: H.264 in this pixel format needs
    both dimensions even, and a 4:2:0 stream is the one that plays everywhere without a codec
    conversation. An odd-height frame otherwise fails at the last moment with an obscure message.
    """
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg is not on PATH; install it or keep the PNG sequence")
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        f"{float(fps):g}",
        "-pattern_type",
        "glob",
        "-i",
        frame_glob,
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        *extra,
        str(out),
    ]
    subprocess.run(command, check=True)
    return out


def overlay_readout(
    frame: NDArray[np.uint8],
    lines: Sequence[str],
    values_k: Mapping[str, float],
    span_k: tuple[float, float] | None,
    *,
    bare: bool = False,
    palette: NDArray[np.uint8] | None = None,
) -> NDArray[np.uint8]:
    """The standard flight-film readout: caption block top-left, node gauges bottom-left.

    One function rather than one per stage, because the two videos of a scene differ only in how
    temperature was mapped to brightness, and the caption is the only thing that says which -- so
    it has to look the same in both or the comparison is doing the reader no favours.

    ``span_k`` may be ``None``, and then the **temperature gauge is left off entirely**. That is
    the reflective-band case (M10.23): the picture is reflected sunlight and its brightness axis
    is not a temperature axis, so a gauge in kelvin beside it would be a scale for a quantity the
    image is not showing -- worse than no gauge, because a reader would use it. The caption block,
    which carries the node temperatures as *numbers*, stays: those are real and are still what the
    thermal model computed.
    """
    arr = np.asarray(frame, dtype=np.uint8)
    if bare:
        return np.ascontiguousarray(arr[..., :3])
    annotated = annotate(arr, lines)
    if span_k is None:
        return np.ascontiguousarray(annotated[..., :3])
    annotated = temperature_bar(annotated, values_k, span_k)
    # `palette` is what turns the gauge into a *scale*: the gauge says what each part is, the
    # colour bar says how any temperature became the pixel beside it. Passed only where the span
    # is fixed, because under AGC the mapping is rebuilt every frame (see `palette_scale`).
    if palette is not None:
        annotated = palette_scale(annotated, palette, span_k)
    return np.ascontiguousarray(annotated[..., :3])
