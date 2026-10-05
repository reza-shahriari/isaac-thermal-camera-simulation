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
    "apparent_target_span_k",
    "target_code_span",
    "interior",
    "with_margin",
    "readout_margin",
    "READOUT_MARGIN",
    "overlay_readout",
    "temperature_bar",
    "palette_scale",
    "encode_mp4",
    "ffmpeg_available",
]

_MARGIN = 10

#: (top, left) pixels of empty canvas that `with_margin` puts before a frame, so the readout lands
#: beside the scene instead of over it. The caption block is ~90 px tall and the gauge panel ~380
#: px wide; both even, so an encoded frame keeps even dimensions.
READOUT_MARGIN: tuple[int, int] = (96, 400)


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


def apparent_target_span_k(
    target_pixels: NDArray[np.floating], *, floor_percentile: float = 1.0
) -> tuple[float, float]:
    """The white-hot span of a demo clip, from the target's own *apparent* pixels (`SC.24`).

    The floor is the target's ``floor_percentile`` (1 %), so a stray rim pixel blended with the
    sky does not pull it down. The top is the **hottest** target pixel, not a percentile: the
    hottest parts are few pixels -- four motor bells on a Phantom 4 -- so a 99th percentile lands
    below them and they clip to one white, which is exactly the contrast the clip exists to show.

    Apparent, not kinetic: a part with ε < 1 reflects the cold sky, so it reads colder than
    `target_span_k`'s node temperatures say (15 … 38 °C kinetic against −26 … +34 °C apparent at
    `phantom4_perpart` frame 96), and a span from the nodes would clip the picture's cold end.
    """
    values = np.asarray(target_pixels, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("need at least one finite target pixel to span")
    lo, hi = float(np.percentile(values, floor_percentile)), float(values.max())
    return (lo, hi if hi > lo else lo + 1.0)


def target_code_span(display8: NDArray[np.uint8], mask: NDArray[np.bool_]) -> int:
    """Grey codes the target spans in an 8-bit frame: max − min over ``mask`` (0 if empty).

    The one number the AGC defects were measured in (S52: 3 of 256 on `phantom4_perpart`), so the
    render and the re-display both report it the same way.
    """
    codes = np.asarray(display8)
    if codes.ndim == 3:
        codes = codes[..., 0]
    picked = codes[np.asarray(mask, dtype=bool)]
    return int(picked.max()) - int(picked.min()) if picked.size else 0


def interior(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """The mask with its one-pixel rim removed: pixels whose four neighbours are all in it.

    A silhouette's rim pixels are part target and part sky once the optics and the supersampling
    have mixed them, so their display codes run all the way to the sky's. Counted as target, they
    make a starved target look well spread: on `phantom4_perpart` the whole aircraft mask read
    112-231 codes where its parts, measured away from the rim, read 18-33. The frame edge counts as
    outside, so a mask touching it loses that row too.
    """
    m: NDArray[np.bool_] = np.asarray(mask, dtype=bool)
    out: NDArray[np.bool_] = m.copy()
    out[1:, :] &= m[:-1, :]
    out[:-1, :] &= m[1:, :]
    out[:, 1:] &= m[:, :-1]
    out[:, :-1] &= m[:, 1:]
    out[0, :] = out[-1, :] = False
    out[:, 0] = out[:, -1] = False
    return out


def with_margin(
    image: NDArray[np.uint8], margin: tuple[int, int] = READOUT_MARGIN, *, right: int = 0
) -> NDArray[np.uint8]:
    """Put a frame below and right of an empty margin, so the readout covers no scene.

    `overlay_readout` draws the caption top-left and the node gauges bottom-left; on
    `phantom4_perpart` the gauges sat over half of the aircraft. ``right`` adds a margin on the
    other side for the colour bar, which `palette_scale` draws against the canvas's right edge.
    The frame itself is untouched, and the result is always RGB.
    """
    arr = np.asarray(image, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    top, left = margin
    if right < 0:
        raise ValueError("right must not be negative")
    height, width = arr.shape[:2]
    canvas = np.zeros((height + top, width + left + right, 3), dtype=np.uint8)
    canvas[top:, left : left + width] = arr[..., :3]
    return canvas


def readout_margin(
    values_k: Mapping[str, float], *, scale: bool, size: int = 13
) -> tuple[tuple[int, int], int]:
    """The ``(top, left)`` margin and the ``right`` margin a readout needs to cover no scene.

    :data:`READOUT_MARGIN`'s 400 px fits a handful of short node names; the solved Phantom 4 has
    nineteen, the longest ``phantom4.motor_mount_front_right``, and its gauges ran ~100 px into
    the frame -- over the aircraft as it left the field on the left (`EV.20`). The width is
    measured with the font the gauges are drawn in, so it is what :func:`temperature_bar` will
    actually occupy. ``scale`` reserves the colour bar's width on the right.
    """
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    font = _font(size)
    label_w = max((int(draw.textlength(f"{n} ", font=font)) for n in values_k), default=0)
    gauge = label_w + _GAUGE_WIDTH_PX + 62 + 2 * _MARGIN
    left = max(READOUT_MARGIN[1], gauge + 2 * _MARGIN)
    right = _SCALE_WIDTH_PX + 52 + 4 * _MARGIN if scale else 0
    # h264 refuses odd dimensions, so each margin is rounded up to an even width.
    return (READOUT_MARGIN[0], left + left % 2), right + right % 2


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


#: :func:`temperature_bar`'s and :func:`palette_scale`'s default widths, named so
#: :func:`readout_margin` can reserve exactly what they draw.
_GAUGE_WIDTH_PX = 150
_SCALE_WIDTH_PX = 26


def temperature_bar(
    frame: NDArray[np.uint8],
    values_k: Mapping[str, float],
    span_k: tuple[float, float],
    *,
    width_px: int = _GAUGE_WIDTH_PX,
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
    width_px: int = _SCALE_WIDTH_PX,
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
