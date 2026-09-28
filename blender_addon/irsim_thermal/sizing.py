"""Is this model the size of the real thing? Only answered when the user asks.

Pure Python, no ``bpy``.

A model in the wrong units still renders a perfectly plausible thermal image, which is what makes
the mistake expensive: the Phantom 4 FBX arrived in centimetres, declared no unit, and would have
entered a scene 41 m wide (ADR 0128). What gives a units mistake away is not that the model is
large or small -- a 300 m ship *is* 300 m -- but that it is wrong by one of a handful of exact
factors: 10, 100, 1000, 2.54 or 0.3048.

So the check is never "is this too big". By default nothing is checked and nothing warns; the size
is simply shown in metres beside a familiar comparison. When the user asks for guidance they name
the kind of object, and a warning appears only if the size is outside that kind's typical range
**and** one of the unit factors brings it inside. With a real published dimension the scale is
exact and no range is needed at all.
"""

import json
import math
import pathlib
from dataclasses import dataclass

__all__ = [
    "COMPARISONS",
    "UNIT_FACTORS",
    "Diagnosis",
    "SizeRange",
    "comparison",
    "diagnose",
    "load_ranges",
    "scale_for_known",
]

DATA = pathlib.Path(__file__).resolve().parent / "data" / "size_ranges.json"

#: ``(factor, what went wrong)``: multiplying the model by ``factor`` undoes the mistake.
UNIT_FACTORS: tuple[tuple[float, str], ...] = (
    (0.001, "millimetres read as metres"),
    (0.01, "centimetres read as metres"),
    (0.0254, "inches read as metres"),
    (0.1, "decimetres read as metres"),
    (0.3048, "feet read as metres"),
    (1.0 / 0.3048, "metres read as feet"),
    (10.0, "metres read as decimetres"),
    (1.0 / 0.0254, "metres read as inches"),
    (100.0, "metres read as centimetres"),
    (1000.0, "metres read as millimetres"),
)

#: A familiar object per order of magnitude, so a bare number has something to be compared with.
COMPARISONS: tuple[tuple[float, str], ...] = (
    (0.01, "a coin"),
    (0.15, "a phone"),
    (0.45, "a small drone"),
    (1.8, "a person"),
    (4.5, "a car"),
    (12.0, "a bus"),
    (40.0, "an airliner's wingspan"),
    (300.0, "a large ship"),
    (1000.0, "a kilometre"),
)

#: How close a known-dimension scale must come to a unit factor to be called that unit. Real models
#: miss published dimensions by a few per cent -- the Phantom 4 FBX is 207 mm tall against DJI's
#: 196 mm, 5.6 % off, and was centimetres all the same -- while the nearest two unit factors are
#: 2.5x apart, so 10 % names the unit without ever confusing two of them.
UNIT_MATCH_TOLERANCE = 0.10


@dataclass(frozen=True)
class SizeRange:
    key: str
    label: str
    min_m: float
    max_m: float
    note: str = ""

    @property
    def centre_m(self) -> float:
        """Geometric centre: sizes span orders of magnitude, so the middle is in log space."""
        return math.sqrt(self.min_m * self.max_m)


@dataclass(frozen=True)
class Diagnosis:
    #: ``within`` the range, explained by a ``unit`` mistake, or ``outside`` with no explanation.
    status: str
    message: str
    factor: float | None = None
    corrected_m: float | None = None


def load_ranges(path: pathlib.Path | None = None) -> list[SizeRange]:
    raw = json.loads((path or DATA).read_text(encoding="utf-8"))
    ranges = [SizeRange(**entry) for entry in raw["ranges"]]
    for r in ranges:
        if not 0.0 < r.min_m < r.max_m:
            raise ValueError(f"size range {r.key!r}: need 0 < min_m < max_m")
    return ranges


def _fmt(metres: float) -> str:
    if metres < 1.0:
        return f"{metres * 100.0:.3g} cm"
    return f"{metres:.3g} m"


def comparison(largest_m: float) -> str:
    """``"about the size of a car"`` -- the closest familiar object in log space."""
    if largest_m <= 0.0:
        return "no geometry"
    size, what = min(COMPARISONS, key=lambda c: abs(math.log(largest_m / c[0])))
    ratio = largest_m / size
    if 0.67 <= ratio <= 1.5:
        return f"about the size of {what}"
    if ratio > 1.0:
        return f"about {ratio:.2g}x {what}"
    return f"about 1/{1.0 / ratio:.2g} of {what}"


def diagnose(largest_m: float, rng: SizeRange) -> Diagnosis:
    """Compare a model's largest dimension with the typical range for its kind.

    Silent unless the size is outside the range; when it is, the unit factor that brings it
    closest to the middle of the range is offered, and only if one brings it inside at all.
    """
    if largest_m <= 0.0:
        return Diagnosis("outside", "The model has no geometry to measure.")
    if rng.min_m <= largest_m <= rng.max_m:
        return Diagnosis(
            "within",
            f"{_fmt(largest_m)} is a normal size for a {rng.label.lower()} "
            f"({_fmt(rng.min_m)} to {_fmt(rng.max_m)}).",
        )
    fits = [
        (factor, why)
        for factor, why in UNIT_FACTORS
        if rng.min_m <= largest_m * factor <= rng.max_m
    ]
    if not fits:
        return Diagnosis(
            "outside",
            f"{_fmt(largest_m)} is outside the usual {_fmt(rng.min_m)} to {_fmt(rng.max_m)} for a "
            f"{rng.label.lower()}, and no unit mistake explains it. It may simply be unusual; "
            "check it against a real dimension if you know one.",
        )
    factor, why = min(fits, key=lambda f: abs(math.log(largest_m * f[0] / rng.centre_m)))
    corrected = largest_m * factor
    return Diagnosis(
        "unit",
        f"{_fmt(largest_m)} is not a usual size for a {rng.label.lower()}. If the file was in "
        f"other units ({why}), scaling by {factor:g} gives {_fmt(corrected)}.",
        factor=factor,
        corrected_m=corrected,
    )


def scale_for_known(measured_m: float, known_m: float) -> tuple[float, str | None]:
    """The exact scale that makes a measured dimension equal a known real one.

    Returns ``(factor, unit explanation or None)``. The explanation names a unit mistake when the
    factor lies within :data:`UNIT_MATCH_TOLERANCE` of one, because then the file was almost
    certainly in that unit and the remainder is modelling tolerance, not a second error.
    """
    if measured_m <= 0.0 or known_m <= 0.0:
        raise ValueError("both dimensions must be positive")
    factor = known_m / measured_m
    for unit, why in UNIT_FACTORS:
        if abs(factor / unit - 1.0) <= UNIT_MATCH_TOLERANCE:
            return factor, why
    return factor, None
