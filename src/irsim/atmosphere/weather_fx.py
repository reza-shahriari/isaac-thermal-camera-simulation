"""Read `isaac-weather-fx`'s cloud field through this project's own deck contract.

The submodule at ``third_party/isaac-weather-fx`` owns the sky: where the sun and the moon are,
what the cloud looks like, what kind of day it is. This module is the seam that lets the infrared
band integrate **that** cloud instead of a second one of its own.

**Why a seam and not a rewrite.** `SkyModel.radiance_field_from_deck` asks a deck for exactly two
things per ray -- the optical depth it accumulated and the height the emission came from -- and
does the radiometry itself. That contract is narrow enough to satisfy from a foreign density
field, so the whole of this project's infrared physics (the band LUT, the clear-sky column, the
atmosphere in front of the cloud, the apparent-temperature inversion) is reused unchanged. What
changes is only *which array* says where the cloud is.

**And that is the point.** Before this, the infrared band marched
:class:`~irsim.atmosphere.cloud_deck.CloudDeck` while the visible companion baked a dome from a
different object, and the two agreed only as well as two implementations of the same idea ever
do -- measured at 26 % of pixels against 57 % at one point (AT.15). Now both bands sample one
``CloudField``: the visible dome through weather-fx's own renderer backend, the infrared band
through this adapter. They cannot disagree about where a cloud is, because there is one cloud.

**The engine-free rule still holds.** ``weather_fx.core`` is pure Python and numpy with no
Omniverse imports -- that is a property its own test suite enforces -- so importing it here does
not breach CLAUDE.md #1, which is about engine modules. The import is still done lazily and with
a message that says what to do, because the submodule may not be initialised.
"""

from __future__ import annotations

import math
import pathlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from irsim.atmosphere.cloud import CLOUD_OD_RATIO
from irsim.atmosphere.cloud_deck import MarchResult

__all__ = [
    "WEATHER_FX_PACKAGE",
    "WeatherFxDeck",
    "cloud_field_from_spec",
    "ensure_weather_fx_on_path",
    "weather_fx_available",
]

#: Where the submodule keeps its importable package. It is a Kit extension rather than a wheel,
#: so the path is the extension directory and not a site-packages entry.
WEATHER_FX_PACKAGE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "third_party"
    / "isaac-weather-fx"
    / "exts"
    / "weather.fx"
)


def ensure_weather_fx_on_path() -> None:
    """Put the submodule's package directory on ``sys.path`` if it is not already importable.

    The same pattern as :func:`irsim_isaac.env.ensure_warp_on_path`: a dependency that ships as an
    extension directory rather than as an installed distribution has to be found rather than
    imported. Idempotent, and a no-op once the package is installed properly.
    """
    try:
        import weather_fx.core  # noqa: F401

        return
    except ImportError:
        pass
    if not WEATHER_FX_PACKAGE.is_dir():
        raise ImportError(
            f"the isaac-weather-fx submodule is not checked out at {WEATHER_FX_PACKAGE}.\n"
            "  run: git submodule update --init third_party/isaac-weather-fx"
        )
    path = str(WEATHER_FX_PACKAGE)
    if path not in sys.path:
        sys.path.insert(0, path)


def weather_fx_available() -> bool:
    """Whether the submodule can be imported, without raising if it cannot."""
    try:
        ensure_weather_fx_on_path()
        import weather_fx.core  # noqa: F401
    except ImportError:
        return False
    return True


def cloud_field_from_spec(
    *,
    cover: float,
    genus: str = "cumulus",
    base_m: float = 0.0,
    temperature_c: float = 20.0,
    dewpoint_c: float = 10.0,
    thickness_m: float = 0.0,
    optical_depth: float = 0.0,
    feature_m: float = 400.0,
    cells: int = 256,
    levels: int = 48,
    cell_m: float = 60.0,
    seed: int = 0,
) -> Any:
    """Build a ``weather_fx.core.clouds.CloudField`` from plain numbers.

    The arguments are exactly the ``clouds`` section of a weather-fx state, so a scene config and
    the extension's own UI describe a cloud the same way and nothing has to be translated twice.
    A ``base_m`` of zero means "compute it from the temperature and dew point", which is why a
    whole field of cumulus has its bases on one level.
    """
    ensure_weather_fx_on_path()
    from weather_fx.core.clouds import CloudField, cloud_profile, lifting_condensation_level_m

    base = float(base_m)
    if base <= 0.0:
        base = lifting_condensation_level_m(float(temperature_c), float(dewpoint_c))
    return CloudField(
        cover=float(cover),
        base_m=base,
        profile=cloud_profile(str(genus)),
        cell_m=float(cell_m),
        cells=int(cells),
        levels=int(levels),
        seed=int(seed),
        thickness_m=float(thickness_m),
        optical_depth=float(optical_depth),
        feature_m=float(feature_m),
    )


@dataclass
class WeatherFxDeck:
    """A weather-fx ``CloudField``, presented through this project's deck march contract.

    Satisfies what :meth:`irsim.atmosphere.sky.SkyModel.radiance_field_from_deck` needs -- a
    ``march`` returning optical depth and emission height -- so the infrared radiometry is
    untouched and only the geometry comes from elsewhere.
    """

    field: Any
    #: Ratio of the band's extinction to the visible one. A cloud carries one optical depth and
    #: each band derives what it needs from it (ADR 0126), which is the same convention the
    #: native deck uses; it is kept here so an infrared caller can override it per band.
    od_ratio: float = CLOUD_OD_RATIO
    #: Steps the march takes through the field. The default matches weather-fx's own, so the
    #: infrared band and the visible dome integrate the same cloud at the same fidelity.
    steps: int = 64

    @property
    def base_m(self) -> float:
        return float(self.field.base_m)

    @property
    def top_m(self) -> float:
        return float(self.field.top_m)

    @property
    def thickness_m(self) -> float:
        return float(self.field.thickness_m)

    @property
    def optical_depth(self) -> float:
        return float(self.field.optical_depth)

    @property
    def cover(self) -> float:
        """The sky fraction the field actually covers, measured rather than requested."""
        return float(self.field.measured_cover)

    def march(
        self,
        elevation_rad: Any,
        azimuth_rad: Any,
        *,
        origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        steps: int | None = None,
    ) -> MarchResult:
        """Integrate the field along each ray and answer in this project's own terms.

        **The two conventions already agree, and that is checked rather than assumed.** Both
        projects place a direction as ``[cos el sin az, sin el, -cos el cos az]`` -- +Y up, -Z at
        azimuth zero, +X at ninety. weather-fx writes it that way in
        ``BodyPosition.direction``; this project writes it that way in
        ``irsim_isaac.visible_sky.stage_direction``. A test pins the pair, because a silent
        disagreement here rotates the whole cloud field about the observer and looks entirely
        plausible.
        """
        el = np.asarray(elevation_rad, dtype=np.float64)
        az = np.asarray(azimuth_rad, dtype=np.float64)
        el, az = np.broadcast_arrays(el, az)
        direction = np.stack(
            [
                np.cos(el) * np.sin(az),
                np.sin(el),
                -np.cos(el) * np.cos(az),
            ],
            axis=-1,
        )
        origin = np.zeros(direction.shape, dtype=np.float64)
        origin[...] = np.asarray(origin_m, dtype=np.float64)

        result = self.field.march(origin, direction, steps=steps or self.steps)
        # This project measures the emission height **above the cloud base**, because the
        # temperature it lapses from is the base's. weather-fx reports it as a height above the
        # ground, which is the frame a renderer wants. Subtracting the base is the whole
        # conversion, and getting it wrong is a cloud that emits at ground temperature.
        above_base = np.maximum(
            np.asarray(result.emission_height_m, dtype=np.float64) - self.base_m, 0.0
        )
        return MarchResult(
            optical_depth=np.asarray(result.optical_depth, dtype=np.float64),
            emission_height_m=above_base,
        )

    def adequate_steps(self, elevation_rad: Any) -> int:
        """How many steps this field wants for the shallowest ray in the array.

        Present so a caller written against the native deck keeps working. weather-fx grows its
        own steps geometrically along the ray, so the count is far less sensitive here than it is
        for a uniform march -- the answer is a sanity bound rather than a requirement.
        """
        el = np.abs(np.asarray(elevation_rad, dtype=np.float64))
        positive = el[el > 0.0]
        lowest = float(np.min(positive)) if positive.size else 0.5 * math.pi
        lowest = max(lowest, math.radians(2.0))
        want = self.thickness_m / math.sin(lowest) / 120.0
        return int(min(256, max(32, math.ceil(want))))
