"""A person's seventeen skin temperatures from JOS-3, on the scene's one weather (HU.4).

docs/physics-model.md §6.1 (the person's surfaces), §6.2 (convection), §16.2 (skin);
roadmap HU.4; ADR 0122 (a person is two surfaces), ADR 0192 (the body is a fixed taxonomy),
ADR 0194 (JOS-3 is the physiology, installed and not rewritten).

PH.12 authored one skin temperature for the whole body from ISO 7730's set point. Measured people
are not one temperature: forehead near 34.7 °C against a nose at 33.5 °C indoors, palms near
28 °C and fingers near 25 °C in a cool room -- face minus fingers is about 10 K, two hundred
NETDs. The reason is thermoregulation: the body defends its core by letting the extremities go,
so skin temperature is a *physiological* field, not a surface energy balance (ADR 0122). The
model of that field with open code is **JOS-3** (Takahashi, Tanabe et al. 2021, Energy &
Buildings 231:110575, doi:10.1016/j.enbuild.2020.110575; MIT), 17 segments × core, muscle, fat
and skin nodes, blood flow, sweating, shivering, as its authors ship it on PyPI (``jos3``,
pure NumPy).
This module drives it and does not reimplement any of it: the segments it solves are, by name
and in order, the segments of ``configs/humans/body_schema.yaml``, so every ``skin.<Segment>``
prim of a ``kind: human`` asset binds to its own node.

**Forcing comes from the one ``WeatherSeries``** (CLAUDE.md #6): air temperature, humidity and
wind at every tick, and a **mean radiant temperature** built from the same sky and sun every
surface in the scene is forced by -- half the body sees the sky (``irsim.thermal.longwave``'s
clear-sky emissivity with the weather's cloud), half the ground, and the sun's beam lands on
Fanger's projected area of a standing person (VDI 3787 Part 2, the outdoor T_mrt every
biometeorological index is built on). So a clear night pulls the radiant temperature below the
air and a summer noon lifts it 20 K above, out of one weather. ASHRAE 55's SolarCal
(the comfort package's ``solar_gain``) was tried first and rejected: built for a person behind a
window over a reflective floor, it gave +55 K at this scene's noon, three times the outdoor
balance. The ground is taken at air temperature and reflects 0.2 of the sun (ESTIMATED; a scene's
own ground field is the upgrade, ADR 0194).

**What a person is here.** :class:`HumanBodySolver` is a ``TemperatureSolver`` for the mean
skin, owns the JOS-3 state, and hands out one :class:`SegmentView` per segment -- also a
``TemperatureSolver`` -- so a scene registers ``<name>.skin_Head`` … as ordinary targets. The
views step the body **once** per tick however many of them are asked: the body remembers the
step it last took. A person is acclimatised before the first frame by running at the scene's
start conditions for :data:`ACCLIMATISE_S`, the same way every other node is spun up.

⚠️ JOS-3 is a comfort model validated indoors; a bare body at 0 °C in a 1 m/s wind for an hour is
its extrapolation, as ISO 7730's was for PH.12. The numbers past its range are the model's.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from irsim.config.humans import JOS3_SEGMENTS, HumanSpec
from irsim.thermal.solvers import SolverState
from irsim.thermal.weather import WeatherSeries

__all__ = [
    "ACCLIMATISE_S",
    "HEAD_LAYERS",
    "HumanBodySolver",
    "SegmentView",
    "clo_by_segment",
    "mean_radiant_temperature_k",
    "projected_area_factor",
]

#: How long the body runs at the scene's start conditions before the first frame, seconds.
#: Measured on JOS-3 at its own neutral (28.8 °C, seated, bare): mean skin 34.28 °C after 1 h,
#: 33.94 after 2 h, 33.75 after 3 h, 33.51 after 6 h -- it creeps for hours. Two hours is where
#: the drift falls under 0.2 K/h and the mean sits within 0.3 K of Fanger's 33.7 °C; it is also
#: about as long as a pedestrian has been outdoors. A person who has just stepped out of a warm
#: building is a different scene and would author a shorter run.
ACCLIMATISE_S = 7200.0
#: Optics-only layers that render at the head's skin temperature.
HEAD_LAYERS = ("eyes", "eyebrows", "eyelashes", "hair")
#: The smallest air speed JOS-3 is driven with: its convection correlations are for v ≥ 0.1 m/s.
MIN_AIR_SPEED_M_S = 0.1
#: JOS-3's activity in met for the postures a scene may author (ISO 8996 Table A.2: standing
#: relaxed 1.2, seated 1.0, lying 0.8).
MET_BY_POSTURE: dict[str, float] = {"standing": 1.25, "sitting": 1.0, "lying": 0.8}

#: Skin's short-wave absorptance (the library's `human_skin` α_sol) and long-wave emissivity
#: (0.97, ISO 7730's; the library's 0.98 is for the camera's band), for the radiant balance.
SKIN_SOLAR_ABSORPTANCE = 0.65
SKIN_EMISSIVITY = 0.97
#: What the ground gives back of the sun (ESTIMATED: grass and dry soil 0.15–0.25).
GROUND_REFLECTANCE = 0.2

Posture = Literal["standing", "sitting", "lying"]


def clo_by_segment(spec: HumanSpec, slots: Mapping[str, Any]) -> NDArray[np.float64]:
    """Insulation per segment, clo: the garments on each slot summed over what they cover.

    ``slots`` is the body schema's ``garment_slots``; a garment's own ``covers`` wins over the
    slot's default. A bare segment is 0 clo. Layered garments on one segment add (ISO 9920's
    first approximation for an ensemble; the proper sum is smaller by the overlap, HU.5).
    """
    out = np.zeros(len(JOS3_SEGMENTS), dtype=np.float64)
    index = {name: i for i, name in enumerate(JOS3_SEGMENTS)}
    for slot, garment in spec.garments.items():
        covers = garment.covers if garment.covers is not None else list(slots[slot].covers)
        for seg in covers:
            out[index[seg]] += garment.clo
    return out


def projected_area_factor(elevation_deg: float) -> float:
    """Fanger's projected-area factor of a standing person for a beam at this elevation.

    The fraction of the body's surface that faces the sun, averaged over azimuth: 0.308 at the
    horizon, falling toward the zenith (Fanger 1970, as fitted in VDI 3787 Part 2 / RayMan:
    f_p = 0.308 · cos(γ · (0.998 − γ² / 50000)), γ in degrees).
    """
    g = float(np.clip(elevation_deg, 0.0, 90.0))
    return float(0.308 * np.cos(np.radians(g * (0.998 - g * g / 50000.0))))


def mean_radiant_temperature_k(
    sample: Any,
    elevation_deg: float,
    dni_w_m2: float,
    dhi_w_m2: float,
    *,
    absorptance: float = SKIN_SOLAR_ABSORPTANCE,
    emissivity: float = SKIN_EMISSIVITY,
    ground_reflectance: float = GROUND_REFLECTANCE,
    t_ground_k: float | None = None,
    sky_view: float = 0.5,
) -> float:
    """The outdoor mean radiant temperature a standing person is exposed to, kelvin.

    VDI 3787 Part 2's balance: the long-wave irradiance from sky and ground (the project's own
    ``longwave_down_from_sample``, with the weather's vapour pressure and cloud), plus the
    short-wave the body absorbs -- the beam on the projected area, half the diffuse, half the
    ground's reflection -- weighted by α_k / ε_p, all divided by σ and raised to the quarter.
    With no sun under an overcast sky and the ground at air temperature this is exactly the
    air temperature; under a clear night sky it is below it.
    """
    from irsim.radiometry.constants import SIGMA_SB
    from irsim.thermal.longwave import longwave_down_from_sample

    t_ground = float(sample.t_air_k) if t_ground_k is None else float(t_ground_k)
    q_lw = float(longwave_down_from_sample(sample, sky_view, t_ground))
    if elevation_deg > 0.0:
        dni, dhi = max(float(dni_w_m2), 0.0), max(float(dhi_w_m2), 0.0)
        ghi = dhi + dni * float(np.sin(np.radians(elevation_deg)))
        q_sw = (
            projected_area_factor(elevation_deg) * dni + 0.5 * dhi + 0.5 * ground_reflectance * ghi
        )
    else:
        q_sw = 0.0
    return float(((q_lw + (absorptance / emissivity) * q_sw) / SIGMA_SB) ** 0.25)


@dataclass(frozen=True)
class _Forcing:
    tdb_c: float
    tr_c: float
    rh_percent: float
    v_m_s: float


class HumanBodySolver:
    """JOS-3 on the scene's weather: the mean skin as a target, every segment as a view."""

    def __init__(
        self,
        spec: HumanSpec,
        weather: WeatherSeries,
        t0_s: float,
        *,
        clo: NDArray[np.float64] | None = None,
        activity_met: float | None = None,
        posture: Posture = "standing",
        site_latitude_deg: float | None = None,
        site_longitude_deg: float | None = None,
        acclimatise_s: float = ACCLIMATISE_S,
        solar: bool = True,
    ) -> None:
        try:
            from jos3 import JOS3
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "a `solver: human` target needs JOS-3: pip install jos3 (a project dependency)"
            ) from exc
        ph = spec.phenotype
        self.spec = spec
        self.weather = weather
        self.posture: Posture = posture
        self.activity_met = float(
            activity_met if activity_met is not None else MET_BY_POSTURE[posture]
        )
        self._lat, self._lon = site_latitude_deg, site_longitude_deg
        self._solar = bool(
            solar and site_latitude_deg is not None and site_longitude_deg is not None
        )
        self._model = JOS3(
            height=float(ph.height_m),
            weight=float(ph.mass_kg),
            age=int(round(ph.age_y)),
            sex=ph.sex,
        )
        names = list(self._model.bodyname)
        if len(names) != len(JOS3_SEGMENTS):
            raise RuntimeError(
                f"JOS-3 has {len(names)} segments, the body schema {len(JOS3_SEGMENTS)}"
            )
        self._clo = (
            np.zeros(len(JOS3_SEGMENTS)) if clo is None else np.asarray(clo, dtype=np.float64)
        )
        if self._clo.shape != (len(JOS3_SEGMENTS),) or np.any(self._clo < 0.0):
            raise ValueError("clo must be one non-negative value per segment")
        self._model.Icl = self._clo
        self._model.PAR = self.activity_met
        self._model.posture = posture
        self._t_s = float(t0_s)
        self._last_step: tuple[float, float] | None = None
        self.steps = 0
        self.last_forcing: _Forcing | None = None
        if acclimatise_s > 0.0:
            self._apply_forcing(self._t_s)
            dt = 60.0
            n = max(1, int(round(acclimatise_s / dt)))
            self._model.simulate(times=n, dtime=dt, output=False)

    # -- forcing ------------------------------------------------------------------------------

    def forcing_at(self, t_s: float) -> _Forcing:
        """Air, radiant, humidity and wind at ``t_s`` on the weather's axis."""
        sample = self.weather.at(t_s)
        tdb = float(sample.t_air_k) - 273.15
        tr = tdb
        if self._solar:
            from irsim.thermal.solar import sun_position_utc

            when = self.weather.epoch_utc + timedelta(seconds=float(t_s))
            sun = sun_position_utc(float(self._lat), float(self._lon), when)  # type: ignore[arg-type]
            elev = float(np.asarray(sun.elevation_deg).reshape(-1)[0])
            dni = 0.0 if elev <= 0.0 else float(sample.dni_w_m2)
            tr = mean_radiant_temperature_k(sample, elev, dni, float(sample.dhi_w_m2)) - 273.15
        return _Forcing(
            tdb_c=tdb,
            tr_c=tr,
            rh_percent=100.0 * float(sample.rh_fraction),
            v_m_s=max(float(sample.wind_speed_m_s), MIN_AIR_SPEED_M_S),
        )

    def _apply_forcing(self, t_s: float) -> None:
        f = self.forcing_at(t_s)
        self._model.Ta = f.tdb_c
        self._model.Tr = f.tr_c
        self._model.RH = f.rh_percent
        self._model.Va = f.v_m_s
        self.last_forcing = f

    # -- the TemperatureSolver protocol -------------------------------------------------------

    def advance(self, t_s: float, dt_s: float) -> float:
        """Step the body from ``t_s`` by ``dt_s`` once; repeating the same step is free."""
        key = (float(t_s), float(dt_s))
        if self._last_step == key:
            return self.temperature()
        if dt_s > 0.0:
            self._apply_forcing(float(t_s))
            self._model.simulate(times=1, dtime=float(dt_s), output=False)
            self.steps += 1
        self._t_s = float(t_s) + float(dt_s)
        self._last_step = key
        return self.temperature()

    def temperature(self) -> float:
        return float(self._model.TskMean) + 273.15

    @property
    def state(self) -> SolverState:
        return SolverState(t_s=self._t_s, temperature_k=self.temperature())

    # -- the segments -------------------------------------------------------------------------

    @property
    def segment_names(self) -> tuple[str, ...]:
        return JOS3_SEGMENTS

    def skin_k(self) -> NDArray[np.float64]:
        """Every segment's skin temperature, kelvin, in the body schema's order."""
        return np.asarray(self._model.Tsk, dtype=np.float64) + 273.15

    def segment_temperature_k(self, name: str) -> float:
        return float(self.skin_k()[JOS3_SEGMENTS.index(name)])

    def derived_targets(self, name: str) -> dict[str, SegmentView]:
        """``<name>.skin_<Segment>`` for every segment, and the head's optics-only layers.

        The spelling is the prim's: a ``kind: human`` asset's segment objects are ``skin_<Segment>``
        (a USD prim name cannot carry a dot), and the renderer looks a prim's thermal node up as
        ``<human>.<leaf>``.
        """
        out = {f"{name}.skin_{seg}": SegmentView(self, seg) for seg in JOS3_SEGMENTS}
        for layer in HEAD_LAYERS:
            out[f"{name}.{layer}"] = SegmentView(self, "Head")
        return out


class SegmentView:
    """One segment's skin temperature as a ``TemperatureSolver``; steps the body it belongs to."""

    def __init__(self, body: HumanBodySolver, segment: str) -> None:
        if segment not in JOS3_SEGMENTS:
            raise KeyError(f"unknown segment {segment!r}")
        self.body = body
        self.segment = segment

    def advance(self, t_s: float, dt_s: float) -> float:
        self.body.advance(t_s, dt_s)
        return self.temperature()

    def temperature(self) -> float:
        return self.body.segment_temperature_k(self.segment)

    @property
    def state(self) -> SolverState:
        return SolverState(t_s=self.body.state.t_s, temperature_k=self.temperature())
