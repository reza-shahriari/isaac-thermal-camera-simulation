"""The same balance over many facets at once, and the spin-up that makes its answer meaningful.

Two things, because they are two halves of one problem. A scene has thousands of surfaces and the
scalar solver of M6.7/M6.8 is the oracle, not the implementation — so :class:`FacetSolver` runs
the identical equations over ``(N,)`` arrays, and a test holds it to **0.1 mK** against N separate
scalar runs rather than to "close enough".

And an initial condition has to come from somewhere. A surface temperature is a *memory* — asphalt
at 06:00 is carrying yesterday afternoon — so starting a scene at the air temperature is starting
it wrong by several kelvin and staying wrong for hours. :func:`spin_up` integrates the same facets
through the preceding day or two and returns the state the weather implies, cached on the things
that actually determine it.

**float64 inside, float32 only at the boundary.** The balance subtracts numbers around 400 W m⁻²
to leave a residual of a few, and a diurnal run accumulates ~10⁵ steps of that; in float32 the
subtraction alone loses four digits before the accumulation starts. The solver refuses a float16
input outright and works in float64 throughout (CLAUDE.md #2).

**Conduction between facets, and the tick that survives it** (TC.1, ADR 0094). A solver built
with a :class:`~irsim.thermal.conduction.ConductionOperator` steps the surface balance exactly as
before and the conduction term implicitly (backward Euler, prefactored once per tick size), so a
60 s scene tick stands over a joint whose own time constant is a tenth of a second. Without an
operator the step is the midpoint rule it always was, bit for bit. Either way the explicit part is
now guarded: §6.4's bound ``2C/(h + 4εσT³)`` is checked every step on the forcing's actual ``h``
and raises with the numbers -- it never was before, and ``tick_s`` up to 3600 s parsed.

docs/physics-model.md §6.4, §15 T1; ADR 0036, ADR 0037, ADR 0094
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import SIGMA_SB

__all__ = [
    "FacetProperties",
    "FacetForcing",
    "FacetSolver",
    "SpinUpResult",
    "SpinUpCache",
    "spin_up",
    "DEFAULT_SPIN_UP_HOURS",
]

#: §6.4 asks for "a day or two". 48 h is the default because it is where the measurement lands:
#: 48 h against 96 h differs by under 0.5 K for asphalt and thin steel (ADR 0037).
DEFAULT_SPIN_UP_HOURS = 48.0


def _f64(value: Any, what: str) -> NDArray[np.float64]:
    arr = np.asarray(value)
    if arr.dtype == np.float16:
        raise TypeError(
            f"{what} is float16 (CLAUDE.md #2): the balance subtracts ~400 W/m² terms to leave a "
            "few, and float16 has no digits left for the residual"
        )
    return np.asarray(arr, dtype=np.float64)


@dataclass(frozen=True)
class FacetProperties:
    """Per-facet material properties, as ``(N,)`` arrays."""

    heat_capacity_j_m2_k: NDArray[np.float64]
    emissivity: NDArray[np.float64]
    solar_absorptivity: NDArray[np.float64]

    def __post_init__(self) -> None:
        shapes = {np.shape(getattr(self, f.name)) for f in self.__dataclass_fields__.values()}
        if len(shapes) != 1:
            raise ValueError(f"facet property arrays disagree on shape: {sorted(shapes)}")
        if np.any(self.heat_capacity_j_m2_k <= 0.0):
            raise ValueError("every facet needs a positive areal heat capacity")
        for name in ("emissivity", "solar_absorptivity"):
            arr = getattr(self, name)
            if np.any(arr < 0.0) or np.any(arr > 1.0):
                raise ValueError(f"{name} must lie in [0, 1] for every facet")

    @property
    def n_facets(self) -> int:
        return int(np.shape(self.heat_capacity_j_m2_k)[0])

    @classmethod
    def stack(cls, properties: list[Any]) -> FacetProperties:
        """Pack a list of scalar :class:`~irsim.thermal.balance.ThermalProperties`."""
        return cls(
            heat_capacity_j_m2_k=_f64([p.heat_capacity_j_m2_k for p in properties], "capacity"),
            emissivity=_f64([p.emissivity for p in properties], "emissivity"),
            solar_absorptivity=_f64([p.solar_absorptivity for p in properties], "absorptivity"),
        )

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for arr in (self.heat_capacity_j_m2_k, self.emissivity, self.solar_absorptivity):
            h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
        return h.hexdigest()


@dataclass(frozen=True)
class FacetForcing:
    """Per-facet forcing at one instant. Scalars broadcast; arrays must be ``(N,)``."""

    t_air_k: Any
    h_w_m2_k: Any
    q_solar_w_m2: Any = 0.0
    q_longwave_down_w_m2: Any = 0.0
    q_internal_w_m2: Any = 0.0
    #: The fraction of each facet's own emission that leaves for good (`SurfaceForcing`'s field
    #: of the same name): 1 under an open sky, ``1 − F (1 − ε_r) ε`` under a grey body that
    #: reflects part of it back (ADR 0088 addendum, PT.7). Not part of :meth:`arrays`, whose
    #: five-tuple callers unpack; read through :meth:`emission_factors`.
    emission_factor: Any = 1.0
    #: The latent term (PH.1): the air's specific humidity, the bulk conductance ``ρ_a C_E U``
    #: (kg m⁻² s⁻¹), the declared wet fraction, a surface resistance (s/m) and the rain rate
    #: (kg m⁻² s⁻¹) that fills a solver's film. ``wet_fraction = 0`` with no film is no term.
    q_air_kg_kg: Any = 0.0
    g_e_kg_m2_s: Any = 0.0
    wet_fraction: Any = 0.0
    r_s_s_m: Any = 0.0
    precip_kg_m2_s: Any = 0.0

    def latent_arrays(self, n_facets: int) -> tuple[NDArray[np.float64], ...]:
        """``(q_air, g_e, wet_fraction, r_s, precip)`` as ``(N,)`` arrays."""
        out = []
        for name in ("q_air_kg_kg", "g_e_kg_m2_s", "wet_fraction", "r_s_s_m", "precip_kg_m2_s"):
            arr = _f64(getattr(self, name), name)
            if arr.ndim == 0:
                arr = np.full(n_facets, float(arr))
            elif arr.shape != (n_facets,):
                raise ValueError(f"{name} has shape {arr.shape}, expected ({n_facets},)")
            if np.any(arr < 0.0):
                raise ValueError(f"{name} cannot be negative")
            out.append(arr)
        if np.any(out[2] > 1.0):
            raise ValueError("wet_fraction must lie in [0, 1]")
        return tuple(out)

    @property
    def has_latent(self) -> bool:
        """Cheap gate: a forcing with no wetness declared and no conductance skips the term."""
        return bool(np.any(np.asarray(self.wet_fraction) > 0.0)) and bool(
            np.any(np.asarray(self.g_e_kg_m2_s) > 0.0)
        )

    def emission_factors(self, n_facets: int) -> NDArray[np.float64]:
        arr = _f64(self.emission_factor, "emission_factor")
        if arr.ndim == 0:
            arr = np.full(n_facets, float(arr))
        elif arr.shape != (n_facets,):
            raise ValueError(f"emission_factor has shape {arr.shape}, expected ({n_facets},)")
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError("emission_factor must lie in [0, 1]")
        return arr

    def arrays(self, n_facets: int) -> tuple[NDArray[np.float64], ...]:
        out = []
        for name in (
            "t_air_k",
            "h_w_m2_k",
            "q_solar_w_m2",
            "q_longwave_down_w_m2",
            "q_internal_w_m2",
        ):
            arr = _f64(getattr(self, name), name)
            if arr.ndim == 0:
                arr = np.full(n_facets, float(arr))
            elif arr.shape != (n_facets,):
                raise ValueError(f"{name} has shape {arr.shape}, expected ({n_facets},)")
            out.append(arr)
        return tuple(out)


class FacetSolver:
    """§6.1's balance over ``(N,)`` facets, stepped with the same midpoint rule as M6.7.

    ``conduction`` adds ``Σ_j K_ij (T_j − T_i)`` between facets and switches the step to IMEX --
    the surface balance explicit on the midpoint rule, the conduction term backward Euler -- see
    `irsim.thermal.conduction`. ``guard`` is the §6.4 explicit bound, on by default; a caller who
    deliberately steps past it (a steady-state search, say) switches it off and says so.
    """

    def __init__(
        self,
        properties: FacetProperties,
        initial_k: Any,
        *,
        conduction: Any = None,
        guard: bool = True,
        film_kg_m2: Any = None,
    ) -> None:
        self.properties = properties
        state = _f64(initial_k, "initial temperature")
        if state.ndim == 0:
            state = np.full(properties.n_facets, float(state))
        if state.shape != (properties.n_facets,):
            raise ValueError(
                f"initial temperature has shape {state.shape}, expected ({properties.n_facets},)"
            )
        if np.any(state <= 0.0):
            raise ValueError("temperatures must be positive (kelvin)")
        if conduction is not None and conduction.n_facets != properties.n_facets:
            raise ValueError(
                f"the conduction operator links {conduction.n_facets} facets, the solver has "
                f"{properties.n_facets}"
            )
        self._state = state
        self.conduction = conduction
        self.guard = guard
        # PH.1: a water film per facet, kg m⁻², that rain fills and evaporation empties. `None`
        # is a surface with no film bookkeeping at all (the declared `wet_fraction` still applies).
        self._film: NDArray[np.float64] | None = None
        if film_kg_m2 is not None:
            film = np.asarray(film_kg_m2, dtype=np.float64)
            if film.ndim == 0:
                film = np.full(properties.n_facets, float(film))
            if film.shape != (properties.n_facets,):
                raise ValueError(
                    f"film_kg_m2 has shape {film.shape}, expected ({properties.n_facets},)"
                )
            if np.any(film < 0.0):
                raise ValueError("a film cannot be negative")
            self._film = film
        self._evaporated = np.zeros(properties.n_facets)
        self._factor: tuple[float, Any] | None = None

    @property
    def temperatures_k(self) -> NDArray[np.float64]:
        return np.asarray(self._state.copy())

    @property
    def film_kg_m2(self) -> NDArray[np.float64] | None:
        """The water film per facet, kg m⁻² (a 0.2 mm film is 0.2 kg m⁻²), or ``None``."""
        return None if self._film is None else np.asarray(self._film.copy())

    @property
    def evaporated_kg_m2(self) -> NDArray[np.float64]:
        """Water each facet's film has given up since the start, kg m⁻² (exact bookkeeping)."""
        return np.asarray(self._evaporated.copy())

    def _wetness(self, forcing: FacetForcing) -> NDArray[np.float64] | None:
        """The wet fraction the balance sees: the declared one, or 1 wherever a film stands."""
        if not forcing.has_latent and self._film is None:
            return None
        _, g_e, wet, _, _ = forcing.latent_arrays(self.properties.n_facets)
        if self._film is not None:
            wet = np.maximum(wet, (self._film > 0.0).astype(np.float64))
        if not np.any(wet > 0.0) or not np.any(g_e > 0.0):
            return None
        return np.asarray(wet)

    def net_flux(
        self,
        temperatures: NDArray[np.float64],
        forcing: FacetForcing,
        wet: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """§6.1's right-hand side per facet; ``wet`` overrides the forcing's wet fraction."""
        t_air, h, q_sol, q_lw, q_int = forcing.arrays(self.properties.n_facets)
        leaves = forcing.emission_factors(self.properties.n_facets)
        # eps multiplies BOTH the absorbed sky radiation and the emitted term, as §6.1 writes
        # it. Kirchhoff: a surface absorbs the same fraction of incident longwave that it emits.
        net = np.asarray(
            self.properties.solar_absorptivity * q_sol
            + self.properties.emissivity * q_lw
            - leaves * self.properties.emissivity * SIGMA_SB * temperatures**4
            - h * (temperatures - t_air)
            + q_int
        )
        wetness = self._wetness(forcing) if wet is None else wet
        if wetness is None:
            return net
        from irsim.thermal.latent import latent_heat_flux_w_m2

        q_air, g_e, _, r_s, _ = forcing.latent_arrays(self.properties.n_facets)
        return np.asarray(net - latent_heat_flux_w_m2(temperatures, q_air, g_e, wetness, r_s))

    def _check_explicit_bound(self, forcing: FacetForcing, dt_s: float) -> None:
        """§6.4's bound on the forcing's actual h and the state's own T, every step."""
        from irsim.thermal.conduction import explicit_bound_s

        h = forcing.arrays(self.properties.n_facets)[1]
        bound = explicit_bound_s(
            self.properties.heat_capacity_j_m2_k, h, self.properties.emissivity, self._state
        )
        worst = int(np.argmin(bound))
        if dt_s > bound[worst]:
            raise ValueError(
                f"dt = {dt_s:g} s exceeds the §6.4 explicit bound {bound[worst]:.3g} s on facet "
                f"{worst} (C = {self.properties.heat_capacity_j_m2_k[worst]:.4g} J/m²/K, "
                f"h = {h[worst]:.3g} W/m²/K, T = {self._state[worst]:.1f} K). The midpoint rule "
                "diverges there; shorten the tick or give the surface a larger areal capacity. "
                "Conduction is not the cause -- it is stepped implicitly (ADR 0094)"
            )

    def advance(self, forcing: FacetForcing, dt_s: float) -> NDArray[np.float64]:
        if dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.guard:
            self._check_explicit_bound(forcing, dt_s)
        capacity = self.properties.heat_capacity_j_m2_k
        wet = self._wetness(forcing)
        half = self._state + 0.5 * dt_s * self.net_flux(self._state, forcing, wet) / capacity
        explicit = self._state + dt_s * self.net_flux(half, forcing, wet) / capacity
        if self._film is not None:
            # The film: rain in, evaporation (at the midpoint temperature) out, never below zero.
            from irsim.thermal.latent import evaporation_kg_m2_s

            q_air, g_e, _, r_s, precip = forcing.latent_arrays(self.properties.n_facets)
            rate = precip.copy()
            if wet is not None:
                rate = rate - evaporation_kg_m2_s(half, q_air, g_e, wet, r_s)
            before_film = self._film
            self._film = np.maximum(0.0, self._film + rate * dt_s)
            # What actually left the film this tick (the clamp means E dt can exceed it), so
            # a mass budget can be checked exactly: film₀ − film + rain = evaporated.
            self._evaporated = self._evaporated + (before_film - self._film + precip * dt_s)
        if self.conduction is None:
            self._state = explicit
            return self.temperatures_k
        # IMEX: the surface balance above is the explicit half; conduction is backward Euler on
        # a matrix that depends only on the tick, factorised once and reused (ADR 0094).
        if self._factor is None or self._factor[0] != float(dt_s):
            self._factor = (float(dt_s), self.conduction.factorise(capacity, float(dt_s)))
        self._state = np.asarray(self._factor[1].solve(explicit), dtype=np.float64)
        return self.temperatures_k

    def as_float32(self) -> NDArray[np.float32]:
        """The boundary: the only place the state is allowed to narrow (CLAUDE.md #2)."""
        return np.asarray(self._state, dtype=np.float32)


# ---------------------------------------------------------------------------------------------
# spin-up
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SpinUpResult:
    temperatures_k: NDArray[np.float64]
    hours: float
    steps: int
    key: str
    from_cache: bool = False


@dataclass
class SpinUpCache:
    """Keyed on what actually determines the answer: the materials, the weather, and t₀.

    Not on the scene, the camera or the frame. Two scenes made of the same materials under the
    same weather at the same hour have the same surface temperatures, and re-integrating 48 h to
    rediscover that is the single most expensive thing this package can be asked to do.
    """

    entries: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, key: str) -> NDArray[np.float64] | None:
        value = self.entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return np.asarray(value.copy())

    def put(self, key: str, value: NDArray[np.float64]) -> None:
        self.entries[key] = value.copy()


def spin_up_key(
    properties: FacetProperties, weather_hash: str, t0_s: float, hours: float, dt_s: float
) -> str:
    """SHA-256 over the material content, the weather hash, t₀, the span and the step."""
    h = hashlib.sha256()
    h.update(properties.content_hash().encode())
    h.update(weather_hash.encode())
    for value in (t0_s, hours, dt_s):
        h.update(np.float64(value).tobytes())
    return h.hexdigest()


def spin_up(
    properties: FacetProperties,
    forcing_at: Callable[[float], FacetForcing],
    weather_hash: str,
    t0_s: float,
    hours: float = DEFAULT_SPIN_UP_HOURS,
    dt_s: float = 60.0,
    initial_k: Any = None,
    cache: SpinUpCache | None = None,
    conduction: Any = None,
) -> SpinUpResult:
    """Integrate the facets through ``hours`` of weather *ending* at ``t0_s``.

    The run ends where the scene begins, so what comes back is the state the weather implies at
    t₀ -- not a state at some earlier time that the caller then has to advance. Starting a scene
    at the air temperature instead is wrong by several kelvin on a sunlit surface and stays wrong
    for hours, which is exactly the part of a diurnal cycle a thermal camera is most interesting in.
    """
    if hours <= 0.0 or dt_s <= 0.0:
        raise ValueError("hours and dt_s must be positive")
    key = spin_up_key(properties, weather_hash, t0_s, hours, dt_s)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return SpinUpResult(cached, hours, 0, key, from_cache=True)

    start = t0_s - hours * 3600.0
    solver = FacetSolver(
        properties,
        forcing_at(start).arrays(properties.n_facets)[0] if initial_k is None else initial_k,
        conduction=conduction,
    )
    steps = int(round(hours * 3600.0 / dt_s))
    for i in range(steps):
        solver.advance(forcing_at(start + i * dt_s), dt_s)
    if cache is not None:
        cache.put(key, solver.temperatures_k)
    return SpinUpResult(solver.temperatures_k, hours, steps, key)
