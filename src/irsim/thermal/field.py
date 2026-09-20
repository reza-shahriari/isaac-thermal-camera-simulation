"""The object the engine glue reads: a fixed-tick solve behind a render-time query (§6.4, §1).

§6.4: "Decouple this from the render loop entirely — run it on a fixed thermal tick (say 1 Hz) and
interpolate." That is one sentence and two separate requirements, and the second is the one that
gets broken:

* the solve advances on **its own clock**, so a render at 240 fps and a render at 1 fps see the
  same temperatures;
* a query **never mutates** anything. A renderer asks for a temperature many times per tick — per
  band, per AOV, per pass — and a query that advanced the solver would make the answer depend on
  how many times it was asked, which is not a bug that shows up as an error message.

So :class:`ThermalField` keeps two bracketing ticks and interpolates between them, and
``temperature_at`` is const. Ticks are produced on demand up to the requested time and then kept;
asking for an earlier time than the ticks already cover raises rather than re-integrating, because
a thermal history is not reversible and silently re-running it would give a different answer.

**float32 is the boundary and only the boundary** (CLAUDE.md #2). The solve is float64 throughout;
``temperature_at`` narrows once, on the way out, because that is what the G-buffer carries. Over
48 h the difference between carrying float32 internally and narrowing at the end is measurable,
which is why the narrowing is in one place with a test on it.

**How much history a field keeps is a choice, and it was not being made** (PT.8, ADR 0093). Every
tick was appended and none pruned, while a query only ever reads the two ticks bracketing it. For
a per-prim field that is a few facets times a few thousand ticks and nobody notices; for a
10 400-cell road patch a 24 h run was ~240 MB, and the 10⁵-cell fields the coupling lane brings
would be gigabytes -- which is what stopped ADR 0074's full-diurnal time-lapse. So ``keep_ticks``
bounds the history to a ring (``None`` keeps everything, as before), the state hash is a running
digest fed tick by tick -- the same bytes in the same order, so it equals the old walk exactly --
and ``on_tick`` is how a caller who *wants* the history records it. A query before the oldest
retained tick raises and names the knob: a field that answered from a fallback would be quietly
telling a time-lapse a different story than it solved.

docs/physics-model.md §6.4, §1; ADR 0036, ADR 0037, ADR 0093
"""

from __future__ import annotations

import collections
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.thermal.facets import FacetForcing, FacetProperties, FacetSolver

__all__ = ["DEFAULT_TICK_S", "ThermalField"]

#: §6.4's "say 1 Hz". Fast enough that the linear interpolation between ticks is far below a
#: millikelvin for every material in the library, and slow enough that a 48 h spin-up is cheap.
DEFAULT_TICK_S = 1.0


@dataclass(frozen=True)
class _Tick:
    t_s: float
    temperatures_k: NDArray[np.float64]


class ThermalField:
    """A facet solve on a fixed tick, queried at arbitrary render times.

    ``forcing_at`` is the scene's own weather-driven forcing; the field never looks at a clock of
    its own. Construction takes the state at ``t0_s`` -- normally M6.10's spin-up result -- so the
    field begins where the scene begins and the first frame is not a transient.
    """

    def __init__(
        self,
        properties: FacetProperties,
        forcing_at: Callable[[float], FacetForcing],
        t0_s: float,
        initial_k: NDArray[np.float64],
        tick_s: float = DEFAULT_TICK_S,
        *,
        keep_ticks: int | None = None,
        on_tick: Callable[[float, NDArray[np.float64]], None] | None = None,
        conduction: Any = None,
    ) -> None:
        if tick_s <= 0.0:
            raise ValueError("tick_s must be positive")
        if keep_ticks is not None and keep_ticks < 2:
            raise ValueError(
                "keep_ticks must be at least 2: a query is a blend of the two ticks bracketing it"
            )
        self.properties = properties
        self.forcing_at = forcing_at
        self.tick_s = float(tick_s)
        self.t0_s = float(t0_s)
        self.keep_ticks = keep_ticks
        self.on_tick = on_tick
        self._solver = FacetSolver(properties, initial_k, conduction=conduction)
        self._ticks: collections.deque[_Tick] = collections.deque(maxlen=keep_ticks)
        self._produced = 0
        self._digest = hashlib.sha256()
        self._push(float(t0_s), self._solver.temperatures_k)

    def _push(self, t_s: float, temperatures_k: NDArray[np.float64]) -> None:
        """Record one tick: into the ring, into the running hash, and out to the hook."""
        self._ticks.append(_Tick(t_s, temperatures_k))
        self._produced += 1
        self._digest.update(np.float64(t_s).tobytes())
        self._digest.update(np.ascontiguousarray(temperatures_k, dtype=np.float64).tobytes())
        if self.on_tick is not None:
            self.on_tick(t_s, temperatures_k)

    # -- the solve ---------------------------------------------------------------------------

    @property
    def n_ticks(self) -> int:
        """Ticks **produced** since t₀, including t₀ itself -- not ticks still held."""
        return self._produced

    @property
    def n_held(self) -> int:
        """Ticks still resident; equals :attr:`n_ticks` only when ``keep_ticks`` is ``None``."""
        return len(self._ticks)

    @property
    def latest_t_s(self) -> float:
        return self._ticks[-1].t_s

    @property
    def latest_state_k(self) -> NDArray[np.float64]:
        """The newest tick's temperatures in the solver's own float64, for energy bookkeeping.

        A query through :meth:`temperature_at` is float32 by contract (CLAUDE.md #2: the
        boundary narrows once); a conservation check summing ``C A T`` over 10⁴ cells needs the
        seven extra digits, and this is the only place they leave the field.
        """
        out: NDArray[np.float64] = np.array(self._ticks[-1].temperatures_k, dtype=np.float64)
        return out

    @property
    def earliest_t_s(self) -> float:
        """The oldest tick still held: the start of the window a query can be answered in."""
        return self._ticks[0].t_s

    def _extend_to(self, t_s: float) -> None:
        while self._ticks[-1].t_s < t_s - 1e-12:
            start = self._ticks[-1].t_s
            self._solver.advance(self.forcing_at(start), self.tick_s)
            self._push(start + self.tick_s, self._solver.temperatures_k)

    def advance_to(self, t_s: float) -> None:
        """Produce ticks up to ``t_s``. The **only** method that changes anything."""
        if t_s < self.t0_s - 1e-12:
            raise ValueError(
                f"cannot advance to {t_s} s, before the field's start {self.t0_s} s: a thermal "
                "history is not reversible, and re-running it would give a different answer"
            )
        self._extend_to(t_s)

    # -- the query ---------------------------------------------------------------------------

    def temperature_at(self, t_s: float) -> NDArray[np.float32]:
        """Per-facet temperature at an arbitrary render time, float32, **without mutating**.

        Linear between the two bracketing ticks. Linear rather than anything cleverer because the
        tick is 1 s and the fastest surface in the library has a time constant of minutes: the
        interpolation error is far below a millikelvin, and a higher-order scheme would need more
        stored ticks to buy nothing.
        """
        if t_s < self.t0_s - 1e-12:
            raise ValueError(f"t_s = {t_s} is before the field's start {self.t0_s}")
        if t_s > self.latest_t_s + 1e-12:
            raise ValueError(
                f"t_s = {t_s} is past the last tick {self.latest_t_s}: call advance_to first. A "
                "query must not advance the solve -- a renderer asks many times per tick, and an "
                "answer that depended on how often it was asked would not look like an error"
            )
        earliest = self.earliest_t_s
        if t_s < earliest - 1e-12:
            raise ValueError(
                f"t_s = {t_s} is before the oldest tick this field still holds ({earliest}): it "
                f"keeps {self.keep_ticks} ticks (keep_ticks), so it answers only inside that "
                "window. Keep more ticks, or record the history with on_tick, rather than "
                "reading a fallback (ADR 0093)"
            )
        index = min(int((t_s - earliest) / self.tick_s), len(self._ticks) - 1)
        lower = self._ticks[index]
        if index + 1 >= len(self._ticks):
            return np.asarray(lower.temperatures_k, dtype=np.float32)
        upper = self._ticks[index + 1]
        span = upper.t_s - lower.t_s
        weight = 0.0 if span <= 0.0 else (t_s - lower.t_s) / span
        blended = lower.temperatures_k + weight * (upper.temperatures_k - lower.temperatures_k)
        return np.asarray(blended, dtype=np.float32)

    @property
    def conduction(self) -> Any:
        """The operator linking the facets, or ``None`` for independent columns (TC.1)."""
        return self._solver.conduction

    def state_hash(self) -> str:
        """SHA-256 over every tick produced so far: what a query must not change.

        A running digest fed at each tick, so it costs nothing to ask for and does not depend on
        how many ticks are still held. It is byte-for-byte the digest a walk over the full
        history would give -- `test_thermal_field` reconstructs that walk from ``on_tick`` and
        checks -- so a field bounded to a ring hashes exactly as the unbounded one did.
        """
        return self._digest.hexdigest()
