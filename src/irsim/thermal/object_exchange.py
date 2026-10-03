"""Heat exchange between objects, as one scene switch (TC.10).

docs/physics-model.md §6.1 (what a surface exchanges with) and §6.6 (heat sources that radiate
onto what faces them); ADR 0088 (the net rule), ADR 0156 (the traced factors), ADR 0157 (this).

Every solved surface in a scene has, until now, exchanged longwave with the sky and with an
air-temperature surround -- never with another solved surface, except where a scene author
hand-placed one of ADR 0088's parallel rectangles. With ``thermal.object_exchange: true`` the
scene's plain patches and meshes become one **exchange group**: TC.9's traced view factors say
which cells face which, and at every tick each cell receives, in place of the sky and surround
it thought it saw there, what the faces it sees actually emit:

    q_c = ε_c · [ Σ_j G_cj · ε_j σ T̄_j⁴  −  cover_c · q_lw,c ]

``G_cj`` is the cell's view of face *j* (the traced factor, symmetrised so that
``A_i F_ij = A_j F_ji`` holds exactly), ``T̄_j`` the face's mean temperature over its cells,
``cover_c = Σ_j G_cj`` the share of the cell's hemisphere the other bodies fill, and ``q_lw,c``
the longwave the cell's own forcing already credits it -- so the second term is ADR 0088's
"net of the sky it hides", with the cell's hemisphere-mean incoming longwave standing for what
each covered steradian replaces (for a cell that sees only sky the two are the same number).

The members keep their own solvers and step in **lockstep**: the group snapshots every member
at the start of a tick, computes every exchange flux from that one snapshot, and advances each
member exactly one tick with the flux added to its ``q_internal``. Radiation between surfaces
is a soft coupling (its conductance ``4 ε σ T³ · cover`` is a few W m⁻² K⁻¹ against areal heat
capacities of 10³-10⁶ J m⁻² K⁻¹), so an explicit, one-tick-lagged exchange is stable where a
bolted contact (ADR 0099) is not, and the members' own IMEX steps are untouched. A member
advanced through the group cannot be advanced past its partners, and a member's forcing asked
for a tick the group has not snapshotted refuses rather than quietly returning no exchange.

Single bounce: what a face reflects of another's emission is lost from the pair, as in
ADR 0088. Switched off, nothing here is constructed and every field is the separate solve it
was, bit for bit.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.radiometry.constants import SIGMA_SB
from irsim.thermal.facets import FacetForcing
from irsim.thermal.mesh_geometry import mesh_soup
from irsim.thermal.raycast import TriangleSoup
from irsim.thermal.view_factors import RadiantMesh, ViewFactors, view_factors

__all__ = ["ExchangeBody", "ExchangedField", "ObjectExchange"]


@dataclass(frozen=True)
class ExchangeBody:
    """One solved surface as a radiating body: its triangles, and how cells and faces map.

    ``receive`` is ``(n_cells, n_faces)`` with unit rows -- a cell's view is the area-weighted
    mean of its faces' views (a planar cell is two triangles); ``emit`` is ``(n_faces, n_cells)``
    with unit rows -- a face's temperature is the mean of the cells on it (a mesh face carries
    ``level²`` congruent cells). Faces radiate from their winding-normal side.
    """

    name: str
    soup: TriangleSoup
    receive: NDArray[np.float64]
    emit: NDArray[np.float64]
    emissivity: NDArray[np.float64]
    cell_areas_m2: NDArray[np.float64]

    def __post_init__(self) -> None:
        n_cells, n_faces = self.receive.shape
        if self.emit.shape != (n_faces, n_cells):
            raise ValueError(f"{self.name!r}: emit must be (n_faces, n_cells)")
        if n_faces != self.soup.n_faces:
            raise ValueError(
                f"{self.name!r}: receive has {n_faces} faces, the soup {self.soup.n_faces}"
            )
        for label, m in (("receive", self.receive), ("emit", self.emit)):
            if not np.allclose(m.sum(axis=1), 1.0, atol=1e-12):
                raise ValueError(f"{self.name!r}: {label} rows must sum to one")
        if self.emissivity.shape != (n_cells,) or self.cell_areas_m2.shape != (n_cells,):
            raise ValueError(f"{self.name!r}: emissivity and cell_areas_m2 are per cell")

    @property
    def n_cells(self) -> int:
        return int(self.receive.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.receive.shape[1])

    @classmethod
    def from_planar(cls, name: str, patch: Any, emissivity: Any) -> ExchangeBody:
        """A `PlanarPatch`: two triangles per cell, wound to the patch's normal ``u × v``."""
        n_u, n_v = int(patch.n_u), int(patch.n_v)
        iu, iv = np.meshgrid(np.arange(n_u), np.arange(n_v), indexing="xy")
        iu, iv = iu.reshape(-1), iv.reshape(-1)  # C order, v slow: the patch's cell order
        o = patch.origin_m
        du, dv = patch.du_m * patch.u_axis, patch.dv_m * patch.v_axis
        p00 = o + iu[:, None] * du + iv[:, None] * dv
        p10, p01, p11 = p00 + du, p00 + dv, p00 + du + dv
        vertices = np.concatenate([p00, p10, p01, p11])
        n = n_u * n_v
        c = np.arange(n)
        faces = np.concatenate(
            [np.stack([c, c + n, c + 3 * n], axis=1), np.stack([c, c + 3 * n, c + 2 * n], axis=1)]
        )
        # face 2c and 2c+1 belong to cell c: interleave so a cell's faces sit together
        order = np.empty(2 * n, dtype=np.int64)
        order[0::2], order[1::2] = np.arange(n), np.arange(n) + n
        faces = faces[order]
        receive = np.zeros((n, 2 * n))
        receive[c, 2 * c] = 0.5
        receive[c, 2 * c + 1] = 0.5
        emit = np.zeros((2 * n, n))
        emit[2 * c, c] = 1.0
        emit[2 * c + 1, c] = 1.0
        return cls(
            name,
            TriangleSoup(vertices, faces),
            receive,
            emit,
            np.broadcast_to(np.asarray(emissivity, dtype=np.float64), (n,)).copy(),
            np.full(n, float(patch.cell_area_m2)),
        )

    @classmethod
    def from_mesh(cls, name: str, patch: Any, emissivity: Any) -> ExchangeBody:
        """A `TriangleMeshPatch`: its own faces, each carrying the cells cut on it."""
        soup = mesh_soup(patch)
        face_of_cell = np.asarray(patch.cell_face, dtype=np.int64)
        n, n_faces = int(face_of_cell.shape[0]), soup.n_faces
        counts = np.bincount(face_of_cell, minlength=n_faces).astype(np.float64)
        if np.any(counts == 0):
            raise ValueError(f"{name!r}: a face carries no cell")
        receive = np.zeros((n, n_faces))
        receive[np.arange(n), face_of_cell] = 1.0
        emit = np.zeros((n_faces, n))
        emit[face_of_cell, np.arange(n)] = 1.0 / counts[face_of_cell]
        face_areas = RadiantMesh(name, soup).face_areas_m2
        return cls(
            name,
            soup,
            receive,
            emit,
            np.broadcast_to(np.asarray(emissivity, dtype=np.float64), (n,)).copy(),
            face_areas[face_of_cell] / counts[face_of_cell],
        )


class ObjectExchange:
    """The exchange group: traced factors between its bodies, and the lockstep that uses them."""

    def __init__(self, bodies: Sequence[ExchangeBody], *, subdivide: int = 2) -> None:
        if len(bodies) < 2:
            raise ValueError("an exchange needs at least two bodies")
        names = [b.name for b in bodies]
        if len(set(names)) != len(names):
            raise ValueError(f"body names must be unique: {names}")
        self.bodies: tuple[ExchangeBody, ...] = tuple(bodies)
        self.names: tuple[str, ...] = tuple(names)
        self.traced: ViewFactors = view_factors(
            [RadiantMesh(b.name, b.soup) for b in bodies], subdivide=subdivide
        )
        cell_sizes = [b.n_cells for b in bodies]
        face_sizes = [b.n_faces for b in bodies]
        self._cell_start = np.concatenate([[0], np.cumsum(cell_sizes)]).astype(int)
        self._face_start = np.concatenate([[0], np.cumsum(face_sizes)]).astype(int)
        n_cells, n_faces = int(self._cell_start[-1]), int(self._face_start[-1])
        receive = np.zeros((n_cells, n_faces))
        emit = np.zeros((n_faces, n_cells))
        for k, b in enumerate(bodies):
            cs, fs = self.cells_of(k), self.faces_of(k)
            receive[cs, fs] = b.receive
            emit[fs, cs] = b.emit
        # reciprocity exact by construction: symmetrise the power matrix A_i F_ij
        areas = self.traced.areas_m2
        power = areas[:, None] * self.traced.face
        power = 0.5 * (power + power.T)
        self.face_factors: NDArray[np.float64] = power / areas[:, None]
        self.cell_factors: NDArray[np.float64] = receive @ self.face_factors  # G_cj
        self.cover: NDArray[np.float64] = self.cell_factors.sum(axis=1)
        self.cell_emissivity: NDArray[np.float64] = np.concatenate([b.emissivity for b in bodies])
        self.cell_areas_m2: NDArray[np.float64] = np.concatenate([b.cell_areas_m2 for b in bodies])
        self.face_areas_m2: NDArray[np.float64] = areas
        self.face_emissivity: NDArray[np.float64] = emit @ self.cell_emissivity
        self._emit = emit
        self._fields: dict[str, Any] = {}
        self._inner_forcing: dict[str, Callable[[float], FacetForcing]] = {}
        self._snapshot_t: float | None = None
        self._incoming: NDArray[np.float64] | None = None
        self.last_flux_w_m2: dict[str, NDArray[np.float64]] = {}

    # -- indexing ----------------------------------------------------------------------------

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(
                f"{name!r} is not a body of this exchange; bodies: {self.names}"
            ) from None

    def cells_of(self, k: int) -> slice:
        return slice(int(self._cell_start[k]), int(self._cell_start[k + 1]))

    def faces_of(self, k: int) -> slice:
        return slice(int(self._face_start[k]), int(self._face_start[k + 1]))

    # -- the physics -------------------------------------------------------------------------

    def face_emission_w_m2(self, temperatures_k: Any) -> NDArray[np.float64]:
        """``ε_j σ T̄_j⁴`` per face from every cell's temperature (all bodies, in order)."""
        t = np.asarray(temperatures_k, dtype=np.float64)
        if t.shape != (self.cell_areas_m2.shape[0],):
            raise ValueError("temperatures are one value per cell of every body, in order")
        t4_face = self._emit @ t**4
        return np.asarray(self.face_emissivity * SIGMA_SB * t4_face)

    def incoming_w_m2(self, temperatures_k: Any) -> NDArray[np.float64]:
        """``Σ_j G_cj ε_j σ T̄_j⁴`` per cell: what the other bodies radiate onto it."""
        return np.asarray(self.cell_factors @ self.face_emission_w_m2(temperatures_k))

    def flux_w_m2(self, temperatures_k: Any, longwave_down_w_m2: Any) -> NDArray[np.float64]:
        """The net exchange flux per cell, W m⁻²: received, less the sky and surround hidden."""
        q_lw = np.broadcast_to(np.asarray(longwave_down_w_m2, dtype=np.float64), self.cover.shape)
        return np.asarray(
            self.cell_emissivity * (self.incoming_w_m2(temperatures_k) - self.cover * q_lw)
        )

    def absorbed_power_w(self, temperatures_k: Any) -> NDArray[np.float64]:
        """``(n_bodies, n_bodies)``: ``[a, b]`` is the power *a* absorbs of *b*'s emission."""
        emission = self.face_emission_w_m2(temperatures_k)
        weighted = self.cell_areas_m2 * self.cell_emissivity
        out = np.zeros((len(self.names), len(self.names)))
        for a in range(len(self.names)):
            g = self.cell_factors[self.cells_of(a)]
            w = weighted[self.cells_of(a)]
            for b in range(len(self.names)):
                fs = self.faces_of(b)
                out[a, b] = float(w @ (g[:, fs] @ emission[fs]))
        return out

    def emitted_toward_power_w(self, temperatures_k: Any) -> NDArray[np.float64]:
        """``(n_bodies, n_bodies)``: ``[b, a]`` is the power *b* emits into *a*'s direction."""
        emission = self.face_emission_w_m2(temperatures_k)
        out = np.zeros((len(self.names), len(self.names)))
        for b in range(len(self.names)):
            fs = self.faces_of(b)
            power = self.face_areas_m2[fs] * emission[fs]
            for a in range(len(self.names)):
                out[b, a] = float(power @ self.face_factors[fs, self.faces_of(a)].sum(axis=1))
        return out

    # -- the lockstep ------------------------------------------------------------------------

    def register(self, name: str, field: Any) -> ExchangedField:
        """Join a solved field to the group and wrap its forcing; returns the proxy to use."""
        k = self.index(name)
        if name in self._fields:
            raise ValueError(f"{name!r} is already registered")
        if int(field.field.properties.n_facets) != self.bodies[k].n_cells:
            raise ValueError(
                f"{name!r}: the field has {field.field.properties.n_facets} cells, the body "
                f"{self.bodies[k].n_cells}"
            )
        if self._fields:
            first = next(iter(self._fields.values()))
            if abs(field.t0_s - first.t0_s) > 1e-9 or abs(field.tick_s - first.tick_s) > 1e-12:
                raise ValueError(
                    f"{name!r}: every member must share t0 and the tick; "
                    f"{first.t0_s}/{first.tick_s} against {field.t0_s}/{field.tick_s}"
                )
        self._inner_forcing[name] = field.field.forcing_at
        field.field.forcing_at = self._wrap(name, field.field.forcing_at, field.t0_s)
        self._fields[name] = field
        return ExchangedField(self, field)

    @property
    def members(self) -> Mapping[str, Any]:
        return dict(self._fields)

    def _wrap(
        self, name: str, inner: Callable[[float], FacetForcing], t0_s: float
    ) -> Callable[[float], FacetForcing]:
        k = self.index(name)
        cells = self.cells_of(k)
        eps, cover = self.cell_emissivity[cells], self.cover[cells]
        n = self.bodies[k].n_cells

        def forcing(t_s: float) -> FacetForcing:
            base = inner(t_s)
            if t_s < t0_s - 1e-9:  # the spin-up: the group does not exist yet
                return base
            if self._snapshot_t is None or abs(t_s - self._snapshot_t) > 1e-9:
                raise RuntimeError(
                    f"{name!r} asked for its forcing at {t_s} s outside the exchange group's "
                    "tick: advance an exchanged field through the group (its proxy), not on "
                    "its own, so every member sees the same snapshot"
                )
            assert self._incoming is not None
            q_lw = base.arrays(n)[3]
            q_int = np.broadcast_to(np.asarray(base.q_internal_w_m2, dtype=np.float64), (n,))
            q = eps * (self._incoming[cells] - cover * q_lw)
            self.last_flux_w_m2[name] = q
            return dataclasses.replace(base, q_internal_w_m2=q_int + q)

        return forcing

    def _members_in_order(self) -> list[Any]:
        if len(self._fields) != len(self.names):
            missing = [n for n in self.names if n not in self._fields]
            raise ValueError(f"the exchange cannot step before every body has a field: {missing}")
        return [self._fields[n] for n in self.names]

    def spin_up(
        self,
        hours: float,
        dt_s: float = 60.0,
        wrap: Callable[[Callable[[float], FacetForcing]], Callable[[float], FacetForcing]]
        | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Spin the whole group up together and restart every member at t₀ from the result.

        docs/physics-model.md §6.1 and §6.4's spin-up; ADR 0157, amendment of 2026-10-03. The
        same integration as :func:`~irsim.thermal.facets.spin_up` -- ``hours`` of weather ending
        at t₀, from the air temperature at the start, on a ``dt_s`` step -- except that the
        members advance in lockstep and each step's exchange flux, from one snapshot, is added
        to every member's ``q_internal``. So a road that has stood under a warm pan all night
        opens the scene with the pan's patch already on it, rather than growing it over the
        first hours of the run.

        ``wrap`` is the scene's wrap of a forcing into its weather series (the spin-up starts
        before a 48 h file does); the exchange itself reads only the members' own forcings.
        Every member must be registered and none advanced past t₀. Returns the state each
        member was restarted from, in float64.
        """
        if hours <= 0.0 or dt_s <= 0.0:
            raise ValueError("hours and dt_s must be positive")
        fields = self._members_in_order()
        advanced = [n for n, f in zip(self.names, fields, strict=True) if f.field.n_ticks != 1]
        if advanced:
            raise RuntimeError(f"spin-up after the run began: {advanced} already ticked")
        from irsim.thermal.facets import FacetSolver

        t0 = float(fields[0].t0_s)
        start = t0 - hours * 3600.0
        steps = int(round(hours * 3600.0 / dt_s))
        inner = [self._inner_forcing[n] for n in self.names]
        forcings = inner if wrap is None else [wrap(f) for f in inner]
        sizes = [b.n_cells for b in self.bodies]
        solvers = [
            FacetSolver(
                f.field.properties,
                fa(start).arrays(n)[0],
                conduction=f.field.conduction,
            )
            for f, fa, n in zip(fields, forcings, sizes, strict=True)
        ]
        for i in range(steps):
            t = start + i * dt_s
            incoming = self.incoming_w_m2(np.concatenate([s.temperatures_k for s in solvers]))
            for k, (solver, fa, n) in enumerate(zip(solvers, forcings, sizes, strict=True)):
                cells = self.cells_of(k)
                base = fa(t)
                q_lw = base.arrays(n)[3]
                q_int = np.broadcast_to(np.asarray(base.q_internal_w_m2, dtype=np.float64), (n,))
                q = self.cell_emissivity[cells] * (incoming[cells] - self.cover[cells] * q_lw)
                solver.advance(dataclasses.replace(base, q_internal_w_m2=q_int + q), dt_s)
        out: dict[str, NDArray[np.float64]] = {}
        for name, f, solver in zip(self.names, fields, solvers, strict=True):
            f.field.restart(solver.temperatures_k)
            out[name] = np.array(solver.temperatures_k, dtype=np.float64)
        return out

    def advance_to(self, t_s: float) -> None:
        """Every member forward to ``t_s``, one tick at a time, from one snapshot per tick."""
        fields = self._members_in_order()
        tick = float(fields[0].tick_s)
        while True:
            latest = [float(f.latest_t_s) for f in fields]
            if max(latest) - min(latest) > 1e-9:
                raise RuntimeError(
                    "exchange members are out of step "
                    f"({dict(zip(self.names, latest, strict=True))}): one "
                    "was advanced on its own"
                )
            now = latest[0]
            if now >= t_s - 1e-12:
                return
            temps = np.concatenate([f.field.latest_state_k for f in fields])
            self._incoming = self.incoming_w_m2(temps)
            self._snapshot_t = now
            for f in fields:
                f.advance_to(now + tick)


class ExchangedField:
    """A member of an exchange group, as the scene hands it out: the field it wraps, except that
    ``advance_to`` moves the whole group, by construction."""

    def __init__(self, exchange: ObjectExchange, inner: Any) -> None:
        self._exchange = exchange
        self._inner = inner

    @property
    def exchange(self) -> ObjectExchange:
        return self._exchange

    @property
    def inner(self) -> Any:
        return self._inner

    def advance_to(self, t_s: float) -> None:
        self._exchange.advance_to(t_s)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return f"ExchangedField({self._inner!r})"
