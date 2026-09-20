"""Linear conduction between facets, and the implicit step that lets a scene tick survive it.

docs/physics-model.md §6.1 (the conduction term), §6.4 (the integrator and its bound); ADR 0094.

Every facet in `FacetSolver` has been an independent column: its own balance, its own root, no
neighbour. That is why a bonnet renders sharper than aluminium can be (PT.11's diffusion lengths)
and why an engine cannot warm the bracket bolted to it (the `TC` lane). Both need one thing first:
a term that moves heat **between** facets, and a way to step it that does not shrink the tick.

**The operator.** A symmetric matrix of conductances ``K_ij`` in W/K, zero on the diagonal, and
the facet areas, so that

    C_i A_i dT_i/dt  +=  Σ_j K_ij (T_j − T_i)

with ``C`` in J m⁻² K⁻¹ as everywhere in this package. Conductances are per link, not per unit
area, because that is what a bolted joint, a contactor or a grid edge naturally is (h_c·A for a
joint, k·δ·w/d for a grid edge); the areas turn them into the areal balance the facets already
run. Symmetry is what makes the exchange conservative for *unequal* cells -- ``Σ_i C_i A_i dT_i``
from conduction is exactly zero -- and it is checked at construction, not assumed.

**Why implicit, and why backward Euler.** Conduction is stiff: a 1.2 mm painted panel on steel
equilibrates across itself in a quarter of a second (ADR 0036's factor of 400), a 5 cm aluminium
cell in ~9 s, while the scene tick is 60 s. Explicit stepping would need a tick two orders of
magnitude smaller for a term whose *transient* nobody images. So the surface balance keeps the
midpoint rule (ADR 0036 chose it over Euler on **bias**: the T⁴ term makes explicit Euler inflate
the diurnal swing, the very quantity §6.3's crossover test measures) and the conduction operator
alone is stepped with backward Euler:

    T* = Tⁿ + ½ dt F(Tⁿ)/C,   rhs = Tⁿ + dt F(T*)/C,   (I + dt D⁻¹ L) Tⁿ⁺¹ = rhs

with ``L = diag(K·1) − K`` and ``D = diag(C A)``. Backward Euler is L-stable: a mode with
``dt/τ = 600`` is damped by 1/601 per tick and gone after a few, which is what a stiff mode should
do. Crank–Nicolson is second order but only A-stable, and damps that same mode by −0.997: it rings
for hours. The price of backward Euler is first-order accuracy on the *slow* modes, an
``O(dt/τ_slow)`` error the tests measure rather than assume (2 % at dt = 60 s for τ = 1 h, halving
with the tick). The matrix is constant for a constant tick, so it is factorised once per tick size
and back-substituted every step.

**The bound that was never enforced.** `FacetSolver` had no stability guard at all -- `tick_s` up
to 3600 s parsed, and a leaf at 630 J m⁻² K⁻¹ with a 60 s tick would have diverged quietly during
spin-up. The explicit part now checks §6.4's bound ``2C/(h + 4εσT³)`` at every step against the
forcing's *actual* h and the state's own T, and raises with the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

__all__ = ["ConductionOperator", "explicit_bound_s", "lateral_operator"]


def explicit_bound_s(
    heat_capacity_j_m2_k: Any, h_w_m2_k: Any, emissivity: Any, temperature_k: Any
) -> NDArray[np.float64]:
    """§6.4's explicit step bound, per facet: ``2C/(h + 4εσT³)`` for the midpoint rule.

    The same expression `two_node.py` enforces at construction, evaluated here on the forcing's
    actual ``h`` and the state's own ``T`` rather than on a worst case, because a scene's forcing
    is a callable the field cannot ask for a maximum.
    """
    from irsim.radiometry.constants import SIGMA_SB

    c = np.asarray(heat_capacity_j_m2_k, dtype=np.float64)
    h = np.asarray(h_w_m2_k, dtype=np.float64)
    eps = np.asarray(emissivity, dtype=np.float64)
    t = np.asarray(temperature_k, dtype=np.float64)
    loss = h + 4.0 * eps * SIGMA_SB * t**3
    # A facet that neither convects nor radiates has no explicit time constant to bound.
    with np.errstate(divide="ignore"):
        return np.asarray(np.where(loss > 0.0, 2.0 * c / np.where(loss > 0.0, loss, 1.0), np.inf))


@dataclass(frozen=True)
class ConductionOperator:
    """Symmetric link conductances in W/K between facets, plus the facets' areas.

    ``conductance_w_k`` may be any SciPy sparse matrix or a dense array; it is stored as CSR.
    """

    conductance_w_k: Any
    area_m2: Any

    def __post_init__(self) -> None:
        k = sp.csr_matrix(self.conductance_w_k, dtype=np.float64)
        area = np.asarray(self.area_m2, dtype=np.float64).reshape(-1)
        n = area.shape[0]
        if k.shape != (n, n):
            raise ValueError(f"conductance is {k.shape} but there are {n} facet areas")
        if np.any(area <= 0.0):
            raise ValueError("every facet needs a positive area")
        if k.diagonal().any():
            raise ValueError("a facet cannot conduct to itself: the diagonal must be zero")
        if k.nnz and k.data.min() < 0.0:
            raise ValueError("a conductance cannot be negative")
        asym = abs(k - k.T)
        if asym.nnz and asym.data.max() > 1e-12 * max(1.0, float(abs(k).data.max())):
            raise ValueError(
                "conductances must be symmetric (K_ij = K_ji): heat leaving i for j is heat "
                "arriving at j from i, or the exchange invents energy"
            )
        k.eliminate_zeros()
        object.__setattr__(self, "conductance_w_k", k)
        object.__setattr__(self, "area_m2", area)

    @property
    def n_facets(self) -> int:
        return int(self.area_m2.shape[0])

    @property
    def laplacian(self) -> Any:
        """``L = diag(K·1) − K``, so that the conduction power into facet i is ``−(L T)_i``."""
        k = self.conductance_w_k
        return sp.diags(np.asarray(k.sum(axis=1)).ravel()) - k

    def power_in_w(self, temperatures_k: Any) -> NDArray[np.float64]:
        """Net conduction power into each facet, W: ``Σ_j K_ij (T_j − T_i)``."""
        t = np.asarray(temperatures_k, dtype=np.float64)
        return np.asarray(-(self.laplacian @ t))

    def implicit_matrix(self, heat_capacity_j_m2_k: Any, dt_s: float) -> Any:
        """``I + dt · diag(1/(C A)) · L`` for backward Euler on the conduction term (CSC)."""
        c = np.asarray(heat_capacity_j_m2_k, dtype=np.float64).reshape(-1)
        if c.shape[0] != self.n_facets:
            raise ValueError(f"{c.shape[0]} capacities for {self.n_facets} facets")
        inv = sp.diags(1.0 / (c * self.area_m2))
        return (sp.identity(self.n_facets, format="csc") + dt_s * (inv @ self.laplacian)).tocsc()

    def factorise(self, heat_capacity_j_m2_k: Any, dt_s: float) -> Any:
        """A prefactored solve for one tick size; call once, back-substitute every tick."""
        return spla.splu(self.implicit_matrix(heat_capacity_j_m2_k, dt_s))


def lateral_operator(patch: Any, conductivity_w_mk: float, thickness_m: float) -> Any:
    """In-plane conduction between a patch's four-neighbour cells (PT.11, ADR 0102).

    A grid edge between cells ``du`` apart along ``u`` with a shared side ``dv`` long carries
    ``K = k δ dv / du`` in W/K (and ``k δ du / dv`` along ``v``): Fourier's law across a slab of
    thickness ``δ``, which is the ``k δ w / d`` link the module docstring names. Returns ``None``
    for ``k δ = 0`` -- the operator-free field, bit for bit -- rather than a matrix of zeros.

    Why it matters: over §6.6's 750 s engine-bay rise a signal spreads ``√(α t)`` with
    ``α = k δ / C``: 270 mm in aluminium and 99 mm in steel against 71–150 mm cells, so a metal
    panel without this renders sharper than aluminium can be. Asphalt spreads 16 mm and barely
    notices. Stepped by ADR 0094's backward Euler, a 60 s tick stands where the explicit limit
    ``du² C / (4 k δ)`` is 6 s for 5 cm of aluminium.
    """
    if conductivity_w_mk < 0.0 or thickness_m < 0.0:
        raise ValueError("conductivity and thickness cannot be negative")
    k_delta = float(conductivity_w_mk) * float(thickness_m)
    if k_delta == 0.0:
        return None
    n_u, n_v = int(patch.n_u), int(patch.n_v)
    du, dv = float(patch.du_m), float(patch.dv_m)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    if n_u > 1:
        g_u = k_delta * dv / du
        iv, iu = np.meshgrid(np.arange(n_v), np.arange(n_u - 1), indexing="ij")
        a = (iv * n_u + iu).ravel()
        rows += a.tolist()
        cols += (a + 1).tolist()
        vals += [g_u] * a.size
    if n_v > 1:
        g_v = k_delta * du / dv
        iv, iu = np.meshgrid(np.arange(n_v - 1), np.arange(n_u), indexing="ij")
        a = (iv * n_u + iu).ravel()
        rows += a.tolist()
        cols += (a + n_u).tolist()
        vals += [g_v] * a.size
    n = n_u * n_v
    upper = sp.coo_matrix((np.asarray(vals), (rows, cols)), shape=(n, n))
    return ConductionOperator(upper + upper.T, np.full(n, patch.cell_area_m2))
