"""PH.9 -- steam and droplet plumes: Mie extinction of condensed water (§7.4).

The headline is `test_what_a_lwir_camera_sees_of_steam_is_droplets_not_vapour`. Everything else
here exists to make that measurement trustworthy: the Mie kernel is checked against the *exact*
Rayleigh limit rather than a remembered book value, and the band comparison is made on the two
real camera responses rather than on monochromatic wavelengths.

`PH.9`'s row was written expecting "`cloud.py`'s Mie tables". There are none -- `cloud.py` carries
one measured band ratio and says deriving it from Mie theory is out of scope -- so the efficiency
is computed here from ``data/nk/water.csv``, which is the input Mie theory needs and which covers
0.65-15.6 um, both bands included.
"""

from __future__ import annotations

import numpy as np
import pytest

from irsim.atmosphere.droplets import (
    RHO_WATER_KG_M3,
    droplet_band_kappa_per_m,
    droplet_q_ext,
    water_refractive_index,
)
from irsim.atmosphere.mie import mie_efficiencies
from irsim.pipeline.gas_slab import GasSlab, slab_optical_depth, slab_radiance
from irsim.pipeline.gas_tables import load_gas_tables
from irsim.radiometry.band_integration import band_radiance
from irsim.radiometry.spectral_response import SpectralResponse, load_spectral_response

STEAM_T_K = 373.15


@pytest.fixture(scope="module")
def lwir(tmp_path_factory: pytest.TempPathFactory) -> SpectralResponse:
    p = tmp_path_factory.mktemp("resp") / "lwir.csv"
    p.write_text("# exact top-hat\n7.5,1.0\n13.5,1.0\n")
    return load_spectral_response(p)


@pytest.fixture(scope="module")
def mwir(tmp_path_factory: pytest.TempPathFactory) -> SpectralResponse:
    p = tmp_path_factory.mktemp("resp") / "mwir.csv"
    p.write_text("# exact top-hat\n3.0,1.0\n5.0,1.0\n")
    return load_spectral_response(p)


# --- the Mie kernel ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("x", "m"),
    [(0.01, complex(1.55, 0.0)), (0.02, complex(1.33, 0.0)), (0.01, complex(1.33, 0.05))],
)
def test_mie_reproduces_the_exact_rayleigh_limit(x: float, m: complex) -> None:
    """The oracle is closed form, not a published table value.

    ``Q_sca = (8/3) x^4 |K|^2`` and ``Q_abs = 4 x Im K`` with ``K = (m^2-1)/(m^2+2)`` are exact as
    ``x -> 0``, and they pin the ``a_1``/``b_1`` coefficients and the ``2/x^2`` normalisation --
    which between them are what the whole series is built from. The absorbing case is included
    deliberately: the downward recursion for the logarithmic derivative exists for it, and an
    implementation that recurred upward would pass the two transparent cases and fail this one.
    """
    k = (m * m - 1.0) / (m * m + 2.0)
    q_ext, q_sca = mie_efficiencies(x, m)
    assert q_sca == pytest.approx(8.0 / 3.0 * x**4 * abs(k) ** 2, rel=2e-4)
    assert q_ext - q_sca == pytest.approx(4.0 * x * k.imag, rel=2e-4, abs=1e-12)


def test_a_transparent_sphere_absorbs_nothing_and_a_large_one_extincts_two() -> None:
    """Two limits that hold for every refractive index, so a sign error cannot hide in them."""
    q_ext, q_sca = mie_efficiencies(5.213, complex(1.55, 0.0))
    assert q_ext == pytest.approx(q_sca, rel=1e-12), "k = 0 means Q_abs is identically zero"
    assert mie_efficiencies(400.0, complex(1.33, 0.0))[0] == pytest.approx(2.0, abs=0.05)
    for x in (0.5, 2.0, 8.0, 30.0):
        q_ext, q_sca = mie_efficiencies(x, complex(1.33, 0.05))
        assert q_sca <= q_ext + 1e-12, "scattering cannot exceed extinction"
        assert q_ext - q_sca >= -1e-12, "absorption cannot be negative"


def test_water_is_absorbing_across_both_infrared_bands() -> None:
    """The premise of the whole module: ``k`` is not negligible where these cameras look."""
    lam = np.array([4.0, 10.0, 12.0])
    m = water_refractive_index(lam)
    assert np.all(m.imag > 0.0)
    assert m.imag[1] > m.imag[0], "water absorbs harder in LWIR than in MWIR"


# --- the band comparison PH.9 asks for --------------------------------------------------------


def test_a_small_droplet_extincts_harder_in_mwir_than_in_lwir(lwir, mwir) -> None:
    """`PH.9`'s criterion, and the size range in which it is the right one to state.

    A 2 um droplet is ``x = 3`` against MWIR and ``x = 1`` against LWIR: the shorter-wave camera is
    already climbing toward the geometric limit while the longer-wave one is still in the Rayleigh
    tail, so the same water is several times more opaque to it. Freshly condensed steam is in
    exactly this range, which is why the criterion is worth having.
    """
    lwc = 1e-3
    for radius_um in (2.0, 5.0):
        beta_mwir = droplet_band_kappa_per_m(mwir, lwc, radius_um, STEAM_T_K)
        beta_lwir = droplet_band_kappa_per_m(lwir, lwc, radius_um, STEAM_T_K)
        assert beta_mwir > 2.0 * beta_lwir, f"r = {radius_um} um: {beta_mwir} vs {beta_lwir}"


def test_the_two_bands_converge_once_both_are_geometric(lwir, mwir) -> None:
    """And the limit of that criterion, asserted rather than left to be discovered.

    By 20 um both bands are past ``x = 12`` and ``Q_ext`` has settled near 2 in each, so the ratio
    goes to one. A reader who took the previous test as "MWIR is always more opaque" would size a
    plume's contrast wrongly for anything but the finest droplets.
    """
    ratio = droplet_band_kappa_per_m(mwir, 1e-3, 20.0, STEAM_T_K) / droplet_band_kappa_per_m(
        lwir, 1e-3, 20.0, STEAM_T_K
    )
    assert 0.9 < ratio < 1.1, f"the bands should agree on a large droplet, got {ratio:.3f}"


def test_extinction_goes_as_one_over_radius_at_fixed_water_content(lwir) -> None:
    """The counter-intuitive identity, checked where ``Q_ext`` is flat so only ``1/r`` is left.

    ``beta = 3 LWC Q_ext / (4 rho r)``: the same kilogram of water is *more* opaque spread over
    smaller droplets. A plume does not clear as it condenses further, and anyone reaching for
    "bigger droplets, thicker plume" has the sign backwards.
    """
    big, bigger = 20.0, 40.0
    q_ratio = droplet_q_ext(bigger, lwir, STEAM_T_K) / droplet_q_ext(big, lwir, STEAM_T_K)
    beta_ratio = droplet_band_kappa_per_m(lwir, 1e-3, bigger, STEAM_T_K) / (
        droplet_band_kappa_per_m(lwir, 1e-3, big, STEAM_T_K)
    )
    assert beta_ratio == pytest.approx(0.5 * q_ratio, rel=1e-9)
    assert beta_ratio < 0.6, "doubling the radius roughly halves the extinction"


def test_the_identity_matches_a_hand_built_droplet_population(lwir) -> None:
    """``beta = N pi r^2 Q_ext`` from first principles, against the LWC form the module uses."""
    radius_um, lwc = 5.0, 2e-3
    radius_m = radius_um * 1e-6
    number_density = lwc / (4.0 / 3.0 * np.pi * radius_m**3 * RHO_WATER_KG_M3)
    expected = number_density * np.pi * radius_m**2 * droplet_q_ext(radius_um, lwir, STEAM_T_K)
    assert droplet_band_kappa_per_m(lwir, lwc, radius_um, STEAM_T_K) == pytest.approx(
        expected, rel=1e-12
    )


# --- what the camera actually sees ------------------------------------------------------------


def test_what_a_lwir_camera_sees_of_steam_is_droplets_not_vapour(lwir) -> None:
    """The headline. Water vapour is nearly transparent through the window; the droplets are not.

    `PH.9`'s row quotes NIRATAM for ``tau_LWIR >= 0.9`` on a pure-gas slab without naming a path,
    and the path is what decides it: at 373 K and a full atmosphere of H2O -- saturated *pure*
    steam, the most opaque vapour there can be -- the measured transmittance is 0.9 only inside
    about half a metre and falls to 0.84 over a metre. So the criterion is stated here as the
    comparison it was standing in for, which is also the one a scene author needs: at a realistic
    plume's water loading the condensed phase dominates the vapour several times over.
    """
    tables = load_gas_tables("lwir")
    vapour = slab_optical_depth(
        GasSlab(t_gas_k=STEAM_T_K, length_m=1.0, p_h2o_atm=1.0), lwir, tables
    )
    assert vapour < 0.2, f"pure steam over a metre must still be a thin absorber, got {vapour:.3f}"

    droplets = slab_optical_depth(
        GasSlab(t_gas_k=STEAM_T_K, length_m=1.0, lwc_kg_m3=5e-3, droplet_radius_um=5.0), lwir
    )
    assert droplets > 4.0 * vapour, f"droplets {droplets:.3f} vs vapour {vapour:.3f}"

    # and the crossover, so the claim carries a number rather than a direction
    per_g = slab_optical_depth(
        GasSlab(t_gas_k=STEAM_T_K, length_m=1.0, lwc_kg_m3=1e-3, droplet_radius_um=5.0), lwir
    )
    crossover_g_m3 = vapour / per_g
    assert 0.5 < crossover_g_m3 < 1.5, f"they should match near 1 g/m^3, got {crossover_g_m3:.2f}"


def test_a_plume_never_looks_hotter_than_its_droplets(lwir) -> None:
    """``tau L_behind + (1 - tau) B(T)`` cannot exceed ``B(T)`` when the background is cooler.

    The bound is what makes a plume safe to composite: however thick it is made, it saturates at
    its own temperature instead of running away. Asserted across two decades of water content,
    because the interesting failure would be an extinction that overshot and produced emission
    above the Planck ceiling.
    """
    ceiling = float(band_radiance(lwir, STEAM_T_K)[0])
    behind = float(band_radiance(lwir, 290.0)[0])
    assert behind < ceiling
    previous = behind
    for lwc in (1e-4, 1e-3, 1e-2, 1e-1):
        slab = GasSlab(t_gas_k=STEAM_T_K, length_m=1.0, lwc_kg_m3=lwc, droplet_radius_um=5.0)
        value = float(slab_radiance(slab, lwir, behind))
        assert behind <= value <= ceiling, f"LWC {lwc}: {value} outside [{behind}, {ceiling}]"
        assert value >= previous, "more water can only bring it closer to the droplet temperature"
        previous = value
    assert previous == pytest.approx(ceiling, rel=1e-3), "a thick plume reaches its own temperature"


# --- what the slab refuses --------------------------------------------------------------------


def test_liquid_water_needs_a_radius_and_refuses_nonsense() -> None:
    with pytest.raises(ValueError, match="needs a droplet radius"):
        GasSlab(t_gas_k=STEAM_T_K, length_m=1.0, lwc_kg_m3=1e-3)
    with pytest.raises(ValueError, match="cannot be negative"):
        GasSlab(t_gas_k=STEAM_T_K, length_m=1.0, lwc_kg_m3=-1e-3, droplet_radius_um=5.0)
    with pytest.raises(ValueError, match="empty air"):
        GasSlab(t_gas_k=STEAM_T_K, length_m=1.0)


def test_a_droplet_only_slab_is_allowed_below_the_gas_tables_floor() -> None:
    """Fog is 285 K and needs no absorption table, so the 300 K floor does not apply to it."""
    fog = GasSlab(t_gas_k=285.0, length_m=50.0, lwc_kg_m3=3e-4, droplet_radius_um=8.0)
    assert fog.droplets_only
    with pytest.raises(ValueError, match="outside"):
        GasSlab(t_gas_k=285.0, length_m=1.0, p_h2o_atm=0.1)  # gas at 285 K still refused
    with pytest.raises(ValueError, match="outside"):
        GasSlab(t_gas_k=250.0, length_m=50.0, lwc_kg_m3=3e-4, droplet_radius_um=8.0)  # ice
