"""Overview plots of the engine-free physics core: radiometry, atmosphere, materials, optics.

Every number on these figures comes from ``irsim`` itself (no re-implemented physics), so the
plots are a picture of what the simulator currently computes, ESTIMATED values included.

    python scripts/plot_physics_overview.py [--out DIR]   # default: outputs/physics_overview/

Sections: docs/physics-model.md §3 (Planck, band integration), §4 (materials, Fresnel),
§7 (Beer-Lambert atmosphere), §8 (aperture factor), §12.1 (bands, spectral responses).
"""

from __future__ import annotations

import argparse
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from irsim.atmosphere.extinction import transmittance_per_band
from irsim.atmosphere.library import load_atmosphere_preset
from irsim.config.bands import NOMINAL_RANGES_UM
from irsim.config.loader import resolve_data_dir
from irsim.materials.library import MaterialLibrary, nominal_response
from irsim.materials.nk import band_directional_emissivity, load_nk_table
from irsim.optics.aperture import aperture_factor
from irsim.radiometry.band_integration import band_radiance, d_band_radiance_dT
from irsim.radiometry.planck import spectral_radiance
from irsim.radiometry.spectral_response import load_spectral_response

BANDS = ("nir", "swir", "mwir", "lwir")
# Categorical slots 1-4 (validated default palette), fixed per band on every figure.
BAND_COLOR = {"nir": "#2a78d6", "swir": "#eb6834", "mwir": "#1baf7a", "lwir": "#eda100"}
BAND_LABEL = {"nir": "NIR", "swir": "SWIR", "mwir": "MWIR", "lwir": "LWIR"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

# Reference weather each preset was checked against (the presets' own provenance notes).
PRESET_WEATHER = {
    "us_standard_clear": (288.15, 0.46, 23_000.0),
    "midlat_winter_dry": (272.2, 0.76, 23_000.0),
    "midlat_summer_humid": (303.15, 0.80, 23_000.0),
    "tropical": (300.0, 0.74, 23_000.0),
    "haze": (288.15, 0.46, 1_500.0),
    "fog_light_200m": (283.15, 1.0, 200.0),
    "fog_dense_50m": (283.15, 1.0, 50.0),
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def _shade_bands(ax: plt.Axes) -> None:
    for b in BANDS:
        lo, hi = NOMINAL_RANGES_UM[b]
        ax.axvspan(lo, hi, color=BAND_COLOR[b], alpha=0.12, lw=0)
        ax.text(
            math.sqrt(lo * hi),
            1.0,
            BAND_LABEL[b],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            color=INK2,
            fontsize=9,
        )


def fig_planck(out: pathlib.Path) -> None:
    """§3.1: spectral radiance for scene-relevant temperatures, nominal bands shaded."""
    lam = np.geomspace(0.5, 25.0, 1200)
    temps = (250.0, 300.0, 400.0, 600.0, 1000.0)
    blues = ("#b7d3f2", "#86b4ea", "#5893de", "#2a78d6", "#174a8a")  # one hue, light -> dark
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for t, c in zip(temps, blues, strict=True):
        ax.plot(lam, spectral_radiance(lam, np.full_like(lam, t)), color=c)
        i = int(np.argmax(spectral_radiance(lam, np.full_like(lam, t))))
        ax.annotate(
            f"{t:.0f} K",
            (lam[i], spectral_radiance(lam[i], t)),
            xytext=(4, 4),
            textcoords="offset points",
            color=INK2,
            fontsize=9,
        )
    _shade_bands(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1e5)
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Spectral radiance (W m⁻² sr⁻¹ µm⁻¹)")
    ax.set_title("Planck spectral radiance, with the four camera bands", pad=18)
    fig.tight_layout()
    fig.savefig(out / "01_planck_bands.png", dpi=150)
    plt.close(fig)


def fig_band_radiance(out: pathlib.Path) -> None:
    """§3.2-3.4: band radiance and relative thermal contrast (1/L)(dL/dT) per band."""
    t = np.linspace(230.0, 1000.0, 400)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for b in BANDS:
        r = nominal_response(b)
        lb = band_radiance(r, t)
        dl = d_band_radiance_dT(r, t)
        a1.plot(t, lb, color=BAND_COLOR[b], label=BAND_LABEL[b])
        a2.plot(t, 100.0 * dl / lb, color=BAND_COLOR[b], label=BAND_LABEL[b])
    a1.set_yscale("log")
    a1.set_xlabel("Blackbody temperature (K)")
    a1.set_ylabel("Band radiance (W m⁻² sr⁻¹)")
    a1.set_title("In-band radiance (nominal top-hat bands)")
    a2.set_yscale("log")
    a2.set_xlabel("Blackbody temperature (K)")
    a2.set_ylabel("(1/L) dL/dT  (% per K)")
    a2.set_title("Relative thermal contrast")
    for a in (a1, a2):
        a.axvline(300.0, color=INK2, lw=0.8, ls="--")
        a.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out / "02_band_radiance_contrast.png", dpi=150)
    plt.close(fig)


def fig_responses(out: pathlib.Path) -> None:
    """§12.1: the detector spectral responses shipped in data/spectra/responses."""
    root = resolve_data_dir() / "spectra" / "responses"
    files = {
        "nir_si": ("Si (NIR)", "#2a78d6"),
        "ingaas": ("InGaAs (SWIR)", "#eb6834"),
        "insb": ("InSb (MWIR)", "#1baf7a"),
        "insb_flame_window": ("InSb flame window", "#e87ba4"),
        "boson_vox": ("VOx bolometer (LWIR)", "#eda100"),
    }
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for stem, (label, color) in files.items():
        r = load_spectral_response(root / f"{stem}.csv")
        ax.plot(r.wavelength_um, r.response, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 3, 5, 8, 10, 14, 20])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Relative response (peak = 1)")
    ax.set_title("Detector spectral responses (ESTIMATED curves)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=5, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "03_spectral_responses.png", dpi=150)
    plt.close(fig)


def fig_atmosphere(out: pathlib.Path) -> None:
    """§7.2: Beer-Lambert band transmittance vs range, each preset at its reference weather."""
    d = np.geomspace(10.0, 20_000.0, 300)
    presets = list(PRESET_WEATHER)
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.6), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, presets, strict=False):
        p = load_atmosphere_preset(name)
        t_air, rh, vis = PRESET_WEATHER[name]
        for b in BANDS:
            tau = [transmittance_per_band(p, t_air, rh, vis, float(x))[b] for x in d]
            ax.plot(d, tau, color=BAND_COLOR[b], label=BAND_LABEL[b])
        ax.set_xscale("log")
        ax.set_ylim(0, 1.02)
        vis_s = f"{vis / 1000:g} km" if vis >= 1000 else f"{vis:g} m"
        ax.set_title(f"{name}\n{t_air - 273.15:.0f} °C, RH {rh:.0%}, V {vis_s}", fontsize=10)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", fontsize=11, title="Band")
    for ax in axes[1]:
        ax.set_xlabel("Range (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Band transmittance τ")
    fig.suptitle(
        "Atmospheric transmittance by preset", x=0.01, ha="left", fontweight="bold", color=INK
    )
    fig.tight_layout()
    fig.savefig(out / "04_atmosphere_transmittance.png", dpi=150)
    plt.close(fig)


def fig_materials(out: pathlib.Path) -> None:
    """§4: band emissivity of every library material, LWIR vs MWIR (Kirchhoff-closed)."""
    lib = MaterialLibrary.load()
    rows = []
    for name in lib.names:
        m = lib[name]
        vals = {}
        for b in ("lwir", "mwir"):
            try:
                vals[b] = m.band_properties(b).emissivity
            except (KeyError, ValueError):
                vals[b] = np.nan
        rows.append((name, vals["lwir"], vals["mwir"]))
    rows = [r for r in rows if not np.isnan(r[1])]
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8.5, 0.22 * len(rows) + 1.2))
    ax.hlines(
        y,
        [min(r[1], r[2]) if not np.isnan(r[2]) else r[1] for r in rows],
        [max(r[1], r[2]) if not np.isnan(r[2]) else r[1] for r in rows],
        color=GRID,
        lw=2,
    )
    ax.scatter(
        [r[2] for r in rows],
        y,
        s=26,
        color=BAND_COLOR["mwir"],
        label="MWIR",
        zorder=3,
        edgecolor=SURFACE,
        linewidth=1,
    )
    ax.scatter(
        [r[1] for r in rows],
        y,
        s=26,
        color=BAND_COLOR["lwir"],
        label="LWIR",
        zorder=3,
        edgecolor=SURFACE,
        linewidth=1,
    )
    ax.set_yticks(y, names, fontsize=7.5)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-1, len(rows))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Band emissivity ε (nominal band, 300 K weighting)")
    ax.set_title(f"Material library: emissivity, {len(rows)} materials")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "05_material_emissivity.png", dpi=150)
    plt.close(fig)


def fig_fresnel(out: pathlib.Path) -> None:
    """§4.2: LWIR directional emissivity ε(θ) from Fresnel on tabulated n/k."""
    theta = np.linspace(0.0, 89.0, 90)
    mu = np.cos(np.radians(theta))
    picks = {
        "water": ("Water", "#2a78d6"),
        "glass": ("Glass", "#eb6834"),
        "paint_proxy": ("Paint (proxy)", "#1baf7a"),
        "aluminium": ("Aluminium", "#eda100"),
    }
    resp = nominal_response("lwir")
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for key, (label, color) in picks.items():
        try:
            eps = band_directional_emissivity(load_nk_table(key), resp, mu)
        except ValueError as e:
            print(f"skip {key}: {e}")
            continue
        ax.plot(theta, eps, color=color, label=label)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(0, 91, 15))
    ax.set_xlabel("View angle from surface normal θ (deg)")
    ax.set_ylabel("Directional emissivity ε(θ), LWIR")
    ax.set_title("Fresnel angular emissivity (grazing views reflect the sky)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "06_fresnel_angular_emissivity.png", dpi=150)
    plt.close(fig)


def fig_aperture(out: pathlib.Path) -> None:
    """§8 / CLAUDE.md #5: π/(4F²+1) against the paraxial π/(4F²)."""
    f = np.linspace(0.8, 4.0, 200)
    exact = np.array([aperture_factor(x) for x in f])
    paraxial = math.pi / (4 * f**2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(f, 100 * (paraxial / exact - 1.0), color="#2a78d6")
    for fn in (1.0, 1.4, 2.0):
        v = 100 * (math.pi / (4 * fn**2) / aperture_factor(fn) - 1)
        ax.plot(fn, v, "o", ms=7, color="#2a78d6", mec=SURFACE, mew=2)
        ax.annotate(
            f"F/{fn:g}: {v:.1f} %",
            (fn, v),
            xytext=(8, 2),
            textcoords="offset points",
            color=INK2,
            fontsize=9,
        )
    ax.set_xlabel("F-number")
    ax.set_ylabel("Paraxial over-estimate of FPA irradiance (%)")
    ax.set_title("Why the aperture factor is π/(4F²+1), not π/(4F²)")
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(out / "07_aperture_factor.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("outputs/physics_overview"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    _style()
    for fn in (
        fig_planck,
        fig_band_radiance,
        fig_responses,
        fig_atmosphere,
        fig_materials,
        fig_fresnel,
        fig_aperture,
    ):
        fn(args.out)
        print(f"done: {fn.__name__}")


if __name__ == "__main__":
    main()
