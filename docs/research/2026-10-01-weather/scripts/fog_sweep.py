"""Mie extinction of the Shettle & Fenn (1979) fog models, per band, relative to 0.55 um.

Scratch computation for the literature note. Uses the repo's BHMIE port and Segelstein water n/k
(0.65-15.6 um); 0.55 um uses Hale & Querry 1973 n=1.333, k=1.96e-9.
"""

import sys

import numpy as np

import importlib.util
_spec = importlib.util.spec_from_file_location("mie", "/home/hunter/irsim-scaffold/src/irsim/atmosphere/mie.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
mie_efficiencies = _mod.mie_efficiencies

_rows = [l for l in open("/home/hunter/irsim-scaffold/data/nk/water.csv") if l.strip() and not l.startswith("#") and l[0].isdigit()]
wl_tab, n_tab, k_tab = np.array([[float(v) for v in l.split(",")[:3]] for l in _rows]).T


def m_water(lam):
    if lam < 0.65:
        return complex(1.333, 1.96e-9)
    return complex(np.interp(lam, wl_tab, n_tab), np.interp(lam, wl_tab, k_tab))


# (name, alpha, b, gamma, N cm^-3); A normalised to N here (see note on the A column).
MODELS = {}
for reff in [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]:
    b = 9.0 / reff  # alpha=6, gamma=1 family: r_eff = (alpha+3)/b, r_mode = 6/b
    MODELS[f"alpha=6 gamma=1, r_eff={reff}um, r_mode={6/b:.2f}um"] = (6, b, 1.0, 100.0)

r = np.geomspace(0.02, 120.0, 260)
lnr = np.log(r)

bands = {
    "0.55": [0.55],
    "NIR 0.7-1.0": list(np.linspace(0.70, 1.0, 7)),
    "SWIR 0.9-1.7": list(np.linspace(0.9, 1.7, 9)),
    "MWIR 3-5": list(np.linspace(3.0, 5.0, 11)),
    "LWIR 8-14": list(np.linspace(8.0, 14.0, 13)),
    "10.0": [10.0],
}

qcache = {}


def qext(rad, lam):
    key = (round(rad, 6), round(lam, 4))
    if key not in qcache:
        x = 2 * np.pi * rad / lam
        qcache[key] = mie_efficiencies(x, m_water(lam))[0]
    return qcache[key]


for name, (alpha, b, g, N) in MODELS.items():
    shape = r**alpha * np.exp(-b * r**g)
    A = N / np.trapezoid(shape * r, lnr)
    nr = A * shape  # cm^-3 um^-1
    m2 = np.trapezoid(nr * r**2 * r, lnr)
    m3 = np.trapezoid(nr * r**3 * r, lnr)
    reff = m3 / m2
    lwc_g_m3 = 4 / 3 * np.pi * m3 * 1e-12 * 1e6  # um^3 cm^-3 -> g m^-3 (rho=1 g/cm3)
    out = {}
    for bn, lams in bands.items():
        vals = []
        for lam in lams:
            q = np.array([qext(ri, lam) for ri in r])
            beta = np.trapezoid(np.pi * r**2 * q * nr * r, lnr) * 1e-3  # km^-1
            vals.append(beta)
        out[bn] = float(np.mean(vals))
    b55 = out["0.55"]
    print(f"{name}: A={A:.5g}  r_eff={reff:.2f} um  LWC={lwc_g_m3:.3f} g/m3  "
          f"beta055={b55:.2f}/km  V={3.912 / b55 * 1000:.0f} m")
    for bn in bands:
        print(f"   {bn:14s} beta={out[bn]:8.3f}/km  ratio={out[bn] / b55:.3f}")
    sys.stdout.flush()
