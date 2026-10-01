"""Mie extinction / absorption of rain over Marshall-Palmer and gamma DSDs.

Research check for research_precip.md. Plain NumPy (no scipy). BHMIE-style
(Bohren & Huffman 1983, App. A) with downward recurrence for D_n.
Water n,k: Segelstein 1981 (refractiveindex.info database copy); 0 C liquid
from Rowe et al. 2020 where available.
"""
import math
import numpy as np

SRC = "./nk/"  # refractiveindex.info YAMLs (h2o_Segelstein.yml, ice_Warren-2008.yml, ice_Rowe-273K.yml), not committed


def load(fn):
    s = open(SRC + fn).read()
    i = s.find("data: |")
    rows = []
    for line in s[i:].split("\n")[1:]:
        p = line.split()
        if len(p) >= 3:
            try:
                rows.append([float(x) for x in p[:3]])
            except ValueError:
                break
        elif not p:
            continue
        else:
            break
    return np.array(rows)


def nk(tab, wl):
    n = np.interp(wl, tab[:, 0], tab[:, 1])
    k = math.exp(np.interp(wl, tab[:, 0], np.log(tab[:, 2])))
    return complex(n, k)


def mie_q(m, x):
    """Return Qext, Qsca, g for sphere of complex index m (n+ik), size param x."""
    nstop = int(x + 4.0 * x ** (1.0 / 3.0) + 2.0)
    mx = m * x
    nmx = int(max(nstop, abs(mx))) + 16
    d = np.zeros(nmx + 1, dtype=complex)
    for n in range(nmx, 0, -1):
        en = n / mx
        d[n - 1] = en - 1.0 / (d[n] + en)
    psi0, psi1 = math.cos(x), math.sin(x)
    chi0, chi1 = -math.sin(x), math.cos(x)
    xi1 = complex(psi1, -chi1)
    qext = qsca = gsum = 0.0
    an1 = bn1 = 0j
    for n in range(1, nstop + 1):
        fn = (2.0 * n + 1.0) / (n * (n + 1.0))
        psi = (2.0 * n - 1.0) * psi1 / x - psi0
        chi = (2.0 * n - 1.0) * chi1 / x - chi0
        xi = complex(psi, -chi)
        da = d[n] / m + n / x
        db = d[n] * m + n / x
        an = (da * psi - psi1) / (da * xi - xi1)
        bn = (db * psi - psi1) / (db * xi - xi1)
        qext += (2 * n + 1) * (an.real + bn.real)
        qsca += (2 * n + 1) * (abs(an) ** 2 + abs(bn) ** 2)
        gsum += fn * (an * bn.conjugate()).real
        if n > 1:
            gsum += ((n - 1.0) * (n + 1.0) / n) * (an1 * an.conjugate() + bn1 * bn.conjugate()).real
        an1, bn1 = an, bn
        psi0, psi1 = psi1, psi
        chi0, chi1 = chi1, chi
        xi1 = complex(psi1, -chi1)
    qext *= 2.0 / x**2
    qsca *= 2.0 / x**2
    g = 2.0 * gsum / (qsca * x**2 / 2.0) / x**2 * 2.0 if qsca > 0 else 0.0
    # g = (4/x^2) * gsum / Qsca
    g = 4.0 * gsum / (x**2 * qsca)
    return qext, qsca, g


def airy_encircled(x, theta):
    """Fraction of Fraunhofer-diffracted power inside half-angle theta (rad).

    E = 1 - J0(x sin th)^2 - J1(x sin th)^2 (Born & Wolf; Airy). Bessel via series/asymptotic.
    """
    u = x * math.sin(theta)
    return 1.0 - j0(u) ** 2 - j1(u) ** 2


def j0(u):
    if u < 8.0:
        s, t, k = 0.0, 1.0, 0
        while abs(t) > 1e-17 or k < 5:
            s += t
            k += 1
            t *= -(u * u / 4.0) / (k * k)
        return s
    w = u - math.pi / 4
    return math.sqrt(2 / (math.pi * u)) * (math.cos(w) * (1 - 9 / (128 * u * u)) + math.sin(w) / (8 * u))


def j1(u):
    if u < 8.0:
        s, t, k = 0.0, u / 2.0, 0
        while abs(t) > 1e-17 or k < 5:
            s += t
            k += 1
            t *= -(u * u / 4.0) / (k * (k + 1))
        return s
    w = u - 3 * math.pi / 4
    return math.sqrt(2 / (math.pi * u)) * (math.cos(w) * (1 + 15 / (128 * u * u)) - 3 * math.sin(w) / (8 * u))


if __name__ == "__main__":
    seg = load("h2o_Segelstein.yml")
    # sanity: known large-sphere Mie: m=1.33, x=1000 -> Qext ~ 2.0
    print("sanity m=1.33+0j x=1000:", mie_q(complex(1.33, 1e-8), 1000.0))
    print("sanity m=1.5+0j x=10 (BH: Qext=2.8820)", mie_q(complex(1.5, 0.0), 10.0))
    D = np.concatenate([np.arange(0.1, 1.0, 0.05), np.arange(1.0, 6.01, 0.1)])  # mm
    dD = np.gradient(D)
    wls = [0.55, 0.85, 1.06, 1.55, 2.2, 3.5, 3.8, 4.0, 4.6, 8.0, 10.0, 11.0, 12.0]
    res = {}
    for wl in wls:
        m = nk(seg, wl)
        qe, qs, gg = [], [], []
        for Dm in D:
            x = math.pi * Dm * 1e3 / wl
            a, b, c = mie_q(m, x)
            qe.append(a)
            qs.append(b)
            gg.append(c)
        res[wl] = (m, np.array(qe), np.array(qs), np.array(gg))
        print(f"wl={wl}: m={m:.4f}  D=0.5mm Qe={qe[8]:.4f} Qa={qe[8]-qs[8]:.4f} g={gg[8]:.4f}; D=2mm Qe={qe[np.argmin(abs(D-2))]:.4f} Qa={qe[np.argmin(abs(D-2))]-qs[np.argmin(abs(D-2))]:.4f}", flush=True)
    np.save(SRC + "../mie_res.npy", {"D": D, "res": res}, allow_pickle=True)
    # Marshall-Palmer: N(D)=N0 exp(-L D), N0=8000 m^-3 mm^-1, L=4.1 R^-0.21 mm^-1
    print("\nMarshall-Palmer, truncated 0.1-6 mm:")
    print("R mm/h | wl um | sigma_ext km^-1 | dB/km | geo(Q=2,0-inf) dB/km | w0=Qs/Qe | abs frac | g")
    for R in [1, 5, 10, 25, 50, 100]:
        L = 4.1 * R ** -0.21
        N = 8000.0 * np.exp(-L * D)  # m^-3 mm^-1
        A = math.pi / 4 * (D * 1e-3) ** 2  # m^2
        geo = math.pi * 8000.0 * 1e-6 / L**3 * 1e3  # km^-1 for Q=2, D 0..inf
        for wl in wls:
            m, qe, qs, gg = res[wl]
            se = np.sum(qe * A * N * dD) * 1e3
            ss = np.sum(qs * A * N * dD) * 1e3
            gm = np.sum(gg * qs * A * N * dD) * 1e3 / ss
            print(f"{R:5d} | {wl:5.2f} | {se:7.4f} | {4.343*se:7.3f} | {4.343*geo:7.3f} | {ss/se:.3f} | {1-ss/se:.3f} | {gm:.3f}")
    # Diffraction lobe kept inside a cone: MP R=10 mm/h, Q=2 split: 1 diffraction
    print("\nFraction of diffracted power within half-angle theta, MP DSD (area-weighted):")
    for R in [2, 10, 50]:
        L = 4.1 * R ** -0.21
        N = 8000.0 * np.exp(-L * D)
        A = math.pi / 4 * (D * 1e-3) ** 2
        w = A * N * dD
        for wl in [0.55, 1.0, 1.55, 4.0, 10.0]:
            row = []
            for th_mrad in [0.25, 0.5, 1.0, 2.0, 5.0]:
                e = np.array([airy_encircled(math.pi * Dm * 1e3 / wl, th_mrad * 1e-3) for Dm in D])
                row.append(np.sum(e * w) / np.sum(w))
            print(f"R={R:3d} wl={wl:5.2f}: " + "  ".join(f"th={t}mrad:{v:.2f}" for t, v in zip([0.25, 0.5, 1, 2, 5], row)))
